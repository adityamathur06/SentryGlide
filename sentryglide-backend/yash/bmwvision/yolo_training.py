"""Train and evaluate the YOLOv8 classifier architecture supplied in the project notebook."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np


MODEL_YAML = """# Adapted from yolov8-n-5-fold.ipynb supplied with the dataset.
nc: 23
ch: 3
backbone:
  - [-1, 1, Conv, [32, 3, 2]]
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 3, Bottleneck, [64]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, Bottleneck, [128]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 1, SPPF, [256]]
head:
  - [-1, 1, Classify, [nc]]
"""


def _metrics(confusion: np.ndarray, names: list[str]) -> dict:
    per_class = {}
    for i, name in enumerate(names):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": int(confusion[i].sum()),
        }
    total = int(confusion.sum())
    return {"accuracy": float(np.trace(confusion) / total) if total else 0.0,
            "classes": names, "per_class": per_class, "confusion_matrix": confusion.tolist()}


def train_yolo(dataset: str | Path, out: str | Path, epochs: int = 15,
               image_size: int = 224, batch_size: int = 32, seed: int = 42) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install training dependencies with: pip install -e ".[yolo]"') from exc

    dataset, out = Path(dataset), Path(out)
    split_root = dataset / "splits"
    for split in ("train", "val", "test"):
        if not (split_root / split).is_dir():
            raise FileNotFoundError(split_root / split)
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / "yolov8n_medical.yaml"
    yaml_path.write_text(MODEL_YAML, encoding="utf-8")

    model = YOLO(str(yaml_path), task="classify")
    run = model.train(
        data=str(split_root), epochs=epochs, imgsz=image_size, batch=batch_size,
        device="cpu", workers=0, seed=seed, deterministic=True, pretrained=False,
        project=str(out / "runs"), name="medical_classifier", exist_ok=True,
        plots=True, verbose=True,
    )
    best_source = Path(run.save_dir) / "weights" / "best.pt"
    if not best_source.exists():
        raise RuntimeError(f"Training did not produce {best_source}")
    best_path = out / "yolov8_medical_best.pt"
    shutil.copy2(best_source, best_path)

    trained = YOLO(str(best_path), task="classify")
    names_map = trained.names
    names = [str(names_map[i]) for i in range(len(names_map))]
    name_to_index = {name: index for index, name in enumerate(names)}
    paths: list[Path] = []
    labels: list[int] = []
    for class_dir in sorted((split_root / "test").iterdir()):
        if not class_dir.is_dir() or class_dir.name not in name_to_index:
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                paths.append(path)
                labels.append(name_to_index[class_dir.name])

    confusion = np.zeros((len(names), len(names)), dtype=int)
    confidences: list[float] = []
    predictions: list[int] = []
    stream = trained.predict(source=[str(path) for path in paths], imgsz=image_size,
                             batch=batch_size, device="cpu", verbose=False, stream=True)
    for true_label, result in zip(labels, stream):
        pred = int(result.probs.top1)
        predictions.append(pred)
        confidences.append(float(result.probs.top1conf))
        confusion[true_label, pred] += 1

    report = _metrics(confusion, names)
    report.update({"model": str(best_path.resolve()), "dataset": str(dataset.resolve()),
                   "test_images": len(paths), "epochs": epochs, "image_size": image_size,
                   "architecture_source": "yolov8-n-5-fold.ipynb",
                   "evaluation_split": "source-group-isolated test split"})
    (out / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream_file:
        writer = csv.writer(stream_file)
        writer.writerow(["true \\ predicted", *names])
        for name, row in zip(names, confusion):
            writer.writerow([name, *row.tolist()])

    import matplotlib.pyplot as plt
    size = max(12, len(names) * 0.68)
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    image = ax.imshow(confusion, cmap="YlGn", interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
           xticklabels=names, yticklabels=names, xlabel="Predicted label", ylabel="True label",
           title=f"YOLOv8 medical-waste error matrix - accuracy {report['accuracy']:.1%}")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor")
    threshold = confusion.max() / 2 if confusion.size else 0
    for i in range(len(names)):
        for j in range(len(names)):
            value = int(confusion[i, j])
            if value:
                ax.text(j, i, str(value), ha="center", va="center",
                        color="white" if value > threshold else "#17352c", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {"model_type": "ultralytics_yolov8_classification", "weights": best_path.name,
                "type_names": names, "input_size": [image_size, image_size],
                "minimum_confidence": 0.45, "test_accuracy": report["accuracy"]}
    (out / "yolo_model.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return report

