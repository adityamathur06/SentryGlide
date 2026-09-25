"""Dataset splitting, training, evaluation, and ONNX export.

The split unit is a physical object, never a frame or placement.  This prevents images of
the same specimen leaking into both training and evaluation.  The exported network follows
the input/output contract documented in :mod:`bmwvision.classifier`.
"""
from __future__ import annotations

import csv
import copy
import json
import random
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .classifier import KnnOOD, ModelMeta
from .features import GEOM_FEATURE_NAMES


def read_manifest(path: str | Path, type_names: list[str]) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    rows = list(csv.DictReader(path.open(newline="")))
    rows = [r for r in rows if r.get("type") in type_names and r.get("occupancy") == "ONE"]
    if not rows:
        raise ValueError("Manifest contains no single-item classifier samples")
    missing = sorted(set(type_names) - {r["type"] for r in rows})
    if missing:
        raise ValueError(f"Dataset has no samples for: {', '.join(missing)}")
    return rows


def split_by_physical_id(rows: list[dict], type_names: list[str], seed: int = 0) -> dict[str, list[dict]]:
    """Stratified 70/15/15 split with specimen-level isolation."""
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for name in type_names:
        ids = sorted({r["physical_id"] for r in rows if r["type"] == name})
        if len(ids) < 3:
            raise ValueError(f"{name} needs at least 3 physical_id values; found {len(ids)}")
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * 0.15))
        n_test = max(1, round(len(ids) * 0.15))
        if n_val + n_test >= len(ids):
            n_val = n_test = 1
        groups = {
            "test": set(ids[:n_test]),
            "val": set(ids[n_test:n_test + n_val]),
            "train": set(ids[n_test + n_val:]),
        }
        for key, selected in groups.items():
            split[key].extend(r for r in rows if r["type"] == name and r["physical_id"] in selected)
    return split


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError('Training requires: pip install -e ".[train]"') from exc


def build_model(n_types: int, n_geom: int, embedding_dim: int = 96, use_geometry: bool = True):
    torch = _torch()
    nn = torch.nn

    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.image = nn.Sequential(
                nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                nn.Conv2d(48, 72, 3, stride=2, padding=1), nn.BatchNorm2d(72), nn.ReLU(),
                nn.Conv2d(72, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.use_geometry = use_geometry
            if use_geometry:
                self.geom = nn.Sequential(nn.Linear(n_geom, 32), nn.ReLU())
            self.embed = nn.Sequential(nn.Linear(128 if use_geometry else 96, embedding_dim), nn.ReLU())
            self.head = nn.Linear(embedding_dim, n_types)

        def forward(self, image, geom=None):
            image_features = self.image(image)
            combined = (torch.cat((image_features, self.geom(geom)), dim=1)
                        if self.use_geometry else image_features)
            embedding = self.embed(combined)
            return self.head(embedding), embedding

    return FusionNet()


def _load_example(row: dict, root: Path, input_size: tuple[int, int], geom_mean, geom_std,
                  augment: bool, rng: np.random.Generator):
    image = cv2.imread(str(root / row["rgb"]), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(root / row["rgb"])
    w, h = input_size
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    if augment:
        if rng.random() < 0.5:
            image = cv2.flip(image, 1)
        gain = float(rng.uniform(0.9, 1.1))
        bias = float(rng.uniform(-8, 8))
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    image = ((image - mean) / std).transpose(2, 0, 1).copy()
    geom = np.array([float(row[n]) for n in GEOM_FEATURE_NAMES], np.float32)
    geom = (geom - geom_mean) / geom_std
    return image, geom


def _predict(model, rows, root, input_size, geom_mean, geom_std, type_names, batch_size=32,
             use_geometry=True):
    torch = _torch()
    model.eval()
    logits, embeddings, labels = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            examples = [_load_example(r, root, input_size, geom_mean, geom_std, False,
                                      np.random.default_rng(0)) for r in rows[start:start + batch_size]]
            x = torch.from_numpy(np.stack([e[0] for e in examples]))
            g = torch.from_numpy(np.stack([e[1] for e in examples]))
            z, emb = model(x, g) if use_geometry else model(x)
            logits.append(z.numpy())
            embeddings.append(emb.numpy())
            labels.extend(type_names.index(r["type"]) for r in rows[start:start + batch_size])
    return np.concatenate(logits), np.concatenate(embeddings), np.asarray(labels, np.int64)


def _save_outputs(path: Path, rows, logits, embedding, labels, metal_types):
    np.savez_compressed(
        path, logits=logits, embedding=embedding, label=labels,
        placement_id=np.asarray([r["placement_id"] for r in rows]),
        physical_id=np.asarray([r["physical_id"] for r in rows]),
        metal=np.asarray([r["type"] in metal_types for r in rows], dtype=bool),
    )


def train(dataset: str | Path, out: str | Path, type_names: list[str], metal_types: set[str],
          input_size=(224, 224), epochs: int = 12, batch_size: int = 32,
          learning_rate: float = 1e-3, seed: int = 0, use_geometry: bool = True) -> dict:
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset, out = Path(dataset), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(dataset / "manifest.csv", type_names)
    split = split_by_physical_id(rows, type_names, seed)

    train_geom = np.array([[float(r[n]) for n in GEOM_FEATURE_NAMES] for r in split["train"]], np.float32)
    geom_mean = train_geom.mean(0)
    geom_std = train_geom.std(0)
    geom_std[geom_std < 1e-6] = 1.0
    meta = ModelMeta(type_names=type_names, input_size=tuple(input_size),
                     geom_mean=geom_mean.tolist() if use_geometry else None,
                     geom_std=geom_std.tolist() if use_geometry else None)
    meta.save(out / "classifier.json")

    model = build_model(len(type_names), len(GEOM_FEATURE_NAMES), use_geometry=use_geometry)
    counts = np.bincount([type_names.index(r["type"]) for r in split["train"]], minlength=len(type_names))
    weights = torch.tensor(len(split["train"]) / (len(type_names) * counts), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history = []
    best_accuracy = -1.0
    best_state = None
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(split["train"]))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_rows = [split["train"][i] for i in order[start:start + batch_size]]
            ex = [_load_example(r, dataset, tuple(input_size), geom_mean, geom_std, True, rng)
                  for r in batch_rows]
            x = torch.from_numpy(np.stack([e[0] for e in ex]))
            g = torch.from_numpy(np.stack([e[1] for e in ex]))
            y = torch.tensor([type_names.index(r["type"]) for r in batch_rows])
            optimizer.zero_grad()
            logits, _ = model(x, g) if use_geometry else model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_rows)
        val_logits, _, val_y = _predict(model, split["val"], dataset, tuple(input_size),
                                        geom_mean, geom_std, type_names, batch_size, use_geometry)
        val_acc = float((val_logits.argmax(1) == val_y).mean())
        history.append({"epoch": epoch + 1, "loss": total_loss / len(split["train"]),
                        "val_accuracy": val_acc})
        print(json.dumps(history[-1]))
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    dummy_image = torch.zeros(1, 3, input_size[1], input_size[0])
    dummy_geom = torch.zeros(1, len(GEOM_FEATURE_NAMES))
    export_inputs = (dummy_image, dummy_geom) if use_geometry else (dummy_image,)
    input_names = ["image", "geom"] if use_geometry else ["image"]
    torch.onnx.export(model, export_inputs, str(out / "classifier.onnx"),
                      input_names=input_names, output_names=["logits", "embedding"],
                      opset_version=17, dynamo=False)

    predictions = {}
    for key in ("train", "val", "test"):
        predictions[key] = _predict(model, split[key], dataset, tuple(input_size),
                                    geom_mean, geom_std, type_names, batch_size, use_geometry)
        z, emb, y = predictions[key]
        _save_outputs(out / f"{key}.npz", split[key], z, emb, y, metal_types)

    train_emb = predictions["train"][1]
    val_emb = predictions["val"][1]
    ood = KnnOOD(train_emb, k=5)
    ood.fit_threshold(val_emb, 99.0)
    ood.save(out / "ood_bank.npz")
    test_z, _, test_y = predictions["test"]
    predicted = test_z.argmax(1)
    confusion = np.zeros((len(type_names), len(type_names)), dtype=int)
    np.add.at(confusion, (test_y, predicted), 1)
    with (out / "confusion_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true \\ predicted", *type_names])
        for name, values in zip(type_names, confusion):
            writer.writerow([name, *values.tolist()])
    per_class = {}
    for i, name in enumerate(type_names):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall)
                           if precision + recall else 0.0,
                           "support": int(confusion[i].sum())}
    classification = {"accuracy": float((predicted == test_y).mean()),
                      "classes": type_names, "per_class": per_class,
                      "confusion_matrix": confusion.tolist()}
    (out / "classification_report.json").write_text(json.dumps(classification, indent=2))
    try:
        import matplotlib.pyplot as plt
        size = max(9, len(type_names) * 0.75)
        fig, ax = plt.subplots(figsize=(size, size * 0.86))
        image = ax.imshow(confusion, cmap="YlGn", interpolation="nearest")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set(xticks=np.arange(len(type_names)), yticks=np.arange(len(type_names)),
               xticklabels=type_names, yticklabels=type_names,
               xlabel="Predicted label", ylabel="True label",
               title=f"SentryGlide error matrix - accuracy {classification['accuracy']:.1%}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        threshold = confusion.max() / 2 if confusion.size else 0
        for i in range(len(type_names)):
            for j in range(len(type_names)):
                ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                        color="white" if confusion[i, j] > threshold else "#17352c", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
    report = {
        "development_model": True,
        "input_mode": "image_plus_geometry" if use_geometry else "image_only",
        "dataset": str(dataset.resolve()),
        "split_frames": {k: len(v) for k, v in split.items()},
        "split_physical_ids": {k: len({r['physical_id'] for r in v}) for k, v in split.items()},
        "test_accuracy": classification["accuracy"],
        "best_validation_accuracy": best_accuracy,
        "history": history,
        "artifacts": ["classifier.onnx", "classifier.json", "ood_bank.npz",
                      "train.npz", "val.npz", "test.npz", "confusion_matrix.csv",
                      "confusion_matrix.png", "classification_report.json"],
    }
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    return report
