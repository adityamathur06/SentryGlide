"""Import a class-folder ZIP and create leakage-safe train/validation/test splits."""
from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_ROBOFLOW_VARIANT = re.compile(r"^(?P<base>.+?)_(?:JPG|JPEG|PNG)\.rf\.[0-9a-f]+$", re.I)


def source_image_id(filename: str) -> str:
    """Return the original-photo id shared by a Roboflow image and its variants."""
    stem = Path(filename).stem
    match = _ROBOFLOW_VARIANT.match(stem)
    return (match.group("base") if match else stem).lower()


def _split_groups(groups: list[str], seed: int) -> dict[str, set[str]]:
    rng = random.Random(seed)
    groups = sorted(groups)
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    if n_test + n_val >= n:
        n_test = n_val = 1
    return {
        "test": set(groups[:n_test]),
        "val": set(groups[n_test:n_test + n_val]),
        "train": set(groups[n_test + n_val:]),
    }


def import_classification_zip(archive: str | Path, out: str | Path, seed: int = 42) -> dict:
    """Extract ``class/image`` entries, then split whole source-image families together.

    Roboflow filenames such as ``item_JPG.rf.<hash>.jpg`` are grouped with ``item.JPG``.
    This prevents augmented views of one photo leaking into evaluation splits.
    """
    archive, out = Path(archive), Path(out)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    source = out / "source"
    split_root = out / "splits"
    source.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            member = PurePosixPath(info.filename)
            if info.is_dir() or member.suffix.lower() not in IMAGE_SUFFIXES or len(member.parts) != 2:
                continue
            class_name, filename = member.parts
            if class_name in {".", ".."} or filename in {".", ".."}:
                continue
            target_dir = source / class_name
            target_dir.mkdir(exist_ok=True)
            target = target_dir / filename
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            records.append({"class": class_name, "filename": filename,
                            "group": source_image_id(filename), "path": str(target)})

    if not records:
        raise ValueError("The ZIP contains no class/image entries")
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_class[record["class"]].append(record)

    split_lookup: dict[tuple[str, str], str] = {}
    group_counts: dict[str, int] = {}
    for class_name, class_records in sorted(by_class.items()):
        groups = sorted({r["group"] for r in class_records})
        if len(groups) < 3:
            raise ValueError(f"{class_name} needs at least 3 source-image groups")
        group_counts[class_name] = len(groups)
        for split, selected in _split_groups(groups, seed).items():
            for group in selected:
                split_lookup[(class_name, group)] = split

    counts: Counter[tuple[str, str]] = Counter()
    manifest_path = out / "manifest.csv"
    fields = ["rgb", "depth", "mask", "type", "physical_id", "placement_id", "frame",
              "occupancy", "t", "area_mm2", "length_mm", "width_mm", "elongation",
              "max_height_mm", "mean_height_mm", "p90_height_mm", "height_std_mm",
              "volume_mm3", "invalid_depth_frac", "source_split", "source_file"]
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            split = split_lookup[(record["class"], record["group"])]
            src = Path(record["path"])
            destination = split_root / split / record["class"] / src.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, destination)
            except OSError:
                shutil.copy2(src, destination)
            counts[(split, record["class"])] += 1
            relative = src.relative_to(out)
            group_id = f"{record['class']}:{record['group']}"
            writer.writerow({
                "rgb": str(relative), "type": record["class"], "physical_id": group_id,
                "placement_id": f"{group_id}:{index}", "frame": 0, "occupancy": "ONE",
                "area_mm2": 0, "length_mm": 0, "width_mm": 0, "elongation": 0,
                "max_height_mm": 0, "mean_height_mm": 0, "p90_height_mm": 0,
                "height_std_mm": 0, "volume_mm3": 0, "invalid_depth_frac": 0,
                "source_split": split, "source_file": f"{record['class']}/{record['filename']}",
            })

    report = {
        "archive": str(archive.resolve()), "images": len(records), "classes": sorted(by_class),
        "class_image_counts": {name: len(by_class[name]) for name in sorted(by_class)},
        "class_source_groups": group_counts,
        "split_images": {split: sum(value for (key, _), value in counts.items() if key == split)
                         for split in ("train", "val", "test")},
        "seed": seed, "leakage_guard": "Roboflow variants grouped by original source filename",
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

