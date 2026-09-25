"""Convert all COCO splits in the supplied medical-waste archive into one training manifest."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from .collect import MANIFEST_FIELDS
from .features import GEOM_FEATURE_NAMES, geometric_features
from .imageprep import prepare_upload


def normalize_label(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def import_coco(source: str | Path, out: str | Path, side: int = 600) -> dict:
    source, out = Path(source), Path(out)
    annotations = source / "annotations" / "coco"
    image_root = source / "images"
    split_files = [annotations / f"{name}.json" for name in ("train", "val", "test")]
    if not all(path.exists() for path in split_files):
        raise FileNotFoundError(f"Expected train.json, val.json and test.json under {annotations}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    image_lookup = {p.name: p for p in image_root.rglob("*.jpeg")}
    manifest_fields = MANIFEST_FIELDS + ["source_split", "source_file"]
    rows, counts = [], {}
    for split_path in split_files:
        data = json.loads(split_path.read_text())
        categories = {c["id"]: normalize_label(c["name"]) for c in data["categories"]}
        images = {i["id"]: i for i in data["images"]}
        for ann in data["annotations"]:
            info = images[ann["image_id"]]
            source_image = image_lookup.get(info["file_name"])
            if source_image is None:
                raise FileNotFoundError(f"Image referenced by COCO was not found: {info['file_name']}")
            label = categories[ann["category_id"]]
            bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Unreadable image: {source_image}")
            # Use exactly the same object preparation as the upload API. COCO boxes remain
            # provenance only; using them here would create a train/serve skew because an
            # arbitrary uploaded photograph has no annotation box.
            view, mask, fraction = prepare_upload(bgr, side)
            if fraction < 0.001:
                raise ValueError(f"No foreground detected in {source_image}")
            stem = Path(info["file_name"]).stem
            rgb_rel = Path("images") / f"{stem}.jpg"
            mask_rel = Path("masks") / f"{stem}.png"
            cv2.imwrite(str(out / rgb_rel), view, [cv2.IMWRITE_JPEG_QUALITY, 94])
            cv2.imwrite(str(out / mask_rel), mask)
            geom = geometric_features(mask, None, None, 0.25)
            rows.append({
                "rgb": str(rgb_rel), "depth": "", "mask": str(mask_rel), "type": label,
                "physical_id": stem, "placement_id": stem, "frame": 0, "occupancy": "ONE", "t": "",
                **{name: round(float(value), 4) for name, value in zip(GEOM_FEATURE_NAMES, geom)},
                "source_split": split_path.stem, "source_file": str(source_image.relative_to(source)),
            })
            counts[label] = counts.get(label, 0) + 1
    with (out / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"source": str(source.resolve()), "samples": len(rows), "classes": counts,
              "combined_source_splits": ["train", "val", "test"], "output_side": side}
    (out / "import_report.json").write_text(json.dumps(report, indent=2))
    return report
