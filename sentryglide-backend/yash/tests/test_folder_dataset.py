import csv
import zipfile
from pathlib import Path

from bmwvision.folder_dataset import import_classification_zip, source_image_id


def test_roboflow_variants_share_source_id():
    assert source_image_id("IMG_6893.JPG") == "img_6893"
    assert source_image_id("IMG_6893_JPG.rf.2afba6ab3930adc5ff0a76bd91a2602e.jpg") == "img_6893"


def test_import_keeps_source_families_in_one_split(tmp_path: Path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for class_name in ("a", "b"):
            for number in range(10):
                zf.writestr(f"{class_name}/item{number}.JPG", b"image")
                zf.writestr(f"{class_name}/item{number}_JPG.rf.{number:032x}.jpg", b"variant")
    out = tmp_path / "dataset"
    report = import_classification_zip(archive, out, seed=7)
    assert report["images"] == 40
    rows = list(csv.DictReader((out / "manifest.csv").open()))
    seen = {}
    for row in rows:
        previous = seen.setdefault(row["physical_id"], row["source_split"])
        assert row["source_split"] == previous
        assert (out / row["rgb"]).exists()

