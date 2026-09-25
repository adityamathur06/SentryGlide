import csv

import pytest

from bmwvision.training import read_manifest, split_by_physical_id


def _rows():
    return [
        {"type": t, "physical_id": f"{t}_{i}", "placement_id": f"{t}_{i}_p0",
         "occupancy": "ONE"}
        for t in ("a", "b") for i in range(5)
    ]


def test_split_is_by_physical_id_and_contains_every_class():
    split = split_by_physical_id(_rows(), ["a", "b"], seed=3)
    ids = [{r["physical_id"] for r in split[name]} for name in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert all({r["type"] for r in rows} == {"a", "b"} for rows in split.values())


def test_manifest_rejects_missing_class(tmp_path):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "physical_id", "placement_id", "occupancy"])
        writer.writeheader()
        writer.writerow({"type": "a", "physical_id": "a_1", "placement_id": "p1", "occupancy": "ONE"})
    with pytest.raises(ValueError, match="no samples for: b"):
        read_manifest(path, ["a", "b"])

