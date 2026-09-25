import csv

import numpy as np

from bmwvision import cli
from bmwvision.classifier import ModelMeta, softmax
from bmwvision.collect import collect_placement
from bmwvision.sim import make_item
from conftest import ROOT


def test_collect_placements(cfg, rig, calib, ref, tmp_path):
    scene, cam = rig
    for i in range(2):
        scene.items = [make_item("needle", (125 + 5 * i, 100), 40 * i)]
        scene.jitter_frames = 2
        r = collect_placement(cam, calib, ref, cfg, tmp_path, "needle", "needle_001")
        assert r["saved"], r
    scene.items = [make_item("gloves", (100, 100)), make_item("needle", (160, 140))]
    assert not collect_placement(cam, calib, ref, cfg, tmp_path, "gloves", "gloves_001")["saved"]
    assert collect_placement(cam, calib, ref, cfg, tmp_path, "multiple", "pair_001")["saved"]
    rows = list(csv.DictReader(open(tmp_path / "manifest.csv")))
    assert len(rows) == 3 * cfg["collect"]["frames_per_placement"]
    assert {r["placement_id"] for r in rows} == {"needle_001_p000", "needle_001_p001", "pair_001_p000"}
    assert (tmp_path / rows[0]["rgb"]).exists() and (tmp_path / rows[0]["depth"]).exists()


def make_outputs(path, type_names, n_items=300, seed=0, temp=2.0):
    """Synthetic validation outputs: overconfident logits, 5 placements x 3 frames per physical item."""
    rng = np.random.default_rng(seed)
    K = len(type_names)
    rows = {k: [] for k in ["logits", "placement_id", "physical_id", "label", "embedding", "metal"]}
    metal_types = {"needle", "syringe_with_fixed_needle", "scalpel_blade", "lancet", "metal_implant"}
    for item in range(n_items):
        c = rng.integers(K)
        for p in range(5):
            base = rng.normal(0, 1, K)
            base[c] += 5.0
            y = rng.choice(K, p=softmax(base)) if p == 0 else y   # base is calibrated; label fixed per item
            for f in range(3):
                rows["logits"].append((base + rng.normal(0, 0.2, K)) * temp)
                rows["placement_id"].append(item * 100 + p)
                rows["physical_id"].append(item)
                rows["label"].append(y)
                e = np.zeros(K); e[y] = 1
                rows["embedding"].append(e + rng.normal(0, 0.1, K))
                rows["metal"].append(type_names[y] in metal_types)
    np.savez(path, **{k: np.array(v) for k, v in rows.items()})


def test_fit_and_verify_decision(tmp_path, type_names, capsys):
    make_outputs(tmp_path / "val.npz", type_names, seed=0)
    make_outputs(tmp_path / "test.npz", type_names, seed=1)
    rng = np.random.default_rng(2)
    bank = np.eye(len(type_names))[rng.integers(len(type_names), size=500)] + rng.normal(0, 0.1, (500, len(type_names)))
    np.savez(tmp_path / "bank.npz", embedding=bank)
    ModelMeta(type_names=type_names).save(tmp_path / "meta.json")
    cfgp = str(ROOT / "config.yaml")
    rc = cli.main(["--config", cfgp, "fit-decision", "--val", str(tmp_path / "val.npz"),
                   "--meta", str(tmp_path / "meta.json"), "--bank", str(tmp_path / "bank.npz"),
                   "--ood-out", str(tmp_path / "ood.npz"), "--write-meta",
                   "--out", str(tmp_path / "report.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "temperature" in out and (tmp_path / "ood.npz").exists()
    assert ModelMeta.load(tmp_path / "meta.json").temperature > 1.2
    cli.main(["--config", cfgp, "verify-decision", "--test", str(tmp_path / "test.npz"),
              "--meta", str(tmp_path / "meta.json")])
    out = capsys.readouterr().out
    assert "true \\ routed" in out


def test_cli_sim_run(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil, yaml
    shutil.copy(ROOT / "taxonomy.yaml", tmp_path)
    c = yaml.safe_load(open(ROOT / "config.yaml"))
    c["reference"]["n_frames"] = 5
    yaml.safe_dump(c, open(tmp_path / "config.yaml", "w"))
    assert cli.main(["--config", "config.yaml", "run", "--sim", "--cycles", "4"]) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert len(lines) == 4
    assert (tmp_path / "runs" / "records.jsonl").exists()
