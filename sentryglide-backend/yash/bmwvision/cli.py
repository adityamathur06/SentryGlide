"""Command line entry points.

  python -m bmwvision make-board       printable marker plate
  python -m bmwvision calibrate        homography, resolution check, empty reference
  python -m bmwvision collect          placement-based dataset capture
  python -m bmwvision run              the station (real camera + bench controller, or --sim)
  python -m bmwvision fit-decision     temperature, OOD threshold, c_hold from validation outputs
  python -m bmwvision verify-decision  frozen check on an untouched test set
Add --sim to calibrate / collect / run to use the simulated rig.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from .calibration import Calibration, calibrate, make_marker_board, resolution_report
from .config import load_config, resolve
from .gating import EmptyReference


# ---------------------------------------------------------------- shared builders
def _sim_rig(cfg, deliveries=None):
    from .sim import SimScene, SimulatedCamera, SimulatedController
    scene = SimScene(cfg["geometry"])
    cam = SimulatedCamera(scene)
    ctrl = SimulatedController(scene, deliveries or [])
    return scene, cam, ctrl


def _camera(args, cfg):
    if args.sim:
        return _sim_rig(cfg)[1]
    from .hardware import RealSenseCamera
    return RealSenseCamera()


def _median_frame(cam, n=10):
    return np.median(np.stack([cam.read().bgr for _ in range(n)]), axis=0).astype(np.uint8)


def _type_names(cfg, meta_path: Path | None):
    if meta_path and meta_path.exists():
        from .classifier import ModelMeta
        return ModelMeta.load(meta_path).type_names
    return list(yaml.safe_load(resolve(cfg, cfg["taxonomy"]).read_text())["types"])


# ---------------------------------------------------------------- commands
def cmd_make_board(args):
    cfg = load_config(args.config)
    px_per_mm = args.dpi / 25.4
    img = make_marker_board(cfg["geometry"], px_per_mm, background=255)
    cv2.imwrite(args.out, img)
    print(f"Wrote {args.out}: print at {args.dpi} DPI, 100% scale, then check a marker "
          f"measures {cfg['geometry']['markers']['size_mm']} mm.")


def cmd_calibrate(args):
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = _camera(args, cfg)
    if not args.sim:
        input("Clear the chamber (markers visible, no item) and press Enter: ")
    calib = calibrate(_median_frame(cam), cfg["geometry"])
    rep = resolution_report(calib, cfg["geometry"])
    calib.save(out / "calibration.json")
    print(json.dumps({"reprojection_error_mm": round(calib.reprojection_error_mm, 3),
                      "markers_used": sorted(calib.marker_corners_px), **rep}, indent=2))
    if not (rep["camera_ok"] and rep["roi_ok"]):
        print("WARNING: resolution too coarse for the thinnest safety-critical feature. "
              "Move the camera closer, use a longer lens or higher resolution, or lower mm_per_px.")
    frames = [cam.read() for _ in range(cfg["reference"]["n_frames"])]
    ref = EmptyReference.build([calib.rectify(f.bgr) for f in frames],
                               [calib.rectify(f.depth, nearest=True) for f in frames]
                               if frames[0].depth is not None else None)
    ref.save(out / "reference.npz")
    cv2.imwrite(str(out / "roi_preview.png"), ref.bgr)
    print(f"Saved {out/'calibration.json'}, {out/'reference.npz'}, {out/'roi_preview.png'}")


def cmd_collect(args):
    from .collect import collect_placement
    cfg = load_config(args.config)
    calib = Calibration.load(Path(args.calib) / "calibration.json")
    if args.sim:
        from .sim import make_item
        scene, cam, _ = _sim_rig(cfg)
    else:
        cam = _camera(args, cfg)
    ref = EmptyReference.load(Path(args.calib) / "reference.npz")
    rng = np.random.default_rng(0)
    done = 0
    while done < args.placements:
        if args.sim:
            scene.items = [] if args.type == "empty" else [
                make_item(args.sim_item or args.type, (130 + rng.uniform(-10, 10), 100 + rng.uniform(-10, 10)),
                          rng.uniform(0, 180))]
            scene.jitter_frames = 3
        else:
            s = input(f"[{args.type} / {args.item}] placement {done + 1}/{args.placements}: "
                      f"drop the item, Enter to capture (q to stop): ").strip()
            if s.lower() == "q":
                break
        r = collect_placement(cam, calib, ref, cfg, args.out, args.type, args.item)
        print(json.dumps(r))
        done += bool(r["saved"])


def cmd_make_sim_dataset(args):
    """Create a complete synthetic dataset for pipeline development, never clinical validation."""
    from .collect import collect_placement
    from .sim import PROTOTYPES, SimScene, SimulatedCamera, make_item
    cfg = load_config(args.config)
    cfg["collect"]["frames_per_placement"] = args.frames
    scene = SimScene(cfg["geometry"])
    cam = SimulatedCamera(scene, seed=args.seed)
    calib = calibrate(_median_frame(cam, 3), cfg["geometry"])
    empty_frames = [cam.read() for _ in range(8)]
    ref = EmptyReference.build([calib.rectify(f.bgr) for f in empty_frames],
                               [calib.rectify(f.depth, nearest=True) for f in empty_frames])
    types = _type_names(cfg, None)
    missing = sorted(set(types) - set(PROTOTYPES))
    if missing:
        raise RuntimeError(f"Simulator has no prototypes for: {missing}")
    rng = np.random.default_rng(args.seed)
    saved = 0
    for type_name in types:
        for specimen in range(args.items_per_type):
            physical_id = f"sim_{type_name}_{specimen:03d}"
            for _ in range(args.placements):
                scene.items = [make_item(type_name,
                                         (130 + rng.uniform(-1, 1), 100 + rng.uniform(-1, 1)),
                                         rng.uniform(0, 180))]
                scene.jitter_frames = 0
                result = collect_placement(cam, calib, ref, cfg, args.out, type_name, physical_id)
                if not result["saved"]:
                    raise RuntimeError(f"Synthetic capture failed for {physical_id}: {result}")
                saved += 1
        print(json.dumps({"type": type_name, "placements": args.items_per_type * args.placements}))
    print(json.dumps({"dataset": str(Path(args.out).resolve()), "placements": saved,
                      "development_only": True}))


def cmd_train(args):
    from .training import train
    cfg = load_config(args.config)
    taxonomy = yaml.safe_load(resolve(cfg, cfg["taxonomy"]).read_text())
    with open(Path(args.dataset) / "manifest.csv", newline="") as stream:
        import csv
        present = {row["type"] for row in csv.DictReader(stream)}
    type_names = [name for name in taxonomy["types"] if name in present]
    unknown = sorted(present - set(taxonomy["types"]))
    if unknown:
        raise SystemExit(f"Dataset labels missing from taxonomy.yaml: {unknown}")
    report = train(args.dataset, args.out, type_names, set(taxonomy["metal_types"]),
                   input_size=(args.size, args.size), epochs=args.epochs,
                   batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed,
                   use_geometry=not args.image_only)
    print(json.dumps(report, indent=2))


def cmd_import_coco(args):
    from .coco_import import import_coco
    print(json.dumps(import_coco(args.source, args.out, args.side), indent=2))


def cmd_import_folder_zip(args):
    from .folder_dataset import import_classification_zip
    print(json.dumps(import_classification_zip(args.source, args.out, args.seed), indent=2))


def cmd_train_yolo(args):
    from .yolo_training import train_yolo
    print(json.dumps(train_yolo(args.dataset, args.out, args.epochs, args.size,
                                args.batch_size, args.seed), indent=2))


def cmd_run(args):
    from .classifier import KnnOOD, ModelMeta, OnnxTypeClassifier
    from .station import RunLogger, Station
    from .taxonomy import Taxonomy
    cfg = load_config(args.config)
    logger = RunLogger(resolve(cfg, cfg["logging"]["dir"]), cfg["logging"]["save_frames"])
    if args.sim:
        from .sim import Delivery, SimulatedClassifier, make_item, PROTOTYPES
        rng = np.random.default_rng(args.seed)
        kinds = [k for k in PROTOTYPES]
        dels = [Delivery(f"sim_{i:04d}", [make_item(kinds[rng.integers(len(kinds))],
                                                    (130 + rng.uniform(-10, 10), 100 + rng.uniform(-10, 10)),
                                                    rng.uniform(0, 180))]) for i in range(args.cycles)]
        scene, cam, ctrl = _sim_rig(cfg, dels)
        calib = calibrate(_median_frame(cam, 3), cfg["geometry"])
        meta = ModelMeta(type_names=_type_names(cfg, None))
        clf = SimulatedClassifier(scene, meta, seed=args.seed)
        ood = KnnOOD(clf.make_bank(40))
        ood.fit_threshold(clf.make_bank(20), 99.0)
        ref = None
    else:
        from .hardware import ConsoleController
        cam = _camera(args, cfg)
        ctrl = ConsoleController()   # replace with your PLC implementation of hardware.Controller
        calib = Calibration.load(Path(args.calib) / "calibration.json")
        ref = EmptyReference.load(Path(args.calib) / "reference.npz")
        meta = ModelMeta.load(resolve(cfg, cfg["model"]["meta"]))
        clf = OnnxTypeClassifier(resolve(cfg, cfg["model"]["onnx"]), meta)
        bank = resolve(cfg, cfg["model"]["ood_bank"])
        ood = KnnOOD.load(bank) if bank.exists() else None
    tax = Taxonomy.from_yaml(resolve(cfg, cfg["taxonomy"]), meta.type_names)
    st = Station(cfg, cam, ctrl, calib, clf, tax, ood, logger, reference=ref)
    try:
        for rec in st.run(max_cycles=args.cycles, stop_when_idle=args.sim):
            print(json.dumps(rec.summary()))
    except KeyboardInterrupt:
        pass
    if st.faulted:
        print(f"Station stopped on FAULT: {st.faulted}", file=sys.stderr)


def _load_outputs(path):
    z = np.load(path, allow_pickle=False)
    need = {"logits", "placement_id", "physical_id", "label"}
    if not need <= set(z.files):
        raise SystemExit(f"{path} must contain {sorted(need)} (optional: embedding, metal)")
    return {k: z[k] for k in z.files}


def _placement_rows(d, idx, T, tax, cfg):
    """Aggregate frames per placement and apply the metal sensor, as at runtime."""
    from .offline import aggregate_placements, placement_labels
    pids = d["placement_id"][idx]
    uniq, probs, inv = aggregate_placements(d["logits"][idx], pids, T)
    y = placement_labels(d["label"][idx], pids)
    ms = cfg["decision"].get("metal_sensor") or {}
    if "metal" in d and ms.get("enabled"):
        metal = np.zeros(len(uniq), bool)
        np.logical_or.at(metal, inv, d["metal"][idx].astype(bool))
        probs = np.stack([tax.apply_metal(p, bool(m), ms["p_detect"], ms["p_false_alarm"])
                          for p, m in zip(probs, metal)])
    return uniq, probs, y


def cmd_fit_decision(args):
    from .classifier import KnnOOD, ModelMeta
    from .decision import CostModel
    from .offline import (ece, fit_temperature, risk_coverage, rows_for, select_c_hold,
                          split_by_physical)
    from .taxonomy import Taxonomy
    cfg = load_config(args.config)
    meta_path = Path(args.meta)
    meta = ModelMeta.load(meta_path)
    tax = Taxonomy.from_yaml(resolve(cfg, cfg["taxonomy"]), meta.type_names)
    d = _load_outputs(args.val)
    a = split_by_physical(d["physical_id"], args.frac_a, args.seed)
    A, B = np.where(a)[0], np.where(~a)[0]
    print(f"frames {len(a)}  physical items A={len(np.unique(d['physical_id'][A]))} "
          f"B={len(np.unique(d['physical_id'][B]))}")

    T = fit_temperature(d["logits"][A], d["placement_id"][A], d["label"][A])
    _, pB1, yB = _placement_rows(d, B, 1.0, tax, {"decision": {}})
    _, pBT, _ = _placement_rows(d, B, T, tax, {"decision": {}})
    print(f"temperature {T:.3f}   ECE on B: {ece(pB1, yB):.4f} -> {ece(pBT, yB):.4f}")

    report = {"temperature": T}
    if args.bank and "embedding" in d:
        from .offline import aggregate_placements
        ood = KnnOOD(np.load(args.bank)["embedding"])
        sc = ood.scores(d["embedding"][A])
        uniq, inv = np.unique(d["placement_id"][A], return_inverse=True)
        med = np.array([np.median(sc[inv == i]) for i in range(len(uniq))])
        ood.threshold = float(np.percentile(med, args.ood_percentile))
        ood.save(args.ood_out)
        report["ood_threshold"] = ood.threshold
        print(f"OOD threshold (p{args.ood_percentile} of per-placement median, split A): "
              f"{ood.threshold:.4f} -> {args.ood_out}")

    cost = CostModel.from_config(cfg["decision"])
    _, pB, yB = _placement_rows(d, B, T, tax, cfg)
    rows = tax.to_rows(pB)
    true_rows = rows_for(tax, yB)
    grid = sorted(set(np.round(np.geomspace(0.01, 100, 41), 4).tolist() + [cost.c_hold]))
    table = risk_coverage(rows, true_rows, cost, grid, cfg["decision"]["dangerous"])
    target = cfg["decision"]["max_dangerous_upper"]
    best = select_c_hold(table, target)
    report.update({"risk_coverage": table, "selected": best, "target_dangerous_upper": target})
    print(f"\n{'c_hold':>8} {'coverage':>9} {'err|acc':>8} {'n_dang':>7} {'dang_ub95':>10} {'sharps_miss_ub95':>17}")
    for r in table[::4]:
        print(f"{r['c_hold']:8.3g} {r['coverage']:9.3f} {r['error_rate_accepted']:8.4f} "
              f"{r['n_dangerous']:7d} {r['dangerous_upper']:10.5f} {r['sharps_missed_upper']:17.5f}")
    if best is None:
        n = len(true_rows)
        print(f"\nNo c_hold meets dangerous upper bound <= {target} with {n} placements. "
              f"With zero dangerous errors the bound is ~3/n = {3 / max(n, 1):.4f}; "
              f"you need about {int(np.ceil(3 / target))} placements in split B.")
    else:
        print(f"\nselected c_hold = {best['c_hold']:.4g}: coverage {best['coverage']:.3f}, "
              f"dangerous upper bound {best['dangerous_upper']:.5f}")
    if args.write_meta:
        meta.temperature = T
        meta.save(meta_path)
        print(f"wrote temperature to {meta_path}")
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"report -> {args.out}. Set decision.c_hold in config.yaml, then run verify-decision "
          f"once on an untouched test set.")


def cmd_verify_decision(args):
    from .classifier import ModelMeta
    from .decision import CostModel
    from .offline import risk_coverage, rows_for
    from .taxonomy import Taxonomy
    from .types import BINS, ROWS
    cfg = load_config(args.config)
    meta = ModelMeta.load(args.meta)
    tax = Taxonomy.from_yaml(resolve(cfg, cfg["taxonomy"]), meta.type_names)
    d = _load_outputs(args.test)
    cost = CostModel.from_config(cfg["decision"])
    _, p, y = _placement_rows(d, np.arange(len(d["label"])), meta.temperature, tax, cfg)
    rows, true_rows = tax.to_rows(p), rows_for(tax, y)
    r = risk_coverage(rows, true_rows, cost, [cost.c_hold], cfg["decision"]["dangerous"])[0]
    R = rows @ cost.C
    k = R.argmin(1)
    routed = np.where(R.min(1) < cost.c_hold, k, len(BINS))
    cols = [b.value for b in BINS] + ["HOLD"]
    cm = np.zeros((len(ROWS), len(cols)), int)
    np.add.at(cm, (true_rows, routed), 1)
    print(f"placements {len(true_rows)}  temperature {meta.temperature:.3f}  c_hold {cost.c_hold}")
    print(json.dumps(r, indent=2))
    print("\ntrue \\ routed " + " ".join(f"{c:>7}" for c in cols))
    for i, name in enumerate(ROWS):
        print(f"{name:>13} " + " ".join(f"{v:7d}" for v in cm[i]))
    ok = r["dangerous_upper"] <= cfg["decision"]["max_dangerous_upper"]
    print(f"\n{'PASS' if ok else 'FAIL'}: dangerous upper bound {r['dangerous_upper']:.5f} "
          f"vs target {cfg['decision']['max_dangerous_upper']}")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="bmwvision")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("make-board")
    s.add_argument("--out", default="marker_board.png")
    s.add_argument("--dpi", type=float, default=300)
    s.set_defaults(fn=cmd_make_board)

    s = sub.add_parser("calibrate")
    s.add_argument("--out", default="calib")
    s.add_argument("--sim", action="store_true")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("collect")
    s.add_argument("--calib", default="calib")
    s.add_argument("--out", default="dataset")
    s.add_argument("--type", required=True, help="object type from taxonomy.yaml, or empty / multiple")
    s.add_argument("--item", required=True, help="physical item id, e.g. needle_017 (label it on the object)")
    s.add_argument("--placements", type=int, default=8)
    s.add_argument("--sim", action="store_true")
    s.add_argument("--sim-item", default=None, help="simulator prototype to place (defaults to --type)")
    s.set_defaults(fn=cmd_collect)

    s = sub.add_parser("make-sim-dataset", help="build a full synthetic development dataset")
    s.add_argument("--out", default="dataset-sim")
    s.add_argument("--items-per-type", type=int, default=6)
    s.add_argument("--placements", type=int, default=3)
    s.add_argument("--frames", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_make_sim_dataset)

    s = sub.add_parser("train", help="train and export the RGB + geometry classifier")
    s.add_argument("--dataset", default="dataset")
    s.add_argument("--out", default="models")
    s.add_argument("--size", type=int, default=224)
    s.add_argument("--epochs", type=int, default=12)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--learning-rate", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--image-only", action="store_true",
                   help="train exclusively from RGB pixels; do not fuse geometric features")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("import-coco", help="combine COCO train/val/test files into one manifest")
    s.add_argument("--source", required=True)
    s.add_argument("--out", default="dataset")
    s.add_argument("--side", type=int, default=600)
    s.set_defaults(fn=cmd_import_coco)

    s = sub.add_parser("import-folder-zip", help="import class/image ZIP with leakage-safe splits")
    s.add_argument("--source", required=True)
    s.add_argument("--out", default="dataset")
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_import_folder_zip)

    s = sub.add_parser("train-yolo", help="train the uploaded YOLOv8 classification architecture")
    s.add_argument("--dataset", default="dataset")
    s.add_argument("--out", default="models")
    s.add_argument("--size", type=int, default=224)
    s.add_argument("--epochs", type=int, default=15)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_train_yolo)

    s = sub.add_parser("run")
    s.add_argument("--calib", default="calib")
    s.add_argument("--cycles", type=int, default=None)
    s.add_argument("--sim", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("fit-decision")
    s.add_argument("--val", required=True, help="npz: logits, placement_id, physical_id, label [, embedding, metal]")
    s.add_argument("--meta", default="models/classifier.json")
    s.add_argument("--bank", default=None, help="npz with training-set 'embedding' for OOD")
    s.add_argument("--ood-out", default="models/ood_bank.npz")
    s.add_argument("--ood-percentile", type=float, default=99.0)
    s.add_argument("--frac-a", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--write-meta", action="store_true", help="store the fitted temperature in --meta")
    s.add_argument("--out", default="decision_report.json")
    s.set_defaults(fn=cmd_fit_decision)

    s = sub.add_parser("verify-decision")
    s.add_argument("--test", required=True)
    s.add_argument("--meta", default="models/classifier.json")
    s.set_defaults(fn=cmd_verify_decision)

    args = p.parse_args(argv)
    if args.cmd == "run" and args.sim and args.cycles is None:
        args.cycles = 20
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
