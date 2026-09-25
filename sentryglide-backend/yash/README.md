# bmw-vision: single-item biomedical waste classifier (Approach D)

Runtime and tooling for the inspection station: fixed ROI, geometric presence and occupancy
gates, a lightweight classifier on the rectified crop, temporal aggregation, and a cost-based
accept or hold rule. Outputs RED, YELLOW, BLUE, WHITE, UNKNOWN (hold) or FAULT.

```
controller "delivered" -> settle -> rectified ROI (fixed mm/px) -> occupancy (colour + depth + depth holes)
   NONE -> FAULT   MULTIPLE / OUT_OF_POSITION -> HOLD
-> per frame: quality gate -> classifier (type logits + embedding) + geometric features
-> mean calibrated type probabilities, OOD score, consistency
-> metal sensor evidence (optional) -> type -> bin table -> expected-cost rule
-> bin command -> drop confirmed + chamber seen empty -> log
```

## Layout

| File | What it does |
|---|---|
| `config.yaml` | Every threshold and cost. Items marked TUNE come from rig data |
| `taxonomy.yaml` | Object type to bin table (BMW Rules 2016 Schedule I) and metal-bearing types |
| `bmwvision/calibration.py` | ArUco homography, ROI rectification in mm, resolution check, drift check, printable board |
| `bmwvision/gating.py` | Empty reference, settle detector, occupancy, image quality |
| `bmwvision/features.py` | Ten geometric features (size, height, volume, depth-hole fraction) for late fusion |
| `bmwvision/classifier.py` | ONNX model wrapper (the model contract is in the docstring) and kNN OOD scorer |
| `bmwvision/decision.py` | Temporal aggregator and the Bayes accept or hold rule |
| `bmwvision/offline.py` | Temperature fitting, risk-coverage table, Clopper-Pearson bounds, c_hold selection |
| `bmwvision/station.py` | State machine, controller handshake, run logging |
| `bmwvision/collect.py` | Placement-based dataset capture with physical item ids |
| `bmwvision/hardware.py` | Camera and Controller interfaces, RealSense adapter, console bench controller |
| `bmwvision/sim.py` | Simulated camera, controller and classifier for testing without hardware |

## Install and test

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m bmwvision run --sim --cycles 20
```

On Windows, open `E:\sih\SentryGlide.code-workspace` in VS Code. The workspace is already configured
to use `.venv`, discover the test suite, and provide launch configurations for the simulator,
simulated calibration, and the full tests. The same actions are available from **Terminal > Run Task**.

The original requirements-file workflow is also supported:

```
pip install -r requirements.txt
python -m pytest -q tests          # 47 tests, about 1 minute
python -m bmwvision run --sim      # 20 simulated cycles
```

## Bring-up on the rig

1. **Marker plate.** `python -m bmwvision make-board --dpi 300`, print at 100 %, check a marker
   measures 20 mm, fix it to the inspection plate. Markers must stay outside the ROI.
2. **Calibrate.** `python -m bmwvision calibrate`. It prints reprojection error and a resolution
   check. If `camera_ok` is false, the camera cannot resolve the thinnest needle: move it closer
   or change the lens before collecting any data. It also saves `calib/reference.npz` and a
   preview of the rectified empty ROI.
3. **Tune the gates** on real frames: `occupancy.lab_thresh`, `quality.min_sharpness`,
   `settle.max_changed_frac`. Collect `empty` and `multiple` placements to check them.
4. **Collect data.** Label every physical object (e.g. `needle_017`) and capture 5 to 10 drops each:
   `python -m bmwvision collect --type needle --item needle_017 --placements 8`.
   Use `--type invalid` for non-waste objects, `--type multiple` for staged pairs, `--type empty`
   for the empty chamber with stains and debris.
5. **Train** a classifier on `dataset/manifest.csv` (split by `physical_id`), export to ONNX
   matching the contract in `classifier.py`, and write `models/classifier.json`.
6. **Fit the decision layer** from per-frame validation outputs:
   `python -m bmwvision fit-decision --val val.npz --bank train_emb.npz --write-meta`.
   It fits the temperature, sets the OOD threshold, prints coverage against the dangerous-error
   upper bound for a sweep of `c_hold`, and recommends one. Put it in `config.yaml`.
7. **Verify once** on untouched physical items: `python -m bmwvision verify-decision --test test.npz`.
8. **Run.** `python -m bmwvision run` uses the console bench controller until you implement
   `hardware.Controller` for your PLC.

## Train and export the classifier

Install the optional training dependency, then train from the placement manifest. The command
splits by `physical_id`, trains an RGB + 10-feature fusion network, exports the ONNX contract,
and writes train/validation/test outputs plus an OOD bank.

```
.venv\Scripts\python -m pip install -e ".[train]"
.venv\Scripts\python -m bmwvision train --dataset dataset --out models --epochs 12
.venv\Scripts\python -m bmwvision fit-decision --val models\val.npz --meta models\classifier.json --bank models\train.npz --ood-out models\ood_bank.npz --write-meta --out models\decision_report.json
.venv\Scripts\python -m bmwvision verify-decision --test models\test.npz --meta models\classifier.json
```

For software development without hardware, generate a full synthetic dataset first:

```
.venv\Scripts\python -m bmwvision make-sim-dataset --out dataset-sim
.venv\Scripts\python -m bmwvision train --dataset dataset-sim --out models --size 160 --epochs 30
```

Synthetic results validate the pipeline only. They are not evidence of real classification safety.

The active model can also be trained from the combined COCO medical-waste archive. The importer
merges its original train/validation/test files by category name and writes one unified manifest;
training then creates a new held-out split for honest evaluation:

```
.venv\Scripts\python -m bmwvision import-coco --source "dataset-source\Medical Waste dataset" --out dataset
.venv\Scripts\python -m bmwvision train --dataset dataset --out models --size 160 --epochs 20
```

See `models/confusion_matrix.png`, `models/confusion_matrix.csv`, and
`models/classification_report.json` for the resulting error analysis.

### Class-folder ZIP + uploaded YOLOv8 notebook

For an archive laid out as `class_name/image.jpg`, import it with source-family isolation and
train the YOLOv8 classifier architecture supplied in `yolov8-n-5-fold.ipynb`:

```powershell
python -m bmwvision import-folder-zip --source "C:\path\archive.zip" --out dataset
python -m bmwvision train-yolo --dataset dataset --out models --epochs 15 --size 224
```

Roboflow variants named `original_JPG.rf.<hash>.jpg` stay in the same split as their source
photo. This avoids reporting an inflated test score caused by near-duplicate leakage. The web
API automatically selects `models/yolov8_medical_best.pt` when that file and
`models/yolo_model.json` are present.

## Local React upload interface

The React frontend is prebuilt into the local FastAPI service. Start both with one command:

```
.venv\Scripts\python -m pip install -e ".[web]"
.venv\Scripts\python -m bmwvision.web_api
```

Open `http://127.0.0.1:8000`. Upload a JPG, PNG, or WebP photo containing one item on a plain
background. Predictions run locally; the app does not send images to a cloud service. The supplied
model is synthetic and the interface is for prototype demonstration, not physical routing.

The `.npz` files need `logits` (frames x types), `placement_id`, `physical_id`, `label`
(type index), and optionally `embedding` and `metal` (sensor reading per frame).

## Design choices worth knowing

- **Fixed physical scale.** The ROI is rectified to 0.25 mm/px and objects are never resized to
  fill the crop, so size is a real feature. Keep the model input equal to the ROI size.
- **No morphological opening** in the occupancy mask. It erases a 2 px needle shaft; speckle is
  removed by an area filter instead (tested: the shaft survives).
- **Depth holes are evidence.** Glass and polished metal return no depth. New holes count as
  object pixels and their fraction is a classifier feature.
- **The cost matrix drives coverage.** With WHITE-to-RED at 1000 and `c_hold` at 1, a RED item is
  only routed when P(sharp) is below about 0.1 %. Without extra evidence many clean RED items
  will be held. A metal sensor reading of "no metal" is what makes those items routable.
- **Metal sensor is fused as a likelihood, not a veto.** `p_detect` must be measured on the rig
  with the hardest case (thin needle, large plastic hub, worst position). Overstating it is the
  one way this module can make routing less safe.
- **Frames of one item are not independent.** Aggregation removes flicker; it does not make a
  systematic error go away. Calibrate and validate per placement, split by physical item.

## Hardware-dependent work

- A PLC implementation of `hardware.Controller` (depends on your protocol).
- Backlit alternate frames. The occupancy code works on them, but the capture loop does not
  alternate lighting yet.
- Hardware testing. Everything here is tested only against the simulator, whose classifier
  reads the true label, so simulated routing results say nothing about real accuracy.
  `RealSenseCamera` has not been run on a device.
- Offline risk tables ignore the OOD and consistency holds, so they slightly overstate
  risk and understate coverage.
