import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bmwvision.calibration import calibrate  # noqa: E402
from bmwvision.classifier import KnnOOD, ModelMeta  # noqa: E402
from bmwvision.config import load_config  # noqa: E402
from bmwvision.gating import EmptyReference  # noqa: E402
from bmwvision.sim import SimScene, SimulatedCamera, SimulatedClassifier  # noqa: E402
from bmwvision.taxonomy import Taxonomy  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    c = load_config(ROOT / "config.yaml")
    c["reference"]["n_frames"] = 8
    return c


@pytest.fixture(scope="session")
def type_names():
    return list(yaml.safe_load((ROOT / "taxonomy.yaml").read_text())["types"])


@pytest.fixture(scope="session")
def tax(type_names):
    return Taxonomy.from_yaml(ROOT / "taxonomy.yaml", type_names)


@pytest.fixture
def rig(cfg):
    scene = SimScene(cfg["geometry"])
    cam = SimulatedCamera(scene)
    return scene, cam


@pytest.fixture(scope="session")
def calib(cfg):
    cam = SimulatedCamera(SimScene(cfg["geometry"]))
    return calibrate(cam.read().bgr, cfg["geometry"])


@pytest.fixture(scope="session")
def ref(cfg, calib):
    cam = SimulatedCamera(SimScene(cfg["geometry"]), seed=7)
    fr = [cam.read() for _ in range(8)]
    return EmptyReference.build([calib.rectify(f.bgr) for f in fr],
                                [calib.rectify(f.depth, nearest=True) for f in fr])


def rect(calib, frame):
    return calib.rectify(frame.bgr), calib.rectify(frame.depth, nearest=True)


@pytest.fixture
def sim_classifier(type_names):
    def make(scene, **kw):
        clf = SimulatedClassifier(scene, ModelMeta(type_names=type_names), **kw)
        ood = KnnOOD(clf.make_bank(40))
        ood.fit_threshold(clf.make_bank(20), 99.0)
        return clf, ood
    return make
