"""Classifier runtime: ONNX model wrapper and kNN out-of-distribution scorer.

Model contract (export your trained network to match):
  inputs : "image" float32 (1, 3, H, W), RGB, normalised with meta.mean / meta.std
           "geom"  float32 (1, F) optional, standardised with meta.geom_mean / meta.geom_std
  outputs: "logits" (1, K) over meta.type_names
           "embedding" (1, D) optional, penultimate features, used for OOD scoring
The sidecar JSON (meta) is the single source of truth for type order, input size,
normalisation and the fitted softmax temperature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .features import GEOM_FEATURE_NAMES


@dataclass
class ModelMeta:
    type_names: list[str]
    input_size: tuple[int, int] = (600, 600)          # (w, h); should equal the ROI size
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    geom_feature_names: list[str] = field(default_factory=lambda: list(GEOM_FEATURE_NAMES))
    geom_mean: list[float] | None = None
    geom_std: list[float] | None = None
    temperature: float = 1.0

    @classmethod
    def load(cls, path: str | Path) -> "ModelMeta":
        d = json.loads(Path(path).read_text())
        d["input_size"] = tuple(d.get("input_size", (600, 600)))
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


@dataclass
class ClassifierOutput:
    logits: np.ndarray               # (K,)
    embedding: np.ndarray | None     # (D,)


class TypeClassifier(Protocol):
    meta: ModelMeta

    def __call__(self, bgr_roi: np.ndarray, geom: np.ndarray) -> ClassifierOutput: ...


class OnnxTypeClassifier:
    def __init__(self, onnx_path: str | Path, meta: ModelMeta, providers: list[str] | None = None):
        import onnxruntime as ort
        self.meta = meta
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers or ort.get_available_providers())
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.outputs = [o.name for o in self.sess.get_outputs()]
        if "image" not in self.inputs or "logits" not in self.outputs:
            raise ValueError(f"ONNX model must have input 'image' and output 'logits'; "
                             f"got inputs {self.inputs}, outputs {self.outputs}")
        self._mean = np.array(meta.mean, np.float32).reshape(3, 1, 1)
        self._std = np.array(meta.std, np.float32).reshape(3, 1, 1)

    def preprocess(self, bgr_roi: np.ndarray) -> np.ndarray:
        w, h = self.meta.input_size
        if bgr_roi.shape[1] != w or bgr_roi.shape[0] != h:
            # Allowed but not ideal: set the ROI size equal to the model input instead.
            bgr_roi = cv2.resize(bgr_roi, (w, h), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1) / 255.0
        return ((x - self._mean) / self._std)[None]

    def __call__(self, bgr_roi: np.ndarray, geom: np.ndarray) -> ClassifierOutput:
        feeds = {"image": self.preprocess(bgr_roi)}
        if "geom" in self.inputs:
            g = geom.astype(np.float32)
            if self.meta.geom_mean is not None:
                g = (g - np.array(self.meta.geom_mean, np.float32)) / np.array(self.meta.geom_std, np.float32)
            feeds["geom"] = g[None]
        res = dict(zip(self.outputs, self.sess.run(self.outputs, feeds)))
        emb = res.get("embedding")
        return ClassifierOutput(res["logits"][0].astype(np.float64), None if emb is None else emb[0])


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class KnnOOD:
    """Distance-to-training-set OOD score: mean cosine distance to the k nearest training embeddings.

    Catches objects nobody thought to collect, which a trained INVALID class cannot.
    Fit the threshold on embeddings of held-out in-distribution items (never the bank itself,
    whose self-distance is zero).
    """

    def __init__(self, bank: np.ndarray, k: int = 5, threshold: float | None = None):
        self.bank = self._norm(np.asarray(bank, np.float32))
        self.k = min(k, len(self.bank))
        self.threshold = threshold

    @staticmethod
    def _norm(x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)

    def scores(self, emb: np.ndarray) -> np.ndarray:
        e = self._norm(np.atleast_2d(np.asarray(emb, np.float32)))
        d = 1.0 - e @ self.bank.T
        part = np.partition(d, self.k - 1, axis=1)[:, :self.k]
        return part.mean(axis=1)

    def score(self, emb: np.ndarray) -> float:
        return float(self.scores(emb)[0])

    def fit_threshold(self, val_emb: np.ndarray, percentile: float = 99.0) -> float:
        self.threshold = float(np.percentile(self.scores(val_emb), percentile))
        return self.threshold

    def is_ood(self, score: float | None) -> bool:
        return score is not None and self.threshold is not None and score > self.threshold

    def save(self, path) -> None:
        np.savez_compressed(path, bank=self.bank, k=self.k,
                            threshold=np.nan if self.threshold is None else self.threshold)

    @classmethod
    def load(cls, path) -> "KnnOOD":
        z = np.load(path)
        t = float(z["threshold"])
        return cls(z["bank"], int(z["k"]), None if np.isnan(t) else t)
