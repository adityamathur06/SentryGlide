"""Local HTTP API for the React demonstration UI."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .classifier import KnnOOD, ModelMeta, OnnxTypeClassifier, softmax
from .config import load_config, resolve
from .decision import CostModel, bayes_route
from .features import GEOM_FEATURE_NAMES, geometric_features
from .imageprep import prepare_upload
from .taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _display_name(value: str) -> str:
    return value.replace("_", " ").title()


class PredictionService:
    def __init__(self):
        cfg = load_config(ROOT / "config.yaml")
        self.cfg = cfg
        yolo_path = ROOT / "models" / "yolov8_medical_best.pt"
        yolo_meta = ROOT / "models" / "yolo_model.json"
        self.mode = "yolo" if yolo_path.exists() and yolo_meta.exists() else "onnx"
        if self.mode == "yolo":
            import json
            from ultralytics import YOLO
            metadata = json.loads(yolo_meta.read_text(encoding="utf-8"))
            self.type_names = list(metadata["type_names"])
            self.minimum_confidence = float(metadata.get("minimum_confidence", 0.45))
            self.classifier = YOLO(str(yolo_path), task="classify")
            self.model_name = yolo_path.name
            self.taxonomy = Taxonomy.from_yaml(resolve(cfg, cfg["taxonomy"]), self.type_names)
            self.ood = None
            self.cost = None
        else:
            self.meta = ModelMeta.load(resolve(cfg, cfg["model"]["meta"]))
            self.type_names = self.meta.type_names
            self.classifier = OnnxTypeClassifier(resolve(cfg, cfg["model"]["onnx"]), self.meta,
                                                 providers=["CPUExecutionProvider"])
            self.ood = KnnOOD.load(resolve(cfg, cfg["model"]["ood_bank"]))
            self.taxonomy = Taxonomy.from_yaml(resolve(cfg, cfg["taxonomy"]), self.type_names)
            self.cost = CostModel.from_config(cfg["decision"])
            self.model_name = "classifier.onnx"

    def predict(self, encoded: bytes) -> dict:
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded file is not a readable JPG, PNG, or WebP image")
        if self.mode == "yolo":
            return self._predict_yolo(image)
        view, mask, area_fraction = prepare_upload(image)
        if area_fraction < 0.001:
            raise ValueError("No foreground object was detected. Use a plain contrasting background")
        # 600 px represents the configured 150 mm inspection ROI.
        geom = geometric_features(mask, None, None, 0.25)
        output = self.classifier(view, geom)
        probabilities = softmax(output.logits, self.meta.temperature)
        ood_score = self.ood.score(output.embedding) if output.embedding is not None else None
        ood = self.ood.is_ood(ood_score)
        route = bayes_route(self.taxonomy.to_rows(probabilities), self.cost)
        if ood:
            route_name, reason = "UNKNOWN", "out_of_distribution"
        else:
            route_name, reason = route.outcome.value, route.reason
        order = np.argsort(probabilities)[::-1][:5]
        top = []
        for i in order:
            type_name = self.meta.type_names[int(i)]
            bin_value = self.taxonomy.type_to_bin[type_name]
            top.append({"type": type_name, "label": _display_name(type_name),
                        "probability": round(float(probabilities[i]), 6),
                        "bin": None if bin_value is None else bin_value.value})
        winner = top[0]
        return {
            "input_mode": "image_plus_geometry" if "geom" in self.classifier.inputs else "image_only",
            "predicted_type": winner["type"],
            "predicted_label": winner["label"],
            "confidence": winner["probability"],
            "taxonomy_bin": winner["bin"],
            "route": route_name,
            "held": route_name == "UNKNOWN",
            "reason": reason,
            "ood_score": None if ood_score is None else round(float(ood_score), 6),
            "ood_threshold": self.ood.threshold,
            "is_ood": ood,
            "foreground_fraction": round(area_fraction, 4),
            "top_predictions": top,
            "geometric_features": ({name: round(float(value), 4)
                                    for name, value in zip(GEOM_FEATURE_NAMES, geom)}
                                   if "geom" in self.classifier.inputs else {}),
            "development_only": True,
            "warning": "Research-dataset model - not approved for physical waste routing.",
        }

    def _predict_yolo(self, image: np.ndarray) -> dict:
        result = self.classifier.predict(image, imgsz=224, device="cpu", verbose=False)[0]
        probabilities = result.probs.data.detach().cpu().numpy().astype(np.float64)
        order = np.argsort(probabilities)[::-1][:5]
        top = []
        for i in order:
            type_name = self.type_names[int(i)]
            bin_value = self.taxonomy.type_to_bin[type_name]
            top.append({"type": type_name, "label": _display_name(type_name),
                        "probability": round(float(probabilities[i]), 6),
                        "bin": None if bin_value is None else bin_value.value})
        winner = top[0]
        confidence = float(winner["probability"])
        held = confidence < self.minimum_confidence or winner["bin"] is None
        if confidence < self.minimum_confidence:
            reason = f"confidence_{confidence:.3f}_below_{self.minimum_confidence:.3f}"
        elif winner["bin"] is None:
            reason = "class_requires_manual_routing"
        else:
            reason = None
        uncertainty = 1.0 - confidence
        return {
            "input_mode": "image_only_yolov8",
            "predicted_type": winner["type"], "predicted_label": winner["label"],
            "confidence": confidence, "taxonomy_bin": winner["bin"],
            "route": "UNKNOWN" if held else winner["bin"], "held": held,
            "reason": reason, "ood_score": round(uncertainty, 6),
            "ood_threshold": round(1.0 - self.minimum_confidence, 6),
            "is_ood": confidence < self.minimum_confidence,
            "foreground_fraction": None, "top_predictions": top, "geometric_features": {},
            "development_only": True,
            "warning": "Research-dataset model - not approved for physical waste routing.",
        }


@lru_cache(maxsize=1)
def service() -> PredictionService:
    return PredictionService()


app = FastAPI(title="SentryGlide Vision API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    model = service()
    return {"status": "ok", "classes": len(model.type_names), "model": model.model_name,
            "input_mode": "image_only_yolov8" if model.mode == "yolo" else "onnx",
            "development_only": True}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPG, PNG, or WebP image")
    encoded = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(encoded) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image must be 10 MB or smaller")
    try:
        return service().predict(encoded)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


WEB_DIST = ROOT / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
