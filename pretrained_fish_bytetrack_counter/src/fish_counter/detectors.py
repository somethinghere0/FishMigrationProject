from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


@dataclass
class DetectionResult:
    boxes: np.ndarray   # [N,4] xyxy
    scores: np.ndarray  # [N]
    class_ids: np.ndarray  # [N], fish=0


class BaseDetector:
    def predict_bgr(self, frame_bgr: np.ndarray) -> DetectionResult:
        raise NotImplementedError


class CFDRFDETRDetector(BaseDetector):
    """Community Fish Detector RF-DETR backend.

    Requires:
      pip install rfdetr supervision
    And a downloaded CFD .pth checkpoint.
    """

    def __init__(self, weights: str, resolution: int = 640, confidence: float = 0.3):
        self.weights = str(weights)
        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"CFD weights not found: {self.weights}\n"
                "Download a .pth checkpoint from the Community Fish Detector GitHub Releases page "
                "and update configs/cfd_counter.yaml -> model.weights."
            )
        self.confidence = float(confidence)
        self.resolution = int(resolution)
        try:
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
        except Exception as e:
            raise ImportError(
                "Could not import rfdetr. Install dependencies with: pip install -r requirements.txt\n"
                f"Original error: {e}"
            )

        # Choose RF-DETR class by resolution/filename hint. Medium/Small are both valid at 1024.
        name = Path(self.weights).name.lower()
        if "medium" in name:
            cls = RFDETRMedium
        elif "small" in name:
            cls = RFDETRSmall
        else:
            cls = RFDETRNano
        self.model = cls(pretrain_weights=self.weights, resolution=self.resolution)

    def predict_bgr(self, frame_bgr: np.ndarray) -> DetectionResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        detections = self.model.predict(pil_img, threshold=self.confidence)

        # supervision.Detections usually exposes xyxy, confidence, class_id.
        boxes = np.asarray(getattr(detections, "xyxy", np.zeros((0, 4))), dtype=np.float32)
        scores = np.asarray(getattr(detections, "confidence", np.zeros((len(boxes),))), dtype=np.float32)
        class_ids = np.zeros((len(boxes),), dtype=np.int32)
        return DetectionResult(boxes=boxes, scores=scores, class_ids=class_ids)


class UltralyticsDetector(BaseDetector):
    """Fallback detector for YOLO .pt weights.

    Use this with a fish-pretrained YOLO .pt model if available, or yolo11n.pt/yolov8n.pt for smoke tests.
    """

    def __init__(self, weights: str, confidence: float = 0.25):
        self.weights = str(weights)
        self.confidence = float(confidence)
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise ImportError(f"Could not import ultralytics. Run: pip install ultralytics\nOriginal error: {e}")
        self.model = YOLO(self.weights)

    def predict_bgr(self, frame_bgr: np.ndarray) -> DetectionResult:
        results = self.model.predict(frame_bgr, conf=self.confidence, verbose=False)
        if not results:
            return DetectionResult(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32))
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return DetectionResult(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32))
        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls = r.boxes.cls.detach().cpu().numpy().astype(np.int32)
        return DetectionResult(boxes=boxes, scores=scores, class_ids=cls)


def build_detector(cfg: dict) -> BaseDetector:
    backend = cfg.get("backend", "cfd_rfdetr").lower()
    if backend == "cfd_rfdetr":
        return CFDRFDETRDetector(
            weights=cfg["weights"],
            resolution=int(cfg.get("resolution", 640)),
            confidence=float(cfg.get("confidence", 0.3)),
        )
    if backend == "ultralytics":
        return UltralyticsDetector(weights=cfg["weights"], confidence=float(cfg.get("confidence", 0.25)))
    raise ValueError(f"Unknown model.backend: {backend}")
