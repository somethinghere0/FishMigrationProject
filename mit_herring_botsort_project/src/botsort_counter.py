from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x, y = x1 + w / 2, y1 + h / 2
    s = w * h
    r = w / max(h, 1e-6)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))


def z_to_bbox(x):
    cx, cy, s, r = x[:4].reshape(-1)
    w = np.sqrt(max(s * r, 1e-6))
    h = s / max(w, 1e-6)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


@dataclass
class Track:
    bbox: np.ndarray
    score: float
    track_id: int
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]
        ], dtype=np.float32)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]
        ], dtype=np.float32)
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = bbox_to_z(self.bbox)
        self.add_center()

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.bbox = z_to_bbox(self.kf.x)
        self.add_center()
        return self.bbox

    def update(self, bbox, score):
        self.time_since_update = 0
        self.hits += 1
        self.score = float(score)
        self.kf.update(bbox_to_z(bbox))
        self.bbox = z_to_bbox(self.kf.x)
        self.add_center()

    def add_center(self):
        x1, y1, x2, y2 = self.bbox
        self.history.append(((x1 + x2) / 2, (y1 + y2) / 2))
        self.history = self.history[-30:]


class BoTSORTCounter:
    """Educational BoT-SORT-inspired counter.

    Real BoT-SORT = detector + Kalman motion + camera-motion compensation + IoU/appearance association.
    This class keeps the core Kalman + IoU association, which is enough for students to learn counting.
    Add a ReID embedding branch later if fish frequently cross/occlude.
    """
    def __init__(self, iou_threshold=0.25, max_age=30, min_hits=2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: List[Track] = []
        self.counted_ids = set()

    def update(self, detections: np.ndarray, scores: np.ndarray):
        # detections: Nx4 xyxy
        predicted = np.array([t.predict() for t in self.tracks], dtype=np.float32) if self.tracks else np.empty((0, 4), dtype=np.float32)
        matches, unmatched_dets, unmatched_tracks = [], list(range(len(detections))), list(range(len(self.tracks)))
        if len(detections) and len(predicted):
            ious = iou_xyxy(detections, predicted)
            cost = 1.0 - ious
            det_idx, trk_idx = linear_sum_assignment(cost)
            matches = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
            for d, t in zip(det_idx, trk_idx):
                if ious[d, t] >= self.iou_threshold:
                    matches.append((d, t))
                    unmatched_dets.remove(d)
                    unmatched_tracks.remove(t)
        for d, t in matches:
            self.tracks[t].update(detections[d], scores[d])
        for d in unmatched_dets:
            self.tracks.append(Track(detections[d], float(scores[d]), self.next_id))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.time_since_update == 0]

    def count_unique_tracks(self):
        for t in self.tracks:
            if t.hits >= self.min_hits:
                self.counted_ids.add(t.track_id)
        return len(self.counted_ids)
