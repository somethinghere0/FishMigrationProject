from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from .utils import xyxy_iou, center_xy


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    score: float
    age: int = 0
    hits: int = 1
    lost: int = 0
    counted: bool = False
    prev_center: np.ndarray | None = None
    center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def __post_init__(self):
        self.box = self.box.astype(np.float32)
        self.center = center_xy(self.box)
        if self.prev_center is None:
            self.prev_center = self.center.copy()

    def update(self, box: np.ndarray, score: float):
        self.prev_center = self.center.copy()
        self.box = box.astype(np.float32)
        self.score = float(score)
        self.center = center_xy(self.box)
        self.hits += 1
        self.lost = 0

    def mark_lost(self):
        self.age += 1
        self.lost += 1
        self.prev_center = self.center.copy()


class ByteTrackLite:
    """Small ByteTrack-style tracker for fish counting.

    This is intentionally lightweight for volunteer laptops. It performs:
      1) high-confidence association
      2) low-confidence association for unmatched tracks
      3) new track creation from unmatched high-confidence detections

    It does not include ReID embeddings or full Kalman filtering. For this project,
    this is enough to turn pretrained detector boxes into stable-ish track IDs.
    """

    def __init__(self, high_thresh=0.35, low_thresh=0.10, match_iou_thresh=0.25, max_lost_frames=30):
        self.high_thresh = float(high_thresh)
        self.low_thresh = float(low_thresh)
        self.match_iou_thresh = float(match_iou_thresh)
        self.max_lost_frames = int(max_lost_frames)
        self.tracks: list[Track] = []
        self.next_id = 1

    def _associate(self, tracks: list[Track], boxes: np.ndarray, iou_thresh: float):
        if len(tracks) == 0 or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        track_boxes = np.stack([t.box for t in tracks], axis=0)
        ious = xyxy_iou(track_boxes, boxes)
        cost = 1.0 - ious
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        matched_t = set()
        matched_d = set()
        for r, c in zip(row_ind, col_ind):
            if ious[r, c] >= iou_thresh:
                matches.append((int(r), int(c)))
                matched_t.add(int(r))
                matched_d.add(int(c))
        unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_d = [i for i in range(len(boxes)) if i not in matched_d]
        return matches, unmatched_t, unmatched_d

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[Track]:
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if len(boxes) == 0:
            for t in self.tracks:
                t.mark_lost()
            self.tracks = [t for t in self.tracks if t.lost <= self.max_lost_frames]
            return self.tracks

        high_idx = np.where(scores >= self.high_thresh)[0]
        low_idx = np.where((scores >= self.low_thresh) & (scores < self.high_thresh))[0]
        high_boxes, high_scores = boxes[high_idx], scores[high_idx]
        low_boxes, low_scores = boxes[low_idx], scores[low_idx]

        # Pass 1: high-confidence detections to all active tracks.
        matches, unmatched_track_idx, unmatched_high_det_idx = self._associate(self.tracks, high_boxes, self.match_iou_thresh)
        for ti, di in matches:
            self.tracks[ti].update(high_boxes[di], float(high_scores[di]))

        # Pass 2: low-confidence detections recover unmatched tracks.
        unmatched_tracks = [self.tracks[i] for i in unmatched_track_idx]
        low_matches, still_unmatched_local, _ = self._associate(unmatched_tracks, low_boxes, self.match_iou_thresh)
        recovered_track_global = set()
        for local_ti, low_di in low_matches:
            global_ti = unmatched_track_idx[local_ti]
            self.tracks[global_ti].update(low_boxes[low_di], float(low_scores[low_di]))
            recovered_track_global.add(global_ti)

        # Mark tracks that did not match high or low as lost.
        for global_ti in unmatched_track_idx:
            if global_ti not in recovered_track_global:
                self.tracks[global_ti].mark_lost()

        # Create new tracks from unmatched high-confidence detections.
        for di in unmatched_high_det_idx:
            self.tracks.append(Track(self.next_id, high_boxes[di], float(high_scores[di])))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost_frames]
        return self.tracks


def _side_of_line(point: np.ndarray, line: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = line
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


class LineCounter:
    def __init__(self, line, direction="right", min_hits_before_count=2):
        self.line = tuple(float(v) for v in line)
        self.direction = str(direction).lower()
        self.min_hits_before_count = int(min_hits_before_count)
        self.count = 0
        self.counted_ids: set[int] = set()

    def update(self, tracks: list[Track]) -> int:
        new_counts = 0
        x1, y1, x2, y2 = self.line
        vertical = abs(x2 - x1) < abs(y2 - y1)

        for t in tracks:
            if t.track_id in self.counted_ids or t.hits < self.min_hits_before_count or t.lost > 0:
                continue
            prev = t.prev_center
            cur = t.center
            crossed = False
            good_direction = True

            if vertical:
                line_x = x1
                crossed = (prev[0] < line_x <= cur[0]) or (prev[0] > line_x >= cur[0])
                if self.direction == "right":
                    good_direction = cur[0] > prev[0]
                elif self.direction == "left":
                    good_direction = cur[0] < prev[0]
            else:
                line_y = y1
                crossed = (prev[1] < line_y <= cur[1]) or (prev[1] > line_y >= cur[1])
                if self.direction == "down":
                    good_direction = cur[1] > prev[1]
                elif self.direction == "up":
                    good_direction = cur[1] < prev[1]

            if self.direction == "any":
                good_direction = True

            if crossed and good_direction:
                self.counted_ids.add(t.track_id)
                t.counted = True
                self.count += 1
                new_counts += 1
        return new_counts
