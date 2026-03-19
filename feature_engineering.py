from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np

from .face_mesh_module import LandmarkFrame

# ── MediaPipe Face Mesh landmark indices ────────────────────────────────────
LEFT_EYE_LIDS        = (159, 145)
RIGHT_EYE_LIDS       = (386, 374)
LEFT_EYE_HORIZONTAL  = (33, 133)
RIGHT_EYE_HORIZONTAL = (362, 263)

# Full eyebrow landmark sets
LEFT_EYEBROW         = (55, 107, 46)
RIGHT_EYEBROW        = (285, 336, 276)

# Inner brow corners (for furrow)
LEFT_INNER_BROW      = 55
RIGHT_INNER_BROW     = 285

# Lips
LEFT_LIP_CORNER  = 61
RIGHT_LIP_CORNER = 291
TOP_LIP          = 13
BOTTOM_LIP       = 14
MOUTH_CENTER     = 0    # nose-base / upper lip centre

# Face structure
NOSE_TIP    = 1
CHIN        = 152
LEFT_CHEEK  = 234
RIGHT_CHEEK = 454


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _average_points(indices: List[int], landmarks: np.ndarray) -> np.ndarray:
    points = np.array([landmarks[idx] for idx in indices], dtype=np.float32)
    return points.mean(axis=0)


@dataclass
class TemporalMetric:
    window_seconds: float
    timestamps: Deque[float] = field(default_factory=deque)

    def add(self, timestamp: float) -> None:
        self.timestamps.append(timestamp)
        while self.timestamps and (timestamp - self.timestamps[0]) > self.window_seconds:
            self.timestamps.popleft()

    @property
    def count(self) -> int:
        return len(self.timestamps)


class FeatureExtractor:
    def __init__(
        self,
        smoothing_window: int = 5,
        blink_threshold: float = 0.23,
        blink_window_seconds: float = 60.0,
    ) -> None:
        self.smoothing_window = smoothing_window
        self.blink_threshold = blink_threshold
        self.previous_blink_state = False
        self.metrics_history: Dict[str, Deque[float]] = {
            "eyebrow":            deque(maxlen=smoothing_window),
            "lip_tension":        deque(maxlen=smoothing_window),
            "nod":                deque(maxlen=smoothing_window),
            "symmetry":           deque(maxlen=smoothing_window),
            "brow_furrow":        deque(maxlen=smoothing_window),
            "eye_openness":       deque(maxlen=smoothing_window),
            "lip_corner_dir":     deque(maxlen=smoothing_window),
        }
        self.blink_events = TemporalMetric(window_seconds=blink_window_seconds)
        self.previous_nose_height: float | None = None

    # ── helpers ─────────────────────────────────────────────────────────────

    def _ear(
        self,
        landmarks: np.ndarray,
        lids: tuple[int, int],
        horizontal_pair: tuple[int, int],
    ) -> float:
        upper     = landmarks[lids[0]]
        lower     = landmarks[lids[1]]
        horiz_len = _distance(landmarks[horizontal_pair[0]], landmarks[horizontal_pair[1]])
        return _distance(upper, lower) / max(horiz_len, 1e-5)

    # ── Feature 1: eyebrow raise (original) ─────────────────────────────────
    def _compute_eyebrow_raise(self, landmarks: np.ndarray) -> float:
        left_brow  = _average_points(list(LEFT_EYEBROW),  landmarks)
        right_brow = _average_points(list(RIGHT_EYEBROW), landmarks)
        anchor = (landmarks[LEFT_EYE_LIDS[0]] + landmarks[RIGHT_EYE_LIDS[0]]) * 0.5
        value  = ((abs(left_brow[1] - anchor[1]) + abs(right_brow[1] - anchor[1])) * 0.5)
        self.metrics_history["eyebrow"].append(value)
        return float(np.mean(self.metrics_history["eyebrow"]))

    # ── Feature 2: lip tension (original) ───────────────────────────────────
    def _compute_lip_tension(self, landmarks: np.ndarray) -> float:
        mouth_width  = _distance(landmarks[LEFT_LIP_CORNER], landmarks[RIGHT_LIP_CORNER])
        mouth_height = _distance(landmarks[TOP_LIP],         landmarks[BOTTOM_LIP])
        raw_ratio = mouth_width / max(mouth_height, 1e-5)
        tension   = float(np.clip((raw_ratio - 5.0) / 55.0, 0.0, 1.0))
        self.metrics_history["lip_tension"].append(tension)
        return float(np.mean(self.metrics_history["lip_tension"]))

    # ── Feature 3: head nod intensity (original) ────────────────────────────
    def _compute_head_nod(self, frame: LandmarkFrame) -> float:
        nose_y     = frame.landmarks[NOSE_TIP][1]
        chin_y     = frame.landmarks[CHIN][1]
        head_len   = abs(chin_y - nose_y)
        if self.previous_nose_height is None:
            self.previous_nose_height = nose_y
            return 0.0
        delta = abs(nose_y - self.previous_nose_height) / max(head_len, 1e-5)
        self.previous_nose_height = nose_y
        self.metrics_history["nod"].append(delta)
        return float(np.mean(self.metrics_history["nod"]))

    # ── Feature 4: facial symmetry (original) ───────────────────────────────
    def _compute_symmetry(self, landmarks: np.ndarray) -> float:
        left_dist  = _distance(landmarks[LEFT_CHEEK],  landmarks[NOSE_TIP])
        right_dist = _distance(landmarks[RIGHT_CHEEK], landmarks[NOSE_TIP])
        score = abs(left_dist - right_dist) / max((left_dist + right_dist) * 0.5, 1e-5)
        self.metrics_history["symmetry"].append(score)
        return float(np.mean(self.metrics_history["symmetry"]))

    # ── Feature 5: blink rate (original) ────────────────────────────────────
    def _compute_blink_rate(self, frame: LandmarkFrame) -> float:
        left_ratio  = self._ear(frame.landmarks, LEFT_EYE_LIDS,  LEFT_EYE_HORIZONTAL)
        right_ratio = self._ear(frame.landmarks, RIGHT_EYE_LIDS, RIGHT_EYE_HORIZONTAL)
        eye_ratio   = (left_ratio + right_ratio) * 0.5
        is_blinking = eye_ratio < self.blink_threshold
        if is_blinking and not self.previous_blink_state:
            self.blink_events.add(frame.timestamp)
        self.previous_blink_state = is_blinking
        minutes = max(self.blink_events.window_seconds / 60.0, 1e-3)
        return self.blink_events.count / minutes

    # ── Feature 6 (NEW): brow furrow ────────────────────────────────────────
    # Inner brow corners moving closer together = furrowed = anger/disgust.
    # Normalized by face width so it's scale-invariant.
    def _compute_brow_furrow(self, landmarks: np.ndarray) -> float:
        inner_gap  = _distance(landmarks[LEFT_INNER_BROW], landmarks[RIGHT_INNER_BROW])
        face_width = _distance(landmarks[LEFT_CHEEK],      landmarks[RIGHT_CHEEK])
        # At rest the gap is ~25-35 % of face width; furrowed it shrinks to ~10-15 %
        # Map: 0.30 → 0.0 (relaxed), 0.10 → 1.0 (fully furrowed)
        ratio   = inner_gap / max(face_width, 1e-5)
        furrow  = float(np.clip((0.30 - ratio) / 0.20, 0.0, 1.0))
        self.metrics_history["brow_furrow"].append(furrow)
        return float(np.mean(self.metrics_history["brow_furrow"]))

    # ── Feature 7 (NEW): eye openness ───────────────────────────────────────
    # Continuous EAR averaged over both eyes, smoothed.
    # Low → squinting/anger; neutral ~0.28; high → surprise/fear.
    def _compute_eye_openness(self, landmarks: np.ndarray) -> float:
        left  = self._ear(landmarks, LEFT_EYE_LIDS,  LEFT_EYE_HORIZONTAL)
        right = self._ear(landmarks, RIGHT_EYE_LIDS, RIGHT_EYE_HORIZONTAL)
        raw   = (left + right) * 0.5
        # Normalize: closed ~0.15, neutral ~0.28, wide ~0.40+
        openness = float(np.clip((raw - 0.10) / 0.35, 0.0, 1.0))
        self.metrics_history["eye_openness"].append(openness)
        return float(np.mean(self.metrics_history["eye_openness"]))

    # ── Feature 8 (NEW): lip corner direction ───────────────────────────────
    # Positive  → corners above mouth centre → HAPPY (smile)
    # Negative  → corners below mouth centre → ANGRY / SAD (frown)
    # Normalized by face height for scale invariance.
    def _compute_lip_corner_direction(self, landmarks: np.ndarray) -> float:
        face_height   = _distance(landmarks[NOSE_TIP], landmarks[CHIN])
        mouth_center_y = (landmarks[TOP_LIP][1] + landmarks[BOTTOM_LIP][1]) * 0.5
        left_corner_y  = landmarks[LEFT_LIP_CORNER][1]
        right_corner_y = landmarks[RIGHT_LIP_CORNER][1]
        avg_corner_y   = (left_corner_y + right_corner_y) * 0.5
        # In image coords Y increases downward, so:
        # corner ABOVE centre → diff negative → smile (map to positive)
        # corner BELOW centre → diff positive → frown (map to negative)
        raw = (mouth_center_y - avg_corner_y) / max(face_height, 1e-5)
        # Clamp to [-1, 1]
        direction = float(np.clip(raw / 0.05, -1.0, 1.0))
        self.metrics_history["lip_corner_dir"].append(direction)
        return float(np.mean(self.metrics_history["lip_corner_dir"]))

    # ── Public interface ─────────────────────────────────────────────────────
    def extract(self, frame: LandmarkFrame) -> Dict[str, float]:
        lm = frame.landmarks
        return {
            "eyebrow_raise":        self._compute_eyebrow_raise(lm),
            "lip_tension":          self._compute_lip_tension(lm),
            "head_nod_intensity":   self._compute_head_nod(frame),
            "symmetry_delta":       self._compute_symmetry(lm),
            "blink_rate":           self._compute_blink_rate(frame),
            "brow_furrow":          self._compute_brow_furrow(lm),
            "eye_openness":         self._compute_eye_openness(lm),
            "lip_corner_direction": self._compute_lip_corner_direction(lm),
        }
