from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# ── Emotion catalogue ────────────────────────────────────────────────────────
# Each entry: (icon, display_label, BGR colour for OpenCV overlay)
EMOTION_INFO: Dict[str, Tuple[str, str, Tuple[int, int, int]]] = {
    "angry":     ("🤬", "Angry",     (0,   30,  220)),   # red
    "happy":     ("😊", "Happy",     (0,   210, 50)),    # green
    "surprised": ("😲", "Surprised", (0,   200, 255)),   # amber
    "sad":       ("😢", "Sad",       (200, 100, 0)),     # blue-ish
    "disgusted": ("🤢", "Disgusted", (40,  160, 40)),    # dark green
    "fearful":   ("😨", "Fearful",   (180, 50,  180)),   # purple
    "neutral":   ("😐", "Neutral",   (130, 130, 130)),   # grey
}


@dataclass
class StressScore:
    """Kept as StressScore for backward-compat with main.py / dashboard.py."""
    score: float           # confidence of the winning emotion (0–1)
    label: str             # display label, e.g. "Angry"
    icon:  str             # emoji
    level: str             # emotion key, e.g. "angry"
    color: Tuple[int, int, int] = (130, 130, 130)  # BGR

    def formatted(self) -> str:
        return f"{self.icon} {self.label} ({self.score:.2f})"


class StressEstimator:
    """
    Rule-based multi-emotion classifier.
    Scores each emotion as a weighted combination of normalised features,
    then picks the winner.  Falls back to 'neutral' when nothing is strong.
    """

    # Minimum confidence to claim a non-neutral emotion
    THRESHOLD = 0.45  # Tuned back to middle-ground to revive all emotions
    def predict(self, features: Dict[str, float]) -> StressScore:
        scores = {
            "angry":     float(np.clip(self._score_angry(features),     0.0, 1.0)),
            "happy":     float(np.clip(self._score_happy(features),     0.0, 1.0)),
            "surprised": float(np.clip(self._score_surprised(features), 0.0, 1.0)),
            "sad":       float(np.clip(self._score_sad(features),       0.0, 1.0)),
            "disgusted": float(np.clip(self._score_disgusted(features), 0.0, 1.0)),
            "fearful":   float(np.clip(self._score_fearful(features),   0.0, 1.0)),
        }
        winner = max(scores, key=lambda k: scores[k])
        confidence = scores[winner]

        if confidence < self.THRESHOLD:
            winner, confidence = "neutral", 1.0 - confidence

        icon, label, color = EMOTION_INFO[winner]
        return StressScore(score=confidence, label=label, icon=icon,
                           level=winner, color=color)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(features: Dict[str, float], key: str,
              lo: float, hi: float) -> float:
        """Linearly map feature value from [lo, hi] → [0, 1]."""
        v = features.get(key, 0.0)
        return float(np.clip((v - lo) / max(hi - lo, 1e-5), 0.0, 1.0))

    # ── Per-emotion scorers ───────────────────────────────────────────────────

    def _score_angry(self, f: Dict[str, float]) -> float:
        """
        Anger cues:
          - Brows pulled DOWN (low eyebrow_raise)   → STRONGEST cue
          - Brows pulled TOGETHER (brow_furrow)      → strong cue
          - Eyes narrowed/squinted  (eye_openness LOW)
          - Lips pressed tight      (lip_tension HIGH)
          - Lip corners down        (lip_corner_direction NEGATIVE)
        """
        # Brows DOWN → eyebrow_raise is LOW for anger
        # Range: relaxed ~0.03-0.05, angry ~0.01-0.025  → invert
        low_brow = 1.0 - self._norm(f, "eyebrow_raise", 0.01, 0.055)

        # Brows pulled TOGETHER — any furrowing counts
        furrow  = self._norm(f, "brow_furrow",    0.0,  0.75)

        # Eyes narrowed (squinting)
        squint  = 1.0 - self._norm(f, "eye_openness", 0.15, 0.75)

        # Lips pressed or jaw set
        tension = self._norm(f, "lip_tension",    0.15, 0.80)

        # Lip corners pulled down
        frown   = float(np.clip(-f.get("lip_corner_direction", 0.0), 0.0, 1.0))

        return (
            0.25 * low_brow   
            + 0.25 * furrow   
            + 0.20 * squint
            + 0.20 * tension
            + 0.10 * frown
        )


    def _score_happy(self, f: Dict[str, float]) -> float:
        """
        Happiness cues:
          - Lip corners raised (smile)          (lip_corner_direction POSITIVE)
          - Eyes slightly narrowed by cheeks    (eye_openness MEDIUM)
          - Brows relaxed                        (brow_furrow LOW)
        """
        smile   = float(np.clip(f.get("lip_corner_direction", 0.0), 0.0, 1.0))
        
        # Bug Fix: Wide-open jaw (Surprise/Shock) mathematically forces the mouth's center 
        # down so far that the lip corners appear technically "raised" relative to it.
        # Since genuine smiles rarely have 100% bulging wide eyes, we aggressively penalize 
        # the "Happy" score when eyes are wide open to let Surprise/Shock win out.
        if f.get("eye_openness", 0.0) > 0.90:
            smile *= 0.30 
            
        relaxed = 1.0 - self._norm(f, "brow_furrow", 0.0, 0.5)
        return 0.65 * smile + 0.35 * relaxed

    def _score_surprised(self, f: Dict[str, float]) -> float:
        """
        Surprise cues:
          - Brows raised HIGH (above typical resting position)  (eyebrow_raise)
          - Eyes very wide open                                  (eye_openness)
          - Mouth open (low lip tension)                         (lip_tension low)
          - NOT furrowed — furrowed brows mean angry, not surprised (penalty)
        """
        brow_up    = self._norm(f, "eyebrow_raise", 0.055, 0.10)  # needs real raise
        wide_eye   = self._norm(f, "eye_openness",  0.72,  1.0)   # needs very wide eyes
        open_mouth = 1.0 - self._norm(f, "lip_tension", 0.0, 0.35)
        # Penalty: if brows are furrowed it's not surprise
        furrow_penalty = self._norm(f, "brow_furrow", 0.0, 0.5)

        return 0.35 * brow_up + 0.40 * wide_eye + 0.25 * open_mouth - 0.25 * furrow_penalty

    def _score_sad(self, f: Dict[str, float]) -> float:
        """
        Sadness cues:
          - Inner brows raised & angled          (eyebrow_raise MEDIUM, not high)
          - Lip corners drooping                 (lip_corner_direction NEGATIVE)
          - Slower blink / lowered gaze          (eye_openness LOW-MEDIUM)
        """
        frown      = float(np.clip(-f.get("lip_corner_direction", 0.0), 0.0, 1.0))
        brow_raise = self._norm(f, "eyebrow_raise", 0.02, 0.06)
        downcast   = 1.0 - self._norm(f, "eye_openness", 0.3, 0.7)

        return 0.45 * frown + 0.30 * brow_raise + 0.25 * downcast

    def _score_disgusted(self, f: Dict[str, float]) -> float:
        """
        Disgust cues:
          - Brows pulled down & together         (brow_furrow HIGH)
          - Upper lip raised / asymmetric mouth  (lip_tension HIGH + symmetry)
          - Lip corners pulled slightly down      (lip_corner_direction slightly NEGATIVE)
        """
        furrow  = self._norm(f, "brow_furrow",  0.2,  1.0)
        tension = self._norm(f, "lip_tension",  0.3,  0.9)
        asym    = self._norm(f, "symmetry_delta", 0.01, 0.05)

        return 0.40 * furrow + 0.35 * tension + 0.25 * asym

    def _score_fearful(self, f: Dict[str, float]) -> float:
        """
        Fear cues:
          - Brows raised HIGH AND furrowed simultaneously
            (product term: BOTH must be present at once)
          - Eyes very wide                       (eye_openness)
          - Slight head movement                 (head_nod_intensity)
        Fearful is distinguished from surprised by requiring brow furrow,
        and from angry by requiring raised (not lowered) brows and open eyes.
        """
        brow_up  = self._norm(f, "eyebrow_raise",     0.055, 0.10)   # needs real raise
        furrow   = self._norm(f, "brow_furrow",        0.25,  0.85)  # needs real furrow
        wide_eye = self._norm(f, "eye_openness",       0.72,  1.0)   # needs very wide eyes
        movement = self._norm(f, "head_nod_intensity", 0.15,  1.0)
        # Key distinguisher: fear needs BOTH brow raise AND brow furrow at the same time
        fear_brow = brow_up * furrow

        return 0.45 * fear_brow + 0.35 * wide_eye + 0.20 * movement
