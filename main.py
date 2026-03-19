from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Tuple

import cv2
import numpy as np

from . import data_logger
from .dashboard import Dashboard
from .face_mesh_module import iter_landmarks_from_camera, LandmarkFrame
from .feature_engineering import FeatureExtractor
from .stress_model import StressEstimator, StressScore

# ── Default fallback colour (BGR) ───────────────────────────────────
DEFAULT_COLOR: Tuple[int, int, int] = (130, 130, 130)  # grey / neutral
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (60, 60, 60)
DARK_BG = (30, 30, 30)
BAR_BG = (50, 50, 50)
LANDMARK_COLOR = (200, 220, 200)

WINDOW = "AI Micro-Expression Analyzer"
PANEL_W = 320  # width of side panel


# ── Draw face-mesh dots on the camera image ─────────────────────────
def draw_landmarks(image: np.ndarray, landmarks: np.ndarray) -> None:
    h, w = image.shape[:2]
    for lm in landmarks:
        x, y = int(lm[0] * w), int(lm[1] * h)
        cv2.circle(image, (x, y), 1, LANDMARK_COLOR, -1)


# ── Draw a horizontal progress bar ──────────────────────────────────
def draw_bar(
    panel: np.ndarray,
    x: int,
    y: int,
    bar_w: int,
    bar_h: int,
    ratio: float,
    color: Tuple[int, int, int],
) -> None:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    cv2.rectangle(panel, (x, y), (x + bar_w, y + bar_h), BAR_BG, -1)
    fill_w = int(bar_w * ratio)
    if fill_w > 0:
        cv2.rectangle(panel, (x, y), (x + fill_w, y + bar_h), color, -1)
    cv2.rectangle(panel, (x, y), (x + bar_w, y + bar_h), WHITE, 1)


# ── Build the side panel ────────────────────────────────────────────
def build_panel(
    height: int,
    features: Dict[str, float],
    stress: StressScore,
) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), DARK_BG, dtype=np.uint8)
    color = getattr(stress, "color", DEFAULT_COLOR)

    # ── Emotion banner ──────────────────────────────────────────
    banner_h = 80
    cv2.rectangle(panel, (0, 0), (PANEL_W, banner_h), color, -1)
    cv2.putText(panel, stress.label.upper(), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2, cv2.LINE_AA)
    cv2.putText(panel, f"Confidence: {stress.score:.2f}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1, cv2.LINE_AA)

    # ── Metric bars (all 8 features) ────────────────────────────
    # (label, raw_scale_max, centre_offset for signed features)
    pretty = [
        ("eyebrow_raise",        "Eyebrow Raise",      0.08,  False),
        ("lip_tension",          "Lip Tension",         1.0,   False),
        ("brow_furrow",          "Brow Furrow",         1.0,   False),
        ("eye_openness",         "Eye Openness",        1.0,   False),
        ("lip_corner_direction", "Lip Corner Dir",      1.0,   True),   # signed
        ("head_nod_intensity",   "Head Nod",            1.5,   False),
        ("symmetry_delta",       "Asymmetry",           0.05,  False),
        ("blink_rate",           "Blinks /min",         30.0,  False),
    ]
    y = banner_h + 22
    bar_w = PANEL_W - 40
    bar_h = 14
    gap   = bar_h + 22

    for key, label, scale, signed in pretty:
        val = features.get(key, 0.0)
        if signed:
            # centre the bar: map [-1, +1] to [0, 1]
            ratio = float(np.clip((val + 1.0) / 2.0, 0.0, 1.0))
            disp  = f"{label}: {val:+.2f}"
        else:
            ratio = float(np.clip(val / scale, 0.0, 1.0))
            disp  = f"{label}: {val:.3f}"
        cv2.putText(panel, disp, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)
        draw_bar(panel, 15, y + 3, bar_w, bar_h, ratio, color)
        y += gap

    # ── Instructions ───────────────────────────────────────────
    cv2.putText(panel, "Press 'q' to quit", (15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, GRAY, 1, cv2.LINE_AA)
    return panel


# ── Combine camera feed + panel into one window ─────────────────────
def render_frame(
    frame: LandmarkFrame,
    features: Dict[str, float],
    stress: StressScore,
) -> np.ndarray:
    image = frame.image.copy()
    draw_landmarks(image, frame.landmarks)

    # Coloured border driven by detected emotion
    border_color = getattr(stress, "color", DEFAULT_COLOR)
    cv2.rectangle(image, (0, 0),
                  (image.shape[1] - 1, image.shape[0] - 1), border_color, 3)

    panel = build_panel(image.shape[0], features, stress)
    combined = np.hstack([image, panel])
    return combined


# ── Main loop ───────────────────────────────────────────────────────
def run(camera_index: int, log_path: pathlib.Path, display: bool, verbose: bool) -> None:
    extractor = FeatureExtractor()
    estimator = StressEstimator()
    fields = [
        "eyebrow_raise",
        "lip_tension",
        "head_nod_intensity",
        "symmetry_delta",
        "blink_rate",
        "brow_furrow",
        "eye_openness",
        "lip_corner_direction",
        "stress_score",
    ]
    dashboard = Dashboard(verbose=verbose)

    with data_logger.DataLogger(log_path, fieldnames=fields) as logger:
        for frame in iter_landmarks_from_camera(camera_index):
            features = extractor.extract(frame)
            stress_score = estimator.predict(features)
            metrics = {**features, "stress_score": stress_score.score}

            # Terminal output (always)
            dashboard.render(features, stress_score)
            logger.log(metrics)

            # OpenCV visual output
            if display and frame.image is not None:
                canvas = render_frame(frame, features, stress_score)
                cv2.imshow(WINDOW, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time micro-expression stress analyzer"
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--log-path",
        type=pathlib.Path,
        default=pathlib.Path("logs/session.csv"),
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV preview window",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full metric breakdown to terminal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        camera_index=args.camera_index,
        log_path=args.log_path,
        display=not args.no_display,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
