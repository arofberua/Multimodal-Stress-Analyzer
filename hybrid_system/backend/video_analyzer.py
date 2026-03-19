import sys
import os
import types
import importlib.util
import time
import cv2
import numpy as np

# Create microexpr package dynamically to import core modules
pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pkg_name = "microexpr"

if pkg_name not in sys.modules:
    package = types.ModuleType(pkg_name)
    package.__path__ = [pkg_dir]
    package.__package__ = pkg_name
    sys.modules[pkg_name] = package

def load_module(name):
    filepath = os.path.join(pkg_dir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", filepath)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[f"{pkg_name}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

# Load required modules from the parent project
data_logger = load_module("data_logger")
face_mesh_module = load_module("face_mesh_module")
feature_engineering = load_module("feature_engineering")
stress_model = load_module("stress_model")
dashboard_mod = load_module("dashboard")
main_mod = load_module("main")

# Mapping human emotions to the 1-5 Stress Scale directly
EMOTION_TO_STRESS = {
    "happy": 1.0,      # Completely relaxed
    "neutral": 2.0,    # Baseline
    "surprised": 3.0,  # Alert / Mild stress
    "sad": 4.0,        # High internal stress
    "disgusted": 4.5,  # High stress / rejection
    "fearful": 5.0,    # Severe stress
    "angry": 5.0       # Severe stress
}

def run_hybrid(ml_score: int, duration_sec: int = 90) -> float:
    print(f"Starting OpenCV analysis for {duration_sec}s...")
    extractor = feature_engineering.FeatureExtractor()
    estimator = stress_model.StressEstimator()
    
    start_time = time.time()
    last_update_time = start_time
    
    display_features = None
    display_stress = None
    scores = []
    
    window_name = "Hybrid Stress Analyzer (Recording)"
    
    for frame in face_mesh_module.iter_landmarks_from_camera(0):
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed > duration_sec:
            break
            
        features = extractor.extract(frame)
        stress_score = estimator.predict(features)
        
        # We log the specific level's impact on stress (1 to 5) instead of the raw confidence! 
        # This fixes the bug where averaging 0.5 confidence always returned EXACTLY 3.0.
        mapped_stress_value = EMOTION_TO_STRESS.get(stress_score.level, 3.0)
        scores.append(mapped_stress_value)
        
        # Update UI instantly every frame (removed the 2-second delay for smoother feedback)
        display_features = features
        display_stress = stress_score
            
        if frame.image is not None and display_features is not None:
            canvas = main_mod.render_frame(frame, display_features, display_stress)
            
            # Simple text overlay for countdown and ML score
            rem = max(0, int(duration_sec - elapsed))
            cv2.putText(canvas, f"Recording: {rem}s remaining", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(canvas, f"ML Base Score: {ml_score}/5", (20, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
            cv2.imshow(window_name, canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    try:
        cv2.destroyWindow(window_name)
    except Exception:
        pass
    
    if not scores:
        return 1.0
        
    # The scores are ALREADY mapped 1 to 5, so we just average them accurately.
    avg_score = sum(scores) / len(scores)
    return float(avg_score)

if __name__ == "__main__":
    # Test script locally
    print("Testing Video Analyzer...")
    res = run_hybrid(ml_score=3, duration_sec=5)
    print(f"Final Heuristic Avg (1-5): {res}")
