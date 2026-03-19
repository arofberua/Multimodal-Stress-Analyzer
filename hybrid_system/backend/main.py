from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import sys

# Add current dir to path to find video_analyzer
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DIR)

import video_analyzer

app = FastAPI(title="Hybrid Stress Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow local frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(DIR, "ml_model.joblib")

# Try to load model
try:
    ml_model = joblib.load(MODEL_PATH)
    print("Successfully loaded ml_model.joblib")
except Exception as e:
    print(f"Warning: Failed to load ml_model.joblib: {e}")
    ml_model = None

class QuestionnaireData(BaseModel):
    sleep_quality: int
    headaches: int
    academic_performance: int
    study_load: int
    extracurriculars: int

@app.post("/run_hybrid_analysis")
def run_hybrid_analysis(data: QuestionnaireData):
    global ml_model
    if not ml_model: # Try to reload on predicting just in case it trained while running
        try:
            ml_model = joblib.load(MODEL_PATH)
        except:
            raise HTTPException(status_code=500, detail="ML Model not loaded.")
            
    # Format for model prediction
    # Pydantic dict fields EXACTLY match the required feature aliases 
    df = pd.DataFrame([data.dict()])
    
    # Predict ML score (1 to 5)
    pred = ml_model.predict(df)
    ml_score = int(pred[0])
    
    # Run the 90 second video analysis
    # This will open cv2 locally on the host machine
    avg_heuristic_score = video_analyzer.run_hybrid(ml_score=ml_score, duration_sec=90)
    
    # Combine scores automatically
    final_score = (0.6 * ml_score) + (0.4 * avg_heuristic_score)
    
    return {
        "ml_score": ml_score,
        "heuristic_score": avg_heuristic_score,
        "final_score": round(final_score, 2)
    }
