import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

DIR = os.path.dirname(os.path.abspath(__file__))

def train_and_save():
    csv_path = os.path.join(DIR, "Stress Factors.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Safely alias the complex columns to simple internal features
    df.columns = [
        "sleep_quality",
        "headaches",
        "academic_performance",
        "study_load",
        "extracurriculars",
        "target_stress"
    ]
    
    print("Preparing data...")
    X = df.drop('target_stress', axis=1)
    y = df['target_stress']
    
    # Use Random Forest - better for these 1-5 scalar relations
    clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    
    acc = clf.score(X_test, y_test)
    print(f"Random Forest Model Accuracy: {acc * 100:.2f}%")
    
    model_path = os.path.join(DIR, "ml_model.joblib")
    # This dumps and overwrites the previous Logistic Regression model
    joblib.dump(clf, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_and_save()
