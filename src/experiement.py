import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from river import drift, metrics, ensemble
import matplotlib.pyplot as plt
from calibration import callibrate_model
from utils import train_words
from create_dataset import create_dataset
from create_graphs import plot_experiment_results






def run_experiment():
    calibrated_delta = callibrate_model()
    
    # Initialize models
    base_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    adaptive_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    base_model_acc = []
    adaptive_model_acc = []

    # Initialize drift detector
    drift_detector = drift.ADWIN(delta=calibrated_delta)
    drifts = []

    # Load Data
    file_path = "Datasets/drebin.parquet.zip"
    raw_data = create_dataset(file_path)

    # Prepare data
    base_data = train_words(raw_data.copy(), 1)
    adaptive_data = base_data.copy()
    
    train_data = base_data[base_data['temporal_bucket'] == 1]
    X_train = train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
    y_train = train_data['label']

    # Train models
    base_model.fit(X_train, y_train)
    adaptive_model.fit(X_train, y_train)

    # Start Simulation Loop
    for i in range(2, int(raw_data['temporal_bucket'].max()) + 1):
        print(f"Processing bin {i}...")
        
        base_train_data = base_data[base_data['temporal_bucket'] == i]
        X_base = base_train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
        y_base = base_train_data['label']
        
        adaptive_train_data = adaptive_data[adaptive_data['temporal_bucket'] == i]
        X_adaptive = adaptive_train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
        y_adaptive = adaptive_train_data['label']
        
        predictions = adaptive_model.predict(X_adaptive)
        errors = (predictions != y_adaptive).astype(int)  # 1 if wrong, 0 if correct

        drift_occurred = False
        for error_val in errors:
            drift_detector.update(float(error_val))
            if drift_detector.drift_detected:
                drifts.append(i)
                drift_occurred = True
                drift_detector = drift.ADWIN(delta=calibrated_delta)
                break
                
        if drift_occurred:
            adaptive_data = train_words(raw_data.copy(), i+1)
            new_adaptive_train_data = adaptive_data[adaptive_data['temporal_bucket'] == i]
            X_adaptive = new_adaptive_train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
            y_adaptive = new_adaptive_train_data['label']
            adaptive_model.fit(X_adaptive, y_adaptive)
        
        base_model_acc.append(accuracy_score(y_base, base_model.predict(X_base)))
        adaptive_model_acc.append(accuracy_score(y_adaptive, adaptive_model.predict(X_adaptive)))

    print("Simulation completed.")

    plot_experiment_results(raw_data, base_model_acc, adaptive_model_acc, drifts)
    

if __name__ == "__main__":
    run_experiment()
