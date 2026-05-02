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






def run_experiment(train_bin=5, graph_title="drebin"):
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
    base_data = train_words(raw_data.copy(), train_bin)
    adaptive_data = base_data.copy()
    
    train_data = base_data[base_data['temporal_bucket'] <= train_bin]
    X_train = train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
    y_train = train_data['label']

    # Train models
    base_model.fit(X_train, y_train)
    adaptive_model.fit(X_train, y_train)

    # Start Simulation Loop
    drift_count = 0
    for i in range(1, int(raw_data['temporal_bucket'].max()) + 1):
        print(f"Processing bin {i}...")
        
        base_train_data = base_data[base_data['temporal_bucket'] == i]
        X_base = base_train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
        y_base = base_train_data['label']
        
        adaptive_train_data = adaptive_data[adaptive_data['temporal_bucket'] == i]
        X_adaptive = adaptive_train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
        y_adaptive = adaptive_train_data['label']
        
        predictions = adaptive_model.predict(X_adaptive)
        errors = (predictions != y_adaptive).astype(int)  # 1 if wrong, 0 if correct

        if i > train_bin:
            for error_val in errors:
                drift_detector.update(float(error_val))
                if drift_detector.drift_detected:
                    drifts.append(i)
        
        drift_occurred = False
        if (len(drifts) > drift_count):
            drift_count+=1
            drift_occurred = True
                
        if drift_occurred:
            adaptive_data = train_words(raw_data.copy(), i)
            new_adaptive_train_data = adaptive_data[adaptive_data['temporal_bucket'] <= i]
            X_adaptive_retrain = new_adaptive_train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
            y_adaptive_retrain = new_adaptive_train_data['label']
            adaptive_model.fit(X_adaptive_retrain, y_adaptive_retrain)
        
        base_model_acc.append(accuracy_score(y_base, base_model.predict(X_base)))
        adaptive_model_acc.append(accuracy_score(y_adaptive, predictions))

    print("Simulation completed.")

    plot_experiment_results(raw_data, base_model_acc, adaptive_model_acc, drifts, train_bin, f"{graph_title}_{train_bin}.png")
    

if __name__ == "__main__":
    run_experiment(train_bin=5, graph_title="drebin")
    run_experiment(train_bin=15, graph_title="drebin")
    run_experiment(train_bin=25, graph_title="drebin")
    run_experiment(train_bin=35, graph_title="drebin")
    run_experiment(train_bin=45, graph_title="drebin")
    run_experiment(train_bin=55, graph_title="drebin")
    run_experiment(train_bin=65, graph_title="drebin")
    run_experiment(train_bin=75, graph_title="drebin")
    run_experiment(train_bin=85, graph_title="drebin")
    run_experiment(train_bin=95, graph_title="drebin")

