from create_dataset import create_dataset
from utils import train_words
from river import drift
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def callibrate_model():
    print("\nStarting ADWIN Calibration...")
    ### Load the dataset with proper sampling and binning
    file_path = "Datasets/drebin.parquet.zip"
    data = create_dataset(file_path)

    ### Train Words on the full dataset
    data = train_words(data, 101)

    ### Train a model on the entire dataset
    print("Training Ideal Model on complete dataset...")
    X = data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
    y = data['label']
    
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, y)

    ### Initialize drift detector calibration
    current_delta = 0.4

    ### Run datashift simulation
    while True:
        print(f"\nTesting ADWIN with delta = {current_delta}")
        drift_detector = drift.ADWIN(delta=current_delta)
        drift_detected = False
        
        for i in range(1, int(data['temporal_bucket'].max()) + 1):
            train_data = data[data['temporal_bucket'] == i]
            X_bin = train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
            y_bin = train_data['label']
            
            predictions = model.predict(X_bin)
            errors = (predictions != y_bin).astype(int)

            for error_val in errors:
                drift_detector.update(float(error_val))
                if drift_detector.drift_detected:
                    drift_detected = True
                    break
                    
            if drift_detected:
                break
                
        if drift_detected:
            print(f"Drift detected at bin {i} with delta {current_delta}. Reducing delta by 2.")
            current_delta /= 2
        else:
            print(f"Calibration successful! No drift detected with delta {current_delta}")
            break

    return current_delta