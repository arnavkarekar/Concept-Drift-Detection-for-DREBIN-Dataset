import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from river import drift, metrics, ensemble
import matplotlib.pyplot as plt

# Initialize models
print("\nInitializing models...")
base_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
adaptive_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

base_model_acc = []
adaptive_model_acc = []
print("Models initialized.")

# Initialize drift detector
print("\nInitializing drift detector...")
drift_detector = drift.ADWIN()
drifts = []
print("Drift detector initialized.")

# Load data
print("\nLoading data...")
data = pickle.load(open('Datasets/drebin.pkl', 'rb'))
print("Data loaded.")

# Prepare data
print("\nPreparing data...")
train_data = data[data['bin_id'] == 0]
X_train = train_data.drop(['label', 'bin_id', 'sha256', 'submission_date'], axis=1)
y_train = train_data['label']
print("Data prepared.")

# Train models
print("\nStarting Simulation...")
base_model.fit(X_train, y_train)
adaptive_model.fit(X_train, y_train)

for i in range(1, int(data['bin_id'].max()) + 1):
    print(f"Processing bin {i}...")
    train_data = data[data['bin_id'] == i]
    X = train_data.drop(['label', 'bin_id', 'sha256', 'submission_date'], axis=1)
    y = train_data['label']
    
    predictions = adaptive_model.predict(X)
    errors = (predictions != y).astype(int)  # 1 if wrong, 0 if correct

    for error_val in errors:
        drift_detector.update(error_val)
        if drift_detector.drift_detected:
            # Handle detected drift point...
            drifts.append(i)
            adaptive_model.fit(X, y)
    
    base_model_acc.append(accuracy_score(y, base_model.predict(X)))
    adaptive_model_acc.append(accuracy_score(y, adaptive_model.predict(X)))

print("Simulation completed.")

# Plot results accuracy vs bin_id
print("\nPlotting results...")
bins = list(range(1, len(base_model_acc) + 1))

plt.plot(bins, base_model_acc, label='Base Model')
plt.plot(bins, adaptive_model_acc, label='Adaptive Model')

# Plot a point on the adaptive model line for each unique drift detection
unique_drifts = sorted(set(drifts))
drift_y = [adaptive_model_acc[drift - 1] for drift in unique_drifts]
plt.scatter(unique_drifts, drift_y, color='red', zorder=5, label='Drift Detections')

plt.xlabel('Bin ID')
plt.ylabel('Accuracy')
plt.legend()

# Save plot to results before showing (show() clears the figure)
import os
os.makedirs("Results", exist_ok=True)
print("\nSaving plot to Results/drebin.png...")
plt.savefig("Results/drebin.png")
print("Plot saved.")

plt.show()
print("Results plotted.")


