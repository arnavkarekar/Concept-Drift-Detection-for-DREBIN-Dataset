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

from src.create_dataset import preprocess_features

# Load data
print("\nLoading data...")
# data = pickle.load(open('Datasets/drebin.pkl', 'rb'))
drebin_data = pickle.load(open('Datasets/androbin_sampled.pkl', 'rb'))
print("Data loaded.")

print("\nPreprocessing features...")
drebin_data = preprocess_features(drebin_data)

# Prepare data
print("\nPreparing data...")
train_data = drebin_data[drebin_data['TemporalBuckets'] == 1]
X_train = train_data.drop(['label', 'TemporalBuckets', 'sha256', 'submission_date'], axis=1)
y_train = train_data['label']
print("Data prepared.")

# Train models
print("\nStarting Simulation...")
base_model.fit(X_train, y_train)
adaptive_model.fit(X_train, y_train)

for i in range(2, int(drebin_data['TemporalBuckets'].max()) + 1):
    print(f"Processing bin {i}...")
    train_data = drebin_data[drebin_data['TemporalBuckets'] == i]
    X = train_data.drop(['label', 'TemporalBuckets', 'sha256', 'submission_date'], axis=1)
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

window = 5
base_model_acc_avg = []
for ind in range(len(base_model_acc) - window + 1):
    base_model_acc_avg.append(np.mean(base_model_acc[ind:ind+window]))

adaptive_model_acc_avg = []
for ind in range(len(adaptive_model_acc) - window + 1):
    adaptive_model_acc_avg.append(np.mean(adaptive_model_acc[ind:ind+window]))

bins_avg = bins[window - 1:]

plt.plot(bins_avg, base_model_acc_avg, label='Base Model')
plt.plot(bins_avg, adaptive_model_acc_avg, label='Adaptive Model')

# Plot a point on the adaptive model line for each unique drift detection
unique_drifts = sorted(set(drifts))
drift_x = []
drift_y = []
for drift in unique_drifts:
    if drift in bins_avg:
        idx = bins_avg.index(drift)
        drift_x.append(drift)
        drift_y.append(adaptive_model_acc_avg[idx])

plt.scatter(drift_x, drift_y, color='red', zorder=5, label='Drift Detections')

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


