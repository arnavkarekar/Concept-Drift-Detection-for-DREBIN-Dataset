import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def display_freq(data, title, filename):
    data = data.copy()
    if not pd.api.types.is_datetime64_any_dtype(data['submission_date']):
        data['submission_date'] = pd.to_datetime(data['submission_date'])
    data['month_str'] = data['submission_date'].dt.strftime('%Y-%m')
    
    goodware_freq = data[data['label'] == 0].groupby('month_str').size().to_dict()
    malware_freq = data[data['label'] == 1].groupby('month_str').size().to_dict()

    dates = sorted(set(goodware_freq.keys()) | set(malware_freq.keys()))
    goodware_counts = [goodware_freq.get(d, 0) for d in dates]
    malware_counts = [malware_freq.get(d, 0) for d in dates]

    x = np.arange(len(dates))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.clf()
    plt.bar(x - width/2, goodware_counts, width, label='Goodware')
    plt.bar(x + width/2, malware_counts, width, label='Malware')
    plt.xticks(x, dates, rotation='vertical')
    plt.yscale('log')
    plt.ylabel('Number of Samples (log)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    os.makedirs("Results", exist_ok=True)
    plt.savefig(f"Results/{filename}")
    print(f"Saved plot to Results/{filename}")

def display_temporal_buckets(data, title, filename):
    # Group by bucket and label, then unstack to ensure alignment
    stats = data.groupby(['temporal_bucket', 'label']).size().unstack(fill_value=0)
    
    # Ensure both labels exist in the columns to prevent KeyErrors
    if 0 not in stats.columns:
        stats[0] = 0
    if 1 not in stats.columns:
        stats[1] = 0
        
    goodware_counts = stats[0]
    malware_counts = stats[1]
    
    buckets = sorted(data['temporal_bucket'].unique())
    x = np.arange(len(buckets))
    width = 0.4

    plt.figure(figsize=(12, 5))
    plt.clf()
    
    plt.bar(x - width/2, goodware_counts, width, label='Benign', color='#3498db')
    plt.bar(x + width/2, malware_counts, width, label='Malware', color='#e74c3c')
    
    # Logic to space out the xticks based on the number of buckets
    step = max(1, len(buckets) // 10)
    ticks = np.arange(0, len(buckets), step)
    tick_labels = [buckets[i] for i in ticks]
    plt.xticks(ticks, tick_labels, rotation=45, fontweight='bold')
    
    # Add a bit of space above the max bars
    max_val = max(goodware_counts.max(), malware_counts.max())
    y_max = max_val + (10 - max_val % 10) + 10 if max_val % 10 != 0 else max_val + 20
    plt.ylim(0, y_max)
    
    plt.ylabel('Number of Samples', fontweight='bold')
    plt.xlabel('Temporal Buckets', fontweight='bold')
    plt.title(title, fontweight='bold', fontsize=14)
    
    plt.legend(loc='upper right', frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    plt.tight_layout()
    
    os.makedirs("Results", exist_ok=True)
    plt.savefig(f"Results/{filename}")
    print(f"Saved plot to Results/{filename}")

def plot_experiment_results(data, base_model_acc, adaptive_model_acc, drifts, train_bin=5, filename="drebin.png"):
    print("\nPlotting results...")
    bins = list(range(1, int(data['temporal_bucket'].max()) + 1))

    window = 5
    base_model_acc_avg = []
    adaptive_model_acc_avg = []

    # Calculate the accuracy of the first (window - 1) bins
    for ind in range(window - 1):
        base_model_acc_avg.append(np.mean(base_model_acc[:ind+1]))
        adaptive_model_acc_avg.append(np.mean(adaptive_model_acc[:ind+1]))

    for ind in range(len(base_model_acc) - window + 1):
        base_model_acc_avg.append(np.mean(base_model_acc[ind:ind+window]))
        adaptive_model_acc_avg.append(np.mean(adaptive_model_acc[ind:ind+window]))

    bins_avg = bins

    plt.figure()
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

    # Plot a vertical at train_bin
    plt.axvline(x=train_bin, color='black', linestyle='-', linewidth=1)

    plt.scatter(drift_x, drift_y, color='red', zorder=5, label='Drift Detections')

    plt.xlabel('Bin ID')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save plot to results before showing (show() clears the figure)
    import os
    os.makedirs("Results", exist_ok=True)
    print(f"\nSaving plot to Results/{filename}...")
    plt.savefig(f"Results/{filename}")
    print("Plot saved.")
    print("Results plotted.")