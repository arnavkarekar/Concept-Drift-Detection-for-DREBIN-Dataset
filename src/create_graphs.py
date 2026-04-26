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