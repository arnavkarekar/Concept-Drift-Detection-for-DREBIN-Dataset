import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from create_dataset import load_dataset

def display_freq(data, title, filename):
    data = data.copy()
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
    goodware_counts = data[data['label'] == 0].groupby('TemporalBuckets').size()
    malware_counts = data[data['label'] == 1].groupby('TemporalBuckets').size()
    
    buckets = sorted(data['TemporalBuckets'].unique())
    x = np.arange(len(buckets))
    width = 0.4

    plt.figure(figsize=(12, 5))
    plt.clf()
    
    plt.bar(x - width/2, goodware_counts, width, label='Benign')
    plt.bar(x + width/2, malware_counts, width, label='Malware')
    
    # Show exactly 11 x-ticks corresponding to buckets 1, 11, 21 ... 101
    ticks = np.arange(0, len(buckets), 10)
    tick_labels = [buckets[i] for i in ticks]
    plt.xticks(ticks, tick_labels, rotation=45, fontweight='bold')
    
    # Add a bit of space above the max bars
    max_val = max(goodware_counts.max(), malware_counts.max())
    y_max = max_val + (10 - max_val % 10) + 10 if max_val % 10 != 0 else max_val + 20
    plt.ylim(0, y_max)
    
    plt.ylabel('Number of Samples', fontweight='bold')
    plt.xlabel('TemporalBuckets', fontweight='bold')
    plt.title(title, fontweight='bold', fontsize=14)
    
    plt.legend(loc='upper left', frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    plt.tight_layout()
    
    os.makedirs("Results", exist_ok=True)
    plt.savefig(f"Results/{filename}")
    print(f"Saved plot to Results/{filename}")


def main():
    path = "Datasets/drebin.parquet.zip"
    if not os.path.exists(path):
        if os.path.exists("Datasets/drebin.parquet"):
            path = "Datasets/drebin.parquet"
            
    print(f"Loading data from {path}...")
    try:
        df = load_dataset(path)
    except Exception as e:
        print(f"Could not load via load_dataset. Falling back to pandas. Error: {e}")
        df = pd.read_parquet(path)

    df['submission_date'] = pd.to_datetime(df['submission_date'])
    
    print("\nGenerating original frequency chart...")
    display_freq(df, "Drebin Frequencies", "drebin_freq.png")
    
    print("\nPerforming 1:1 sampled ordinal temporal bucketing...")
    malware = df[df['label'] == 1].sort_values('submission_date').reset_index(drop=True)
    benign = df[df['label'] == 0].sort_values('submission_date').reset_index(drop=True)
    
    num_bins = 101
    min_len = min(len(malware), len(benign))
    elements_per_bin = min_len // num_bins
    malware_elements_per_bin = len(malware) // num_bins
    benign_elements_per_bin = len(benign) // num_bins
    
    print(f"Total Malware: {len(malware)}, Total Benign: {len(benign)}")
    print(f"Using {num_bins} bins, {elements_per_bin} elements per bin per class.")
    
    # Split the sorted data evenly into num_bins chunks (automatically handles remainders!)
    malware_chunks = np.array_split(malware, num_bins)
    benign_chunks = np.array_split(benign, num_bins)
    
    sampled_list = []
    for i in range(num_bins):
        m_bin = malware_chunks[i].copy()
        b_bin = benign_chunks[i].copy()
        
        m_bin['TemporalBuckets'] = i + 1
        b_bin['TemporalBuckets'] = i + 1
        
        m_sampled = m_bin.sample(n=elements_per_bin, replace=False, random_state=42)
        b_sampled = b_bin.sample(n=elements_per_bin, replace=False, random_state=42)
        
        sampled_list.append(m_sampled)
        sampled_list.append(b_sampled)
        
    sampled_df = pd.concat(sampled_list, ignore_index=True)
    
    print("\nGenerating 1:1 sampled frequency chart...")
    display_temporal_buckets(sampled_df, "Distribution of Benign and Malware Samples Across Months", "drebin_freq_undersampled.png")
    print("\nDone.")
    
if __name__ == "__main__":
    main()
