import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pyarrow.parquet as pq
import pickle as pkl

def display_freq(goodware_freq, malware_freq, title, filename):
    dates = sorted(set(goodware_freq.keys()) | set(malware_freq.keys()))
    goodware_counts = [goodware_freq.get(d, 0) for d in dates]
    malware_counts = [malware_freq.get(d, 0) for d in dates]

    x = np.arange(len(dates))
    width = 0.35

    plt.figure(figsize=(15, 6))
    plt.clf()
    plt.bar(x - width/2, goodware_counts, width, label='Goodware')
    plt.bar(x + width/2, malware_counts, width, label='Malware')
    
    # Androbin spans many dates, space ticks reasonably
    tick_step = max(1, len(dates) // 30)
    ticks = np.arange(0, len(dates), tick_step)
    tick_labels = [dates[i] for i in ticks]

    plt.xticks(ticks, tick_labels, rotation='vertical')
    plt.yscale('log')
    plt.ylabel('Number of Samples (log)')
    plt.xlabel('Date (YYYY-MM)')
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
    path = "Datasets/androbin.parquet"
    if not os.path.exists(path):
        if os.path.exists("Datasets/androbin.parquet.zip"):
            path = "Datasets/androbin.parquet.zip"
            
    print(f"Aggregating data from {path} in batches (memory efficient)...")
    
    goodware_freq = {}
    malware_freq = {}
    
    # Load efficiently
    pf = pq.ParquetFile(path)
    total_processed = 0
    
    malware_dates = []
    goodware_dates = []

    # Read 'Unnamed: 0', 'label' and 'meta.vt.date' columns to minimize RAM exhaustion
    for batch in pf.iter_batches(batch_size=50000, columns=['Unnamed: 0', 'label', 'meta.vt.date']):
        df_chunk = batch.to_pandas()
        
        # Ensure correct columns exist
        if 'label' not in df_chunk.columns or 'meta.vt.date' not in df_chunk.columns:
            # print("Missing required columns in batch, skipping...")
            continue
            
        df_chunk = df_chunk.dropna(subset=['meta.vt.date', 'label'])
        
        df_chunk['meta.vt.date'] = pd.to_datetime(df_chunk['meta.vt.date'])
        df_chunk['month_str'] = df_chunk['meta.vt.date'].dt.strftime('%Y-%m')
        
        g_counts = df_chunk[df_chunk['label'] == 0].groupby('month_str').size()
        m_counts = df_chunk[df_chunk['label'] == 1].groupby('month_str').size()
        
        for k, v in g_counts.items():
            goodware_freq[k] = goodware_freq.get(k, 0) + v
            
        for k, v in m_counts.items():
            malware_freq[k] = malware_freq.get(k, 0) + v
            
        total_processed += len(df_chunk)
        
        m_chunk = df_chunk[df_chunk['label'] == 1][['Unnamed: 0', 'meta.vt.date', 'label']]
        b_chunk = df_chunk[df_chunk['label'] == 0][['Unnamed: 0', 'meta.vt.date', 'label']]
        malware_dates.append(m_chunk)
        goodware_dates.append(b_chunk)
        
        print(f"Processed {total_processed} rows...", end='\r')
        
    print("\n\nGenerating frequency chart...")
    display_freq(goodware_freq, malware_freq, "Androbin Frequencies", "androbin_freq.png")
    print("Done.")

    ### Temporal Sampling ###
    print("\nPerforming 1:1 sampled ordinal temporal bucketing...")
    
    malware = pd.concat(malware_dates, ignore_index=True)
    benign = pd.concat(goodware_dates, ignore_index=True)
    
    malware = malware.sort_values('meta.vt.date').reset_index(drop=True)
    benign = benign.sort_values('meta.vt.date').reset_index(drop=True)
    
    num_bins = 101
    min_len = min(len(malware), len(benign))
    elements_per_bin = min_len // num_bins
    
    print(f"Total Malware: {len(malware)}, Total Benign: {len(benign)}")
    print(f"Using {num_bins} bins, {elements_per_bin} elements per bin per class.")
    
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

    # Convert the sampled DataFrame IDs to a set for fast lookup
    sampled_ids = set(sampled_df['Unnamed: 0'].tolist())
    
    # Map the IDs back to their temporal buckets
    id_to_bucket = sampled_df.set_index('Unnamed: 0')['TemporalBuckets'].to_dict()

    print(f"Sampled {len(sampled_ids)} items. Running second pass to extract full features...")
    
    full_chunks = []
    total_processed = 0
    # Process original dataset again to get ALL columns for the sampled rows
    for batch in pf.iter_batches(batch_size=50000):
        df_chunk = batch.to_pandas()
        
        # Filter to keep only the rows we've sampled
        mask = df_chunk['Unnamed: 0'].isin(sampled_ids)
        if mask.any():
            full_chunks.append(df_chunk[mask].copy())
            
        total_processed += len(df_chunk)
        print(f"Second pass: Processed {total_processed} rows...", end='\r')
        
    print("\nAssembling final dataset...")
    final_df = pd.concat(full_chunks, ignore_index=True)
    
    # Attach the assigned temporal buckets!
    final_df['TemporalBuckets'] = final_df['Unnamed: 0'].map(id_to_bucket)

    print("Saving fully featured sampled dataset to Datasets/androbin_sampled.pkl...")
    with open("Datasets/androbin_sampled.pkl", "wb") as file:
        pkl.dump(final_df, file)
    print("Sampled dataset saved successfully.")
    
    print("\nGenerating 1:1 sampled frequency chart...")
    display_temporal_buckets(sampled_df, "Distribution of Benign and Malware Samples Across Months (Androbin)", "androbin_freq_undersampled.png")


if __name__ == "__main__":
    main()
