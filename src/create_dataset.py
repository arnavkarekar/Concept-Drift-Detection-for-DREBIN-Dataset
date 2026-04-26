import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from src.create_graphs import display_freq, display_temporal_buckets

def load_dataset(file_path):
    """
    Load the dataset from a parquet file.
    
    Parameters:
    file_path (str): The path to the parquet file.
    
    Returns:
    pd.DataFrame: The dataset.
    """
    try:
        if os.path.exists(file_path):
            print(f"Loading dataset from {file_path}...")
            df = pd.read_parquet(file_path)
            print("Dataset loaded successfully.")
            return df
        else:
            print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

def create_temporal_bins(data, num_bins=101):
    """
    Create temporal bins for the given dataset.
    
    Parameters:
    data (pd.DataFrame): The dataset.
    num_bins (int): The number of bins to create.
    
    Returns:
    pd.DataFrame: The dataset with temporal bins.
    """
    print("\nCreating Temporal Bins...")
    # Split data into malware and benign
    malware = data[data['label'] == 1]
    benign = data[data['label'] == 0]

    # Sort by submission date
    malware = malware.sort_values('submission_date')
    benign = benign.sort_values('submission_date')

    # Get the unique dates
    unique_dates = sorted(set(malware['submission_date']) | set(benign['submission_date']))

    # Get the number of samples in each bin
    for date in unique_dates:
        # print(date)
        malware_count = len(malware[malware['submission_date'] == date])
        benign_count = len(benign[benign['submission_date'] == date])
        diff = abs(malware_count - benign_count)
        if diff <= 0:
            continue
        if malware_count > benign_count:
            # Filter for the specific date, sample 'diff' rows, and drop them all at once
            to_drop = malware[malware['submission_date'] == date].sample(n=diff).index
            malware = malware.drop(to_drop)
        elif benign_count > malware_count:
            to_drop = benign[benign['submission_date'] == date].sample(n=diff).index
            benign = benign.drop(to_drop)
    
    # Split malware and benign into num_bins assign "temporal_bucket" column
    elements_per_bin = len(malware) // num_bins
    
    malware_chunks = np.array_split(malware, num_bins)
    benign_chunks = np.array_split(benign, num_bins)

    for i in range(num_bins):
        malware_chunks[i]['temporal_bucket'] = i + 1
        benign_chunks[i]['temporal_bucket'] = i + 1

    malware = pd.concat(malware_chunks)
    benign = pd.concat(benign_chunks)

    # Merge the two dataframes
    data = pd.concat([malware, benign])

    print("Finished creating temporal bins.")

    return data

def preprocess_features(df):
    """
    Convert date to datetime and sort by submission date.
    """
    date_col = 'submission_date' if 'submission_date' in df.columns else ('meta.vt.date' if 'meta.vt.date' in df.columns else None)

    if date_col:
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df[date_col] = df[date_col].dt.to_period('M')

        # Sort by date
        df = df.sort_values(date_col)
        
    print("Finished preprocessing features.")

    return df
    
def create_dataset(file_path):
    """
    Create a dataset from a parquet file.
    
    Parameters:
    file_path (str): The path to the parquet file.
    
    Returns:
    pd.DataFrame: The dataset.
    """
    df = load_dataset(file_path)
    
    if df is None:
        return None
    df = preprocess_features(df)
    df = create_temporal_bins(df)

    print("Saving dataset to Datasets/drebin.pkl...")
    with open("Datasets/drebin.pkl", "wb") as file:
        pickle.dump(df, file)
    print("Dataset saved successfully.")
    
    return df

if __name__ == "__main__":
    file_path = "Datasets/drebin.parquet.zip"
    
    dataset = load_dataset(file_path)
    display_freq(dataset, "Frequencies", "freq_original.png")
    
    dataset = create_dataset(file_path)
    display_temporal_buckets(dataset, "Frequencies", "freq_temporal.png")

    # print("\n--- Dataset Summary ---")
    # print(f"Total Rows: {len(dataset)}")
    # print(f"Total Columns: {len(dataset.columns)}")

    # print("\n--- First 5 Rows ---")
    # print(dataset.head())
    
    # print("\n--- Feature Examples ---")
    # print(dataset.columns.tolist()[:20])
