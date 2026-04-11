import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_features(df, max_features=1000):
    """
    Converts string columns into a numerical bag-of-words representation.
    """
    print("\nPreprocessing text features into numerical representation...")
    # List of columns containing string data
    text_cols = ['api_call', 'permission', 'url', 'provider', 'feature', 'intent', 
                 'activity', 'call', 'service_receiver', 'real_permission']
    
    # Keep only the text columns that actually exist in the dataframe
    existing_text_cols = [col for col in text_cols if col in df.columns]
    
    # Fill any missing values with empty strings
    for col in existing_text_cols:
        df[col] = df[col].fillna('')
            
    # Combine all text features for each sample into one space-separated string
    print("Combining text columns...")
    all_text = df[existing_text_cols].agg(' '.join, axis=1)
    
    # Use CountVectorizer with binary=True to create presence/absence numerical features
    print(f"Vectorizing combined text into top {max_features} numerical features...")
    vectorizer = CountVectorizer(binary=True, token_pattern=r'\S+', max_features=max_features)
    X = vectorizer.fit_transform(all_text)
    
    # Create a dense DataFrame to hold our new features, using int8 to optimize memory usage
    feature_names = vectorizer.get_feature_names_out()
    vec_df = pd.DataFrame(X.toarray(), columns=feature_names, dtype='int8')
    
    # Drop original text columns
    df = df.drop(columns=existing_text_cols)
    
    # Concatenate the new numerical features side by side
    merged_df = pd.concat([df.reset_index(drop=True), vec_df.reset_index(drop=True)], axis=1)
    return merged_df

def create_bins(data, num_bins):
    """
    Create bins for the given data.

    Parameters:
    data (pd.Series): The data to be binned.
    num_bins (int): The number of bins to create.

    Returns:
    pd.Series: The binned data.
    """
    try:
        # Create bins using pandas cut function
        print(f"\nCreating {num_bins} bins for the data...")

        data['submission_date'] = pd.to_datetime(data['submission_date'])
        data = data.sort_values(by='submission_date').reset_index(drop=True)
        data['bin_id'] = pd.qcut(data['submission_date'], q=35, labels=False)

        return data
    
    except Exception as e:
        print(f"An error occurred while creating bins: {e}")
        return None

def create_dataset(file_path):
    """
    Create a dataset from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    try:
        # Load the dataset using pandas
        print(f"\nLoading dataset from {file_path}...")
        df = pd.read_parquet(file_path)
        print("Dataset loaded successfully.")

        # Convert string sequence data into tabular numerical data
        df = preprocess_features(df, max_features=1000)
        print("Features preprocessed successfully.")

        # # Creating bins
        print("\nCreating bins for the dataset...")
        df = create_bins(df, num_bins=35)
        print("Bins created successfully.")

        # Save the dataset to a pickle file for faster future loading
        print("\nSaving the dataset to a pickle file...")
        df.to_pickle(file_path.replace(".parquet.zip", ".pkl"))
        print("Dataset saved to pickle file.")

        return df
        
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None
    
if __name__ == "__main__":
    file_path = "Datasets/drebin.parquet.zip"  # Update this to your dataset path
    dataset = create_dataset(file_path)

    print("\n--- Dataset Summary ---")
    print(f"Total Rows: {len(dataset)}")
    print(f"Total Columns: {len(dataset.columns)}")

    # Understand the layout: Look at the first 5 entries
    print("\n--- First 5 Rows ---")
    print(dataset.head())
    
    # Check column names (features)
    print("\n--- Feature Examples ---")
    print(dataset.columns.tolist()[:20]) # Shows first 20 features