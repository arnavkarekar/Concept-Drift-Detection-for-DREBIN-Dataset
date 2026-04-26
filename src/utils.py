import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def train_words(df, bin_id=None):
    """
    Converts string columns into a numerical bag-of-words representation.
    """
    print("\nPreprocessing Features...")
    # List of columns containing string data
    text_cols = ['api_call', 'permission', 'url', 'provider', 'feature', 'intent', 
                 'activity', 'call', 'service_receiver', 'real_permission',
                 'resource.entry', 'source.class.package', 'manifest.permission', 
                 'manifest.activity', 'manifest.action', 'manifest.category', 
                 'manifest.feature', 'source.method.name']
    
    # Keep only the text columns that actually exist in the dataframe
    existing_text_cols = [col for col in text_cols if col in df.columns]
    
    # Fill any missing values with empty strings
    for col in existing_text_cols:
        df[col] = df[col].fillna('')
            
    # Combine all text features for each sample into one space-separated string
    all_text = df[existing_text_cols].agg(' '.join, axis=1)
    
    # Use CountVectorizer with binary=True to create presence/absence numerical features
    vectorizer = CountVectorizer(binary=True, token_pattern=r'\S+')
    
    if bin_id is None:
        X = vectorizer.fit_transform(all_text)
    else:
        # Fit only on the training subset up to bin_id
        train_text = all_text[df['temporal_bucket'] <= bin_id]
        vectorizer.fit(train_text)
        X = vectorizer.transform(all_text)
    
    # Create a dense DataFrame to hold our new features, using int8 to optimize memory usage
    feature_names = vectorizer.get_feature_names_out()
    vec_df = pd.DataFrame(X.toarray(), columns=feature_names, dtype='int8')
    
    # Drop original text columns
    df = df.drop(columns=existing_text_cols)
    
    # Concatenate the new numerical features side by side
    merged_df = pd.concat([df.reset_index(drop=True), vec_df.reset_index(drop=True)], axis=1)

    # Save the Vectorize
    with open("Datasets/vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)
    
    return merged_df