from create_dataset import load_dataset




def callibration():
    ### Load the dataset with proper sampling and binning
    file_path = "Datasets/drebin.parquet.zip"
    data = create_dataset(file_path)

    ### Initialize model
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

    ### Train Words
    train_words(data)

    ### Train a model on the entire dataset
    X = data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
    y = data['label']
    model.fit(X, y)

    ### Initialize drift detector
    drift_detector = drift.ADWIN()

    ### Run datashift simulation
    for i in range(2, int(data['temporal_bucket'].max()) + 1):
        print(f"Processing bin {i}...")
        train_data = data[data['temporal_bucket'] == i]
        X = train_data.drop(['label', 'temporal_bucket', 'sha256', 'submission_date'], axis=1)
        y = train_data['label']
        
        predictions = model.predict(X)
        errors = (predictions != y).astype(int)  # 1 if wrong, 0 if correct

        for error_val in errors:
            drift_detector.update(error_val)
            if drift_detector.drift_detected:
                ### Adjust calibration
                drift_detector.reset()
                continue


    ### Adjust calibration




    ### Repeat