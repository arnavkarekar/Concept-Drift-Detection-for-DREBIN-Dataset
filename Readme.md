# Malware Detection: Concept Drift Analysis

This repository contains the source code for analyzing and detecting concept drift in Android malware datasets, specifically focusing on the Drebin and Androbin datasets. The core experiment simulates a data stream over time, comparing a static machine learning model against an adaptive model that retrains when concept drift is detected.

## Repository Structure

- `Datasets/`: Directory intended for raw and processed datasets (e.g., `drebin.parquet.zip`, `androbin.parquet`).
- `Results/`: Directory where generated graphs, frequency charts, and experiment results are saved.
- `src/`: Contains the core source code modules for the experiment pipeline.
  - `create_dataset.py`: Handles loading dataset files and performing temporal binning (creating balanced time-based chunks of malware and goodware).
  - `create_graphs.py`: Contains utility functions for plotting temporal distributions and experiment results (accuracy over time, drift points).
  - `experiement.py`: The main experiment script. It simulates a data stream across temporal bins, evaluating a base Random Forest model (no retraining) against an adaptive Random Forest model that uses ADWIN for drift detection and retrains when necessary.
  - `calibration.py`: Logic for calibrating the ADWIN drift detector's delta parameter.
  - `utils.py`: Utility functions for feature engineering and dataset transformations.
- `drebin.py`: An exploratory data analysis script specifically for the Drebin dataset. It generates frequency charts and applies a 1:1 sampled ordinal temporal bucketing strategy to handle data chronological ordering.
- `androbin.py`: A similar exploratory data analysis script for the larger Androbin dataset, implemented with batch processing to manage memory efficiency while generating frequency charts and temporal buckets.

## Overview of the Experiment

1. Temporal Binning: The data is sorted chronologically and split into equal-sized temporal bins, ensuring a 1:1 ratio of malware to benign samples per bin.
2. Simulation Loop: The experiment iterates through these bins chronologically to simulate a real-world data stream.
3. Models:
   - Base Model: A Random Forest classifier trained only on the first temporal bin. Its performance typically degrades over time due to concept drift.
   - Adaptive Model: Starts the same as the base model but monitors its prediction error using River's ADWIN drift detector. When drift is detected, it retrains on the most recent data to maintain high accuracy.

## How to Run

To execute the main experiment and generate accuracy comparison graphs:
1. Ensure the datasets are placed in the `Datasets/` folder.
2. Run `python src/experiement.py` from the root directory.
3. Check the `Results/` folder for the generated plots and visual analysis.
