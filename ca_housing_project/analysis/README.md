# Analysis Directory

This folder contains the analysis notebooks and preprocessing script for the California Housing Project.

---

## Files

### `ida.ipynb` — Initial Data Analysis (IDA)
- **Purpose**: Prepare raw data and create stratified train/test splits.  
- **Sections**:
  - Data loading
  - Data type analysis
  - Stratified train/test splitting
  - Saving raw train/test splits
- **Outputs**:
  - `data/train/housing_train.csv` (13 columns)
  - `data/test/housing_test.csv` (13 columns)

### `eda.ipynb` — Exploratory Data Analysis (EDA)
- **Purpose**: Perform exploratory analysis, visualize patterns, and create engineered features.  
- **Sections**:
  - Geographic data visualization
  - Feature correlation analysis
  - Feature engineering and creation
  - Saving processed dataset
- **Outputs**:
  - `data/train/housing_train_processed.csv` (24 features + target)
  - Plots saved in `/images`

### `preprocessing_pipeline.py`
- **Purpose**: Executable Python script that builds a scikit-learn preprocessing pipeline and saves the processed dataset.  
- **Usage**:
  ```bash
  python analysis/preprocessing_pipeline.py
