
---

`/models/README.md`

```markdown
# Models Directory

This folder contains model training notebooks for the California Housing Project.

---

## Files

Each notebook follows the same structure with clearly marked Markdown headers:

### Sections
1. **# Data Loading**
   - Reads processed dataset (`data/train/housing_train_processed.csv`)
   - Uses 24 engineered features (target excluded)

2. **# Model Fitting**
   - Initializes and trains the model
   - Displays training results

3. **# Cross-Validation**
   - Performs CV evaluation
   - Reports RMSE scores and statistics

4. **# Hyperparameter Tuning**
   - Uses `GridSearchCV` or `RandomizedSearchCV`
   - Reports best parameters and improved performance

5. **# Model Saving**
   - Saves trained model as `.pkl` into `/models`

---

## Notebooks

- **`LinearRegression.ipynb`**
  - Fits a linear regression model
  - Saves `linear_regression_model.pkl`

- **`DecisionTree.ipynb`**
  - Fits a decision tree regressor
  - Saves `decision_tree_model.pkl`

- **`RandomForest.ipynb`**
  - Fits a random forest regressor
  - Saves `random_forest_model.pkl`

- **`SVR.ipynb`**
  - Fits a support vector regressor
  - Saves `svr_model.pkl`

---

## Notes
- All notebooks use the **processed training dataset** with 24 features.  
- Trained models are saved locally in this folder.  
- Paths are relative to the project root (`ca_housing_project/`).  
