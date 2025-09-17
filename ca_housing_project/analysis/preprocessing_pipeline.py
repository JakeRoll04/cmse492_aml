#!/usr/bin/env python3
"""
Preprocessing pipeline

Input:  data/train/housing_train.csv (13 columns incl. target)
Output: data/train/housing_train_processed.csv (24 features + target)
Also saves fitted pipeline to models/pipeline.pkl
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(".")
TRAIN = ROOT / "data" / "train"
MODELS = ROOT / "models"

RAW_TRAIN = TRAIN / "housing_train.csv"
PROCESSED = TRAIN / "housing_train_processed.csv"
PIPE_OUT  = MODELS / "pipeline.pkl"

TARGET = "median_house_value"

# Order matters; these indices are used when input becomes a NumPy array
NUM_FEATS = [
    "longitude",            # 0
    "latitude",             # 1
    "housing_median_age",   # 2
    "total_rooms",          # 3
    "total_bedrooms",       # 4
    "population",           # 5
    "households",           # 6
    "median_income"         # 7
]
CAT_FEAT = ["ocean_proximity"]

# Helper to detect DataFrame
def _is_df(x):
    return isinstance(x, pd.DataFrame)

class RatioAdder(BaseEstimator, TransformerMixin):
    """Add numeric ratios; works with pandas DataFrame or NumPy array."""
    def __init__(self, eps: float = 1e-9):
        self.eps = eps
        # indices matching NUM_FEATS order
        self.idx = {
            "total_rooms": 3,
            "total_bedrooms": 4,
            "population": 5,
            "households": 6,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if _is_df(X):
            X = X.copy()
            X["rooms_per_household"] = X["total_rooms"] / np.maximum(X["households"], self.eps)
            X["bedrooms_per_room"]   = X["total_bedrooms"] / np.maximum(X["total_rooms"], self.eps)
            X["population_per_household"] = X["population"] / np.maximum(X["households"], self.eps)
            X["rooms_per_bedroom"]   = X["total_rooms"] / np.maximum(X["total_bedrooms"], self.eps)
            return X
        else:
            tr = X[:, self.idx["total_rooms"]]
            tb = X[:, self.idx["total_bedrooms"]]
            pop = X[:, self.idx["population"]]
            hh = X[:, self.idx["households"]]
            rooms_per_household      = tr / np.maximum(hh, self.eps)
            bedrooms_per_room        = tb / np.maximum(tr, self.eps)
            population_per_household = pop / np.maximum(hh, self.eps)
            rooms_per_bedroom        = tr / np.maximum(tb, self.eps)
            return np.column_stack([
                X,
                rooms_per_household,
                bedrooms_per_room,
                population_per_household,
                rooms_per_bedroom
            ])

class GeoCluster(BaseEstimator, TransformerMixin):
    """Add KMeans cluster indicators based on (longitude, latitude). Works with DF or ndarray."""
    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.km_ = None
        # indices per NUM_FEATS
        self.lon_idx = 0
        self.lat_idx = 1

    def fit(self, X, y=None):
        if _is_df(X):
            coords = X[["longitude", "latitude"]].to_numpy()
        else:
            coords = X[:, [self.lon_idx, self.lat_idx]]
        self.km_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto")
        self.km_.fit(coords)
        return self

    def transform(self, X):
        if _is_df(X):
            coords = X[["longitude", "latitude"]].to_numpy()
            labels = self.km_.predict(coords)
            ind = np.eye(self.n_clusters, dtype=int)[labels]
            out = X.copy()
            for k in range(self.n_clusters):
                out[f"geo_cluster_{k}"] = ind[:, k]
            return out
        else:
            coords = X[:, [self.lon_idx, self.lat_idx]]
            labels = self.km_.predict(coords)
            ind = np.eye(self.n_clusters, dtype=int)[labels]
            return np.column_stack([X, ind])

def build_pipeline():
    # Numeric pipe: impute -> ratios (+4) -> geo clusters (+8) -> scale
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("ratios", RatioAdder()),
        ("geoclust", GeoCluster(n_clusters=8, random_state=42)),
        ("scale", StandardScaler()),
    ])

    # Categorical pipe: impute -> OHE (drop first)
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUM_FEATS),
            ("cat", cat_pipe, CAT_FEAT),
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )
    return pre

def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_TRAIN)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].copy()

    pre = build_pipeline()
    Xp = pre.fit_transform(X, y)
    feat_names = pre.get_feature_names_out()
    # When custom transformers return ndarray, feature names for those parts
    # are auto-generated by ColumnTransformer, which is fine for our use.
    Xp_df = pd.DataFrame(Xp, columns=feat_names)

    if Xp_df.shape[1] != 24:
        raise ValueError(f"Expected 24 features, got {Xp_df.shape[1]}")

    out = Xp_df.copy()
    out[TARGET] = y.values
    out.to_csv(PROCESSED, index=False)

    joblib.dump(pre, PIPE_OUT)
    print(f"Saved processed train → {PROCESSED}")
    print(f"Saved pipeline → {PIPE_OUT}")

if __name__ == "__main__":
    main()
