"""
Project configuration - constants, feature lists, paths.
"""
import os
from pathlib import Path

# ─── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_CV_REPEATS = 10
N_CV_FOLDS = 10

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"

LONGITUDINAL_CSV = DATA_RAW / "oasis_longitudinal.csv"
CROSS_SECTIONAL_CSV = DATA_RAW / "oasis_cross-sectional.csv"

# ─── Feature Sets for MMSE Ablation ─────────────────────────────────────────
# Columns in OASIS longitudinal (after preprocessing)
DEMOGRAPHIC_FEATURES = ["Age", "EDUC", "SES", "M/F_encoded"]
MRI_FEATURES = ["eTIV", "nWBV", "ASF"]
COGNITIVE_FEATURES = ["MMSE"]

ALL_FEATURES = DEMOGRAPHIC_FEATURES + MRI_FEATURES + COGNITIVE_FEATURES
NO_MMSE_FEATURES = DEMOGRAPHIC_FEATURES + MRI_FEATURES
MRI_ONLY_FEATURES = MRI_FEATURES
DEMO_ONLY_FEATURES = DEMOGRAPHIC_FEATURES

# Feature set definitions for ablation study
FEATURE_SETS = {
    "All Features": ALL_FEATURES,
    "No MMSE": NO_MMSE_FEATURES,
    "MRI Only": MRI_ONLY_FEATURES,
    "Demographics Only": DEMO_ONLY_FEATURES,
}

# ─── Engineered Feature Names ───────────────────────────────────────────────
CGFIN_FEATURES = [
    "brain_age_gap",           # Age - expected_age_from_nWBV
    "cognitive_reserve_index",  # EDUC * (1/SES) normalized
    "brain_volume_ratio",      # nWBV / eTIV * 1000
    "atrophy_score",           # (1 - nWBV) * Age / 100
]

CPCRI_FEATURES = [
    "educ_ses_interaction",    # EDUC * SES
    "age_nwbv_interaction",    # Age * nWBV
    "cognitive_brain_ratio",   # MMSE / (nWBV * 100)
]

LONGITUDINAL_FEATURES = [
    "nWBV_slope",              # Linear slope of nWBV over visits
    "MMSE_slope",              # Linear slope of MMSE over visits
    "n_visits",                # Number of visits
    "visit_span_days",         # Time between first and last visit
    "nWBV_change",             # nWBV_last - nWBV_first
    "MMSE_change",             # MMSE_last - MMSE_first
]

# ─── CDR to Group Mapping ───────────────────────────────────────────────────
CDR_GROUP_MAP = {
    0.0: "Nondemented",
    0.5: "Very Mild Dementia",
    1.0: "Mild Dementia",
    2.0: "Moderate Dementia",
}

# Binary target mapping
GROUP_BINARY_MAP = {
    "Nondemented": 0,
    "Demented": 1,
    "Converted": 1,  # Converters are clinically demented
}

# Three-group mapping (for detailed analysis)
GROUP_THREE_MAP = {
    "Nondemented": 0,
    "Demented": 1,
    "Converted": 2,
}

# ─── Visualization ───────────────────────────────────────────────────────────
COLOR_PALETTE = {
    "Nondemented": "#2ecc71",
    "Demented": "#e74c3c",
    "Converted": "#f39c12",
}

MODEL_COLORS = {
    "Logistic Regression": "#3498db",
    "Random Forest": "#2ecc71",
    "Gradient Boosting": "#e74c3c",
    "SVM": "#9b59b6",
    "XGBoost": "#1abc9c",
    "LightGBM": "#e67e22",
    "CatBoost": "#34495e",
    "TabPFN": "#f1c40f",
}

FIGURE_DPI = 150
FIGURE_SIZE_STANDARD = (10, 6)
FIGURE_SIZE_WIDE = (14, 6)
FIGURE_SIZE_TALL = (10, 10)

# ─── Model Hyperparameters (conservative for n=150) ─────────────────────────
MODEL_PARAMS = {
    "logistic_regression": {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_leaf": 5,
        "random_state": RANDOM_SEED,
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_leaf": 5,
        "random_state": RANDOM_SEED,
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": RANDOM_SEED,
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "eval_metric": "logloss",
    },
    "lightgbm": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    },
    "catboost": {
        "iterations": 100,
        "depth": 3,
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "random_seed": RANDOM_SEED,
        "verbose": 0,
    },
}
