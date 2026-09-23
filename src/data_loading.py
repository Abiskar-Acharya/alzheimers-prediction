"""
Data loading and cleaning utilities for OASIS Alzheimer's dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    LONGITUDINAL_CSV, CROSS_SECTIONAL_CSV,
    GROUP_BINARY_MAP, GROUP_THREE_MAP, RANDOM_SEED
)


def load_longitudinal_raw() -> pd.DataFrame:
    """Load raw longitudinal data without any processing."""
    return pd.read_csv(LONGITUDINAL_CSV)


def load_cross_sectional_raw() -> pd.DataFrame:
    """Load raw cross-sectional data without any processing."""
    return pd.read_csv(CROSS_SECTIONAL_CSV)


def impute_ses(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing SES values.

    Args:
        df: DataFrame with SES column
        strategy: 'median' (default), 'mode', or 'drop'

    Returns:
        DataFrame with SES imputed (or rows dropped)
    """
    df = df.copy()
    if strategy == "drop":
        df = df.dropna(subset=["SES"])
    elif strategy == "median":
        df["SES"] = df["SES"].fillna(df["SES"].median())
    elif strategy == "mode":
        df["SES"] = df["SES"].fillna(df["SES"].mode()[0])
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Encode M/F as binary: M=0, F=1."""
    df = df.copy()
    df["M/F_encoded"] = (df["M/F"] == "F").astype(int)
    return df


def load_first_visit(impute_strategy: str = "median") -> pd.DataFrame:
    """
    Load first-visit-only data for cross-sectional analysis.

    This gives 150 unique subjects (one row each).
    SES is imputed by default (not dropped) to preserve all subjects.
    Gender is encoded as binary.

    Returns:
        DataFrame with 150 rows, one per subject
    """
    df = load_longitudinal_raw()
    # Keep only first visit per subject
    first_visit = df[df["Visit"] == 1].copy()
    # Impute SES
    first_visit = impute_ses(first_visit, strategy=impute_strategy)
    # Encode gender
    first_visit = encode_gender(first_visit)
    # Create binary target
    first_visit["target_binary"] = first_visit["Group"].map(GROUP_BINARY_MAP)
    # Create three-group target
    first_visit["target_three"] = first_visit["Group"].map(GROUP_THREE_MAP)
    return first_visit


def load_all_visits(impute_strategy: str = "median") -> pd.DataFrame:
    """
    Load all visits for longitudinal analysis.

    Returns:
        DataFrame with all 373 rows, SES imputed, gender encoded
    """
    df = load_longitudinal_raw()
    df = impute_ses(df, strategy=impute_strategy)
    df = encode_gender(df)
    df["target_binary"] = df["Group"].map(GROUP_BINARY_MAP)
    df["target_three"] = df["Group"].map(GROUP_THREE_MAP)
    return df


def get_converter_subjects() -> list:
    """
    Get list of Subject IDs for converters (Group == 'Converted').

    Returns:
        List of subject ID strings
    """
    df = load_longitudinal_raw()
    return df[df["Group"] == "Converted"]["Subject ID"].unique().tolist()


def get_converter_trajectories() -> pd.DataFrame:
    """
    Get all visits for converter subjects.

    Returns:
        DataFrame with all visits for the ~8 converter subjects
    """
    df = load_all_visits()
    converters = get_converter_subjects()
    return df[df["Subject ID"].isin(converters)].sort_values(["Subject ID", "Visit"])


def load_cross_sectional_with_labels() -> pd.DataFrame:
    """
    Load cross-sectional data and derive dementia group from CDR.

    Only includes subjects with CDR data (excludes young healthy).
    CDR > 0 -> Demented, CDR == 0 -> Nondemented

    Returns:
        DataFrame with derived Group and target columns
    """
    df = load_cross_sectional_raw()
    # Drop subjects without CDR (young healthy controls)
    df = df.dropna(subset=["CDR"]).copy()
    # Derive group from CDR
    df["Group"] = np.where(df["CDR"] > 0, "Demented", "Nondemented")
    df["target_binary"] = (df["CDR"] > 0).astype(int)
    # Encode gender
    df = encode_gender(df)
    # Rename Educ to EDUC for consistency
    if "Educ" in df.columns:
        df = df.rename(columns={"Educ": "EDUC"})
    # Impute SES
    df = impute_ses(df, strategy="median")
    return df


def get_feature_matrix(df: pd.DataFrame, feature_list: list) -> tuple:
    """
    Extract feature matrix X and target y from a DataFrame.

    Args:
        df: DataFrame with features and target_binary column
        feature_list: List of column names to include

    Returns:
        (X, y) tuple of numpy arrays
    """
    X = df[feature_list].values
    y = df["target_binary"].values
    return X, y
