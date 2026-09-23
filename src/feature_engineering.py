"""
Feature engineering for Alzheimer's prediction.
CGFIN: Composite Geriatric Feature Interaction Network
CP-CRI: Cross-Product Cognitive Reserve Index
Longitudinal feature extraction from multi-visit data.
"""
import pandas as pd
import numpy as np
from scipy import stats


def create_cgfin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create CGFIN (Composite Geriatric Feature Interaction Network) features.

    These capture non-linear relationships between age, brain volume,
    education, and socioeconomic status.

    Args:
        df: DataFrame with Age, nWBV, EDUC, SES, eTIV columns

    Returns:
        DataFrame with additional CGFIN feature columns
    """
    df = df.copy()

    # Brain Age Gap: how much older/younger the brain looks vs chronological age
    # Higher nWBV = younger brain. Use linear relationship from healthy controls.
    # Simplified: residual of Age ~ nWBV regression
    df["brain_age_gap"] = df["Age"] - (df["nWBV"] * (-200) + 220)
    # Positive = brain looks older than expected, negative = brain looks younger

    # Cognitive Reserve Index: education protects against decline
    # Higher education + lower SES = more resilient
    ses_inv = 6 - df["SES"]  # Invert SES (1=highest → 5=lowest protection)
    df["cognitive_reserve_index"] = (df["EDUC"] * ses_inv) / 25  # Normalize to ~0-4 range

    # Brain Volume Ratio: normalized brain volume relative to cranial capacity
    df["brain_volume_ratio"] = (df["nWBV"] / df["eTIV"]) * 1e6

    # Atrophy Score: compound measure of age-related brain loss
    df["atrophy_score"] = (1 - df["nWBV"]) * df["Age"] / 100

    return df


def create_cpcri_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create CP-CRI (Cross-Product Cognitive Reserve Index) features.

    These capture interaction effects between demographic, cognitive,
    and neuroimaging variables.

    Args:
        df: DataFrame with EDUC, SES, Age, nWBV, MMSE columns

    Returns:
        DataFrame with additional CP-CRI feature columns
    """
    df = df.copy()

    # Education-SES interaction: socioeconomic context of education
    df["educ_ses_interaction"] = df["EDUC"] * df["SES"]

    # Age-nWBV interaction: age-adjusted brain volume
    df["age_nwbv_interaction"] = df["Age"] * df["nWBV"]

    # Cognitive-Brain ratio: cognitive function relative to brain volume
    # Higher ratio = more cognitive function per unit brain volume
    # (Only meaningful when MMSE is included)
    if "MMSE" in df.columns:
        df["cognitive_brain_ratio"] = df["MMSE"] / (df["nWBV"] * 100)

    return df


def create_longitudinal_features(df_all_visits: pd.DataFrame) -> pd.DataFrame:
    """
    Extract longitudinal features from multi-visit data.

    For each subject, computes slopes, changes, and visit statistics.

    Args:
        df_all_visits: DataFrame with all visits (Subject ID, Visit, MR Delay,
                       nWBV, MMSE columns)

    Returns:
        DataFrame with one row per subject and longitudinal features
    """
    results = []

    for subject_id, group in df_all_visits.groupby("Subject ID"):
        group = group.sort_values("Visit")

        record = {"Subject ID": subject_id}

        # Number of visits
        record["n_visits"] = len(group)

        # Visit span in days
        record["visit_span_days"] = group["MR Delay"].max() - group["MR Delay"].min()

        # nWBV slope and change
        if len(group) >= 2:
            days = group["MR Delay"].values
            nwbv = group["nWBV"].values

            if days[-1] > days[0]:  # Has time progression
                # Linear regression slope (per year)
                slope, _, _, _, _ = stats.linregress(days / 365.25, nwbv)
                record["nWBV_slope"] = slope
            else:
                record["nWBV_slope"] = 0.0

            record["nWBV_change"] = nwbv[-1] - nwbv[0]

            # MMSE slope and change
            mmse = group["MMSE"].dropna().values
            if len(mmse) >= 2:
                mmse_days = group.dropna(subset=["MMSE"])["MR Delay"].values
                if mmse_days[-1] > mmse_days[0]:
                    slope_mmse, _, _, _, _ = stats.linregress(mmse_days / 365.25, mmse)
                    record["MMSE_slope"] = slope_mmse
                else:
                    record["MMSE_slope"] = 0.0
                record["MMSE_change"] = mmse[-1] - mmse[0]
            else:
                record["MMSE_slope"] = 0.0
                record["MMSE_change"] = 0.0
        else:
            record["nWBV_slope"] = 0.0
            record["nWBV_change"] = 0.0
            record["MMSE_slope"] = 0.0
            record["MMSE_change"] = 0.0

        results.append(record)

    return pd.DataFrame(results)


def add_engineered_features(
    df: pd.DataFrame,
    include_cgfin: bool = True,
    include_cpcri: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to add all engineered features.

    Args:
        df: Base DataFrame
        include_cgfin: Whether to add CGFIN features
        include_cpcri: Whether to add CP-CRI features

    Returns:
        DataFrame with engineered features added
    """
    if include_cgfin:
        df = create_cgfin_features(df)
    if include_cpcri:
        df = create_cpcri_features(df)
    return df
