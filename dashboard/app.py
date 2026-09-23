"""
Alzheimer's Prediction Research Dashboard
==========================================
Honest ML methodology for small-sample AD research.
Streamlit dashboard with model comparison, MMSE ablation, uncertainty, and explainability.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    ALL_FEATURES, NO_MMSE_FEATURES, MRI_ONLY_FEATURES, DEMO_ONLY_FEATURES,
    FEATURE_SETS, MODEL_PARAMS, RANDOM_SEED, COLOR_PALETTE, MODEL_COLORS,
    RESULTS_TABLES, RESULTS_FIGURES, DATA_PROCESSED,
)
from src.data_loading import load_first_visit, load_all_visits, get_converter_trajectories
from src.visualization import set_style

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's Prediction Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🧠 AD Prediction Research")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "MMSE Ablation",
        "Model Comparison",
        "Uncertainty",
        "Trajectories",
        "Explainability",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Key Insight:** Reported 89% accuracy relies on MMSE, "
    "which is circular with the diagnostic label."
)


# ─── Helper Functions ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and cache the first-visit dataset."""
    return load_first_visit(impute_strategy="median")


@st.cache_data
def load_results_table(name: str):
    """Load a results CSV if it exists."""
    path = RESULTS_TABLES / name
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_all_visit_data():
    """Load all visits data."""
    return load_all_visits(impute_strategy="median")


# ─── Pages ───────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("Honest ML for Alzheimer's Prediction")
    st.markdown("""
    ## The Problem with "89% Accuracy"

    Many Alzheimer's prediction studies using the OASIS dataset report high accuracy
    (85-95%) but fail to account for a critical methodological flaw: **MMSE circularity**.

    ### What is MMSE Circularity?

    - **CDR (Clinical Dementia Rating)** defines the diagnostic group (Demented vs Nondemented)
    - **MMSE (Mini-Mental State Exam)** is a cognitive test highly correlated with CDR
    - Using MMSE as a predictive feature is essentially **predicting the label from itself**

    ### What This Project Does

    1. **Reproduces** the original 89% single-split result
    2. **Exposes** that proper 10×10 CV gives ~74-80% with MMSE
    3. **Shows** that removing MMSE drops accuracy to ~62-70%
    4. **Argues** that ~65% from MRI + demographics is still clinically meaningful for screening
    5. **Adds** uncertainty quantification, longitudinal analysis, and modern models (TabPFN)
    """)

    # Show data overview
    df = load_data()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Subjects", len(df))
    col2.metric("Features", len(ALL_FEATURES))
    col3.metric("Nondemented", len(df[df["Group"] == "Nondemented"]))
    col4.metric("Demented + Converted", len(df[df["Group"] != "Nondemented"]))

    st.markdown("### Group Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = df["Group"].value_counts()
    colors = [COLOR_PALETTE.get(g, "#95a5a6") for g in counts.index]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    for i, (g, c) in enumerate(zip(counts.index, counts.values)):
        ax.text(i, c + 1, str(c), ha="center", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title("Subject Distribution by Group")
    st.pyplot(fig)
    plt.close()

    st.markdown("### Dataset Preview")
    st.dataframe(df[["Subject ID", "Group", "Age", "M/F", "EDUC", "SES", "MMSE", "CDR", "nWBV", "eTIV"]].head(20))


elif page == "MMSE Ablation":
    st.title("MMSE Ablation Study")
    st.markdown("""
    The central finding: **removing MMSE reveals the true predictive signal**.

    This ablation systematically removes features to show how much each contributes:
    - **All Features**: Includes MMSE (circular)
    - **No MMSE**: Removes the circular feature
    - **MRI Only**: Only brain imaging features
    - **Demographics Only**: Age, education, SES, gender
    """)

    # Load ablation results
    ablation_df = load_results_table("mmse_ablation_baseline.csv")

    if ablation_df is not None:
        st.markdown("### Ablation Results (10×10 Repeated Stratified K-Fold)")

        # Pivot for display
        pivot = ablation_df.pivot(
            index="Model", columns="Feature_Set", values="Accuracy_Formatted"
        )
        col_order = ["All Features", "No MMSE", "MRI Only", "Demographics Only"]
        existing = [c for c in col_order if c in pivot.columns]
        st.dataframe(pivot[existing], use_container_width=True)

        # Bar chart
        st.markdown("### Accuracy by Feature Set")
        fig, ax = plt.subplots(figsize=(12, 6))
        feature_sets = ablation_df["Feature_Set"].unique()
        models = ablation_df["Model"].unique()
        x = np.arange(len(models))
        width = 0.8 / len(feature_sets)

        for i, fs in enumerate(feature_sets):
            subset = ablation_df[ablation_df["Feature_Set"] == fs].set_index("Model").reindex(models)
            offset = (i - len(feature_sets) / 2 + 0.5) * width
            yerr_l = subset["Accuracy_Mean"].values - subset["CI_Lower"].values
            yerr_u = subset["CI_Upper"].values - subset["Accuracy_Mean"].values
            ax.bar(x + offset, subset["Accuracy_Mean"].values, width, label=fs,
                   yerr=[yerr_l, yerr_u], capsize=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.set_ylabel("Accuracy (10×10 CV)")
        ax.set_ylim(0.4, 1.0)
        ax.legend(title="Feature Set")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("MMSE Ablation: Accuracy Drops When Circular Feature Removed")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        ### Key Finding

        Removing MMSE causes a **~15-20 percentage point accuracy drop**, confirming
        that most of the "predictive signal" was actually circularity. The true
        MRI + demographic signal is ~62-70%.
        """)
    else:
        st.warning("Run notebook 03 first to generate ablation results.")


elif page == "Model Comparison":
    st.title("Model Comparison")

    master_df = load_results_table("master_comparison.csv")

    if master_df is not None:
        st.markdown("### All Models - 10×10 Repeated Stratified K-Fold")

        feature_set = st.selectbox(
            "Feature Set",
            ["All Features", "No MMSE", "No MMSE + Engineered"]
        )

        filtered = master_df[master_df["Feature_Set"] == feature_set] if "Feature_Set" in master_df.columns else master_df
        st.dataframe(filtered, use_container_width=True)

        # Horizontal bar chart
        if "Accuracy_Mean" in filtered.columns and "Model" in filtered.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sorted_df = filtered.sort_values("Accuracy_Mean", ascending=True)
            colors = [MODEL_COLORS.get(m, "#95a5a6") for m in sorted_df["Model"]]
            y_pos = np.arange(len(sorted_df))

            if "CI_Lower" in sorted_df.columns:
                xerr_l = sorted_df["Accuracy_Mean"].values - sorted_df["CI_Lower"].values
                xerr_u = sorted_df["CI_Upper"].values - sorted_df["Accuracy_Mean"].values
                ax.barh(y_pos, sorted_df["Accuracy_Mean"].values,
                        xerr=[xerr_l, xerr_u], color=colors, capsize=4)
            else:
                ax.barh(y_pos, sorted_df["Accuracy_Mean"].values, color=colors)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_df["Model"])
            ax.set_xlabel("Accuracy")
            ax.set_xlim(0.4, 1.0)
            ax.set_title(f"Model Comparison - {feature_set}")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("Run notebooks 03-04 first to generate comparison results.")

    st.markdown("""
    ### The Honest Story

    | Condition | Accuracy Range |
    |-----------|---------------|
    | Single split + MMSE (original claim) | ~85-89% |
    | 10×10 CV + MMSE | ~74-80% |
    | 10×10 CV, no MMSE | ~62-70% |
    | Cross-dataset, no MMSE | ~58-65% |

    **The true predictive signal is ~65%**, which is still above chance (50%)
    and meaningful for population-level screening.
    """)


elif page == "Uncertainty":
    st.title("Uncertainty Quantification")
    st.markdown("""
    Honest prediction requires honest uncertainty. Three approaches:

    1. **Calibration**: Are predicted probabilities reliable?
    2. **Conformal Prediction**: Guaranteed coverage prediction sets
    3. **Clinical Thresholds**: Screening vs confirmatory cutoffs
    """)

    # Show calibration figures if they exist
    cal_path = RESULTS_FIGURES / "calibration_gradient_boosting.png"
    if cal_path.exists():
        st.markdown("### Calibration (Reliability Diagram)")
        st.image(str(cal_path))

    conformal_path = RESULTS_FIGURES / "conformal_prediction_sets.png"
    if conformal_path.exists():
        st.markdown("### Conformal Prediction Sets")
        st.image(str(conformal_path))

    st.markdown("""
    ### Conformal Prediction Interpretation

    - **Singleton set {0}**: Confidently nondemented
    - **Singleton set {1}**: Confidently demented
    - **Set {0, 1}**: Model is uncertain — both classes possible

    At 90% confidence level, we expect ~10-30% of predictions to be uncertain
    (set size = 2). This is **honest** — better to say "I don't know" than to
    make a confident wrong prediction.
    """)

    st.markdown("""
    ### Clinical Decision Thresholds

    | Use Case | Threshold | Sensitivity | Specificity |
    |----------|-----------|-------------|-------------|
    | Screening | Low (~0.3) | High (~0.90) | Lower (~0.40) |
    | Confirmatory | High (~0.7) | Lower (~0.50) | High (~0.90) |

    For **screening**: maximize sensitivity (catch most cases, accept false positives)
    For **confirmation**: maximize specificity (minimize false positives)
    """)


elif page == "Trajectories":
    st.title("Longitudinal Trajectories")
    st.markdown("""
    The OASIS dataset includes 2-5 visits per subject over ~2-5 years.
    This allows tracking of brain volume (nWBV) and cognitive (MMSE) trajectories.
    """)

    try:
        converter_df = get_converter_trajectories()
        all_visits = load_all_visit_data()

        feature = st.selectbox("Feature to Plot", ["nWBV", "MMSE", "CDR"])

        st.markdown("### Converter Subject Trajectories")
        fig, ax = plt.subplots(figsize=(10, 6))
        for sid, group in converter_df.groupby("Subject ID"):
            group = group.sort_values("Visit")
            ax.plot(group["MR Delay"] / 365.25, group[feature],
                    marker="o", label=sid, alpha=0.8, linewidth=2)
        ax.set_xlabel("Years from Baseline")
        ax.set_ylabel(feature)
        ax.set_title(f"Converter Trajectories: {feature}")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Group Mean Trajectories")
        fig, ax = plt.subplots(figsize=(10, 6))
        for group_name in ["Nondemented", "Demented", "Converted"]:
            subset = all_visits[all_visits["Group"] == group_name]
            means = subset.groupby("Visit")[feature].mean()
            stds = subset.groupby("Visit")[feature].std()
            color = COLOR_PALETTE.get(group_name, "#95a5a6")
            ax.plot(means.index, means.values, marker="o", color=color,
                    label=group_name, linewidth=2)
            ax.fill_between(means.index, means.values - stds.values,
                           means.values + stds.values, alpha=0.15, color=color)
        ax.set_xlabel("Visit Number")
        ax.set_ylabel(f"Mean {feature}")
        ax.set_title(f"Group Mean Trajectories: {feature}")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error loading trajectory data: {e}")


elif page == "Explainability":
    st.title("Model Explainability (SHAP)")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) reveals which features drive predictions.

    **Critical insight**: When MMSE is included, it dominates all other features.
    Removing it reveals the true contribution of MRI and demographic features.
    """)

    # Show SHAP figures if they exist
    for name, title in [
        ("shap_summary_all_features.png", "SHAP Summary - All Features (MMSE Dominates)"),
        ("shap_summary_no_mmse.png", "SHAP Summary - Without MMSE (True Feature Importance)"),
        ("shap_importance_comparison.png", "Feature Importance: With vs Without MMSE"),
    ]:
        path = RESULTS_FIGURES / name
        if path.exists():
            st.markdown(f"### {title}")
            st.image(str(path))

    st.markdown("""
    ### Clinical Interpretation

    After removing MMSE, the most important features are:

    1. **nWBV** (normalized whole-brain volume) — established AD biomarker
    2. **Age** — strongest risk factor for AD
    3. **EDUC** (education) — relates to cognitive reserve
    4. **eTIV** (estimated total intracranial volume) — head size proxy
    5. **ASF** (atlas scaling factor) — brain normalization factor

    These rankings **align with clinical literature**, confirming that the model
    is learning real biological signal, not just statistical artifacts.
    """)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with honest methodology.  \n"
    "OASIS Longitudinal Dataset (150 subjects)."
)
