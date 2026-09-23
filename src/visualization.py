"""
Visualization utilities for Alzheimer's prediction research.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.config import (
    COLOR_PALETTE, MODEL_COLORS,
    FIGURE_DPI, FIGURE_SIZE_STANDARD, FIGURE_SIZE_WIDE, FIGURE_SIZE_TALL,
    RESULTS_FIGURES,
)


def set_style():
    """Set consistent plotting style."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.figsize": FIGURE_SIZE_STANDARD,
    })


def save_figure(fig, name: str, tight: bool = True):
    """Save figure to results/figures/."""
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(RESULTS_FIGURES / f"{name}.png", dpi=FIGURE_DPI, bbox_inches="tight")


def plot_group_distribution(df: pd.DataFrame, title: str = "Group Distribution") -> plt.Figure:
    """Bar chart of Nondemented/Demented/Converted counts."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df["Group"].value_counts()
    colors = [COLOR_PALETTE.get(g, "#95a5a6") for g in counts.index]

    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontweight="bold")

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Group")

    return fig


def plot_mmse_ablation(results_df: pd.DataFrame, title: str = "MMSE Ablation Study") -> plt.Figure:
    """
    Grouped bar chart showing accuracy across feature sets per model.

    Args:
        results_df: DataFrame with Model, Feature_Set, Accuracy_Mean, CI_Lower, CI_Upper
    """
    set_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    feature_sets = results_df["Feature_Set"].unique()
    models = results_df["Model"].unique()
    x = np.arange(len(models))
    width = 0.8 / len(feature_sets)

    for i, fs in enumerate(feature_sets):
        subset = results_df[results_df["Feature_Set"] == fs]
        # Ensure model order matches
        subset = subset.set_index("Model").reindex(models)

        offset = (i - len(feature_sets) / 2 + 0.5) * width

        yerr_lower = subset["Accuracy_Mean"].values - subset["CI_Lower"].values
        yerr_upper = subset["CI_Upper"].values - subset["Accuracy_Mean"].values

        bars = ax.bar(
            x + offset,
            subset["Accuracy_Mean"].values,
            width,
            label=fs,
            yerr=[yerr_lower, yerr_upper],
            capsize=3,
            alpha=0.85,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (10\u00d710 CV)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend(title="Feature Set", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")

    return fig


def plot_cv_comparison(
    results_df: pd.DataFrame,
    metric: str = "Accuracy_Mean",
    title: str = "Model Comparison (10\u00d710 CV)"
) -> plt.Figure:
    """
    Horizontal bar chart comparing models on a single feature set.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    results_sorted = results_df.sort_values(metric, ascending=True)
    colors = [MODEL_COLORS.get(m, "#95a5a6") for m in results_sorted["Model"]]

    y_pos = np.arange(len(results_sorted))
    xerr_lower = results_sorted[metric].values - results_sorted["CI_Lower"].values
    xerr_upper = results_sorted["CI_Upper"].values - results_sorted[metric].values

    ax.barh(
        y_pos,
        results_sorted[metric].values,
        xerr=[xerr_lower, xerr_upper],
        color=colors,
        capsize=4,
        edgecolor="white",
        linewidth=1,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted["Model"])
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0.4, 1.0)

    # Add value labels
    for i, (val, ci_l, ci_u) in enumerate(zip(
        results_sorted[metric], results_sorted["CI_Lower"], results_sorted["CI_Upper"]
    )):
        ax.text(val + xerr_upper[i] + 0.01, i, f"{val:.3f}", va="center", fontsize=9)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a confusion matrix heatmap."""
    set_style()
    if labels is None:
        labels = ["Nondemented", "Demented"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")

    return fig


def plot_converter_trajectories(
    converter_df: pd.DataFrame,
    feature: str = "nWBV",
    title: str = None,
) -> plt.Figure:
    """
    Spaghetti plot of converter subject trajectories over visits.

    Args:
        converter_df: DataFrame with converter subjects' visit data
        feature: Feature to plot (e.g., 'nWBV', 'MMSE')
        title: Plot title
    """
    set_style()
    if title is None:
        title = f"Converter Trajectories: {feature}"

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    for subject_id, group in converter_df.groupby("Subject ID"):
        group = group.sort_values("Visit")
        ax.plot(
            group["MR Delay"] / 365.25,
            group[feature],
            marker="o",
            label=subject_id,
            alpha=0.8,
            linewidth=2,
        )

    ax.set_xlabel("Years from Baseline")
    ax.set_ylabel(feature)
    ax.set_title(title, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    return fig


def plot_calibration(
    calibration_results: Dict,
    model_name: str = "Model",
) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve).

    Args:
        calibration_results: Output from uncertainty.calibration_analysis()
    """
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE)

    # Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(
        calibration_results["mean_predicted_value"],
        calibration_results["fraction_of_positives"],
        "s-", label=f"{model_name} (ECE={calibration_results['ece']:.3f})",
        color="#3498db",
    )
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram", fontweight="bold")
    ax.legend(loc="lower right")

    # Prediction histogram
    ax = axes[1]
    ax.hist(
        calibration_results["y_proba"],
        bins=20, range=(0, 1), alpha=0.7, color="#3498db", edgecolor="white",
    )
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Distribution", fontweight="bold")

    return fig


def plot_feature_importance_comparison(
    importance_dict: Dict[str, pd.Series],
    title: str = "Feature Importance Across Models",
) -> plt.Figure:
    """
    Compare feature importances across multiple models.

    Args:
        importance_dict: {model_name: pd.Series of feature importances}
    """
    set_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)

    all_features = sorted(set().union(*[s.index for s in importance_dict.values()]))
    x = np.arange(len(all_features))
    width = 0.8 / len(importance_dict)

    for i, (name, importances) in enumerate(importance_dict.items()):
        offset = (i - len(importance_dict) / 2 + 0.5) * width
        values = [importances.get(f, 0) for f in all_features]
        color = MODEL_COLORS.get(name, f"C{i}")
        ax.bar(x + offset, values, width, label=name, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_features, rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title(title, fontweight="bold")
    ax.legend()

    return fig


def plot_survival_curves(
    time: np.ndarray,
    event: np.ndarray,
    groups: np.ndarray = None,
    title: str = "Kaplan-Meier Survival Curves",
) -> plt.Figure:
    """
    Kaplan-Meier plot (requires lifelines).

    Args:
        time: Time to event (days)
        event: Event indicator (1=event occurred)
        groups: Optional group labels for stratification
    """
    from lifelines import KaplanMeierFitter

    set_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    kmf = KaplanMeierFitter()

    if groups is not None:
        for group_name in np.unique(groups):
            mask = groups == group_name
            kmf.fit(
                time[mask], event[mask],
                label=group_name,
            )
            kmf.plot_survival_function(ax=ax)
    else:
        kmf.fit(time, event, label="All subjects")
        kmf.plot_survival_function(ax=ax)

    ax.set_xlabel("Days from Baseline")
    ax.set_ylabel("Survival Probability")
    ax.set_title(title, fontweight="bold")
    ax.legend()

    return fig
