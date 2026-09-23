"""
Render the 5 publication figures for the Alzheimer's MMSE-circularity paper.

Reads:
  - results/tables/mmse_ablation.csv
  - results/tables/cv_results.csv
  - results/tables/conformal_coverage.csv

Writes (300 DPI PNG):
  - results/figures/fig1_mmse_circularity.png
  - results/figures/fig2_mmse_ablation.png
  - results/figures/fig3_shap_beeswarm.png
  - results/figures/fig4_conformal_intervals.png
  - results/figures/fig5_cv_distribution.png

Run from project root:  python scripts/build_figures.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURE_SETS,
    MODEL_COLORS,
    MODEL_PARAMS,
    RANDOM_SEED,
    RESULTS_FIGURES,
    RESULTS_TABLES,
)
from src.data_loading import load_first_visit

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DPI = 300
plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig: plt.Figure, name: str) -> Path:
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = RESULTS_FIGURES / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {path}")
    return path


# ----------------------------------------------------------------------------
# Figure 1: MMSE circularity diagram
# ----------------------------------------------------------------------------

def fig1_circularity():
    """Conceptual diagram: MMSE feeds both the diagnosis (target) and the model (feature)."""
    nodes = {
        # name: (x, y, label_position_relative_to_node)
        "MMSE": (0.0, 1.0, "left"),
        "CDR\n(clinical rating)": (1.6, 1.7, "top"),
        '"Demented" label\n(target variable)': (3.4, 1.0, "right"),
        "ML model\nfeature set": (1.6, 0.3, "bottom"),
        "Predicted label": (3.4, 0.3, "right"),
    }

    diag_edges = [
        ("MMSE", "CDR\n(clinical rating)"),
        ("CDR\n(clinical rating)", '"Demented" label\n(target variable)'),
    ]
    feat_edges = [
        ("MMSE", "ML model\nfeature set"),
        ("ML model\nfeature set", "Predicted label"),
    ]
    eval_edges = [('"Demented" label\n(target variable)', "Predicted label")]

    fig, ax = plt.subplots(figsize=(11, 6))

    pos = {n: (x, y) for n, (x, y, _) in nodes.items()}

    diag_color = "#3498db"
    feat_color = "#e67e22"
    eval_color = "#c0392b"
    node_colors = {
        "MMSE": "#e74c3c",
        '"Demented" label\n(target variable)': "#27ae60",
        "Predicted label": "#9b59b6",
        "CDR\n(clinical rating)": "#2c3e50",
        "ML model\nfeature set": "#2c3e50",
    }

    label_offsets = {
        "left": (-0.18, 0, "right", "center"),
        "right": (0.18, 0, "left", "center"),
        "top": (0, 0.22, "center", "bottom"),
        "bottom": (0, -0.22, "center", "top"),
    }

    for n, (x, y, where) in nodes.items():
        ax.scatter([x], [y], s=900, c=node_colors[n], zorder=3,
                   edgecolors="white", linewidths=2)
        dx, dy, ha, va = label_offsets[where]
        ax.text(x + dx, y + dy, n, ha=ha, va=va, fontsize=10,
                fontweight="bold", color=node_colors[n], zorder=4)

    def draw_edge(src, dst, color, label=None, curve=0.0, label_pos=0.5,
                  label_offset=(0, 0)):
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=2.2,
                shrinkA=14, shrinkB=14,
                connectionstyle=f"arc3,rad={curve}",
                mutation_scale=18,
            ),
            zorder=2,
        )
        if label:
            mx = x1 + (x2 - x1) * label_pos + label_offset[0]
            my = y1 + (y2 - y1) * label_pos + label_offset[1]
            ax.text(mx, my, label, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc="white",
                              ec=color, lw=0.8))

    for s, d in diag_edges:
        draw_edge(s, d, diag_color, "informs", label_offset=(0, 0.06))
    for s, d in feat_edges:
        draw_edge(s, d, feat_color, "input to", label_offset=(0, -0.06))
    for s, d in eval_edges:
        draw_edge(s, d, eval_color, "compared to",
                  curve=-0.35, label_offset=(0.45, 0))

    ax.set_title(
        "The MMSE-CDR Circularity in Alzheimer's ML\n"
        "MMSE is both a predictor and an input to the diagnosis it predicts",
        fontweight="bold", fontsize=12,
    )

    # Legend at bottom
    legend_y = -0.35
    ax.text(0.2, legend_y, "Diagnostic path",
            color=diag_color, fontweight="bold", fontsize=10)
    ax.text(1.7, legend_y, "Modelling path",
            color=feat_color, fontweight="bold", fontsize=10)
    ax.text(3.2, legend_y, "Evaluation",
            color=eval_color, fontweight="bold", fontsize=10)

    ax.set_xlim(-0.8, 4.4)
    ax.set_ylim(-0.6, 2.3)
    ax.axis("off")

    save(fig, "fig1_mmse_circularity")


# ----------------------------------------------------------------------------
# Figure 2: MMSE ablation grouped bar chart with CIs
# ----------------------------------------------------------------------------

def fig2_ablation(summary_df: pd.DataFrame):
    feature_sets = ["All Features", "No MMSE", "MRI Only", "Demographics Only"]
    feature_sets = [fs for fs in feature_sets if fs in summary_df["feature_set"].unique()]
    models = list(summary_df["model"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(feature_sets)

    set_colors = {
        "All Features": "#e74c3c",     # red — inflated
        "No MMSE": "#27ae60",          # green — honest
        "MRI Only": "#3498db",         # blue — partial
        "Demographics Only": "#95a5a6",  # gray — baseline
    }
    set_hatches = {
        "All Features": "",
        "No MMSE": "//",
        "MRI Only": "..",
        "Demographics Only": "xx",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, fs in enumerate(feature_sets):
        subset = summary_df[summary_df["feature_set"] == fs].set_index("model").reindex(models)
        means = subset["mean_accuracy"].values
        ci_lo = subset["ci_lower_95_accuracy"].values
        ci_hi = subset["ci_upper_95_accuracy"].values
        yerr = np.vstack([means - ci_lo, ci_hi - means])
        offset = (i - len(feature_sets) / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            label=fs,
            color=set_colors.get(fs, f"C{i}"),
            hatch=set_hatches.get(fs, ""),
            yerr=yerr,
            capsize=3,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
        )

    ax.axhline(0.5, color="#7f8c8d", ls=":", lw=1, alpha=0.7, zorder=0)
    ax.text(len(models) - 0.5, 0.51, "chance", color="#7f8c8d", fontsize=9, ha="right")
    ax.axhline(0.89, color="#c0392b", ls="--", lw=1, alpha=0.6, zorder=0)
    ax.text(len(models) - 0.5, 0.90, "literature 'best' (with MMSE)",
            color="#c0392b", fontsize=9, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("Accuracy (10x10 stratified CV)")
    ax.set_title("MMSE Ablation: Removing the Circular Feature Drops Accuracy ~20pp",
                 fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.legend(title="Feature set", bbox_to_anchor=(1.02, 1), loc="upper left",
              frameon=False)
    ax.grid(axis="y", alpha=0.3)

    save(fig, "fig2_mmse_ablation")


# ----------------------------------------------------------------------------
# Figure 3: SHAP beeswarm on best No-MMSE model
# ----------------------------------------------------------------------------

def fig3_shap(summary_df: pd.DataFrame):
    """SHAP beeswarm on the best No-MMSE model.

    For non-tree models, prefer a tree surrogate (best tree model from the
    panel) so we get a proper TreeExplainer beeswarm — KernelExplainer on
    SVM with the new SHAP API can return shapes that summary_plot mis-reads
    as interactions, producing a degenerate 2-feature plot.
    """
    import shap
    from sklearn.preprocessing import StandardScaler

    no_mmse = summary_df[summary_df["feature_set"] == "No MMSE"].copy()
    best_overall = no_mmse.sort_values("mean_accuracy", ascending=False).iloc[0]["model"]
    tree_models = {"Random Forest", "Gradient Boosting", "XGBoost",
                   "LightGBM", "CatBoost"}

    if best_overall in tree_models:
        target_name = best_overall
        surrogate = False
    else:
        # Use the strongest tree model in No MMSE as a surrogate explainer
        tree_subset = no_mmse[no_mmse["model"].isin(tree_models)]
        if tree_subset.empty:
            target_name = best_overall
            surrogate = False
        else:
            target_name = tree_subset.sort_values(
                "mean_accuracy", ascending=False
            ).iloc[0]["model"]
            surrogate = True
    print(f"  SHAP target model: {target_name}"
          + (f"  (surrogate; best No-MMSE was {best_overall})" if surrogate else ""))

    df = load_first_visit()
    feats = [f for f in FEATURE_SETS["No MMSE"] if f in df.columns]
    X = df[feats].values
    y = df["target_binary"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = _instantiate(target_name)
    model.fit(X_s, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)
    # Newer SHAP returns 3D (n, n_features, n_classes) for binary classifiers
    sv = np.asarray(shap_values)
    if sv.ndim == 3:
        # Pick positive class along the last axis
        sv = sv[:, :, 1]
    elif isinstance(shap_values, list) and len(shap_values) == 2:
        sv = np.asarray(shap_values[1])
    sv = np.asarray(sv)
    assert sv.shape == X_s.shape, (
        f"SHAP value shape {sv.shape} doesn't match X {X_s.shape}"
    )

    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        sv,
        features=X_s,
        feature_names=feats,
        plot_type="dot",
        show=False,
        max_display=10,
    )
    fig = plt.gcf()
    title = f"SHAP feature attributions: {target_name} (no MMSE)"
    if surrogate:
        title += f"\nused as tree surrogate for {best_overall}"
    fig.suptitle(title, fontweight="bold", y=1.02)
    save(fig, "fig3_shap_beeswarm")


# ----------------------------------------------------------------------------
# Figure 4: Conformal prediction intervals on a held-out sample
# ----------------------------------------------------------------------------

def fig4_conformal(summary_df: pd.DataFrame, cp_df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    no_mmse = summary_df[summary_df["feature_set"] == "No MMSE"]
    best_name = no_mmse.sort_values("mean_accuracy", ascending=False).iloc[0]["model"]
    print(f"  Conformal target model: {best_name}")

    df = load_first_visit()
    feats = [f for f in FEATURE_SETS["No MMSE"] if f in df.columns]
    X = df[feats].values
    y = df["target_binary"].values

    # 60/20/20 split (train / calibrate / test)
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_SEED
    )
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=RANDOM_SEED
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_cal_s = scaler.transform(X_cal)
    X_te_s = scaler.transform(X_te)

    model = _instantiate(best_name)
    model.fit(X_tr_s, y_tr)

    # Split conformal at alpha=0.10
    alpha = 0.10
    cal_proba = model.predict_proba(X_cal_s)
    cal_scores = 1 - cal_proba[np.arange(len(y_cal)), y_cal.astype(int)]
    n_cal = len(y_cal)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q = np.quantile(cal_scores, q_level)

    test_proba = model.predict_proba(X_te_s)
    pred_sets = []
    for i in range(len(y_te)):
        s = []
        for cls in range(test_proba.shape[1]):
            if test_proba[i, cls] >= 1 - q:
                s.append(cls)
        if not s:
            s = [int(np.argmax(test_proba[i]))]
        pred_sets.append(s)

    coverage = float(np.mean([y_te[i] in s for i, s in enumerate(pred_sets)]))

    # Take up to 20 samples sorted by P(demented)
    p_dem = test_proba[:, 1]
    order = np.argsort(p_dem)[:20] if len(p_dem) <= 20 else np.argsort(p_dem)
    if len(order) > 20:
        # Pick a spread sample
        idx = np.linspace(0, len(order) - 1, 20).astype(int)
        order = order[idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    for plot_i, i in enumerate(order):
        true = y_te[i]
        prob = p_dem[i]
        s = pred_sets[i]
        # Interval = range of class probabilities in the prediction set
        lower = min([test_proba[i, c] for c in s])
        upper = max([test_proba[i, c] for c in s])
        color = "#c0392b" if true == 1 else "#2980b9"
        ax.errorbar(
            plot_i, prob,
            yerr=[[max(0, prob - lower)], [max(0, upper - prob)]],
            fmt="o", color=color, ecolor=color, elinewidth=2,
            capsize=4, markersize=7, alpha=0.85,
        )

    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax.text(len(order) - 1, 0.51, "0.5 decision threshold",
            color="gray", fontsize=8, ha="right")

    ax.set_xlabel("Test sample (sorted by P(demented))")
    ax.set_ylabel("P(demented)  [conformal interval]")
    ax.set_title(
        f"Split conformal intervals  -  {best_name} (no MMSE)\n"
        f"alpha = {alpha:.2f}  ->  empirical coverage on test fold = {coverage:.0%}",
        fontweight="bold",
    )
    ax.set_ylim(-0.05, 1.05)

    # Legend for colours
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c0392b",
               markersize=8, label="True: demented"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9",
               markersize=8, label="True: nondemented"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)

    save(fig, "fig4_conformal_intervals")


# ----------------------------------------------------------------------------
# Figure 5: Per-fold accuracy distributions (boxplots, with vs without MMSE)
# ----------------------------------------------------------------------------

def fig5_distribution(fold_df: pd.DataFrame):
    # Only models with full CV distributions (>= 10 folds) get a box.
    counts = fold_df.groupby(["model", "feature_set"]).size()
    eligible = sorted({m for (m, _), n in counts.items() if n >= 10})
    fig, ax = plt.subplots(figsize=(13, 6))

    positions = []
    data = []
    colors = []
    labels = []
    for i, m in enumerate(eligible):
        all_feat = fold_df[(fold_df["model"] == m) & (fold_df["feature_set"] == "All Features")]
        no_mmse = fold_df[(fold_df["model"] == m) & (fold_df["feature_set"] == "No MMSE")]
        if len(all_feat) < 10 or len(no_mmse) < 10:
            continue
        positions.extend([i * 3 + 0.6, i * 3 + 1.4])
        data.extend([all_feat["accuracy"].values, no_mmse["accuracy"].values])
        colors.extend(["#e74c3c", "#27ae60"])
        labels.append(m)

    bp = ax.boxplot(
        data, positions=positions, widths=0.7, patch_artist=True,
        showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.4),
        medianprops=dict(color="white", lw=1.5),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")

    ax.set_xticks([i * 3 + 1 for i in range(len(labels))])
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Accuracy (per fold; 100 folds = 10 repeats x 10 stratified k-fold)")
    ax.set_title(
        "Cross-validation distribution: with vs. without MMSE",
        fontweight="bold",
    )
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_ylim(0.2, 1.05)
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#e74c3c", alpha=0.85, label="With MMSE (All Features)"),
        Patch(facecolor="#27ae60", alpha=0.85, label="Without MMSE"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False)

    save(fig, "fig5_cv_distribution")


# ----------------------------------------------------------------------------
# Helper: instantiate a model by name (mirrors build_results)
# ----------------------------------------------------------------------------

def _instantiate(name: str):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    table = {
        "Logistic Regression": lambda: LogisticRegression(
            **MODEL_PARAMS["logistic_regression"], random_state=RANDOM_SEED),
        "Random Forest": lambda: RandomForestClassifier(**MODEL_PARAMS["random_forest"]),
        "Gradient Boosting": lambda: GradientBoostingClassifier(**MODEL_PARAMS["gradient_boosting"]),
        "SVM": lambda: SVC(**MODEL_PARAMS["svm"]),
        "XGBoost": lambda: XGBClassifier(**MODEL_PARAMS["xgboost"]),
        "LightGBM": lambda: LGBMClassifier(**MODEL_PARAMS["lightgbm"]),
        "CatBoost": lambda: CatBoostClassifier(**MODEL_PARAMS["catboost"]),
    }
    if name in table:
        return table[name]()
    if name == "TabPFN":
        from tabpfn import TabPFNClassifier
        try:
            return TabPFNClassifier(device="cpu", N_ensemble_configurations=16)
        except TypeError:
            return TabPFNClassifier(device="cpu")
    raise KeyError(f"Unknown model: {name}")


def main():
    summary_path = RESULTS_TABLES / "mmse_ablation.csv"
    fold_path = RESULTS_TABLES / "cv_results.csv"
    cp_path = RESULTS_TABLES / "conformal_coverage.csv"

    if not summary_path.exists():
        sys.exit(f"Missing {summary_path}. Run scripts/build_results.py first.")

    summary_df = pd.read_csv(summary_path)
    fold_df = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
    cp_df = pd.read_csv(cp_path) if cp_path.exists() else pd.DataFrame()

    print("Generating figures (300 DPI) ...")
    fig1_circularity()
    fig2_ablation(summary_df)
    fig3_shap(summary_df)
    fig4_conformal(summary_df, cp_df)
    if not fold_df.empty:
        fig5_distribution(fold_df)
    else:
        print("  Skipping fig5: cv_results.csv missing")

    print("Done.")


if __name__ == "__main__":
    main()
