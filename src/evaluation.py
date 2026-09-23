"""
Evaluation utilities with proper cross-validation methodology.
Repeated stratified k-fold with confidence intervals.
MMSE ablation comparison framework.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    make_scorer
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional
from src.config import RANDOM_SEED, N_CV_REPEATS, N_CV_FOLDS, FEATURE_SETS


def create_pipeline(model, scale: bool = True) -> Pipeline:
    """
    Wrap a model in a pipeline with optional scaling.

    Args:
        model: sklearn-compatible classifier
        scale: Whether to add StandardScaler

    Returns:
        Pipeline with scaler + model
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def repeated_stratified_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = N_CV_REPEATS,
    n_folds: int = N_CV_FOLDS,
    scoring: str = "accuracy",
    scale: bool = True,
    return_scores: bool = False,
) -> Dict:
    """
    Perform repeated stratified k-fold cross-validation with CIs.

    Args:
        model: sklearn-compatible classifier
        X: Feature matrix
        y: Target array
        n_repeats: Number of CV repetitions (default 10)
        n_folds: Number of folds (default 10)
        scoring: Scoring metric
        scale: Whether to scale features
        return_scores: Whether to include raw fold scores

    Returns:
        Dict with mean, std, ci_lower, ci_upper, n_scores
    """
    pipe = create_pipeline(model, scale=scale)
    cv = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=RANDOM_SEED
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    result = {
        "mean": np.mean(scores),
        "std": np.std(scores),
        "ci_lower": np.percentile(scores, 2.5),
        "ci_upper": np.percentile(scores, 97.5),
        "n_scores": len(scores),
    }

    if return_scores:
        result["scores"] = scores

    return result


def evaluate_model_comprehensive(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = N_CV_REPEATS,
    n_folds: int = N_CV_FOLDS,
    scale: bool = True,
) -> Dict:
    """
    Comprehensive evaluation with multiple metrics.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc (each with CIs)
    """
    metrics = {}
    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        metrics[metric_name] = repeated_stratified_cv(
            model, X, y,
            n_repeats=n_repeats,
            n_folds=n_folds,
            scoring=metric_name,
            scale=scale,
        )
    return metrics


def mmse_ablation_comparison(
    models: Dict,
    df: pd.DataFrame,
    feature_sets: Optional[Dict] = None,
    n_repeats: int = N_CV_REPEATS,
    n_folds: int = N_CV_FOLDS,
) -> pd.DataFrame:
    """
    Run MMSE ablation: evaluate all models across multiple feature sets.

    This is the intellectual core of the project - showing how much
    accuracy drops when MMSE (which is circular with CDR/diagnosis)
    is removed.

    Args:
        models: Dict of {name: model_instance}
        df: DataFrame with features and target_binary
        feature_sets: Dict of {set_name: feature_list} (default: config.FEATURE_SETS)
        n_repeats: CV repetitions
        n_folds: CV folds

    Returns:
        DataFrame with columns: Model, Feature_Set, Accuracy_Mean, Accuracy_Std,
                                CI_Lower, CI_Upper, Accuracy_Formatted
    """
    if feature_sets is None:
        feature_sets = FEATURE_SETS

    results = []
    for model_name, model in models.items():
        for set_name, features in feature_sets.items():
            # Check all features exist
            available = [f for f in features if f in df.columns]
            if not available:
                continue

            X = df[available].values
            y = df["target_binary"].values

            cv_result = repeated_stratified_cv(
                model, X, y,
                n_repeats=n_repeats,
                n_folds=n_folds,
                scale=True,
            )

            results.append({
                "Model": model_name,
                "Feature_Set": set_name,
                "Accuracy_Mean": cv_result["mean"],
                "Accuracy_Std": cv_result["std"],
                "CI_Lower": cv_result["ci_lower"],
                "CI_Upper": cv_result["ci_upper"],
                "Accuracy_Formatted": f"{cv_result['mean']:.3f} ± {cv_result['std']:.3f}",
            })

    return pd.DataFrame(results)


def single_split_evaluation(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scale: bool = True,
) -> Dict:
    """
    Single train-test split evaluation (for reproducing original results).

    Returns:
        Dict with accuracy, precision, recall, f1, confusion_matrix, report
    """
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    if y_proba is not None:
        result["roc_auc"] = roc_auc_score(y_test, y_proba)

    return result


def format_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format results DataFrame for display with clean CI formatting.

    Args:
        results_df: DataFrame from mmse_ablation_comparison

    Returns:
        Pivoted DataFrame: rows=Models, columns=Feature Sets
    """
    pivot = results_df.pivot(
        index="Model", columns="Feature_Set", values="Accuracy_Formatted"
    )
    # Reorder columns to match logical progression
    col_order = ["All Features", "No MMSE", "MRI Only", "Demographics Only"]
    existing_cols = [c for c in col_order if c in pivot.columns]
    return pivot[existing_cols]


def statistical_comparison(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    test: str = "wilcoxon",
) -> Dict:
    """
    Statistical significance test between two sets of CV scores.

    Args:
        scores_a: CV scores from model/feature set A
        scores_b: CV scores from model/feature set B
        test: 'wilcoxon' (paired, non-parametric) or 'ttest' (paired t-test)

    Returns:
        Dict with statistic, p_value, significant (at p<0.05)
    """
    from scipy import stats

    if test == "wilcoxon":
        stat, p_value = stats.wilcoxon(scores_a, scores_b)
    elif test == "ttest":
        stat, p_value = stats.ttest_rel(scores_a, scores_b)
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
