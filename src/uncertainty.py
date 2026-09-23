"""
Uncertainty quantification for Alzheimer's prediction.
Conformal prediction, calibration analysis, Bayesian stacking.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
from src.config import RANDOM_SEED


def calibration_analysis(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict:
    """
    Analyze model calibration using cross-validated predictions.

    Args:
        model: sklearn-compatible classifier with predict_proba
        X: Feature matrix
        y: Target array
        n_bins: Number of calibration bins
        strategy: 'uniform' or 'quantile'

    Returns:
        Dict with fraction_of_positives, mean_predicted_value, ece
    """
    # Get cross-validated probability predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_proba = cross_val_predict(model, X_scaled, y, cv=cv, method="predict_proba")[:, 1]

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(
        y, y_proba, n_bins=n_bins, strategy=strategy
    )

    # Expected Calibration Error
    bin_counts = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
    total = len(y)
    ece = 0.0
    for i in range(len(fraction_pos)):
        if i < len(bin_counts) and bin_counts[i] > 0:
            ece += (bin_counts[i] / total) * abs(fraction_pos[i] - mean_pred[i])

    return {
        "fraction_of_positives": fraction_pos,
        "mean_predicted_value": mean_pred,
        "ece": ece,
        "y_proba": y_proba,
    }


def conformal_prediction(
    model,
    X: np.ndarray,
    y: np.ndarray,
    alpha_levels: List[float] = None,
) -> Dict:
    """
    Conformal prediction using split conformal method.

    Provides prediction sets with guaranteed coverage.

    Args:
        model: sklearn-compatible classifier with predict_proba
        X: Feature matrix
        y: Target array
        alpha_levels: Significance levels (default [0.05, 0.10])

    Returns:
        Dict with prediction_sets, coverage, avg_set_size per alpha
    """
    if alpha_levels is None:
        alpha_levels = [0.05, 0.10]

    results = {}
    n = len(y)

    # Use cross-validated predictions for conformal scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_proba = cross_val_predict(model, X_scaled, y, cv=cv, method="predict_proba")

    for alpha in alpha_levels:
        # Non-conformity scores: 1 - probability of true class
        scores = 1 - y_proba[np.arange(n), y.astype(int)]

        # Quantile threshold
        q = np.quantile(scores, 1 - alpha)

        # Prediction sets: include class if its probability > 1 - q
        prediction_sets = []
        for i in range(n):
            pset = []
            for cls in range(y_proba.shape[1]):
                if y_proba[i, cls] >= 1 - q:
                    pset.append(cls)
            if not pset:
                # Always include at least the most likely class
                pset = [np.argmax(y_proba[i])]
            prediction_sets.append(pset)

        # Coverage: fraction where true label is in prediction set
        coverage = np.mean([y[i] in pset for i, pset in enumerate(prediction_sets)])

        # Average set size
        avg_size = np.mean([len(pset) for pset in prediction_sets])

        # Fraction of singleton sets (precise predictions)
        singleton_frac = np.mean([len(pset) == 1 for pset in prediction_sets])

        results[alpha] = {
            "prediction_sets": prediction_sets,
            "coverage": coverage,
            "avg_set_size": avg_size,
            "singleton_fraction": singleton_frac,
            "threshold": q,
        }

    return results


def repeated_split_conformal(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    alpha_levels: List[float] = None,
    n_seeds: int = 50,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    base_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Classical split-conformal prediction repeated over multiple random seeds.

    Each seed: split data into train (60%) / calibration (20%) / test (20%),
    fit on train, derive threshold from calibration scores, evaluate empirical
    coverage and set sizes on the held-out test partition. This separates
    calibration from evaluation — the single 80/20 split used in
    `conformal_prediction()` evaluates coverage on the same points used to
    estimate the threshold, which is anti-conservative.

    Args:
        model_factory: Callable returning a fresh sklearn classifier each call
                       (so each seed gets an independent fit). Must support
                       predict_proba after fit().
        X: Feature matrix (will be standardized within each seed's training
           partition).
        y: Binary or multiclass target.
        alpha_levels: Significance levels (default [0.10, 0.05, 0.01]).
        n_seeds: Number of train/cal/test splits to run (default 50).
        train_frac: Training fraction (default 0.6).
        cal_frac: Calibration fraction (default 0.2). Test fraction is
                  inferred as 1 - train_frac - cal_frac.
        base_seed: Starting seed; seeds run base_seed .. base_seed + n_seeds - 1.

    Returns:
        DataFrame with one row per (alpha, seed) plus a summary block. Columns:
        alpha, seed, empirical_coverage, mean_set_size, singleton_fraction,
        n_test, n_calibration, threshold.

        Aggregate the result by alpha to get mean ± SE across seeds — this
        is the quantity the paper should report instead of single-shot
        coverage.
    """
    if alpha_levels is None:
        alpha_levels = [0.10, 0.05, 0.01]

    test_frac = 1.0 - train_frac - cal_frac
    if test_frac <= 0:
        raise ValueError(
            f"train_frac + cal_frac = {train_frac + cal_frac:.2f} >= 1; "
            f"no points left for the test partition."
        )

    n_classes = len(np.unique(y))
    rows = []

    for offset in range(n_seeds):
        seed = base_seed + offset

        # First split: train (train_frac) vs (cal + test)
        X_train, X_caltest, y_train, y_caltest = train_test_split(
            X, y,
            train_size=train_frac,
            stratify=y,
            random_state=seed,
        )
        # Second split: cal vs test, both inside the (cal + test) partition
        cal_share_of_remainder = cal_frac / (cal_frac + test_frac)
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_caltest, y_caltest,
            train_size=cal_share_of_remainder,
            stratify=y_caltest,
            random_state=seed,
        )

        # Standardize using train statistics only — no leakage into cal/test.
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_cal_s = scaler.transform(X_cal)
        X_test_s = scaler.transform(X_test)

        # Fresh model fit on the training partition.
        clf = model_factory()
        clf.fit(X_train_s, y_train)

        # Calibration: nonconformity score = 1 - p(true_class) on cal set.
        cal_proba = clf.predict_proba(X_cal_s)
        cal_scores = 1 - cal_proba[np.arange(len(y_cal)), y_cal.astype(int)]

        # Test predictions.
        test_proba = clf.predict_proba(X_test_s)

        for alpha in alpha_levels:
            # Finite-sample-corrected quantile (Romano et al., 2019).
            n_cal = len(cal_scores)
            quantile_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
            quantile_level = min(quantile_level, 1.0)
            q = np.quantile(cal_scores, quantile_level)

            # Test prediction sets: include class iff its nonconformity <= q.
            test_scores = 1 - test_proba  # shape (n_test, n_classes)
            in_set = test_scores <= q  # boolean (n_test, n_classes)
            # Guarantee non-empty sets (always include argmax).
            no_set = ~in_set.any(axis=1)
            if no_set.any():
                argmax_cls = test_proba[no_set].argmax(axis=1)
                rows_idx = np.where(no_set)[0]
                in_set[rows_idx, argmax_cls] = True

            true_in_set = in_set[np.arange(len(y_test)), y_test.astype(int)]
            empirical_coverage = float(np.mean(true_in_set))
            mean_set_size = float(in_set.sum(axis=1).mean())
            singleton_fraction = float(np.mean(in_set.sum(axis=1) == 1))

            rows.append({
                "alpha": alpha,
                "seed": seed,
                "empirical_coverage": empirical_coverage,
                "mean_set_size": mean_set_size,
                "singleton_fraction": singleton_fraction,
                "n_test": len(y_test),
                "n_calibration": n_cal,
                "threshold": q,
            })

    return pd.DataFrame(rows)


def summarize_repeated_split_conformal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the per-seed output of `repeated_split_conformal()` to mean ± SE
    per alpha level. SE is computed as std / sqrt(n_seeds), treating the
    seeds as independent draws (which they are by construction).
    """
    grouped = df.groupby("alpha")
    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        coverage_mean=("empirical_coverage", "mean"),
        coverage_se=("empirical_coverage", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        coverage_min=("empirical_coverage", "min"),
        coverage_max=("empirical_coverage", "max"),
        set_size_mean=("mean_set_size", "mean"),
        set_size_se=("mean_set_size", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        singleton_mean=("singleton_fraction", "mean"),
        singleton_se=("singleton_fraction", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    ).reset_index()
    summary["theoretical_coverage"] = 1 - summary["alpha"]
    cols = [
        "alpha", "theoretical_coverage", "coverage_mean", "coverage_se",
        "coverage_min", "coverage_max",
        "set_size_mean", "set_size_se",
        "singleton_mean", "singleton_se",
        "n_seeds",
    ]
    return summary[cols]


def bayesian_stacking_weights(
    models: Dict,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
) -> Dict:
    """
    Compute Bayesian-inspired stacking weights using CV log-likelihoods.

    Args:
        models: Dict of {name: model_instance}
        X: Feature matrix
        y: Target array
        n_folds: Number of CV folds

    Returns:
        Dict with weights (Dict[str, float]) and log_likelihoods
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log_liks = {}

    for name, model in models.items():
        # Get cross-validated probability predictions
        try:
            y_proba = cross_val_predict(
                model, X_scaled, y, cv=cv, method="predict_proba"
            )
            # Log-likelihood of true labels
            proba_true = y_proba[np.arange(len(y)), y.astype(int)]
            # Clip to avoid log(0)
            proba_true = np.clip(proba_true, 1e-10, 1.0)
            log_liks[name] = np.sum(np.log(proba_true))
        except Exception:
            log_liks[name] = -np.inf

    # Convert to weights using softmax
    log_lik_values = np.array(list(log_liks.values()))
    # Shift for numerical stability
    log_lik_shifted = log_lik_values - np.max(log_lik_values)
    exp_values = np.exp(log_lik_shifted)
    weights_array = exp_values / np.sum(exp_values)

    weights = dict(zip(log_liks.keys(), weights_array))

    return {
        "weights": weights,
        "log_likelihoods": log_liks,
    }


def clinical_decision_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Evaluate model at different decision thresholds for clinical use.

    Screening threshold: high sensitivity (catch most cases)
    Confirmatory threshold: high specificity (few false positives)

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        thresholds: List of thresholds to evaluate

    Returns:
        DataFrame with threshold, sensitivity, specificity, ppv, npv
    """
    from sklearn.metrics import recall_score, precision_score

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05).tolist()

    results = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        results.append({
            "threshold": t,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
        })

    return pd.DataFrame(results)
