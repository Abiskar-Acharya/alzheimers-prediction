"""
OASIS-3 replication script — second-cohort validation of the MMSE inflation
finding from the OASIS-2 audit.

USAGE
-----
    python scripts/replicate_oasis3.py \\
        --oasis3-csv path/to/oasis3_freesurfer_demographic_table.csv \\
        --output results/tables/oasis3_mmse_ablation.csv

WHAT THIS SCRIPT DOES
---------------------
For each of the 7 model families used in the main OASIS-2 audit
(Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost, LightGBM,
CatBoost), run 10x10 stratified CV under three feature conditions:

    1. All Features      — demographics + MRI morphometry + MMSE
    2. No MMSE           — demographics + MRI morphometry only
    3. MRI Only          — eTIV / nWBV / ASF only

and produce a CSV with mean accuracy, std, and per-fold spread for each
(model, feature_set) combination — directly comparable to the main paper's
`mmse_ablation.csv` from OASIS-2.

DATA REQUIREMENTS
-----------------
OASIS-3 is gated behind https://www.oasis-brains.org/ and requires a free
research-use registration. After approval, download the
"freesurfer_demographic_table" or equivalent that contains, at minimum, these
columns (or their OASIS-3 equivalents — see FEATURE_MAPPING below):

    Subject ID, Age, M/F, EDUC, SES (optional), MMSE, CDR,
    eTIV, nWBV, ASF

The script will:
  1. Load that CSV
  2. Apply the FEATURE_MAPPING below to align column names with OASIS-2
  3. Filter to one row per subject (first visit / earliest scan)
  4. Recode CDR=0 -> non-demented (0), CDR>0 -> demented (1)
  5. Run the 10x10 audit identical to the OASIS-2 protocol

KNOWN CAVEATS
-------------
- OASIS-3 has more than 1000 subjects vs OASIS-2's 150. Per-fold variance
  will be much smaller; the question is whether the *mean* MMSE inflation
  effect (≈13 pp on OASIS-2) replicates within an SE of ~0.5 pp.
- OASIS-3 column names DIFFER from OASIS-2. The FEATURE_MAPPING dict below
  is a best-guess based on OASIS-3's documented schema. Verify against the
  actual CSV before running.
- OASIS-3 includes amyloid PET and richer FreeSurfer parcellations. The
  primary replication uses ONLY the OASIS-2-comparable features for an
  apples-to-apples comparison. A secondary "extended features" run is
  scaffolded but commented out — enable once the primary replication is
  done.
- We exclude subjects with CDR=0.5 (questionable dementia) from the
  binary task to match the OASIS-2 protocol. Modify GROUP_FILTER if the
  paper's framing changes.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

from src.config import MODEL_PARAMS, RANDOM_SEED


# ---------------------------------------------------------------------------
# Column-name mapping from OASIS-3 -> OASIS-2 schema.
#
# The keys are the column names in the OASIS-3 CSV; the values are the
# OASIS-2 names this script expects. Update this once you have the actual
# OASIS-3 download in front of you.
# ---------------------------------------------------------------------------
FEATURE_MAPPING = {
    # OASIS-3 likely names              OASIS-2 names
    "OASISID": "Subject ID",
    "AgeAtEntry": "Age",
    "GENDER": "M/F",
    "Education": "EDUC",
    "SocioeconomicStatus": "SES",
    "MMSE": "MMSE",
    "CDR": "CDR",
    "eTIV": "eTIV",
    "nWBV": "nWBV",
    "ASF": "ASF",
}

# Features used in each comparison. Identical to the OASIS-2 audit.
ALL_FEATURES = ["M/F_encoded", "Age", "EDUC", "SES", "MMSE", "eTIV", "nWBV", "ASF"]
NO_MMSE_FEATURES = ["M/F_encoded", "Age", "EDUC", "SES", "eTIV", "nWBV", "ASF"]
MRI_ONLY_FEATURES = ["eTIV", "nWBV", "ASF"]

# Drop subjects with CDR=0.5 to match the OASIS-2 binary protocol.
# Set to {0.0, 0.5, 1.0, 2.0} to keep them all (multi-class), or
# {0.0, 1.0, 2.0} to drop questionable dementia.
ALLOWED_CDR_VALUES = {0.0, 1.0, 2.0}


def get_models() -> dict:
    """Return the same 7-model panel used in the OASIS-2 audit."""
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(**MODEL_PARAMS["random_forest"]),
        "Gradient Boosting": GradientBoostingClassifier(
            **MODEL_PARAMS["gradient_boosting"]
        ),
        "SVM": SVC(**MODEL_PARAMS["svm"]),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            **MODEL_PARAMS["xgboost"], use_label_encoder=False
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(**MODEL_PARAMS["lightgbm"])
    if CatBoostClassifier is not None:
        models["CatBoost"] = CatBoostClassifier(**MODEL_PARAMS["catboost"])
    return models


def load_and_align_oasis3(csv_path: Path) -> pd.DataFrame:
    """Load OASIS-3 CSV and rename columns to match the OASIS-2 schema."""
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")
    print(f"  Raw columns: {list(df.columns)[:15]}{'...' if len(df.columns) > 15 else ''}")

    # Apply mapping (only rename keys that exist in the file)
    mapping_present = {k: v for k, v in FEATURE_MAPPING.items() if k in df.columns}
    missing_inputs = [k for k in FEATURE_MAPPING if k not in df.columns]
    if missing_inputs:
        print(
            f"  WARNING: these expected OASIS-3 columns are missing: "
            f"{missing_inputs}. Update FEATURE_MAPPING in this script."
        )
    df = df.rename(columns=mapping_present)

    # Numeric encoding for sex.
    if "M/F" in df.columns and "M/F_encoded" not in df.columns:
        df["M/F_encoded"] = (df["M/F"].astype(str).str.upper() == "M").astype(int)

    # First-visit-only filter: keep earliest scan per subject.
    if "Subject ID" in df.columns:
        # If a date / age-at-scan column exists, sort by it; otherwise just dedupe.
        sort_col = (
            "AgeAtEntry"
            if "AgeAtEntry" in df.columns
            else "Age"
            if "Age" in df.columns
            else None
        )
        if sort_col is not None:
            df = df.sort_values([sort_col]).drop_duplicates(subset=["Subject ID"], keep="first")
        else:
            df = df.drop_duplicates(subset=["Subject ID"], keep="first")

    # CDR filter (drop questionable dementia by default).
    if "CDR" in df.columns:
        df = df[df["CDR"].isin(ALLOWED_CDR_VALUES)]
        df["target_binary"] = (df["CDR"] > 0).astype(int)
    else:
        raise ValueError(
            "CDR column not found in OASIS-3 CSV after column mapping. "
            "Check FEATURE_MAPPING."
        )

    # Median-impute SES if present (matches OASIS-2 preprocessing).
    if "SES" in df.columns and df["SES"].isna().any():
        df["SES"] = df["SES"].fillna(df["SES"].median())

    print(f"  Aligned shape: {df.shape}")
    print(f"  Class balance: {df['target_binary'].value_counts().to_dict()}")
    return df


def run_audit(df: pd.DataFrame, n_repeats: int = 10, n_folds: int = 10) -> pd.DataFrame:
    """Run the same 10x10 audit on the OASIS-3 dataframe."""
    feature_sets = {
        "All Features": ALL_FEATURES,
        "No MMSE": NO_MMSE_FEATURES,
        "MRI Only": MRI_ONLY_FEATURES,
    }
    cv = RepeatedStratifiedKFold(
        n_splits=n_folds, n_repeats=n_repeats, random_state=RANDOM_SEED
    )
    y = df["target_binary"].values

    rows = []
    models = get_models()
    for set_name, feats in feature_sets.items():
        feats_present = [f for f in feats if f in df.columns]
        if len(feats_present) < len(feats):
            missing = set(feats) - set(feats_present)
            print(f"  [{set_name}] missing features (skipping): {missing}")
        if not feats_present:
            continue
        X = df[feats_present].values
        for model_name, model in models.items():
            print(f"  {set_name:15s} | {model_name}", end="  ", flush=True)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            scores = cross_val_score(
                pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1
            )
            row = {
                "model": model_name,
                "feature_set": set_name,
                "mmse_included": "MMSE" in feats_present,
                "n_features": len(feats_present),
                "n_folds": len(scores),
                "mean_accuracy": float(np.mean(scores)),
                "std_accuracy": float(np.std(scores)),
                "ci_lower_95_accuracy": float(np.percentile(scores, 2.5)),
                "ci_upper_95_accuracy": float(np.percentile(scores, 97.5)),
                # SE on the mean (Nadeau-Bengio for repeated 10-fold).
                "se_nb_mean_accuracy": float(
                    np.mean(
                        [np.std(scores[i * n_folds : (i + 1) * n_folds]) for i in range(n_repeats)]
                    )
                    * np.sqrt(1 / n_folds + 1 / (n_folds - 1))
                    / np.sqrt(n_repeats)
                ),
            }
            rows.append(row)
            print(
                f"acc={row['mean_accuracy']:.3f} ± {row['se_nb_mean_accuracy']:.3f} (NB SE)"
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oasis3-csv",
        type=Path,
        required=True,
        help="Path to OASIS-3 demographics+FreeSurfer CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables" / "oasis3_mmse_ablation.csv",
        help="Output CSV path",
    )
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=10)
    args = parser.parse_args()

    if not args.oasis3_csv.exists():
        print(f"ERROR: {args.oasis3_csv} does not exist.")
        print(
            "Register at https://www.oasis-brains.org/ and download the "
            "OASIS-3 demographics+FreeSurfer table."
        )
        return 1

    df = load_and_align_oasis3(args.oasis3_csv)
    if len(df) < 30:
        print(f"ERROR: only {len(df)} subjects after filtering. Check FEATURE_MAPPING.")
        return 1

    print()
    print(f"Running 10x10 audit on n={len(df)} OASIS-3 subjects...")
    results = run_audit(df, n_repeats=args.n_repeats, n_folds=args.n_folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")

    # Summary block: mean inflation per model, comparable to OASIS-2 paper.
    pivot = results.pivot(index="model", columns="feature_set", values="mean_accuracy")
    pivot["MMSE Inflation (pp)"] = (pivot["All Features"] - pivot["No MMSE"]) * 100
    print("\n=== OASIS-3 Replication Summary (mean accuracy by feature set) ===")
    print(pivot.round(3).to_string())
    print(
        f"\nMean MMSE inflation across models: "
        f"{pivot['MMSE Inflation (pp)'].mean():.2f} pp"
    )
    print(
        "\nCompare against the OASIS-2 finding (13.1 pp average inflation, "
        "range 6.4-17.7 pp). If OASIS-3 replicates within ~3 pp of that "
        "average, the paper's central claim generalizes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
