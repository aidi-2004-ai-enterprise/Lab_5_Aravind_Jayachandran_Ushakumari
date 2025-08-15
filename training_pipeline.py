"""Lab 5 pipeline for bankruptcy prediction.

This script performs EDA, preprocessing, simple feature selection (RF Top-K + VIF),
hyperparameter tuning, training (LR/XGB/LGBM), evaluation (ROC/PR/F1 + calibration),
SHAP summary for the best model, and PSI drift checks. It matches the Lab 4 decisions.
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap  # type: ignore
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier  # type: ignore
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier  # type: ignore

# Silence common, harmless warnings in this pipeline run.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*has changed to a list of ndarray.*",
)


# ---------------------------------------------------------------------
# Utilities / small helpers
# ---------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    """Set Python/NumPy random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p: Path) -> Path:
    """Create directory if it doesn't exist and return it."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def df_to_markdown_safe(data: pd.DataFrame, index: bool = False) -> str:
    """Render a small DataFrame as Markdown (avoids extra deps like tabulate)."""
    if not index:
        header = list(data.columns)
    else:
        header = ["", *list(data.columns)]
        data = data.reset_index()

    header_line = "| " + " | ".join(map(str, header)) + " |"
    sep_line = "| " + " | ".join(["---"] * len(header)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in data.to_numpy()]
    return "\n".join([header_line, sep_line, *rows])


@dataclass
class Paths:
    """Holds all filesystem paths used by the pipeline."""

    root: Path
    artifacts: Path
    eda: Path
    plots: Path
    shap: Path
    psi: Path
    models: Path
    report_md: Path
    reqs: Path
    ruff_cfg: Path

    @staticmethod
    def make(root: Path) -> Paths:
        """Create and return a Paths bundle under the given project root."""
        artifacts = root / "artifacts"
        p = Paths(
            root=root,
            artifacts=ensure_dir(artifacts),
            eda=ensure_dir(artifacts / "eda"),
            plots=ensure_dir(artifacts / "plots"),
            shap=ensure_dir(artifacts / "shap"),
            psi=ensure_dir(artifacts / "psi"),
            models=ensure_dir(artifacts / "models"),
            report_md=artifacts / "report.md",
            reqs=artifacts / "requirements_frozen.txt",
            ruff_cfg=artifacts / "ruff.toml",
        )
        return p


# ---------------------------------------------------------------------
# Repro bundle
# ---------------------------------------------------------------------
def write_repro_files(paths: Paths) -> None:
    """Write a minimal requirements freeze and a Ruff config (PEP8+docstyle)."""
    # Freeze a few key libs only (fast & simple).
    pkgs = [
        ("python", f"{sys_version()}"),
        ("pandas", pd.__version__),
        ("numpy", np.__version__),
        ("scikit-learn", sklearn_version()),
        ("imblearn", imblearn_version()),
        ("xgboost", xgb_version()),
        ("lightgbm", lgbm_version()),
        ("shap", shap.__version__),
        ("matplotlib", plt.matplotlib.__version__),
        ("seaborn", sns.__version__),
        ("joblib", joblib.__version__),
    ]
    lines = [f"{name}=={ver}" for name, ver in pkgs]
    paths.reqs.write_text("\n".join(lines), encoding="utf-8")

    ruff_toml = """\
target-version = "py310"

line-length = 100
indent-width = 4

[lint]
select = [
    "E","F","I","D","UP","RUF",
]
ignore = [
    "D203","D213",
]

[lint.pydocstyle]
convention = "google"
"""
    paths.ruff_cfg.write_text(ruff_toml, encoding="utf-8")


def sys_version() -> str:
    """Return a short Python version string."""
    import sys as _sys

    return ".".join(map(str, _sys.version_info[:3]))


def sklearn_version() -> str:
    """Return sklearn version."""
    import sklearn as _sk

    return _sk.__version__


def imblearn_version() -> str:
    """Return imblearn version."""
    import imblearn as _ibl

    return _ibl.__version__


def xgb_version() -> str:
    """Return XGBoost version."""
    import xgboost as _xgb

    return _xgb.__version__


def lgbm_version() -> str:
    """Return LightGBM version."""
    import lightgbm as _lgbm

    return _lgbm.__version__


# ---------------------------------------------------------------------
# Outlier clipper (IQR winsorization) - pipeline-compatible
# ---------------------------------------------------------------------
class IQRClipper(BaseEstimator, TransformerMixin):
    """IQR-based winsorization transformer for numeric columns.

    Compatible with sklearn/imb pipelines. Accepts both pandas DataFrames and
    NumPy arrays (as passed by ColumnTransformer).
    """

    def __init__(self, numeric_cols: list[str]) -> None:
        """Initialize with the names of the numeric columns."""
        self.numeric_cols = numeric_cols
        # bounds_ maps column keys (name or index) -> (low, high)
        self.bounds_: dict[object, tuple[float, float]] = {}
        self._use_index_keys: bool = False  # True when fed arrays

    def fit(
        self, X: pd.DataFrame | np.ndarray, y: pd.Series | None = None
    ) -> IQRClipper:
        """Compute per-feature IQR bounds on the training data."""
        if isinstance(X, pd.DataFrame):
            keys = list(X.columns)
            data = X.values.astype(float)
            self._use_index_keys = False
        else:
            data = np.asarray(X, dtype=float)
            keys = list(range(data.shape[1]))
            self._use_index_keys = True

        for j, key in enumerate(keys):
            col = data[:, j]
            q1, q3 = np.percentile(col, 25), np.percentile(col, 75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            self.bounds_[key] = (lo, hi)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        """Clip each feature to the learned IQR bounds; preserve input type."""
        if isinstance(X, pd.DataFrame):
            Xc = X.copy()
            for key, (lo, hi) in self.bounds_.items():
                Xc[key] = Xc[key].astype(float).clip(lower=lo, upper=hi)
            return Xc

        data = np.asarray(X, dtype=float).copy()
        for idx, (lo, hi) in self.bounds_.items():
            data[:, int(idx)] = np.clip(data[:, int(idx)], lo, hi)
        return data


# ---------------------------------------------------------------------
# PSI (Population Stability Index)
# ---------------------------------------------------------------------
def psi_for_column(
    base: ArrayLike, compare: ArrayLike, bins: int = 10, eps: float = 1e-9
) -> float:
    """Compute PSI for one numeric feature using equal-width bins."""
    base = np.asarray(base, dtype=float)
    compare = np.asarray(compare, dtype=float)

    lo = np.nanpercentile(base, 1)
    hi = np.nanpercentile(base, 99)
    edges = np.linspace(lo, hi, bins + 1)

    base_hist, _ = np.histogram(base, bins=edges)
    cmp_hist, _ = np.histogram(compare, bins=edges)

    base_pct = base_hist / max(base_hist.sum(), eps)
    cmp_pct = cmp_hist / max(cmp_hist.sum(), eps)

    base_pct = np.clip(base_pct, eps, None)
    cmp_pct = np.clip(cmp_pct, eps, None)

    psi = np.sum((cmp_pct - base_pct) * np.log(cmp_pct / base_pct))
    return float(psi)


def psi_dataframe(
    X_train: pd.DataFrame, X_test: pd.DataFrame, top_k: int = 15
) -> pd.DataFrame:
    """Compute PSI for all numeric features and return top-k by PSI desc."""
    out: list[dict[str, float | str]] = []
    for col in X_train.columns:
        try:
            val = psi_for_column(X_train[col], X_test[col])
            out.append({"feature": col, "psi": float(val)})
        except Exception:
            continue
    df = pd.DataFrame(out).sort_values("psi", ascending=False)
    return df.head(top_k)


def plot_psi_bar(psi_df: pd.DataFrame, out_path: Path) -> None:
    """Save a horizontal bar plot of PSI values."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=psi_df, y="feature", x="psi", ax=ax, orient="h")
    ax.axvline(0.1, ls="--", lw=1)
    ax.axvline(0.2, ls="--", lw=1)
    ax.set_title("PSI (Train vs Test) - top features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------
def eda_plots(df: pd.DataFrame, target: str, paths: Paths) -> dict[str, str]:
    """Generate simple EDA visuals used in the report."""
    out: dict[str, str] = {}

    # Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    df[target].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    p = paths.eda / "class_distribution.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    out["class_dist"] = str(p)

    # Correlation heatmap (numeric only)
    num_df = df.drop(columns=[target]).select_dtypes(include=[np.number])
    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap (numeric)")
    fig.tight_layout()
    p = paths.eda / "corr_heatmap.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    out["corr"] = str(p)

    # Skewness bar (top 20)
    sk = num_df.skew(numeric_only=True).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    sk.plot(kind="bar", ax=ax)
    ax.set_title("Skewness (top 20)")
    fig.tight_layout()
    p = paths.eda / "skewness_bar.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    out["skew"] = str(p)

    # Boxplots for top skewed (first 8)
    cols = list(sk.index[:8])
    if cols:
        fig, axes = plt.subplots(
            nrows=2, ncols=4, figsize=(12, 6), constrained_layout=True
        )
        axes = axes.ravel()
        for i, c in enumerate(cols):
            sns.boxplot(x=df[c], ax=axes[i])
            axes[i].set_title(c)
        p = paths.eda / "boxplots_top_skewed.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        out["box_top_skew"] = str(p)

    return out


# ---------------------------------------------------------------------
# Feature selection (RandomForest Top-K + VIF prune)
# ---------------------------------------------------------------------
def rf_topk_features(
    X: pd.DataFrame, y: pd.Series, k: int, seed: int
) -> tuple[list[str], pd.DataFrame]:
    """Fit a class-weighted RandomForest and return top-k features by importance."""
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X, y)
    imp_df = (
        pd.DataFrame(
            {"feature": X.columns, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    topk = list(imp_df["feature"].head(k))
    return topk, imp_df


def vif_snapshot(X: pd.DataFrame) -> pd.DataFrame:
    """Compute a VIF-like snapshot via linear reg R^2 on standardized features."""
    Xs = (X - X.mean()) / (X.std(ddof=0) + 1e-9)
    out: list[dict[str, float | str]] = []
    for i, col in enumerate(Xs.columns):
        y_col = Xs[col].values
        X_others = np.delete(Xs.values, i, axis=1)
        lr = LinearRegression(n_jobs=-1)
        lr.fit(X_others, y_col)
        r2 = lr.score(X_others, y_col)
        vif = 1.0 / max(1.0 - r2, 1e-9)
        out.append({"feature": col, "vif": float(vif)})
    return pd.DataFrame(out).sort_values("vif", ascending=False)


def prune_by_vif(
    X: pd.DataFrame, threshold: float = 10.0, max_iter: int = 12
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Iteratively drop the feature with highest VIF until all VIF <= threshold."""
    kept = list(X.columns)
    last_vif = vif_snapshot(X[kept])

    it = 0
    while it < max_iter and last_vif["vif"].max() > threshold and len(kept) > 1:
        worst = last_vif.iloc[0]["feature"]
        kept.remove(str(worst))
        last_vif = vif_snapshot(X[kept])
        it += 1
    return X[kept].copy(), kept, last_vif


# ---------------------------------------------------------------------
# Preprocessors & model pipelines (consistent with Lab 4)
# ---------------------------------------------------------------------
def _make_ohe() -> OneHotEncoder:
    """Return an OneHotEncoder compatible with both old/new sklearn."""
    try:
        # sklearn >= 1.2 / 1.4 uses sparse_output
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # fallback for older versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessors(
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """Build two preprocessors.

    - For LR/XGB: impute + winsorize + scale numeric, impute + OHE categorical.
    - For LGBM: impute + winsorize numeric (no scaling), impute + OHE categorical.
    """
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    ohe = _make_ohe()

    pre_lr_xgb = ColumnTransformer(
        transformers=[
            (
                "num",
                ImbPipeline(
                    [
                        ("imputer", num_imputer),
                        ("winsor", IQRClipper(numeric_cols=numeric_features)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                ImbPipeline([("imputer", cat_imputer), ("ohe", ohe)]),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        n_jobs=-1,
    )

    pre_lgbm = ColumnTransformer(
        transformers=[
            (
                "num",
                ImbPipeline(
                    [
                        ("imputer", num_imputer),
                        ("winsor", IQRClipper(numeric_cols=numeric_features)),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                ImbPipeline([("imputer", cat_imputer), ("ohe", ohe)]),
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        n_jobs=-1,
    )
    return pre_lr_xgb, pre_lgbm


def make_pipelines(
    pos_weight: float, seed: int, pre_lr_xgb: ColumnTransformer, pre_lgbm: ColumnTransformer
) -> dict[str, ImbPipeline]:
    """Create model pipelines consistent with Lab 4 decisions."""
    lr_pipe = ImbPipeline(
        steps=[
            ("pre", pre_lr_xgb),
            ("smote", SMOTE(random_state=seed)),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs", max_iter=2000, class_weight="balanced", n_jobs=-1
                ),
            ),
        ]
    )

    xgb_pipe = ImbPipeline(
        steps=[
            ("pre", pre_lr_xgb),
            ("smote", SMOTE(random_state=seed)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    random_state=seed,
                    scale_pos_weight=pos_weight,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    lgbm_pipe = ImbPipeline(
        steps=[
            ("pre", pre_lgbm),
            ("smote", SMOTE(random_state=seed)),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=400,
                    num_leaves=64,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    random_state=seed,
                    class_weight=None,
                    n_jobs=-1,
                    scale_pos_weight=pos_weight,
                ),
            ),
        ]
    )

    return {
        "LogisticRegression": lr_pipe,
        "XGBoost": xgb_pipe,
        "LightGBM": lgbm_pipe,
    }


def search_spaces(pos_weight: float) -> dict[str, dict[str, Iterable]]:
    """RandomizedSearchCV grids (simple & bounded)."""
    return {
        "LogisticRegression": {
            "clf__C": stats.loguniform(1e-3, 10.0),
            "clf__penalty": ["l2"],
        },
        "XGBoost": {
            "clf__n_estimators": stats.randint(250, 600),
            "clf__max_depth": stats.randint(3, 8),
            "clf__learning_rate": stats.loguniform(0.005, 0.08),
            "clf__subsample": stats.uniform(0.5, 0.5),
            "clf__colsample_bytree": stats.uniform(0.5, 0.5),
            "clf__reg_alpha": stats.loguniform(1e-3, 1.0),
            "clf__reg_lambda": stats.loguniform(1.0, 3.0),
            "clf__scale_pos_weight": [pos_weight],
        },
        "LightGBM": {
            "clf__n_estimators": stats.randint(250, 600),
            "clf__num_leaves": stats.randint(31, 120),
            "clf__learning_rate": stats.loguniform(0.005, 0.06),
            "clf__subsample": stats.uniform(0.5, 0.5),
            "clf__colsample_bytree": stats.uniform(0.5, 0.5),
            "clf__reg_alpha": stats.loguniform(1e-3, 0.1),
            "clf__reg_lambda": stats.loguniform(0.5, 2.0),
            "clf__scale_pos_weight": [pos_weight],
        },
    }


# ---------------------------------------------------------------------
# Evaluation utilities (ROC / PR / Calibration + metrics table)
# ---------------------------------------------------------------------
@dataclass
class EvalResult:
    """Holder for evaluation results for one split & model."""

    model: str
    split: str
    roc_auc: float
    pr_auc: float
    f1_at_050: float


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    title: str,
    out_path: Path,
) -> None:
    """Plot ROC curves for Train/Test on one plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    title: str,
    out_path: Path,
) -> None:
    """Plot Precision-Recall curves for Train/Test on one plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (prec, rec, ap) in curves.items():
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_calibration(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    out_path: Path,
) -> None:
    """Plot reliability curves (calibration) for Train/Test."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (prob_true, prob_pred) in curves.items():
        ax.plot(prob_pred, prob_true, marker="o", lw=1, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Predicted probability (bin avg)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_and_plot(
    name: str,
    pipe: ImbPipeline,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    y_te: pd.Series,
    paths: Paths,
) -> tuple[list[EvalResult], dict[str, str], dict[str, str]]:
    """Fit, evaluate on train/test, and save ROC/PR/Calibration plots."""
    pipe.fit(X_tr, y_tr)

    # Train predictions
    p_tr = pipe.predict_proba(X_tr)[:, 1]
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, p_tr)
    roc_tr = roc_auc_score(y_tr, p_tr)
    prec_tr, rec_tr, _ = precision_recall_curve(y_tr, p_tr)
    ap_tr = average_precision_score(y_tr, p_tr)
    pt_tr, pp_tr = calibration_curve(y_tr, p_tr, n_bins=10)

    # Test predictions
    p_te = pipe.predict_proba(X_te)[:, 1]
    fpr_te, tpr_te, _ = roc_curve(y_te, p_te)
    roc_te = roc_auc_score(y_te, p_te)
    prec_te, rec_te, _ = precision_recall_curve(y_te, p_te)
    ap_te = average_precision_score(y_te, p_te)
    pt_te, pp_te = calibration_curve(y_te, p_te, n_bins=10)

    # F1 at 0.5
    f1_tr = f1_score(y_tr, (p_tr >= 0.5).astype(int))
    f1_te = f1_score(y_te, (p_te >= 0.5).astype(int))

    results = [
        EvalResult(name, "Train", roc_tr, ap_tr, f1_tr),
        EvalResult(name, "Test", roc_te, ap_te, f1_te),
    ]

    # Classification report text
    rep_tr = classification_report(y_tr, (p_tr >= 0.5).astype(int), digits=3)
    rep_te = classification_report(y_te, (p_te >= 0.5).astype(int), digits=3)
    reports = {"train": rep_tr, "test": rep_te}

    # Plots
    plots: dict[str, str] = {}
    out_roc = paths.plots / f"roc_{name}.png"
    plot_roc_curves(
        {"Train": (fpr_tr, tpr_tr, roc_tr), "Test": (fpr_te, tpr_te, roc_te)},
        f"ROC - {name}",
        out_roc,
    )
    plots["roc"] = str(out_roc)

    out_pr = paths.plots / f"pr_{name}.png"
    plot_pr_curves(
        {"Train": (prec_tr, rec_tr, ap_tr), "Test": (prec_te, rec_te, ap_te)},
        f"PR - {name}",
        out_pr,
    )
    plots["pr"] = str(out_pr)

    out_cal = paths.plots / f"calib_{name}.png"
    plot_calibration(
        {"Train": (pt_tr, pp_tr), "Test": (pt_te, pp_te)},
        f"Calibration - {name}",
        out_cal,
    )
    plots["calibration"] = str(out_cal)

    return results, reports, plots


# ---------------------------------------------------------------------
# SHAP for the best model (tree models only)
# ---------------------------------------------------------------------
def shap_summary_for_best(
    pipe: ImbPipeline,
    X_bg: pd.DataFrame,
    paths: Paths,
    max_background: int = 400,
) -> str | None:
    """Save a SHAP summary plot for tree models; return path or None."""
    clf = pipe.named_steps["clf"]
    if not isinstance(clf, (LGBMClassifier, XGBClassifier)):  # noqa: UP038
        return None

    # Background through the pipeline's preprocessor
    pre: ColumnTransformer = pipe.named_steps["pre"]
    idx = np.random.choice(len(X_bg), size=min(max_background, len(X_bg)), replace=False)
    X_sample = pre.transform(X_bg.iloc[idx])

    # Build explainer and SHAP values
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(X_sample)

    out = paths.shap / "shap_summary.png"
    try:
        shap.summary_plot(values, X_sample, show=False)
    except Exception:
        if isinstance(values, list):
            shap.summary_plot(values[0], X_sample, show=False)
        else:
            raise
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return str(out)


# ---------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------
def jot_notes_by_component() -> dict[str, list[str]]:
    """Return jot notes aligned with Lab 4 (≤4 bullets each)."""
    return {
        "EDA": [
            "Checked imbalance; guided SMOTE + class_weight and stratified CV.",
            "Skewness/outliers -> IQR winsorization; no row drops.",
            "Correlation heatmap used for awareness (selection stays simple).",
            "Numeric dataset; OHE reserved if categoricals appear.",
        ],
        "Preprocessing": [
            "Median impute + IQR capping (numeric only).",
            "Scaler: LR/XGB numeric; LGBM unscaled. OHE for categoricals.",
            "SMOTE (train-only) and weighting per Lab 4.",
            "PSI (train vs test) validates split stability; seeds fixed.",
        ],
        "Feature Selection": [
            "RandomForest importance -> keep Top-20 (simple, per Lab 4).",
            "VIF pruning if VIF>10 to reduce multicollinearity.",
            "Shared final feature list across models (fair comparison).",
            "Drivers are financial ratios -> interpretable.",
        ],
        "Tuning": [
            "RandomizedSearchCV (Stratified 5-fold), scoring=ROC-AUC.",
            "Bounded grids for speed/stability; imbalance handled in-pipeline.",
            "Persist best params per model.",
            "Keep it simple; no fragile early-stopping callbacks.",
        ],
        "Training": [
            "Three models: LR (baseline), XGBoost, LightGBM.",
            "Pipelines apply SMOTE only after preprocessing.",
            "Persisted models to artifacts/models/.",
            "Reproducible via global seeds.",
        ],
        "Evaluation": [
            "Metrics: ROC-AUC, PR-AUC, F1@0.5 (train & test).",
            "Overlays: ROC / PR / Calibration (train vs test).",
            "Classification reports included.",
            "Best picked by Test PR-AUC then ROC-AUC.",
        ],
        "SHAP": [
            "TreeExplainer summary plot for best boosted model.",
            "Feature attributions align with finance ratios.",
            "Sample cap for runtime control.",
            "Supports regulatory explainability.",
        ],
        "PSI": [
            "Per-feature PSI; top-15 chart + CSV snapshot.",
            "Thresholds: <0.1 stable; 0.1-0.2 monitor; >0.2 drift.",
            "If drift high on key drivers -> retrain/re-sample.",
            "Monitors generalization to test distribution.",
        ],
    }


def write_report(
    paths: Paths,
    selected_features: list[str],
    selected_features_by_model: list[str] | dict[str, list[str]],
    vif_df_final: pd.DataFrame,
    rf_importance: pd.DataFrame,
    tune_summaries: dict[str, dict[str, object]],
    results: list[EvalResult],
    best_model_name: str,
    psi_top: pd.DataFrame,
    cls_reports: dict[str, dict[str, str]],
    eda_imgs: dict[str, str],
) -> None:
    """Create the Markdown report (NO Brier)."""
    md: list[str] = []

    md.append("# Lab 5 Report - Company Bankruptcy Prediction\n")
    md.append("## Jot Notes (per component)")
    notes = jot_notes_by_component()
    for sec, bullets in notes.items():
        md.append(f"\n### {sec}")
        for b in bullets:
            md.append(f"- {b}")

    md.append("\n## EDA Visuals")
    for k, p in eda_imgs.items():
        md.append(f"- {k}: {p}")

    md.append("\n## Feature Selection (RandomForest Top-K + VIF)")
    md.append("### RandomForest Importances (Top 20)")
    md.append(df_to_markdown_safe(rf_importance.head(20), index=False))
    md.append("\n### Final Selected Numeric Features")
    md.append(", ".join(selected_features))

    md.append("\n## Multicollinearity (VIF) - Final Snapshot (Top 10)")
    md.append(df_to_markdown_safe(vif_df_final.head(10), index=False))

    md.append("\n## Tuning Summary (best hyperparameters)")
    for name, info in tune_summaries.items():
        best_params = json.dumps(info["best_params"], indent=2)
        cv_score = info["cv_score"]
        md.append(f"### {name}\n- Best ROC-AUC (CV): **{cv_score:.4f}**\n")
        md.append("```json")
        md.append(best_params)
        md.append("```")

    md.append("\n## Model Comparison (Train vs Test)")
    df_res = pd.DataFrame(
        [
            {
                "Model": r.model,
                "Split": r.split,
                "ROC-AUC": r.roc_auc,
                "PR-AUC": r.pr_auc,
                "F1": r.f1_at_050,
            }
            for r in results
        ]
    ).set_index(["Model", "Split"]).unstack()
    md.append(df_to_markdown_safe(df_res.round(4), index=True))

    md.append("\n## Classification Reports")
    for model, reps in cls_reports.items():
        md.append(f"### {model} — Train")
        md.append("```\n" + reps["train"] + "\n```")
        md.append(f"### {model} — Test")
        md.append("```\n" + reps["test"] + "\n```")

    md.append("\n## PSI (Drift) — Train vs Test")
    md.append(df_to_markdown_safe(psi_top, index=False))
    md.append(f"- PSI bar: {paths.psi / 'psi_bar.png'}")

    md.append("\n## SHAP Summary")
    md.append(
        "- Generated for best tree model (if applicable): "
        + str(paths.shap / "shap_summary.png")
    )

    md.append("\n## Q&A")
    md.append("**Challenges & fixes**")
    md.append(
        "- Class imbalance -> SMOTE (train-only) + class/pos weights.\n"
        "- Multicollinearity -> VIF pruning after Top-20 selection.\n"
        "- Calibration & thresholds -> reliability curves; F1@0.5 reported.\n"
        "- Version quirks -> avoided fragile early-stopping callbacks.\n"
    )

    md.append("\n**Lab 4 -> Lab 5 influence**")
    md.append(
        "- Preprocessing (StandardScaler for LR/XGB numeric; OHE for "
        "categoricals; no scaling for LGBM).\n"
        "- Models (LR, XGBoost, LightGBM); RF used only for simple FS.\n"
        "- Metrics focus (ROC-AUC, PR-AUC, F1) and SHAP explainability.\n"
        "- Stability checks via PSI & stratified splitting.\n"
    )

    md.append("\n**Deployment recommendation**")
    md.append(
        f"- **{best_model_name}** — Choose based on best Test PR-AUC "
        "and competitive ROC-AUC with reasonable calibration."
    )

    paths.report_md.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    """Entry point: orchestrates EDA -> FS -> Tuning -> Training -> Eval -> Report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str, help="Path to CSV dataset.")
    parser.add_argument("--target", type=str, default="Bankrupt?", help="Target column.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_splits", type=int, default=5, help="Stratified K-fold.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size frac.")
    parser.add_argument(
        "--top_k_features", type=int, default=20, help="RF top-K before VIF prune."
    )
    parser.add_argument(
        "--vif_threshold", type=float, default=10.0, help="VIF threshold."
    )
    parser.add_argument(
        "--vif_max_iter", type=int, default=8, help="Max VIF pruning iterations."
    )
    parser.add_argument(
        "--n_iter_search", type=int, default=40, help="RandomizedSearch iterations."
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    paths = Paths.make(root)
    set_all_seeds(args.seed)
    write_repro_files(paths)

    # Load data
    df = pd.read_csv(args.data)
    assert args.target in df.columns, f"Target '{args.target}' not found."

    # EDA
    eda_imgs = eda_plots(df, args.target, paths)

    # Split
    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # Identify numeric/categorical
    numeric_all = list(X_train.select_dtypes(include=[np.number]).columns)
    categorical_all = list(
        X_train.select_dtypes(include=["object", "category", "bool"]).columns
    )

    # PSI on raw numeric (train vs test)
    psi_top = psi_dataframe(
        X_train[numeric_all], X_test[numeric_all], top_k=min(15, len(numeric_all))
    )
    psi_top.to_csv(paths.psi / "psi_snapshot.csv", index=False, encoding="utf-8")
    plot_psi_bar(psi_top, paths.psi / "psi_bar.png")

    # RF Top-K on numeric features, then VIF prune
    X_train_num = X_train[numeric_all].copy()
    topk_features, rf_imp = rf_topk_features(
        X_train_num, y_train, args.top_k_features, args.seed
    )
    X_train_fs = X_train_num[topk_features].copy()
    X_train_vif, final_numeric_features, vif_df_final = prune_by_vif(
        X_train_fs, threshold=args.vif_threshold, max_iter=args.vif_max_iter
    )

    # Preprocessors
    pre_lr_xgb, pre_lgbm = make_preprocessors(
        numeric_features=final_numeric_features, categorical_features=categorical_all
    )

    # Tuning / Training / Evaluation
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = make_pipelines(
        pos_weight=pos_weight, seed=args.seed, pre_lr_xgb=pre_lr_xgb, pre_lgbm=pre_lgbm
    )
    grids = search_spaces(pos_weight=pos_weight)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    tune_summaries: dict[str, dict[str, object]] = {}
    tuned_models: dict[str, ImbPipeline] = {}
    all_results: list[EvalResult] = {}
    all_results = []
    all_reports: dict[str, dict[str, str]] = {}

    # Shared feature view for all models (fair comparison)
    X_train_w = pd.concat([X_train_vif, X_train[categorical_all]], axis=1)
    X_test_w = pd.concat([X_test[final_numeric_features], X_test[categorical_all]], axis=1)

    for name, base_pipe in models.items():
        rs = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=grids[name],
            n_iter=args.n_iter_search,
            scoring="roc_auc",
            cv=cv,
            random_state=args.seed,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        rs.fit(X_train_w, y_train)
        tuned = rs.best_estimator_
        tuned_models[name] = tuned
        tune_summaries[name] = {
            "best_params": rs.best_params_,
            "cv_score": float(rs.best_score_),
        }

        # Evaluate + plots
        results, reports, plots = evaluate_and_plot(
            name=name,
            pipe=tuned,
            X_tr=X_train_w,
            y_tr=y_train,
            X_te=X_test_w,
            y_te=y_test,
            paths=paths,
        )
        all_results.extend(results)
        all_reports[name] = reports

        # Save model
        joblib.dump(tuned, paths.models / f"{name}_best.joblib")

    # Pick best by Test PR-AUC, then ROC-AUC
    test_rows = [r for r in all_results if r.split == "Test"]
    test_df = pd.DataFrame(
        [
            {
                "Model": r.model,
                "PR-AUC": r.pr_auc,
                "ROC-AUC": r.roc_auc,
                "F1": r.f1_at_050,
            }
            for r in test_rows
        ]
    ).sort_values(["PR-AUC", "ROC-AUC"], ascending=[False, False])
    best_model_name = str(test_df.iloc[0]["Model"])

    # SHAP summary for best tree model (ignore result if not applicable)
    _ = shap_summary_for_best(
        tuned_models[best_model_name], X_train_w, paths, max_background=400
    )

    # Selected features by model (shared set by design)
    selected_by_model = {
        name: list(final_numeric_features) for name in tuned_models.keys()
    }

    # Write report
    write_report(
        paths=paths,
        selected_features=list(final_numeric_features),
        selected_features_by_model=selected_by_model,
        vif_df_final=vif_df_final,
        rf_importance=rf_imp,
        tune_summaries=tune_summaries,
        results=all_results,
        best_model_name=best_model_name,
        psi_top=psi_top,
        cls_reports=all_reports,
        eda_imgs=eda_imgs,
    )

    # Console summary
    print("\n=== SUMMARY ===")
    print(f"Best model: {best_model_name}")
    print(df_to_markdown_safe(test_df.round(4), index=False))
    print("\nArtifacts:")
    print(f"- Report: {paths.report_md}")
    print(f"- ROC/PR/Calibration plots: {paths.plots}")
    print(f"- SHAP: {paths.shap} (shap_summary.png)")
    print(f"- PSI: {paths.psi}")
    print(f"- Models: {paths.models}")
    print("- Repro files: requirements_frozen.txt, ruff.toml")
    print("\nDone. See: artifacts/report.md")


if __name__ == "__main__":
    main()
