"""
regression_single.py  (cleaned)

ZIP-level model. One row per ZIP from NewData/CleanedDataJune.csv.
Predicts log(Median_Price_June) from external area-level features.

Major changes from the previous version:
    1. No longer manufactures collinear features (z-scored duplicates of raw
       counts, composite indices that are linear combinations of columns
       already in the model, definitional negations).
    2. Adds a LANDMARK_STRATEGY flag: 'centrality_only' (parsimonious),
       'all_distances' (information-rich, requires regularization),
       or 'pca' (compromise via top principal components).
    3. Detects whether Count_Fire_Stations and Count_Police_Stations are
       identical or near-identical and warns you, since their OLS
       coefficients in the previous run suggested upstream duplication.
    4. Adds Ridge regression alongside OLS, with cross-validated alpha.
    5. Replaces the single 75/25 split with 5-fold cross-validation
       for Ridge and GBM. Reports mean and std across folds.

Run:
    python regression_single.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
NEW_DATA_DIR = BASE_DIR / "NewData"
OUTPUT_DIR = BASE_DIR / "model_outputs_adjusted"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_FILE = NEW_DATA_DIR / "CleanedDataJune.csv"

TARGET_COL = "Median_Price_June"

RANDOM_STATE = 42
N_SPLITS = 5  # k for k-fold CV

# How to handle the 17 individual landmark distance columns.
#   "centrality_only" : drop all individual distances, keep centrality_index
#   "all_distances"   : keep all individual distances, drop centrality_index
#   "pca"             : replace all distances with top N principal components
# LANDMARK_STRATEGY = "centrality_only"
LANDMARK_STRATEGY = "all_distances"
PCA_N_COMPONENTS = 3

# Set to True only if you want to include Airbnb supply/density as predictors.
INCLUDE_AIRBNB_SUPPLY_FEATURES = False


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def standardize_zip(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.where((s >= 10000) & (s <= 99999))
    return s.astype("Int64")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def plot_actual_vs_predicted(y_true, y_pred, title, outpath):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual ZIP median price")
    plt.ylabel("Predicted ZIP median price")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, outpath: Path, top_n: int = 20):
    if importance_df.empty:
        return
    d = importance_df.sort_values("importance_mean", ascending=False).head(top_n)
    d = d.iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(d["feature"], d["importance_mean"])
    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.title("ZIP-only model feature importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ============================================================
# DATA LOADING + FEATURE ENGINEERING (cleaned)
# ============================================================

def load_zip_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_FILE}. Expected NewData/CleanedDataJune.csv"
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]

    if "ZIP_Code" in df.columns:
        df = df.rename(columns={"ZIP_Code": "ZIP_CODE"})
    elif "ZIP_CODE" not in df.columns:
        raise ValueError("CleanedDataJune.csv must contain ZIP_Code or ZIP_CODE.")

    df["ZIP_CODE"] = standardize_zip(df["ZIP_CODE"])
    for col in df.columns:
        if col != "ZIP_CODE":
            df[col] = safe_numeric(df[col])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} was not found.")

    raw_rows = len(df)
    df = df.dropna(subset=["ZIP_CODE", TARGET_COL]).copy()
    df = df[(df[TARGET_COL] > 0) & (df[TARGET_COL] < 5000)].copy()
    df["log_median_price_june"] = np.log(df[TARGET_COL])

    # ---- Sanity check: are Fire and Police stations literally duplicates? ----
    fire_police_warning = None
    if {"Count_Fire_Stations", "Count_Police_Stations"}.issubset(df.columns):
        a = df["Count_Fire_Stations"].astype(float)
        b = df["Count_Police_Stations"].astype(float)
        if a.equals(b):
            fire_police_warning = (
                "Count_Fire_Stations and Count_Police_Stations are IDENTICAL columns. "
                "This is almost certainly a data error upstream. Dropping Count_Police_Stations."
            )
            df = df.drop(columns=["Count_Police_Stations"])
        else:
            r = a.corr(b)
            if pd.notna(r) and abs(r) > 0.99:
                fire_police_warning = (
                    f"Count_Fire_Stations and Count_Police_Stations have correlation {r:.4f}. "
                    "Near-perfect collinearity. Dropping Count_Police_Stations to be safe."
                )
                df = df.drop(columns=["Count_Police_Stations"])
    if fire_police_warning:
        print("WARNING:", fire_police_warning)

    # ---- Useful transformations (keep) ----
    if "Population" in df.columns:
        df["log_population"] = np.log1p(df["Population"].clip(lower=0))

    if "Part_1_Crime_Count" in df.columns:
        df["log_part_1_crime_count"] = np.log1p(df["Part_1_Crime_Count"].clip(lower=0))

    if "Part_1_Crime_Per_Capita" in df.columns:
        df["crime_per_10k"] = df["Part_1_Crime_Per_Capita"] * 10000

    # ---- Centrality index (kept as the lone landmark summary) ----
    distance_cols = [
        c for c in df.columns
        if ("Distance" in c or "distance" in c)
        and c != "ZIP_CODE"
        and "Sept" not in c
    ]
    if distance_cols:
        # Use raw means/stds rather than producing both a z-score and its negation.
        dist_z = pd.DataFrame({
            c: (df[c] - df[c].mean()) / df[c].std() for c in distance_cols
        })
        df["centrality_index"] = -dist_z.mean(axis=1)

    # ---- PCA on distances (only if requested) ----
    pca_components = []
    if LANDMARK_STRATEGY == "pca" and distance_cols:
        clean = df[distance_cols].fillna(df[distance_cols].median(numeric_only=True))
        scaler = StandardScaler()
        clean_s = scaler.fit_transform(clean)
        n_comp = min(PCA_N_COMPONENTS, clean_s.shape[1])
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        pcs = pca.fit_transform(clean_s)
        for i in range(n_comp):
            col = f"distance_pc_{i + 1}"
            df[col] = pcs[:, i]
            pca_components.append(col)
        print(f"PCA explained variance ratios: {pca.explained_variance_ratio_.round(4)}")

    diagnostics = {
        "raw_rows": raw_rows,
        "rows_after_target_cleanup": len(df),
        "unique_zips": int(df["ZIP_CODE"].nunique()),
        "target": TARGET_COL,
        "distance_columns_found": len(distance_cols),
        "landmark_strategy": LANDMARK_STRATEGY,
        "fire_police_warning": fire_police_warning,
    }

    return df, diagnostics, distance_cols, pca_components


def get_predictor_columns(df: pd.DataFrame, distance_cols, pca_components):
    """
    Cleaned predictor list. No z-scored duplicates, no composite indices
    that are linear combinations of columns already in the model.
    """
    base = [
        "log_population",
        "log_part_1_crime_count",
        "crime_per_10k",
        "Count_Educational_Facility",
        "Count_Public_Safety",
        "Count_Fire_Stations",
        "Count_Police_Stations",  # may have been dropped above; filtered below
        "Count_Corrective_Facility",
    ]

    if LANDMARK_STRATEGY == "centrality_only":
        landmark = ["centrality_index"]
    elif LANDMARK_STRATEGY == "all_distances":
        landmark = list(distance_cols)
    elif LANDMARK_STRATEGY == "pca":
        landmark = list(pca_components)
    else:
        raise ValueError(f"Unknown LANDMARK_STRATEGY: {LANDMARK_STRATEGY}")

    supply = []
    if INCLUDE_AIRBNB_SUPPLY_FEATURES:
        supply = [c for c in df.columns if "Airbnb_Listings" in c and "June" in c]

    predictors = []
    for c in base + landmark + supply:
        if c in df.columns and c not in predictors:
            predictors.append(c)

    # Hard exclusions against target leakage
    predictors = [
        c for c in predictors
        if not c.startswith("Median_Price")
        and c != TARGET_COL
        and c != "log_median_price_june"
        and c != "ZIP_CODE"
    ]
    return predictors


# ============================================================
# MODELS
# ============================================================

def prepare_X_y(df, predictors):
    model_df = df[["ZIP_CODE", TARGET_COL, "log_median_price_june"] + predictors].dropna().copy()
    X = model_df[predictors].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.dropna(axis=1, how="all").fillna(X.median(numeric_only=True))
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print("Dropping constant columns:", constant_cols)
        X = X.drop(columns=constant_cols)
    y = model_df["log_median_price_june"]
    return model_df, X, y


def run_ols(df, predictors):
    model_df, X, y = prepare_X_y(df, predictors)
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const).fit(cov_type="HC3")
    model_df["pred_log_price_ols"] = model.predict(X_const)
    model_df["pred_price_ols"] = np.exp(model_df["pred_log_price_ols"])
    formula = "log_median_price_june ~ " + " + ".join(X.columns)
    return model, model_df, formula


def run_ridge_cv(df, predictors):
    model_df, X, y = prepare_X_y(df, predictors)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25), cv=N_SPLITS)),
    ])
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    pipe.fit(X, y)

    pred_log = pipe.predict(X)
    metrics = {
        "ridge_cv_r2_log_mean": float(cv_r2.mean()),
        "ridge_cv_r2_log_std": float(cv_r2.std()),
        "ridge_chosen_alpha": float(pipe.named_steps["ridge"].alpha_),
        "ridge_full_fit_r2_log": float(r2_score(y, pred_log)),
        "ridge_full_fit_mae_price": float(mean_absolute_error(np.exp(y), np.exp(pred_log))),
        "ridge_full_fit_rmse_price": rmse(np.exp(y), np.exp(pred_log)),
    }

    coefs = pd.DataFrame({
        "feature": X.columns,
        "ridge_coef_standardized": pipe.named_steps["ridge"].coef_,
    }).sort_values("ridge_coef_standardized", key=np.abs, ascending=False)

    model_df["pred_log_price_ridge"] = pred_log
    model_df["pred_price_ridge"] = np.exp(pred_log)
    return pipe, model_df, metrics, coefs


def run_gbm_cv(df, predictors):
    model_df, X, y = prepare_X_y(df, predictors)

    gbm = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=10,
        min_samples_leaf=8,
        l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", gbm),
    ])

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    pipe.fit(X, y)
    pred_log = pipe.predict(X)
    metrics = {
        "gbm_cv_r2_log_mean": float(cv_r2.mean()),
        "gbm_cv_r2_log_std": float(cv_r2.std()),
        "gbm_full_fit_r2_log": float(r2_score(y, pred_log)),
        "gbm_full_fit_mae_price": float(mean_absolute_error(np.exp(y), np.exp(pred_log))),
        "gbm_full_fit_rmse_price": rmse(np.exp(y), np.exp(pred_log)),
    }

    try:
        result = permutation_importance(
            pipe, X, y, n_repeats=10, random_state=RANDOM_STATE, scoring="r2"
        )
        importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)
    except Exception:
        importance_df = pd.DataFrame({"feature": X.columns,
                                      "importance_mean": np.nan,
                                      "importance_std": np.nan})

    model_df["pred_log_price_gbm"] = pred_log
    model_df["pred_price_gbm"] = np.exp(pred_log)
    return pipe, model_df, metrics, importance_df


# ============================================================
# OUTPUTS
# ============================================================

def save_outputs(df, diagnostics, predictors,
                 ols_model, ols_df, ols_formula,
                 ridge_df, ridge_metrics, ridge_coefs,
                 gbm_df, gbm_metrics, importance_df):

    # Datasets
    df.to_csv(OUTPUT_DIR / "single_model_dataset.csv", index=False)
    ols_df.to_csv(OUTPUT_DIR / "single_ols_predictions.csv", index=False)
    ridge_df.to_csv(OUTPUT_DIR / "single_ridge_predictions.csv", index=False)
    gbm_df.to_csv(OUTPUT_DIR / "single_gbm_predictions.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "single_gbm_feature_importance.csv", index=False)
    ridge_coefs.to_csv(OUTPUT_DIR / "single_ridge_coefficients.csv", index=False)

    # OLS summary + coefficients
    write_text(OUTPUT_DIR / "single_ols_summary.txt", str(ols_model.summary()))
    pd.DataFrame({
        "term": ols_model.params.index,
        "coef": ols_model.params.values,
        "std_err": ols_model.bse.values,
        "z_value": ols_model.tvalues.values,
        "p_value": ols_model.pvalues.values,
    }).to_csv(OUTPUT_DIR / "single_ols_coefficients.csv", index=False)

    ols_metrics = {
        "ols_r2_log": r2_score(ols_df["log_median_price_june"], ols_df["pred_log_price_ols"]),
        "ols_mae_price": mean_absolute_error(ols_df[TARGET_COL], ols_df["pred_price_ols"]),
        "ols_rmse_price": rmse(ols_df[TARGET_COL], ols_df["pred_price_ols"]),
        "ols_rows": len(ols_df),
    }

    # Plots
    plot_actual_vs_predicted(
        ols_df[TARGET_COL].values, ols_df["pred_price_ols"].values,
        "ZIP-only OLS: Actual vs Predicted Median Price",
        OUTPUT_DIR / "single_actual_vs_predicted_ols.png",
    )
    plot_actual_vs_predicted(
        ridge_df[TARGET_COL].values, ridge_df["pred_price_ridge"].values,
        "ZIP-only Ridge: Actual vs Predicted Median Price",
        OUTPUT_DIR / "single_actual_vs_predicted_ridge.png",
    )
    plot_actual_vs_predicted(
        gbm_df[TARGET_COL].values, gbm_df["pred_price_gbm"].values,
        "ZIP-only GBM: Actual vs Predicted Median Price",
        OUTPUT_DIR / "single_actual_vs_predicted_gbm.png",
    )
    plot_feature_importance(importance_df, OUTPUT_DIR / "single_gbm_feature_importance.png")

    # Diagnostics report
    lines = [
        "ZIP-only model diagnostics (cleaned)",
        "=" * 70,
        "",
        "Data diagnostics:",
        *[f"{k}: {v}" for k, v in diagnostics.items()],
        "",
        "OLS formula:",
        ols_formula,
        "",
        "OLS metrics (in-sample, log scale):",
        *[f"{k}: {v}" for k, v in ols_metrics.items()],
        "",
        f"Ridge metrics ({N_SPLITS}-fold CV, log scale):",
        *[f"{k}: {v}" for k, v in ridge_metrics.items()],
        "",
        f"GBM metrics ({N_SPLITS}-fold CV, log scale):",
        *[f"{k}: {v}" for k, v in gbm_metrics.items()],
        "",
        "Predictors included:",
        *[f"- {p}" for p in predictors],
        "",
        "Notes:",
        "- Median price columns excluded from predictors to avoid target leakage.",
        "- Airbnb supply variables excluded unless INCLUDE_AIRBNB_SUPPLY_FEATURES=True.",
        "- Cross-validation uses 5 folds with shuffled splits.",
        "- For Ridge, the chosen alpha is selected via internal CV across the same folds.",
    ]
    write_text(OUTPUT_DIR / "single_model_diagnostics.txt", "\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting cleaned ZIP-only model...")

    df, diagnostics, distance_cols, pca_components = load_zip_data()
    predictors = get_predictor_columns(df, distance_cols, pca_components)

    if not predictors:
        raise ValueError("No usable external predictors were found.")

    print(f"Rows: {len(df):,}")
    print(f"Predictors ({len(predictors)}):")
    for p in predictors:
        print(f"  - {p}")

    ols_model, ols_df, ols_formula = run_ols(df, predictors)
    print(f"\nOLS R^2 (in-sample, log): {ols_model.rsquared:.4f}")
    print(f"OLS condition number:     {ols_model.condition_number:.2e}")

    ridge_pipe, ridge_df, ridge_metrics, ridge_coefs = run_ridge_cv(df, predictors)
    print(f"\nRidge CV R^2 (log):       "
          f"{ridge_metrics['ridge_cv_r2_log_mean']:.4f} "
          f"+/- {ridge_metrics['ridge_cv_r2_log_std']:.4f}")
    print(f"Ridge chosen alpha:       {ridge_metrics['ridge_chosen_alpha']:.4f}")

    gbm_pipe, gbm_df, gbm_metrics, importance_df = run_gbm_cv(df, predictors)
    print(f"\nGBM CV R^2 (log):         "
          f"{gbm_metrics['gbm_cv_r2_log_mean']:.4f} "
          f"+/- {gbm_metrics['gbm_cv_r2_log_std']:.4f}")

    save_outputs(
        df=df, diagnostics=diagnostics, predictors=predictors,
        ols_model=ols_model, ols_df=ols_df, ols_formula=ols_formula,
        ridge_df=ridge_df, ridge_metrics=ridge_metrics, ridge_coefs=ridge_coefs,
        gbm_df=gbm_df, gbm_metrics=gbm_metrics, importance_df=importance_df,
    )

    print(f"\nDone. Outputs in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
