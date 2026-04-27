"""
regression_zip_only.py

Purpose:
    ZIP-level model only.

    This model uses only one row per ZIP code from:
        NewData/CleanedDataJune.csv

    It predicts:
        Median_Price_June

    using only external / area-level variables:
        crime, safety, education, population, and distances to landmarks.

    ZIP_Code is used only as an identifier. It is NOT used as a numeric predictor.

Expected project layout:
    AirBNBRegression/
        regression_zip_only.py
        NewData/
            CleanedDataJune.csv
        model_outputs_zip_only/   <-- created automatically

Run:
    python regression_zip_only.py

Install:
    pip install pandas numpy matplotlib scikit-learn statsmodels
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
NEW_DATA_DIR = BASE_DIR / "NewData"
OUTPUT_DIR = BASE_DIR / "model_outputs_single"
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_FILE = NEW_DATA_DIR / "CleanedDataJune.csv"

TARGET_COL = "Median_Price_June"

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Set to True if you want to include Airbnb supply/density as predictors.
# For a purer "external factors only" model, this should stay False.
INCLUDE_AIRBNB_SUPPLY_FEATURES = False


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def standardize_zip(series: pd.Series) -> pd.Series:
    """
    ZIP is only an identifier / join key, never a numeric predictor.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.where((s >= 10000) & (s <= 99999))
    return s.astype("Int64")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


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
# DATA LOADING + FEATURE ENGINEERING
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

    # Basic target cleanup
    df = df[(df[TARGET_COL] > 0) & (df[TARGET_COL] < 5000)].copy()
    df["log_median_price_june"] = np.log(df[TARGET_COL])

    # Safer transformed/scaled versions of external columns
    if "Population" in df.columns:
        df["log_population"] = np.log1p(df["Population"].clip(lower=0))

    if "Part_1_Crime_Count" in df.columns:
        df["log_part_1_crime_count"] = np.log1p(df["Part_1_Crime_Count"].clip(lower=0))

    if "Part_1_Crime_Per_Capita" in df.columns:
        df["crime_per_10k"] = df["Part_1_Crime_Per_Capita"] * 10000
        df["crime_z"] = zscore(df["Part_1_Crime_Per_Capita"])

    if "Count_Educational_Facility" in df.columns:
        df["education_facilities_z"] = zscore(df["Count_Educational_Facility"])

    if "Count_Public_Safety" in df.columns:
        df["public_safety_z"] = zscore(df["Count_Public_Safety"])

    # Composite amenities / safety index from external facility counts
    amenity_parts = []
    for c in [
        "Count_Educational_Facility",
        "Count_Public_Safety",
        "Count_Fire_Stations",
        "Count_Police_Stations",
    ]:
        if c in df.columns:
            amenity_parts.append(zscore(df[c]))

    if amenity_parts:
        df["external_amenities_index"] = sum(amenity_parts) / len(amenity_parts)

    # Distances to attractions / landmarks
    distance_cols = [
        c for c in df.columns
        if ("Distance" in c or "distance" in c) and c != "ZIP_CODE"
    ]

    if distance_cols:
        # Smaller distance means closer to major destinations.
        # This index is the average z-scored distance to listed landmarks.
        distance_zs = [zscore(df[c]) for c in distance_cols]
        df["avg_landmark_distance_z"] = sum(distance_zs) / len(distance_zs)
        df["centrality_index"] = -df["avg_landmark_distance_z"]

    diagnostics = {
        "raw_rows": raw_rows,
        "rows_after_target_cleanup": len(df),
        "unique_zips": int(df["ZIP_CODE"].nunique()),
        "target": TARGET_COL,
        "distance_columns_found": len(distance_cols),
    }

    return df, diagnostics


def get_predictor_columns(df: pd.DataFrame):
    """
    Only external / ZIP-level variables.
    Excludes ZIP_CODE and any median price columns to avoid target leakage.
    """

    explicit_external = [
        "log_population",
        "log_part_1_crime_count",
        "crime_per_10k",
        "crime_z",
        "Count_Educational_Facility",
        "education_facilities_z",
        "Count_Public_Safety",
        "public_safety_z",
        "Count_Fire_Stations",
        "Count_Police_Stations",
        "Count_Corrective_Facility",
        "external_amenities_index",
        "avg_landmark_distance_z",
        "centrality_index",
    ]

    distance_cols = [
        c for c in df.columns
        if ("Distance" in c or "distance" in c)
        and c != "ZIP_CODE"
        and "Sept" not in c  # June-only model should not use September-specific distance variables by default
    ]

    supply_cols = []
    if INCLUDE_AIRBNB_SUPPLY_FEATURES:
        supply_cols = [
            c for c in df.columns
            if "Airbnb_Listings" in c and "June" in c
        ]

    predictors = []
    for c in explicit_external + distance_cols + supply_cols:
        if c in df.columns and c not in predictors:
            predictors.append(c)

    # Hard exclusions against leakage
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

def run_ols(df: pd.DataFrame, predictors):
    """
    Formula-free OLS.

    This avoids Patsy/statsmodels formula parsing problems caused by column names
    containing parentheses, spaces, slashes, or other special characters.
    """

    model_df = df[["ZIP_CODE", TARGET_COL, "log_median_price_june"] + predictors].dropna().copy()

    y = model_df["log_median_price_june"]

    X = model_df[predictors].copy()

    # Force all predictors to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Drop columns that are entirely missing
    X = X.dropna(axis=1, how="all")

    # Fill remaining missing values with medians
    X = X.fillna(X.median(numeric_only=True))

    # Drop constant columns, because they do not help the model
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        print("Dropping constant OLS columns:")
        for c in constant_cols:
            print(f"  - {c}")
        X = X.drop(columns=constant_cols)

    # Add intercept
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(y, X).fit(cov_type="HC3")

    model_df["pred_log_price_ols"] = model.predict(X)
    model_df["pred_price_ols"] = np.exp(model_df["pred_log_price_ols"])

    formula_description = "Formula-free OLS: log_median_price_june ~ " + " + ".join(
        [c for c in X.columns if c != "const"]
    )

    return model, model_df, formula_description

def run_gbm(df: pd.DataFrame, predictors):
    model_df = df[["ZIP_CODE", TARGET_COL, "log_median_price_june"] + predictors].dropna().copy()

    X = model_df[predictors]
    y = model_df["log_median_price_june"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        model_df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, predictors),
        ],
        remainder="drop",
    )

    # Conservative model because there are only ~250 ZIPs.
    gbm = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=300,
        max_leaf_nodes=10,
        min_samples_leaf=8,
        l2_regularization=2.0,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", gbm),
    ])

    pipe.fit(X_train, y_train)

    pred_train_log = pipe.predict(X_train)
    pred_test_log = pipe.predict(X_test)

    pred_train_price = np.exp(pred_train_log)
    pred_test_price = np.exp(pred_test_log)

    y_train_price = np.exp(y_train)
    y_test_price = np.exp(y_test)

    metrics = {
        "gbm_train_r2_log": r2_score(y_train, pred_train_log),
        "gbm_test_r2_log": r2_score(y_test, pred_test_log),
        "gbm_train_mae_price": mean_absolute_error(y_train_price, pred_train_price),
        "gbm_test_mae_price": mean_absolute_error(y_test_price, pred_test_price),
        "gbm_train_rmse_price": rmse(y_train_price, pred_train_price),
        "gbm_test_rmse_price": rmse(y_test_price, pred_test_price),
        "gbm_train_rows": len(X_train),
        "gbm_test_rows": len(X_test),
    }

    pred_df = pd.DataFrame({
        "ZIP_CODE": model_df.loc[idx_test, "ZIP_CODE"].values,
        "actual_price": y_test_price.values,
        "predicted_price_gbm": pred_test_price,
        "actual_log_price": y_test.values,
        "predicted_log_price_gbm": pred_test_log,
    }, index=idx_test)

    try:
        result = permutation_importance(
            pipe,
            X_test,
            y_test,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="r2",
        )

        importance_df = pd.DataFrame({
            "feature": predictors,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)
    except Exception as e:
        importance_df = pd.DataFrame({
            "feature": predictors,
            "importance_mean": np.nan,
            "importance_std": np.nan,
        })

    return pipe, pred_df, metrics, importance_df


# ============================================================
# OUTPUTS
# ============================================================

def save_outputs(df, diagnostics, predictors, ols_model, ols_df, ols_formula, gbm_pred, gbm_metrics, importance_df):
    df.to_csv(OUTPUT_DIR / "single_model_dataset.csv", index=False)
    ols_df.to_csv(OUTPUT_DIR / "single_ols_predictions.csv", index=False)
    gbm_pred.to_csv(OUTPUT_DIR / "single_gbm_test_predictions.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "single_gbm_feature_importance.csv", index=False)

    write_text(OUTPUT_DIR / "single_ols_summary.txt", str(ols_model.summary()))

    coef_df = pd.DataFrame({
        "term": ols_model.params.index,
        "coef": ols_model.params.values,
        "std_err": ols_model.bse.values,
        "z_value": ols_model.tvalues.values,
        "p_value": ols_model.pvalues.values,
    })
    coef_df.to_csv(OUTPUT_DIR / "single_ols_coefficients.csv", index=False)

    ols_metrics = {
        "ols_r2_log": r2_score(ols_df["log_median_price_june"], ols_df["pred_log_price_ols"]),
        "ols_mae_price": mean_absolute_error(ols_df[TARGET_COL], ols_df["pred_price_ols"]),
        "ols_rmse_price": rmse(ols_df[TARGET_COL], ols_df["pred_price_ols"]),
        "ols_rows": len(ols_df),
    }

    lines = []
    lines.append("ZIP-only model diagnostics")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Data diagnostics:")
    for k, v in diagnostics.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Important methodological note:")
    lines.append("ZIP_CODE is used only as a row identifier, not as a numeric predictor.")
    lines.append("This model is one-row-per-ZIP and predicts ZIP-level median Airbnb price.")
    lines.append("")
    lines.append("OLS formula:")
    lines.append(ols_formula)
    lines.append("")
    lines.append("OLS metrics:")
    for k, v in ols_metrics.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("GBM metrics:")
    for k, v in gbm_metrics.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Predictors included:")
    for p in predictors:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Median price columns were excluded from predictors to avoid target leakage.")
    lines.append("- Airbnb supply variables are excluded unless INCLUDE_AIRBNB_SUPPLY_FEATURES=True.")
    lines.append("- With only ZIP-level rows, interpret nonlinear model performance cautiously.")

    write_text(OUTPUT_DIR / "single_model_diagnostics.txt", "\n".join(lines))

    plot_actual_vs_predicted(
        ols_df[TARGET_COL].values,
        ols_df["pred_price_ols"].values,
        "ZIP-only OLS: Actual vs Predicted Median Price",
        OUTPUT_DIR / "single_actual_vs_predicted_ols.png",
    )

    plot_actual_vs_predicted(
        gbm_pred["actual_price"].values,
        gbm_pred["predicted_price_gbm"].values,
        "ZIP-only GBM: Actual vs Predicted Median Price",
        OUTPUT_DIR / "single_actual_vs_predicted_gbm.png",
    )

    plot_feature_importance(
        importance_df,
        OUTPUT_DIR / "single_gbm_feature_importance.png",
        top_n=20,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting ZIP-only model...")

    df, diagnostics = load_zip_data()
    predictors = get_predictor_columns(df)

    if not predictors:
        raise ValueError("No usable external predictors were found.")

    print(f"Rows: {len(df):,}")
    print(f"Predictors: {len(predictors)}")

    ols_model, ols_df, ols_formula = run_ols(df, predictors)
    print(f"OLS R^2 on log median price: {ols_model.rsquared:.4f}")

    gbm_pipe, gbm_pred, gbm_metrics, importance_df = run_gbm(df, predictors)
    print(f"GBM test R^2 on log median price: {gbm_metrics['gbm_test_r2_log']:.4f}")
    print(f"GBM test MAE on price: {gbm_metrics['gbm_test_mae_price']:.2f}")

    save_outputs(
        df=df,
        diagnostics=diagnostics,
        predictors=predictors,
        ols_model=ols_model,
        ols_df=ols_df,
        ols_formula=ols_formula,
        gbm_pred=gbm_pred,
        gbm_metrics=gbm_metrics,
        importance_df=importance_df,
    )

    print(f"Done. Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
