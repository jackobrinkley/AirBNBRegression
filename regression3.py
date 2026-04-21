import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "model_outputs_v3"
OUTPUT_DIR.mkdir(exist_ok=True)

AIRBNB_JUNE_FILE = BASE_DIR / "AirBNB_June.csv"
AIRBNB_SEPT_FILE = BASE_DIR / "AirBNB_September.csv"
CRIME_FILE = BASE_DIR / "CrimeData.csv"

if (BASE_DIR / "CleanedData.csv").exists():
    CLEANED_FILE = BASE_DIR / "CleanedData.csv"
elif (BASE_DIR / "Cleaned_Data.csv").exists():
    CLEANED_FILE = BASE_DIR / "Cleaned_Data.csv"
else:
    raise FileNotFoundError("Could not find CleanedData.csv or Cleaned_Data.csv.")

CRIME_CHUNKSIZE = 200_000
MIN_PRICE = 10
MAX_PRICE = 5000
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def standardize_zip(series):
    s = safe_numeric(series)
    s = s.where((s >= 10000) & (s <= 99999))
    return s.astype("Int64")


def zscore(series):
    s = safe_numeric(series)
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def write_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def add_log1p_feature(df: pd.DataFrame, source_col: str, new_col: str):
    if source_col in df.columns:
        s = safe_numeric(df[source_col]).fillna(0)
        s = s.clip(lower=0)
        df[new_col] = np.log1p(s)


def plot_actual_vs_predicted(y_true, y_pred, title, outpath):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.15)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame, outpath: Path, top_n: int = 20):
    d = importance_df.sort_values("importance_mean", ascending=False).head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(d["feature"], d["importance_mean"])
    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def first_existing(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ============================================================
# LOAD AIRBNB
# ============================================================

def load_airbnb() -> Tuple[pd.DataFrame, Dict[str, float]]:
    june = pd.read_csv(AIRBNB_JUNE_FILE)
    sept = pd.read_csv(AIRBNB_SEPT_FILE)

    june["month"] = "June"
    sept["month"] = "September"

    june.columns = [c.strip() for c in june.columns]
    sept.columns = [c.strip() for c in sept.columns]

    airbnb = pd.concat([june, sept], ignore_index=True)

    # Normalize common variants if present
    rename_map = {}
    if "latitude" in airbnb.columns:
        rename_map["latitude"] = "LATITUDE"
    if "longitude" in airbnb.columns:
        rename_map["longitude"] = "LONGITUDE"
    if "zip_code" in airbnb.columns:
        rename_map["zip_code"] = "ZIP_CODE"
    if rename_map:
        airbnb = airbnb.rename(columns=rename_map)

    raw_rows = len(airbnb)

    needed = ["price", "room_type", "ZIP_CODE", "month"]
    missing = [c for c in needed if c not in airbnb.columns]
    if missing:
        raise ValueError(f"Missing required Airbnb columns: {missing}")

    numeric_candidates = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "number_of_reviews_ltm",
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms",
        "LATITUDE",
        "LONGITUDE",
        "latitude",
        "longitude",
    ]
    for c in numeric_candidates:
        if c in airbnb.columns:
            airbnb[c] = safe_numeric(airbnb[c])

    airbnb["ZIP_CODE"] = standardize_zip(airbnb["ZIP_CODE"])

    airbnb = airbnb.dropna(subset=["price", "ZIP_CODE", "room_type", "month"]).copy()
    after_required = len(airbnb)

    airbnb = airbnb[(airbnb["price"] >= MIN_PRICE) & (airbnb["price"] <= MAX_PRICE)].copy()
    after_price_trim = len(airbnb)

    for c in ["reviews_per_month", "number_of_reviews", "number_of_reviews_ltm"]:
        if c in airbnb.columns:
            airbnb[c] = airbnb[c].fillna(0)

    airbnb["log_price"] = np.log(airbnb["price"])

    # Transformations
    add_log1p_feature(airbnb, "minimum_nights", "log_minimum_nights")
    add_log1p_feature(airbnb, "number_of_reviews", "log_number_of_reviews")
    add_log1p_feature(airbnb, "reviews_per_month", "log_reviews_per_month")
    add_log1p_feature(airbnb, "availability_365", "log_availability_365")
    add_log1p_feature(airbnb, "number_of_reviews_ltm", "log_number_of_reviews_ltm")
    add_log1p_feature(airbnb, "calculated_host_listings_count", "log_host_listings_count")

    # Listing-level size variables if present
    for c in ["accommodates", "bedrooms", "beds", "bathrooms"]:
        if c in airbnb.columns:
            airbnb[c] = airbnb[c].fillna(airbnb[c].median())

    diagnostics = {
        "airbnb_raw_rows": raw_rows,
        "airbnb_after_required_filters": after_required,
        "airbnb_after_price_trim": after_price_trim,
        "airbnb_unique_zips": int(airbnb["ZIP_CODE"].nunique()),
    }

    return airbnb, diagnostics


# ============================================================
# LOAD CLEANED ZIP FEATURES
# ============================================================

def load_cleaned_zip_features() -> Tuple[pd.DataFrame, Dict[str, float]]:
    cleaned = pd.read_csv(CLEANED_FILE)
    cleaned.columns = [c.strip() for c in cleaned.columns]

    if "ZIP_Code" in cleaned.columns:
        cleaned = cleaned.rename(columns={"ZIP_Code": "ZIP_CODE"})
    elif "ZIP_CODE" not in cleaned.columns:
        raise ValueError("CleanedData file must contain ZIP_Code or ZIP_CODE.")

    cleaned["ZIP_CODE"] = standardize_zip(cleaned["ZIP_CODE"])
    cleaned = cleaned.dropna(subset=["ZIP_CODE"]).copy()

    for col in cleaned.columns:
        if col != "ZIP_CODE":
            cleaned[col] = safe_numeric(cleaned[col])

    # Drop leakage columns
    leakage_cols = [c for c in cleaned.columns if c.startswith("Median_")]
    cleaned = cleaned.drop(columns=leakage_cols, errors="ignore")

    keep_candidates = [
        "ZIP_CODE",
        "Population",
        "Count_Public_Safety",
        "Count_Fire_Stations",
        "Count_Police_Stations",
        "Count_Corrective_Facility",
        "Count_Education_Facility",
        "Crime_Per_Capita",
        "Education_Per_Capita",
        "Public_Safety_Per_Capita",
    ]
    keep_cols = [c for c in keep_candidates if c in cleaned.columns]
    cleaned = cleaned[keep_cols].copy()

    # Composite amenities index
    amenity_parts = []
    for c in ["Count_Fire_Stations", "Count_Police_Stations", "Count_Education_Facility"]:
        if c in cleaned.columns:
            amenity_parts.append(zscore(cleaned[c]))
    if amenity_parts:
        cleaned["amenities_index"] = sum(amenity_parts) / len(amenity_parts)

    # Scaled / standardized versions to prevent exploding coefficients
    if "Education_Per_Capita" in cleaned.columns:
        cleaned["Education_Per_10k"] = cleaned["Education_Per_Capita"] * 10000
        cleaned["Education_z"] = zscore(cleaned["Education_Per_Capita"])

    if "Public_Safety_Per_Capita" in cleaned.columns:
        cleaned["Public_Safety_Per_10k"] = cleaned["Public_Safety_Per_Capita"] * 10000
        cleaned["Public_Safety_z"] = zscore(cleaned["Public_Safety_Per_Capita"])

    if "Crime_Per_Capita" in cleaned.columns:
        cleaned["Crime_Per_10k"] = cleaned["Crime_Per_Capita"] * 10000
        cleaned["Crime_z"] = zscore(cleaned["Crime_Per_Capita"])

    diagnostics = {
        "cleaned_zip_rows": len(cleaned),
        "cleaned_zip_unique_zips": int(cleaned["ZIP_CODE"].nunique()),
    }
    return cleaned, diagnostics


# ============================================================
# CRIME AGGREGATION
# ============================================================

def classify_violent(desc_series):
    pattern = (
        r"ASSAULT|ROBBERY|RAPE|SEX|HOMICIDE|MANSLAUGHTER|KIDNAP|"
        r"SHOTS FIRED|SHOOTING|FIREARM|WEAPON|BATTERY|CHILD ABUSE|"
        r"INTIMATE PARTNER|CRUELTY"
    )
    return desc_series.fillna("").str.upper().str.contains(pattern, regex=True, na=False)


def classify_property(desc_series):
    pattern = (
        r"BURGLARY|THEFT|STOLEN|VANDALISM|SHOPLIFTING|BIKE|VEHICLE|"
        r"AUTO|IDENTITY|EMBEZZLEMENT|FRAUD|ARSON|TRESPASS"
    )
    return desc_series.fillna("").str.upper().str.contains(pattern, regex=True, na=False)


def aggregate_crime_by_zip() -> Tuple[pd.DataFrame, Dict[str, float]]:
    usecols = ["ZIP_CODE", "Crm Cd Desc", "Part 1-2"]

    total_rows_read = 0
    total_rows_valid_zip = 0
    agg = {}
    chunk_count = 0

    for chunk in pd.read_csv(CRIME_FILE, usecols=usecols, chunksize=CRIME_CHUNKSIZE):
        chunk_count += 1
        total_rows_read += len(chunk)

        chunk["ZIP_CODE"] = standardize_zip(chunk["ZIP_CODE"])
        total_rows_valid_zip += chunk["ZIP_CODE"].notna().sum()

        chunk = chunk.dropna(subset=["ZIP_CODE"]).copy()
        chunk["violent_flag"] = classify_violent(chunk["Crm Cd Desc"]).astype(int)
        chunk["property_flag"] = classify_property(chunk["Crm Cd Desc"]).astype(int)
        chunk["part1_flag"] = (safe_numeric(chunk["Part 1-2"]) == 1).astype(int)

        grouped = chunk.groupby("ZIP_CODE", observed=False).agg(
            crime_events=("ZIP_CODE", "size"),
            violent_crime_events=("violent_flag", "sum"),
            property_crime_events=("property_flag", "sum"),
            part1_events=("part1_flag", "sum"),
        )

        for zip_code, row in grouped.iterrows():
            if zip_code not in agg:
                agg[zip_code] = {
                    "crime_events": 0,
                    "violent_crime_events": 0,
                    "property_crime_events": 0,
                    "part1_events": 0,
                }
            agg[zip_code]["crime_events"] += int(row["crime_events"])
            agg[zip_code]["violent_crime_events"] += int(row["violent_crime_events"])
            agg[zip_code]["property_crime_events"] += int(row["property_crime_events"])
            agg[zip_code]["part1_events"] += int(row["part1_events"])

        print(
            f"Processed crime chunk {chunk_count}: "
            f"rows read so far={total_rows_read:,}, valid ZIP rows so far={total_rows_valid_zip:,}"
        )

    crime_zip = pd.DataFrame.from_dict(agg, orient="index").reset_index()
    crime_zip = crime_zip.rename(columns={"index": "ZIP_CODE"})
    crime_zip["ZIP_CODE"] = crime_zip["ZIP_CODE"].astype("Int64")

    diagnostics = {
        "crime_chunks_processed": chunk_count,
        "crime_total_rows_read": total_rows_read,
        "crime_rows_with_valid_zip": total_rows_valid_zip,
        "crime_aggregated_total_events": int(crime_zip["crime_events"].sum()) if len(crime_zip) else 0,
        "crime_unique_zip_after_aggregation": int(crime_zip["ZIP_CODE"].nunique()) if len(crime_zip) else 0,
    }

    return crime_zip, diagnostics


# ============================================================
# BUILD MODEL DATASET
# ============================================================

def build_model_dataset() -> Tuple[pd.DataFrame, Dict[str, float]]:
    airbnb, d_airbnb = load_airbnb()
    cleaned, d_cleaned = load_cleaned_zip_features()
    crime_zip, d_crime = aggregate_crime_by_zip()

    zip_all = cleaned.merge(crime_zip, on="ZIP_CODE", how="left")

    if "Population" in zip_all.columns:
        pop = zip_all["Population"].replace(0, np.nan)

        if "crime_events" in zip_all.columns:
            zip_all["crime_events_per_capita"] = zip_all["crime_events"] / pop
            zip_all["crime_events_per_10k"] = zip_all["crime_events_per_capita"] * 10000
            zip_all["crime_events_z"] = zscore(zip_all["crime_events_per_capita"])

        if "violent_crime_events" in zip_all.columns:
            zip_all["violent_crime_per_capita"] = zip_all["violent_crime_events"] / pop
            zip_all["violent_crime_per_10k"] = zip_all["violent_crime_per_capita"] * 10000

        if "property_crime_events" in zip_all.columns:
            zip_all["property_crime_per_capita"] = zip_all["property_crime_events"] / pop
            zip_all["property_crime_per_10k"] = zip_all["property_crime_per_capita"] * 10000

    for col in zip_all.columns:
        if col != "ZIP_CODE" and pd.api.types.is_numeric_dtype(zip_all[col]):
            zip_all[col] = zip_all[col].fillna(zip_all[col].median())

    merged = airbnb.merge(zip_all, on="ZIP_CODE", how="left")
    merged_before = len(merged)

    diagnostics = {}
    diagnostics.update(d_airbnb)
    diagnostics.update(d_cleaned)
    diagnostics.update(d_crime)
    diagnostics["zip_all_rows"] = len(zip_all)
    diagnostics["zip_all_unique_zips"] = int(zip_all["ZIP_CODE"].nunique())
    diagnostics["merged_rows_before_drop"] = merged_before
    diagnostics["merged_unique_zips"] = int(merged["ZIP_CODE"].nunique())
    diagnostics["crime_share_retained_after_zip_cleaning"] = (
        d_crime["crime_aggregated_total_events"] / d_crime["crime_total_rows_read"]
        if d_crime["crime_total_rows_read"] else np.nan
    )

    zip_all.to_csv(OUTPUT_DIR / "crime_zip_aggregates_and_features_v3.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "merged_airbnb_zip_dataset_v3.csv", index=False)

    return merged, diagnostics


# ============================================================
# OLS SPECIFICATION
# ============================================================

def run_ols(df: pd.DataFrame):
    # Keep one main proxy per concept
    predictors = []

    for c in [
        "log_minimum_nights",
        "log_number_of_reviews",
        "log_reviews_per_month",
        "log_host_listings_count",
        "log_availability_365",
        "log_number_of_reviews_ltm",
    ]:
        if c in df.columns:
            predictors.append(c)

    # listing-level size / location if present
    for c in ["accommodates", "bedrooms", "beds", "bathrooms", "LATITUDE", "LONGITUDE"]:
        if c in df.columns:
            predictors.append(c)

    # ZIP-level concept variables
    zip_candidates = [
        "crime_events_z",
        "Education_z",
        "Public_Safety_z",
        "amenities_index",
    ]
    for c in zip_candidates:
        if c in df.columns:
            predictors.append(c)

    # Interactions
    interaction_terms = []
    if "crime_events_z" in df.columns:
        interaction_terms.append("C(room_type):crime_events_z")
    if "amenities_index" in df.columns:
        interaction_terms.append("C(room_type):amenities_index")

    formula = "log_price ~ " + " + ".join(predictors + ["C(room_type)", "C(month)"] + interaction_terms)

    model_cols = ["log_price", "price", "room_type", "month"] + [c for c in predictors if c in df.columns]
    model_df = df[model_cols].dropna().copy()

    model = smf.ols(formula=formula, data=model_df).fit(cov_type="HC3")

    model_df["pred_log_price_ols"] = model.predict(model_df)
    model_df["pred_price_ols"] = np.exp(model_df["pred_log_price_ols"])

    return model, model_df, predictors, formula


# ============================================================
# NONLINEAR MODEL
# ============================================================

def run_nonlinear(df: pd.DataFrame):
    work = df.copy()

    # Candidate features
    numeric_candidates = [
        "log_minimum_nights",
        "log_number_of_reviews",
        "log_reviews_per_month",
        "log_host_listings_count",
        "log_availability_365",
        "log_number_of_reviews_ltm",
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms",
        "LATITUDE",
        "LONGITUDE",
        "crime_events_z",
        "Education_z",
        "Public_Safety_z",
        "amenities_index",
        "crime_events_per_10k",
        "Education_Per_10k",
        "Public_Safety_Per_10k",
    ]
    numeric_features = [c for c in numeric_candidates if c in work.columns]

    categorical_candidates = [
        "room_type",
        "month"
    ]

    """
    categorical_candidates = [
        "room_type",
        "month",
        "property_type",
        "neighbourhood_cleansed",
        "neighborhood",
        "neighbourhood",
    ]
    """

    categorical_features = [c for c in categorical_candidates if c in work.columns]

    feature_cols = numeric_features + categorical_features
    model_df = work[feature_cols + ["price", "log_price"]].dropna(subset=["price", "log_price"]).copy()

    X = model_df[feature_cols]
    y = model_df["log_price"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, model_df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # New try statement
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot),
        # ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    gbm = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
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

    # Save predictions
    pred_df = pd.DataFrame({
        "actual_price": y_test_price.values,
        "predicted_price_gbm": pred_test_price,
        "actual_log_price": y_test.values,
        "predicted_log_price_gbm": pred_test_log,
    }, index=idx_test)

    # Feature importance
    try:
        result = permutation_importance(
            pipe, X_test, y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="r2"
        )

        feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)
    except Exception:
        importance_df = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    return pipe, pred_df, metrics, importance_df, feature_cols


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    diagnostics: Dict[str, float],
    ols_model,
    ols_model_df: pd.DataFrame,
    ols_formula: str,
    ols_predictors: List[str],
    gbm_pred_df: pd.DataFrame,
    gbm_metrics: Dict[str, float],
    gbm_importance_df: pd.DataFrame,
    gbm_feature_cols: List[str],
):
    # OLS outputs
    write_text(OUTPUT_DIR / "ols_summary_v3.txt", str(ols_model.summary()))

    ols_coef = pd.DataFrame({
        "term": ols_model.params.index,
        "coef": ols_model.params.values,
        "std_err": ols_model.bse.values,
        "z_value": ols_model.tvalues.values,
        "p_value": ols_model.pvalues.values,
    })
    ols_coef.to_csv(OUTPUT_DIR / "ols_coefficients_v3.csv", index=False)

    # OLS metrics on full fitted sample
    ols_mae = mean_absolute_error(ols_model_df["price"], ols_model_df["pred_price_ols"])
    ols_rmse = rmse(ols_model_df["price"], ols_model_df["pred_price_ols"])
    ols_r2_log = r2_score(ols_model_df["log_price"], ols_model_df["pred_log_price_ols"])

    # Diagnostics report
    lines = []
    lines.append("Diagnostics report v3")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Pipeline diagnostics:")
    for k, v in diagnostics.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("OLS formula:")
    lines.append(ols_formula)
    lines.append("")
    lines.append("OLS metrics:")
    lines.append(f"ols_r2_log: {ols_r2_log:.6f}")
    lines.append(f"ols_mae_price: {ols_mae:.6f}")
    lines.append(f"ols_rmse_price: {ols_rmse:.6f}")
    lines.append(f"ols_observations: {len(ols_model_df):,}")
    lines.append("")
    lines.append("GBM metrics:")
    for k, v in gbm_metrics.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("OLS predictors included:")
    for p in ols_predictors:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("GBM feature columns included:")
    for p in gbm_feature_cols:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- ZIP_CODE was used only as a merge key.")
    lines.append("- OLS remains the interpretable benchmark.")
    lines.append("- GBM is the nonlinear model for predictive performance.")
    lines.append("- If GBM test R2 improves meaningfully over OLS, nonlinearity is helping.")
    lines.append("- If both remain weak, the main problem is missing explanatory variables.")
    write_text(OUTPUT_DIR / "diagnostics_report_v3.txt", "\n".join(lines))

    # Save predictions
    ols_model_df.to_csv(OUTPUT_DIR / "ols_predictions_v3.csv", index=False)
    gbm_pred_df.to_csv(OUTPUT_DIR / "gbm_test_predictions_v3.csv", index=False)

    # Save feature importance
    gbm_importance_df.to_csv(OUTPUT_DIR / "gbm_feature_importance_v3.csv", index=False)

    # Plots
    plot_actual_vs_predicted(
        ols_model_df["price"].values,
        ols_model_df["pred_price_ols"].values,
        "Actual vs Predicted Airbnb Price (OLS v3)",
        OUTPUT_DIR / "actual_vs_predicted_ols_v3.png"
    )

    plot_actual_vs_predicted(
        gbm_pred_df["actual_price"].values,
        gbm_pred_df["predicted_price_gbm"].values,
        "Actual vs Predicted Airbnb Price (GBM v3)",
        OUTPUT_DIR / "actual_vs_predicted_gbm_v3.png"
    )

    if not gbm_importance_df.empty:
        plot_feature_importance(
            gbm_importance_df,
            OUTPUT_DIR / "gbm_feature_importance_v3.png",
            top_n=20
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting v3 regression build...")

    df, diagnostics = build_model_dataset()
    print(f"Merged rows available: {len(df):,}")

    # OLS
    ols_model, ols_model_df, ols_predictors, ols_formula = run_ols(df)
    print("OLS finished.")
    print(f"OLS observations: {len(ols_model_df):,}")
    print(f"OLS R^2 on log price: {ols_model.rsquared:.4f}")

    # GBM
    gbm_pipe, gbm_pred_df, gbm_metrics, gbm_importance_df, gbm_feature_cols = run_nonlinear(df)
    print("GBM finished.")
    print(f"GBM test R^2 on log price: {gbm_metrics['gbm_test_r2_log']:.4f}")
    print(f"GBM test MAE on price: {gbm_metrics['gbm_test_mae_price']:.2f}")
    print(f"GBM test RMSE on price: {gbm_metrics['gbm_test_rmse_price']:.2f}")

    save_outputs(
        diagnostics=diagnostics,
        ols_model=ols_model,
        ols_model_df=ols_model_df,
        ols_formula=ols_formula,
        ols_predictors=ols_predictors,
        gbm_pred_df=gbm_pred_df,
        gbm_metrics=gbm_metrics,
        gbm_importance_df=gbm_importance_df,
        gbm_feature_cols=gbm_feature_cols,
    )

    print(f"Done. Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
