"""
regression_external_only.py

Purpose:
    Edited version of the listing-level model that excludes house/property/listing-level
    predictors from the feature set.

    This model predicts individual Airbnb listing price, but it uses ONLY external
    / area-level factors attached by ZIP code plus optional pure location coordinates.

    It does NOT use:
        room_type
        bedrooms
        bathrooms
        beds
        accommodates
        reviews
        availability
        host listing count
        minimum nights
        property type
        any other listing/property characteristics

    ZIP_CODE is used only as a merge key.

Expected project layout:
    AirBNBRegression/
        regression_external_only.py
        OldData/
            AirBNB_June.csv
            AirBNB_September.csv
            CleanedData.csv or Cleaned_Data.csv
            CrimeData.csv
        model_outputs_external_only/   <-- created automatically

Run:
    python regression_external_only.py

Install:
    pip install pandas numpy matplotlib scikit-learn statsmodels
"""

from pathlib import Path
from typing import Dict, Tuple, List
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OLD_DATA_DIR = BASE_DIR / "OldData"
OUTPUT_DIR = BASE_DIR / "model_outputs_external_only"
OUTPUT_DIR.mkdir(exist_ok=True)

AIRBNB_JUNE_FILE = OLD_DATA_DIR / "AirBNB_June.csv"
AIRBNB_SEPT_FILE = OLD_DATA_DIR / "AirBNB_September.csv"
CRIME_FILE = OLD_DATA_DIR / "CrimeData.csv"

if (OLD_DATA_DIR / "CleanedData.csv").exists():
    CLEANED_FILE = OLD_DATA_DIR / "CleanedData.csv"
elif (OLD_DATA_DIR / "Cleaned_Data.csv").exists():
    CLEANED_FILE = OLD_DATA_DIR / "Cleaned_Data.csv"
else:
    raise FileNotFoundError(
        "Could not find OldData/CleanedData.csv or OldData/Cleaned_Data.csv."
    )

CRIME_CHUNKSIZE = 200_000
MIN_PRICE = 10
MAX_PRICE = 5000
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Location is not a house/property characteristic. It is external geography.
# Keep True if you want "location" as an external predictor.
INCLUDE_LAT_LONG = True


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def standardize_zip(series):
    """
    ZIP is a merge key only, never a numeric predictor.
    """
    s = safe_numeric(series)
    s = s.where((s >= 10000) & (s <= 99999))
    return s.astype("Int64")


def zscore(series):
    s = safe_numeric(series)
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


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


def plot_feature_importance(importance_df: pd.DataFrame, outpath: Path, top_n=20):
    if importance_df.empty:
        return

    d = importance_df.sort_values("importance_mean", ascending=False).head(top_n)
    d = d.iloc[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(d["feature"], d["importance_mean"])
    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.title("External-only GBM feature importance")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ============================================================
# LOAD AIRBNB TARGET DATA
# ============================================================

def load_airbnb_target_data() -> Tuple[pd.DataFrame, Dict[str, float]]:
    june = pd.read_csv(AIRBNB_JUNE_FILE)
    sept = pd.read_csv(AIRBNB_SEPT_FILE)

    june["month"] = "June"
    sept["month"] = "September"

    june.columns = [c.strip() for c in june.columns]
    sept.columns = [c.strip() for c in sept.columns]

    df = pd.concat([june, sept], ignore_index=True)

    # Normalize common column variants
    rename_map = {}
    if "latitude" in df.columns:
        rename_map["latitude"] = "LATITUDE"
    if "longitude" in df.columns:
        rename_map["longitude"] = "LONGITUDE"
    if "zip_code" in df.columns:
        rename_map["zip_code"] = "ZIP_CODE"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["price", "ZIP_CODE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Airbnb columns: {missing}")

    raw_rows = len(df)

    df["price"] = safe_numeric(df["price"])
    df["ZIP_CODE"] = standardize_zip(df["ZIP_CODE"])

    if "LATITUDE" in df.columns:
        df["LATITUDE"] = safe_numeric(df["LATITUDE"])
    if "LONGITUDE" in df.columns:
        df["LONGITUDE"] = safe_numeric(df["LONGITUDE"])

    df = df.dropna(subset=["price", "ZIP_CODE"]).copy()
    after_required = len(df)

    df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)].copy()
    after_price_trim = len(df)

    df["log_price"] = np.log(df["price"])

    diagnostics = {
        "airbnb_raw_rows": raw_rows,
        "airbnb_after_required_filters": after_required,
        "airbnb_after_price_trim": after_price_trim,
        "airbnb_unique_zips": int(df["ZIP_CODE"].nunique()),
    }

    # Drop all columns except target + external join/location variables.
    keep_cols = ["price", "log_price", "ZIP_CODE", "month"]
    if INCLUDE_LAT_LONG:
        for c in ["LATITUDE", "LONGITUDE"]:
            if c in df.columns:
                keep_cols.append(c)

    df = df[[c for c in keep_cols if c in df.columns]].copy()

    return df, diagnostics


# ============================================================
# CLEANED ZIP FEATURES
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

    # Drop target-leakage columns. These are derived from Airbnb prices and should not
    # be used to predict Airbnb price.
    leakage_cols = [c for c in cleaned.columns if c.startswith("Median_")]
    cleaned = cleaned.drop(columns=leakage_cols, errors="ignore")

    keep_candidates = [
        "ZIP_CODE",
        "Population",
        "Count_Part_1_Crime",
        "Part_1_Crime_Count",
        "Crime_Per_Capita",
        "Part_1_Crime_Per_Capita",
        "Count_Public_Safety",
        "Count_Fire_Stations",
        "Count_Police_Stations",
        "Count_Corrective_Facility",
        "Count_Education_Facility",
        "Count_Educational_Facility",
        "Education_Per_Capita",
        "Public_Safety_Per_Capita",
    ]

    keep_cols = [c for c in keep_candidates if c in cleaned.columns]
    cleaned = cleaned[keep_cols].copy()

    # Harmonize similar names across old/new cleaned files
    rename_map = {}
    if "Part_1_Crime_Count" in cleaned.columns and "Count_Part_1_Crime" not in cleaned.columns:
        rename_map["Part_1_Crime_Count"] = "Count_Part_1_Crime"
    if "Part_1_Crime_Per_Capita" in cleaned.columns and "Crime_Per_Capita" not in cleaned.columns:
        rename_map["Part_1_Crime_Per_Capita"] = "Crime_Per_Capita"
    if "Count_Educational_Facility" in cleaned.columns and "Count_Education_Facility" not in cleaned.columns:
        rename_map["Count_Educational_Facility"] = "Count_Education_Facility"

    if rename_map:
        cleaned = cleaned.rename(columns=rename_map)

    # External indices
    if "Population" in cleaned.columns:
        cleaned["log_population"] = np.log1p(cleaned["Population"].clip(lower=0))

    if "Crime_Per_Capita" in cleaned.columns:
        cleaned["crime_per_10k_cleaned"] = cleaned["Crime_Per_Capita"] * 10000
        cleaned["crime_z_cleaned"] = zscore(cleaned["Crime_Per_Capita"])

    if "Education_Per_Capita" in cleaned.columns:
        cleaned["education_per_10k"] = cleaned["Education_Per_Capita"] * 10000
        cleaned["education_z"] = zscore(cleaned["Education_Per_Capita"])
    elif "Count_Education_Facility" in cleaned.columns and "Population" in cleaned.columns:
        cleaned["education_per_10k"] = (cleaned["Count_Education_Facility"] / cleaned["Population"].replace(0, np.nan)) * 10000
        cleaned["education_z"] = zscore(cleaned["education_per_10k"])

    if "Public_Safety_Per_Capita" in cleaned.columns:
        cleaned["public_safety_per_10k"] = cleaned["Public_Safety_Per_Capita"] * 10000
        cleaned["public_safety_z"] = zscore(cleaned["Public_Safety_Per_Capita"])
    elif "Count_Public_Safety" in cleaned.columns and "Population" in cleaned.columns:
        cleaned["public_safety_per_10k"] = (cleaned["Count_Public_Safety"] / cleaned["Population"].replace(0, np.nan)) * 10000
        cleaned["public_safety_z"] = zscore(cleaned["public_safety_per_10k"])

    amenity_parts = []
    for c in [
        "Count_Public_Safety",
        "Count_Fire_Stations",
        "Count_Police_Stations",
        "Count_Education_Facility",
    ]:
        if c in cleaned.columns:
            amenity_parts.append(zscore(cleaned[c]))

    if amenity_parts:
        cleaned["external_amenities_index"] = sum(amenity_parts) / len(amenity_parts)

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
    chunk_count = 0
    agg = {}

    for chunk in pd.read_csv(CRIME_FILE, usecols=usecols, chunksize=CRIME_CHUNKSIZE):
        chunk_count += 1
        total_rows_read += len(chunk)

        chunk["ZIP_CODE"] = standardize_zip(chunk["ZIP_CODE"])
        total_rows_valid_zip += int(chunk["ZIP_CODE"].notna().sum())

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
# BUILD EXTERNAL-ONLY DATASET
# ============================================================

def build_dataset():
    airbnb, d_airbnb = load_airbnb_target_data()
    cleaned, d_cleaned = load_cleaned_zip_features()
    crime_zip, d_crime = aggregate_crime_by_zip()

    zip_all = cleaned.merge(crime_zip, on="ZIP_CODE", how="left")

    if "Population" in zip_all.columns:
        pop = zip_all["Population"].replace(0, np.nan)

        if "crime_events" in zip_all.columns:
            zip_all["crime_events_per_10k"] = (zip_all["crime_events"] / pop) * 10000
            zip_all["crime_events_z"] = zscore(zip_all["crime_events_per_10k"])

        if "violent_crime_events" in zip_all.columns:
            zip_all["violent_crime_per_10k"] = (zip_all["violent_crime_events"] / pop) * 10000
            zip_all["violent_crime_z"] = zscore(zip_all["violent_crime_per_10k"])

        if "property_crime_events" in zip_all.columns:
            zip_all["property_crime_per_10k"] = (zip_all["property_crime_events"] / pop) * 10000
            zip_all["property_crime_z"] = zscore(zip_all["property_crime_per_10k"])

    # Fill ZIP-level features
    for c in zip_all.columns:
        if c != "ZIP_CODE" and pd.api.types.is_numeric_dtype(zip_all[c]):
            zip_all[c] = zip_all[c].fillna(zip_all[c].median())

    merged = airbnb.merge(zip_all, on="ZIP_CODE", how="left")

    diagnostics = {}
    diagnostics.update(d_airbnb)
    diagnostics.update(d_cleaned)
    diagnostics.update(d_crime)
    diagnostics["zip_all_rows"] = len(zip_all)
    diagnostics["zip_all_unique_zips"] = int(zip_all["ZIP_CODE"].nunique())
    diagnostics["merged_rows_before_drop"] = len(merged)
    diagnostics["merged_unique_zips"] = int(merged["ZIP_CODE"].nunique())
    diagnostics["crime_share_retained_after_zip_cleaning"] = (
        d_crime["crime_aggregated_total_events"] / d_crime["crime_total_rows_read"]
        if d_crime["crime_total_rows_read"] else np.nan
    )

    zip_all.to_csv(OUTPUT_DIR / "external_zip_features.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "external_only_model_dataset.csv", index=False)

    return merged, diagnostics


def get_external_predictors(df: pd.DataFrame) -> List[str]:
    predictors = []

    # Pure location
    if INCLUDE_LAT_LONG:
        for c in ["LATITUDE", "LONGITUDE"]:
            if c in df.columns:
                predictors.append(c)

    # Area-level external variables only
    candidates = [
        "log_population",
        "crime_z_cleaned",
        "crime_events_z",
        "violent_crime_z",
        "property_crime_z",
        "education_z",
        "public_safety_z",
        "external_amenities_index",
        "crime_per_10k_cleaned",
        "crime_events_per_10k",
        "violent_crime_per_10k",
        "property_crime_per_10k",
        "education_per_10k",
        "public_safety_per_10k",
        "Count_Corrective_Facility",
    ]

    for c in candidates:
        if c in df.columns and c not in predictors:
            predictors.append(c)

    # Remove anything that is not external by construction
    forbidden_patterns = [
        "room",
        "bed",
        "bath",
        "accommodates",
        "review",
        "availability",
        "host",
        "minimum_nights",
        "property_type",
        "Median_",
    ]

    clean = []
    for c in predictors:
        lowered = c.lower()
        if any(p.lower() in lowered for p in forbidden_patterns):
            continue
        if c in ["price", "log_price", "ZIP_CODE", "month"]:
            continue
        clean.append(c)

    return clean


# ============================================================
# MODELS
# ============================================================

def run_ols(df: pd.DataFrame, predictors: List[str]):
    # month is not a house/property variable; it is a time control.
    rhs = predictors + ["C(month)"] if "month" in df.columns else predictors
    formula = "log_price ~ " + " + ".join(rhs)

    model_cols = ["price", "log_price"] + predictors
    if "month" in df.columns:
        model_cols.append("month")

    model_df = df[model_cols].dropna().copy()

    model = smf.ols(formula=formula, data=model_df).fit(cov_type="HC3")
    model_df["pred_log_price_ols"] = model.predict(model_df)
    model_df["pred_price_ols"] = np.exp(model_df["pred_log_price_ols"])

    return model, model_df, formula


def run_gbm(df: pd.DataFrame, predictors: List[str]):
    categorical_features = []
    if "month" in df.columns:
        categorical_features.append("month")

    feature_cols = predictors + categorical_features
    model_df = df[["price", "log_price"] + feature_cols].dropna(subset=["price", "log_price"]).copy()

    X = model_df[feature_cols]
    y = model_df["log_price"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    numeric_features = predictors

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # To avoid sparse-matrix issues with HistGradientBoostingRegressor, use month as numeric flag if needed.
    # Month is simple enough to encode manually.
    X_train_work = X_train.copy()
    X_test_work = X_test.copy()

    if "month" in X_train_work.columns:
        X_train_work["month_september"] = (X_train_work["month"].astype(str).str.lower() == "september").astype(int)
        X_test_work["month_september"] = (X_test_work["month"].astype(str).str.lower() == "september").astype(int)
        X_train_work = X_train_work.drop(columns=["month"])
        X_test_work = X_test_work.drop(columns=["month"])
        numeric_features = numeric_features + ["month_september"]

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=20,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )),
    ])

    pipe.fit(X_train_work, y_train)

    pred_train_log = pipe.predict(X_train_work)
    pred_test_log = pipe.predict(X_test_work)

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
        "gbm_train_rows": len(X_train_work),
        "gbm_test_rows": len(X_test_work),
    }

    pred_df = pd.DataFrame({
        "actual_price": y_test_price.values,
        "predicted_price_gbm": pred_test_price,
        "actual_log_price": y_test.values,
        "predicted_log_price_gbm": pred_test_log,
    }, index=idx_test)

    try:
        result = permutation_importance(
            pipe,
            X_test_work,
            y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            scoring="r2",
        )

        importance_df = pd.DataFrame({
            "feature": X_test_work.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance_mean", ascending=False)
    except Exception:
        importance_df = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    return pipe, pred_df, metrics, importance_df, list(X_test_work.columns)


# ============================================================
# OUTPUTS
# ============================================================

def save_outputs(diagnostics, predictors, ols_model, ols_df, ols_formula, gbm_pred, gbm_metrics, importance_df, gbm_features):
    write_text(OUTPUT_DIR / "external_only_ols_summary.txt", str(ols_model.summary()))

    coef_df = pd.DataFrame({
        "term": ols_model.params.index,
        "coef": ols_model.params.values,
        "std_err": ols_model.bse.values,
        "z_value": ols_model.tvalues.values,
        "p_value": ols_model.pvalues.values,
    })
    coef_df.to_csv(OUTPUT_DIR / "external_only_ols_coefficients.csv", index=False)

    ols_df.to_csv(OUTPUT_DIR / "external_only_ols_predictions.csv", index=False)
    gbm_pred.to_csv(OUTPUT_DIR / "external_only_gbm_test_predictions.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "external_only_gbm_feature_importance.csv", index=False)

    ols_metrics = {
        "ols_r2_log": r2_score(ols_df["log_price"], ols_df["pred_log_price_ols"]),
        "ols_mae_price": mean_absolute_error(ols_df["price"], ols_df["pred_price_ols"]),
        "ols_rmse_price": rmse(ols_df["price"], ols_df["pred_price_ols"]),
        "ols_rows": len(ols_df),
    }

    lines = []
    lines.append("External-only model diagnostics")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Pipeline diagnostics:")
    for k, v in diagnostics.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Methodological note:")
    lines.append("This model predicts listing price but uses only external/location variables.")
    lines.append("It excludes house/property/listing characteristics such as room_type, beds, baths, accommodates, reviews, availability, host variables, and minimum nights.")
    lines.append("ZIP_CODE was used only as a merge key.")
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
    lines.append("External predictors included:")
    for p in predictors:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("GBM features included:")
    for p in gbm_features:
        lines.append(f"- {p}")

    write_text(OUTPUT_DIR / "external_only_diagnostics.txt", "\n".join(lines))

    plot_actual_vs_predicted(
        ols_df["price"].values,
        ols_df["pred_price_ols"].values,
        "External-only OLS: Actual vs Predicted Listing Price",
        OUTPUT_DIR / "external_only_actual_vs_predicted_ols.png",
    )

    plot_actual_vs_predicted(
        gbm_pred["actual_price"].values,
        gbm_pred["predicted_price_gbm"].values,
        "External-only GBM: Actual vs Predicted Listing Price",
        OUTPUT_DIR / "external_only_actual_vs_predicted_gbm.png",
    )

    plot_feature_importance(
        importance_df,
        OUTPUT_DIR / "external_only_gbm_feature_importance.png",
        top_n=20,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting external-only listing-level model...")

    df, diagnostics = build_dataset()
    predictors = get_external_predictors(df)

    if not predictors:
        raise ValueError("No external predictors were found.")

    print(f"Merged rows: {len(df):,}")
    print(f"External predictors: {len(predictors)}")

    ols_model, ols_df, ols_formula = run_ols(df, predictors)
    print(f"External-only OLS R^2 on log price: {ols_model.rsquared:.4f}")

    gbm_pipe, gbm_pred, gbm_metrics, importance_df, gbm_features = run_gbm(df, predictors)
    print(f"External-only GBM test R^2 on log price: {gbm_metrics['gbm_test_r2_log']:.4f}")
    print(f"External-only GBM test MAE on price: {gbm_metrics['gbm_test_mae_price']:.2f}")

    save_outputs(
        diagnostics=diagnostics,
        predictors=predictors,
        ols_model=ols_model,
        ols_df=ols_df,
        ols_formula=ols_formula,
        gbm_pred=gbm_pred,
        gbm_metrics=gbm_metrics,
        importance_df=importance_df,
        gbm_features=gbm_features,
    )

    print(f"Done. Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
