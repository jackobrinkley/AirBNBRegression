import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.copy_on_write = True


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(".")  # folder where your CSV files live
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

AIRBNB_JUNE_FILE = BASE_DIR / "AirBNB_June.csv"
AIRBNB_SEPT_FILE = BASE_DIR / "AirBNB_September.csv"
CRIME_FILE = BASE_DIR / "CrimeData.csv"

# The user said the cleaned file is now called CleanedData.csv.
# We support both names just in case.
if (BASE_DIR / "CleanedData.csv").exists():
    CLEANED_FILE = BASE_DIR / "CleanedData.csv"
elif (BASE_DIR / "Cleaned_Data.csv").exists():
    CLEANED_FILE = BASE_DIR / "Cleaned_Data.csv"
else:
    raise FileNotFoundError(
        "Could not find CleanedData.csv or Cleaned_Data.csv in the current folder."
    )

CRIME_CHUNKSIZE = 200_000

# Price trimming to remove obvious outliers / bad entries
MIN_PRICE = 10
MAX_PRICE = 5000

# ============================================================
# HELPERS
# ============================================================

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def standardize_zip(series):
    """
    Convert ZIP values into a nullable integer-like representation.
    ZIP is used only as a JOIN KEY, not as a predictor.
    """
    s = safe_numeric(series)
    s = s.where((s >= 10000) & (s <= 99999))
    return s.astype("Int64")


def zscore(series):
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def plot_binned_relationship(df, x_col, y_col, output_file, bins=20):
    """
    Makes a simple binned trend plot so you can see the relationship
    between a ZIP-level feature and listing price.
    """
    d = df[[x_col, y_col]].dropna().copy()
    if len(d) < 10:
        return

    # Use quantile bins when possible
    try:
        d["bin"] = pd.qcut(d[x_col], q=min(bins, d[x_col].nunique()), duplicates="drop")
    except Exception:
        return

    grouped = d.groupby("bin", observed=False).agg(
        x_mean=(x_col, "mean"),
        y_mean=(y_col, "mean"),
        count=(y_col, "size"),
    ).dropna()

    if grouped.empty:
        return

    plt.figure(figsize=(9, 6))
    plt.plot(grouped["x_mean"], grouped["y_mean"], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col} (binned means)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


# ============================================================
# LOAD AIRBNB DATA
# ============================================================

def load_airbnb():
    print("Loading Airbnb files...")

    june = pd.read_csv(AIRBNB_JUNE_FILE)
    september = pd.read_csv(AIRBNB_SEPT_FILE)

    june["month"] = "June"
    september["month"] = "September"

    # Normalize column names
    june.columns = [c.strip() for c in june.columns]
    september.columns = [c.strip() for c in september.columns]

    df = pd.concat([june, september], ignore_index=True)

    # Required columns expected from your screenshots / files
    needed_cols = [
        "price",
        "room_type",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "number_of_reviews_ltm",
        "ZIP_CODE",
        "month",
    ]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected Airbnb columns: {missing}")

    # Clean numeric fields
    numeric_cols = [
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "number_of_reviews_ltm",
    ]
    for col in numeric_cols:
        df[col] = safe_numeric(df[col])

    df["ZIP_CODE"] = standardize_zip(df["ZIP_CODE"])

    # Clean rows
    df = df.dropna(subset=["price", "ZIP_CODE", "room_type", "month"]).copy()
    df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)].copy()

    # Some Airbnb datasets have missing review rates
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # Target variable
    df["log_price"] = np.log(df["price"])

    print(f"Airbnb rows after cleaning: {len(df):,}")
    print(f"Unique ZIP codes in Airbnb data: {df['ZIP_CODE'].nunique():,}")

    return df


# ============================================================
# LOAD CLEANED ZIP-LEVEL DATA
# ============================================================

def load_cleaned_zip_features():
    print(f"Loading ZIP-level features from: {CLEANED_FILE.name}")
    df = pd.read_csv(CLEANED_FILE)

    df.columns = [c.strip() for c in df.columns]

    if "ZIP_Code" in df.columns:
        df = df.rename(columns={"ZIP_Code": "ZIP_CODE"})
    elif "ZIP_CODE" not in df.columns:
        raise ValueError("CleanedData file must contain ZIP_Code or ZIP_CODE.")

    df["ZIP_CODE"] = standardize_zip(df["ZIP_CODE"])
    df = df.dropna(subset=["ZIP_CODE"]).copy()

    # Convert all non-ZIP columns that should be numeric
    for col in df.columns:
        if col != "ZIP_CODE":
            df[col] = safe_numeric(df[col])

    # IMPORTANT:
    # Exclude median Airbnb / room price columns from CleanedData,
    # because they are derived from Airbnb prices themselves and would leak the target.
    leakage_pattern = re.compile(r"(?i)^Median_")
    leakage_cols = [c for c in df.columns if leakage_pattern.search(c)]
    if leakage_cols:
        print("Dropping target-leakage columns from CleanedData:")
        for c in leakage_cols:
            print(f"  - {c}")
        df = df.drop(columns=leakage_cols)

    # Keep only sensible ZIP-level predictors
    expected_candidate_cols = [
        "Count_Part_1_Crime",
        "Count_Public_Safety",
        "Count_Fire_Stations",
        "Count_Police_Stations",
        "Count_Corrective_Facility",
        "Count_Education_Facility",
        "Population",
        "Crime_Per_Capita",
        "Education_Per_Capita",
        "Public_Safety_Per_Capita",
    ]

    existing = [c for c in expected_candidate_cols if c in df.columns]
    missing = [c for c in expected_candidate_cols if c not in df.columns]

    print("ZIP-level columns found and kept:")
    for c in existing:
        print(f"  - {c}")

    if missing:
        print("ZIP-level columns not found (will be skipped):")
        for c in missing:
            print(f"  - {c}")

    keep_cols = ["ZIP_CODE"] + existing
    df = df[keep_cols].copy()

    # Build a compact amenities index
    amenity_parts = []
    if "Count_Fire_Stations" in df.columns:
        amenity_parts.append(zscore(df["Count_Fire_Stations"]))
    if "Count_Police_Stations" in df.columns:
        amenity_parts.append(zscore(df["Count_Police_Stations"]))
    if "Count_Education_Facility" in df.columns:
        amenity_parts.append(zscore(df["Count_Education_Facility"]))
    if "Count_Public_Safety" in df.columns:
        amenity_parts.append(zscore(df["Count_Public_Safety"]))

    if amenity_parts:
        df["amenities_index"] = sum(amenity_parts) / len(amenity_parts)

    # Public safety can be interpreted positively, crime negatively
    if "Crime_Per_Capita" in df.columns:
        df["crime_index"] = zscore(df["Crime_Per_Capita"])
    elif "Count_Part_1_Crime" in df.columns and "Population" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["crime_index"] = zscore(df["Count_Part_1_Crime"] / df["Population"].replace(0, np.nan))

    if "Education_Per_Capita" in df.columns:
        df["education_index"] = zscore(df["Education_Per_Capita"])
    elif "Count_Education_Facility" in df.columns and "Population" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["education_index"] = zscore(df["Count_Education_Facility"] / df["Population"].replace(0, np.nan))

    return df


# ============================================================
# AGGREGATE CRIME DATA BY ZIP IN CHUNKS
# ============================================================

def classify_violent(desc_series):
    """
    Crude but practical text-based classification.
    This is not a legal taxonomy. It is just a usable modeling feature.
    """
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


def aggregate_crime_by_zip():
    print("Aggregating CrimeData.csv by ZIP in chunks...")
    usecols = ["ZIP_CODE", "Crm Cd Desc", "Part 1-2"]

    agg = {}

    chunk_num = 0
    for chunk in pd.read_csv(CRIME_FILE, usecols=usecols, chunksize=CRIME_CHUNKSIZE):
        chunk_num += 1
        print(f"  Processing crime chunk {chunk_num}...")

        chunk["ZIP_CODE"] = standardize_zip(chunk["ZIP_CODE"])
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

    crime_by_zip = pd.DataFrame.from_dict(agg, orient="index").reset_index()
    crime_by_zip = crime_by_zip.rename(columns={"index": "ZIP_CODE"})
    crime_by_zip["ZIP_CODE"] = crime_by_zip["ZIP_CODE"].astype("Int64")

    print(f"Crime ZIP rows aggregated: {len(crime_by_zip):,}")
    return crime_by_zip


# ============================================================
# MERGE EVERYTHING
# ============================================================

def build_model_dataset():
    airbnb = load_airbnb()
    zip_features = load_cleaned_zip_features()
    crime_by_zip = aggregate_crime_by_zip()

    # Merge ZIP-level features + crime aggregates
    zip_all = zip_features.merge(crime_by_zip, on="ZIP_CODE", how="left")

    # Build per-capita crime features from chunked crime data if population exists
    if "Population" in zip_all.columns:
        pop = zip_all["Population"].replace(0, np.nan)

        if "crime_events" in zip_all.columns:
            zip_all["crime_events_per_capita"] = zip_all["crime_events"] / pop
        if "violent_crime_events" in zip_all.columns:
            zip_all["violent_crime_per_capita"] = zip_all["violent_crime_events"] / pop
        if "property_crime_events" in zip_all.columns:
            zip_all["property_crime_per_capita"] = zip_all["property_crime_events"] / pop
        if "part1_events" in zip_all.columns:
            zip_all["part1_events_per_capita"] = zip_all["part1_events"] / pop

    # Merge onto each Airbnb listing by ZIP_CODE
    df = airbnb.merge(zip_all, on="ZIP_CODE", how="left")

    # Keep only rows with ZIP features present
    before = len(df)
    df = df.dropna(subset=["ZIP_CODE"]).copy()
    after_zip = len(df)

    print(f"Merged model rows: {after_zip:,} (from {before:,})")

    # Fill listing-level missing values where sensible
    listing_fill_zero = [
        "reviews_per_month",
        "number_of_reviews",
        "number_of_reviews_ltm",
    ]
    for col in listing_fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # For ZIP-level predictors, use median fill so model doesn't die on sparse ZIPs
    for col in df.columns:
        if col not in ["room_type", "month", "log_price", "price"] and pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med)

    # Save merged dataset
    df.to_csv(OUTPUT_DIR / "merged_airbnb_zip_dataset.csv", index=False)
    zip_all.to_csv(OUTPUT_DIR / "crime_zip_aggregates_and_features.csv", index=False)

    return df


# ============================================================
# MODEL
# ============================================================

def run_regression(df):
    """
    ZIP_CODE is NOT included as a numeric predictor.
    It is only the join key that attached area features to each listing.

    We also do NOT include C(ZIP_CODE) fixed effects here because that would
    absorb the ZIP-level variation you are explicitly trying to estimate.
    """

    candidate_predictors = [
        # Listing-level controls
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "number_of_reviews_ltm",

        # ZIP-level base counts / rates
        "Count_Part_1_Crime",
        "Count_Public_Safety",
        "Count_Fire_Stations",
        "Count_Police_Stations",
        "Count_Corrective_Facility",
        "Count_Education_Facility",
        "Population",
        "Crime_Per_Capita",
        "Education_Per_Capita",
        "Public_Safety_Per_Capita",

        # Constructed indices
        "amenities_index",
        "crime_index",
        "education_index",

        # Chunked crime aggregates
        "crime_events",
        "violent_crime_events",
        "property_crime_events",
        "part1_events",
        "crime_events_per_capita",
        "violent_crime_per_capita",
        "property_crime_per_capita",
        "part1_events_per_capita",
    ]

    existing_predictors = [c for c in candidate_predictors if c in df.columns]

    # Remove highly redundant raw totals if per-capita versions exist,
    # to reduce multicollinearity.
    redundant_pairs = [
        ("crime_events", "crime_events_per_capita"),
        ("violent_crime_events", "violent_crime_per_capita"),
        ("property_crime_events", "property_crime_per_capita"),
        ("part1_events", "part1_events_per_capita"),
        ("Count_Part_1_Crime", "Crime_Per_Capita"),
    ]
    for raw_col, rate_col in redundant_pairs:
        if raw_col in existing_predictors and rate_col in existing_predictors:
            existing_predictors.remove(raw_col)

    # Formula
    rhs = existing_predictors + ["C(room_type)", "C(month)"]
    formula = "log_price ~ " + " + ".join(rhs)

    print("\nRunning regression with formula:\n")
    print(formula)
    print()

    model_df = df[["log_price", "price", "room_type", "month"] + existing_predictors].dropna().copy()
    print(f"Rows used in regression: {len(model_df):,}")

    model = smf.ols(formula=formula, data=model_df).fit(cov_type="HC3")
    return model, model_df, existing_predictors


# ============================================================
# OUTPUTS
# ============================================================

def save_outputs(model, model_df, predictors):
    # Summary text
    summary_path = OUTPUT_DIR / "regression_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.summary()))

    # Coefficients table
    coef_df = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df.sort_values("abs_coef", ascending=False).to_csv(
        OUTPUT_DIR / "regression_coefficients.csv", index=False
    )

    # Predictions
    model_df["predicted_log_price"] = model.predict(model_df)
    model_df["predicted_price"] = np.exp(model_df["predicted_log_price"])
    model_df.to_csv(OUTPUT_DIR / "model_predictions.csv", index=False)

    # Actual vs predicted plot
    plt.figure(figsize=(8, 8))
    plt.scatter(model_df["price"], model_df["predicted_price"], alpha=0.2)
    mn = min(model_df["price"].min(), model_df["predicted_price"].min())
    mx = max(model_df["price"].max(), model_df["predicted_price"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Actual vs Predicted Airbnb Price")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted_price.png", dpi=150)
    plt.close()

    # Binned trend plots
    y_col = "price"
    for feature in [
        "Crime_Per_Capita",
        "Education_Per_Capita",
        "Public_Safety_Per_Capita",
        "crime_events_per_capita",
        "violent_crime_per_capita",
        "property_crime_per_capita",
        "amenities_index",
        "education_index",
        "crime_index",
    ]:
        if feature in model_df.columns:
            plot_binned_relationship(
                model_df,
                feature,
                y_col,
                OUTPUT_DIR / f"trend_{feature}.png"
            )

    # Short plain-English interpretation file
    lines = []
    lines.append("Regression interpretation notes")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Dependent variable: log_price")
    lines.append("That means small coefficients can be read approximately as percentage effects.")
    lines.append("")
    lines.append("Important methodological note:")
    lines.append("ZIP_CODE was used only to merge listing rows to area-level variables.")
    lines.append("ZIP_CODE was NOT entered as a numeric regression predictor.")
    lines.append("That is the correct way to use ZIP here.")
    lines.append("")
    lines.append(f"R-squared: {model.rsquared:.4f}")
    lines.append(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    lines.append("")
    lines.append("Top statistically significant predictors (p < 0.05):")

    sig = coef_df[(coef_df["p_value"] < 0.05) & (~coef_df["term"].str.contains("Intercept", na=False))].copy()
    sig = sig.sort_values("abs_coef", ascending=False).head(20)

    if sig.empty:
        lines.append("No predictors met the p < 0.05 threshold.")
    else:
        for _, row in sig.iterrows():
            lines.append(
                f"- {row['term']}: coef={row['coef']:.6f}, p={row['p_value']:.6g}"
            )

    with open(OUTPUT_DIR / "interpretation_notes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\nSaved outputs to:", OUTPUT_DIR.resolve())
    print(f"  - {summary_path.name}")
    print("  - regression_coefficients.csv")
    print("  - model_predictions.csv")
    print("  - actual_vs_predicted_price.png")
    print("  - interpretation_notes.txt")
    print("  - merged_airbnb_zip_dataset.csv")
    print("  - crime_zip_aggregates_and_features.csv")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting Airbnb ZIP regression build...\n")
    df = build_model_dataset()
    model, model_df, predictors = run_regression(df)
    save_outputs(model, model_df, predictors)
    print("\nDone.")


if __name__ == "__main__":
    main()
