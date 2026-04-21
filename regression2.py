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

BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "model_outputs_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

AIRBNB_JUNE_FILE = BASE_DIR / "AirBNB_June.csv"
AIRBNB_SEPT_FILE = BASE_DIR / "AirBNB_September.csv"
CRIME_FILE = BASE_DIR / "CrimeData.csv"

if (BASE_DIR / "CleanedData.csv").exists():
    CLEANED_FILE = BASE_DIR / "CleanedData.csv"
elif (BASE_DIR / "Cleaned_Data.csv").exists():
    CLEANED_FILE = BASE_DIR / "Cleaned_Data.csv"
else:
    raise FileNotFoundError(
        "Could not find CleanedData.csv or Cleaned_Data.csv in the current folder."
    )

CRIME_CHUNKSIZE = 200_000

MIN_PRICE = 10
MAX_PRICE = 5000


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def standardize_zip(series):
    """
    ZIP is a join key only, not a regression variable.
    """
    s = safe_numeric(series)
    s = s.where((s >= 10000) & (s <= 99999))
    return s.astype("Int64")


def zscore(series):
    s = pd.to_numeric(series, errors="coerce")
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def pct_effect_from_log_coef(beta):
    return (np.exp(beta) - 1) * 100


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_actual_vs_predicted(df, output_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(df["price"], df["predicted_price"], alpha=0.15)
    mn = min(df["price"].min(), df["predicted_price"].min())
    mx = max(df["price"].max(), df["predicted_price"].max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.title("Actual vs Predicted Airbnb Price")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_binned_means(df, x_col, y_col, output_path, q=12):
    d = df[[x_col, y_col]].dropna().copy()
    if len(d) < 20:
        return

    # Avoid pathological plots from too few unique values
    nunique = d[x_col].nunique()
    if nunique < 4:
        return

    q = min(q, nunique)

    try:
        d["bin"] = pd.qcut(d[x_col], q=q, duplicates="drop")
    except Exception:
        return

    grouped = d.groupby("bin", observed=False).agg(
        x_mean=(x_col, "mean"),
        y_mean=(y_col, "mean"),
        n=(y_col, "size"),
    ).dropna()

    if len(grouped) < 4:
        return

    plt.figure(figsize=(9, 6))
    plt.plot(grouped["x_mean"], grouped["y_mean"], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col} (binned means)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================================================
# LOAD AIRBNB
# ============================================================

def load_airbnb():
    june = pd.read_csv(AIRBNB_JUNE_FILE)
    sept = pd.read_csv(AIRBNB_SEPT_FILE)

    june["month"] = "June"
    sept["month"] = "September"

    june.columns = [c.strip() for c in june.columns]
    sept.columns = [c.strip() for c in sept.columns]

    airbnb = pd.concat([june, sept], ignore_index=True)

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
    missing = [c for c in needed_cols if c not in airbnb.columns]
    if missing:
        raise ValueError(f"Missing expected Airbnb columns: {missing}")

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
        airbnb[col] = safe_numeric(airbnb[col])

    airbnb["ZIP_CODE"] = standardize_zip(airbnb["ZIP_CODE"])

    raw_rows = len(airbnb)
    airbnb = airbnb.dropna(subset=["price", "ZIP_CODE", "room_type", "month"]).copy()
    after_required = len(airbnb)

    airbnb = airbnb[(airbnb["price"] >= MIN_PRICE) & (airbnb["price"] <= MAX_PRICE)].copy()
    after_price_trim = len(airbnb)

    airbnb["reviews_per_month"] = airbnb["reviews_per_month"].fillna(0)
    airbnb["number_of_reviews"] = airbnb["number_of_reviews"].fillna(0)
    airbnb["number_of_reviews_ltm"] = airbnb["number_of_reviews_ltm"].fillna(0)

    airbnb["log_price"] = np.log(airbnb["price"])

    diagnostics = {
        "airbnb_raw_rows": raw_rows,
        "airbnb_after_required_filters": after_required,
        "airbnb_after_price_trim": after_price_trim,
        "airbnb_unique_zips": int(airbnb["ZIP_CODE"].nunique()),
    }
    return airbnb, diagnostics


# ============================================================
# LOAD CLEANED ZIP DATA
# ============================================================

def load_cleaned_zip_features():
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

    # Keep a tight set
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

    # Build composite indices
    amenity_components = []
    for c in ["Count_Fire_Stations", "Count_Police_Stations", "Count_Education_Facility"]:
        if c in cleaned.columns:
            amenity_components.append(zscore(cleaned[c]))

    if amenity_components:
        cleaned["amenities_index"] = sum(amenity_components) / len(amenity_components)

    if "Crime_Per_Capita" in cleaned.columns:
        cleaned["crime_index"] = zscore(cleaned["Crime_Per_Capita"])

    if "Education_Per_Capita" in cleaned.columns:
        cleaned["education_index"] = zscore(cleaned["Education_Per_Capita"])

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


def aggregate_crime_by_zip():
    usecols = ["ZIP_CODE", "Crm Cd Desc", "Part 1-2"]

    total_rows_read = 0
    total_rows_valid_zip = 0
    total_part1_flag_rows = 0

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

        total_part1_flag_rows += int(chunk["part1_flag"].sum())

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

    crime_by_zip = pd.DataFrame.from_dict(agg, orient="index").reset_index()
    crime_by_zip = crime_by_zip.rename(columns={"index": "ZIP_CODE"})
    crime_by_zip["ZIP_CODE"] = crime_by_zip["ZIP_CODE"].astype("Int64")

    diagnostics = {
        "crime_chunks_processed": chunk_count,
        "crime_total_rows_read": total_rows_read,
        "crime_rows_with_valid_zip": total_rows_valid_zip,
        "crime_aggregated_total_events": int(crime_by_zip["crime_events"].sum()) if len(crime_by_zip) else 0,
        "crime_total_part1_rows": total_part1_flag_rows,
        "crime_unique_zip_after_aggregation": int(crime_by_zip["ZIP_CODE"].nunique()) if len(crime_by_zip) else 0,
    }

    return crime_by_zip, diagnostics


# ============================================================
# BUILD MODEL DATASET
# ============================================================

def build_model_dataset():
    airbnb, d_airbnb = load_airbnb()
    cleaned, d_cleaned = load_cleaned_zip_features()
    crime_zip, d_crime = aggregate_crime_by_zip()

    zip_all = cleaned.merge(crime_zip, on="ZIP_CODE", how="left")

    if "Population" in zip_all.columns:
        pop = zip_all["Population"].replace(0, np.nan)

        if "crime_events" in zip_all.columns:
            zip_all["crime_events_per_capita"] = zip_all["crime_events"] / pop
        if "violent_crime_events" in zip_all.columns:
            zip_all["violent_crime_per_capita"] = zip_all["violent_crime_events"] / pop
        if "property_crime_events" in zip_all.columns:
            zip_all["property_crime_per_capita"] = zip_all["property_crime_events"] / pop

    # Safer fill for ZIP-level features after merge
    for col in zip_all.columns:
        if col != "ZIP_CODE" and pd.api.types.is_numeric_dtype(zip_all[col]):
            zip_all[col] = zip_all[col].fillna(zip_all[col].median())

    merged = airbnb.merge(zip_all, on="ZIP_CODE", how="left")
    merged_rows_before_drop = len(merged)

    # Require the core ZIP variables to exist after merge
    core_zip_vars = []
    for c in ["amenities_index", "crime_events_per_capita", "Education_Per_Capita", "Public_Safety_Per_Capita"]:
        if c in merged.columns:
            core_zip_vars.append(c)

    if core_zip_vars:
        merged = merged.dropna(subset=core_zip_vars).copy()

    merged_rows_after_drop = len(merged)

    diagnostics = {}
    diagnostics.update(d_airbnb)
    diagnostics.update(d_cleaned)
    diagnostics.update(d_crime)
    diagnostics["zip_all_rows"] = len(zip_all)
    diagnostics["zip_all_unique_zips"] = int(zip_all["ZIP_CODE"].nunique())
    diagnostics["merged_rows_before_core_drop"] = merged_rows_before_drop
    diagnostics["merged_rows_after_core_drop"] = merged_rows_after_drop
    diagnostics["merged_unique_zips"] = int(merged["ZIP_CODE"].nunique())
    diagnostics["share_crime_rows_retained_after_zip_cleaning"] = (
        d_crime["crime_aggregated_total_events"] / d_crime["crime_total_rows_read"]
        if d_crime["crime_total_rows_read"] else np.nan
    )

    zip_all.to_csv(OUTPUT_DIR / "crime_zip_aggregates_and_features_v2.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "merged_airbnb_zip_dataset_v2.csv", index=False)

    return merged, diagnostics


# ============================================================
# MODEL
# ============================================================

def run_regression(df):
    """
    Tightened specification:
    - keep only one main crime measure
    - one main education measure
    - one amenities measure
    - one public-safety measure
    - listing-level controls
    """

    predictors = []

    # Listing controls
    for c in [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "number_of_reviews_ltm",
    ]:
        if c in df.columns:
            predictors.append(c)

    # ZIP-level concept variables: one main proxy per concept
    preferred_zip_vars = [
        "crime_events_per_capita",   # main crime burden
        "Education_Per_Capita",      # education access
        "Public_Safety_Per_Capita",  # public safety availability
        "amenities_index",           # composite amenities
    ]
    for c in preferred_zip_vars:
        if c in df.columns:
            predictors.append(c)

    formula = "log_price ~ " + " + ".join(predictors + ["C(room_type)", "C(month)"])

    model_cols = ["log_price", "price", "room_type", "month"] + predictors
    model_df = df[model_cols].dropna().copy()

    model = smf.ols(formula=formula, data=model_df).fit(cov_type="HC3")

    model_df["predicted_log_price"] = model.predict(model_df)
    model_df["predicted_price"] = np.exp(model_df["predicted_log_price"])

    return model, model_df, predictors, formula


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(model, model_df, predictors, formula, diagnostics):
    # Summary
    write_text(OUTPUT_DIR / "regression_summary_v2.txt", str(model.summary()))

    # Coefficients
    coef_df = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "std_err": model.bse.values,
        "z_value": model.tvalues.values,
        "p_value": model.pvalues.values,
    })
    coef_df["pct_effect_if_log_model"] = coef_df["coef"].apply(pct_effect_from_log_coef)
    coef_df.to_csv(OUTPUT_DIR / "regression_coefficients_v2.csv", index=False)

    # Predictions
    model_df.to_csv(OUTPUT_DIR / "model_predictions_v2.csv", index=False)

    # Diagnostics text
    lines = []
    lines.append("Diagnostics report")
    lines.append("=" * 60)
    lines.append("")
    for k, v in diagnostics.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("Regression formula:")
    lines.append(formula)
    lines.append("")
    lines.append(f"R-squared: {model.rsquared:.6f}")
    lines.append(f"Adjusted R-squared: {model.rsquared_adj:.6f}")
    lines.append(f"Observations used in regression: {len(model_df):,}")
    lines.append("")
    lines.append("Predictors included:")
    for p in predictors:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- ZIP_CODE was used only for merging, never as a numeric predictor.")
    lines.append("- This model uses a reduced set of ZIP-level variables to limit multicollinearity.")
    lines.append("- The crime file should be validated by comparing crime_total_rows_read to crime_aggregated_total_events.")
    lines.append("- If those two are close, nearly all valid crime rows were used after ZIP cleaning.")

    write_text(OUTPUT_DIR / "diagnostics_report_v2.txt", "\n".join(lines))

    # Simple interpretation file
    sig = coef_df[
        (coef_df["p_value"] < 0.05) &
        (~coef_df["term"].str.contains("Intercept", na=False))
    ].copy()

    sig = sig.reindex(sig["coef"].abs().sort_values(ascending=False).index)

    interp = []
    interp.append("Interpretation notes v2")
    interp.append("=" * 60)
    interp.append("")
    interp.append("Dependent variable: log_price")
    interp.append("Small coefficients can be read approximately as percent changes.")
    interp.append("")
    interp.append(f"R-squared: {model.rsquared:.4f}")
    interp.append("")
    interp.append("Significant predictors (p < 0.05):")
    if sig.empty:
        interp.append("None.")
    else:
        for _, row in sig.iterrows():
            interp.append(
                f"- {row['term']}: coef={row['coef']:.6f}, "
                f"approx_pct_effect={row['pct_effect_if_log_model']:.2f}%, "
                f"p={row['p_value']:.6g}"
            )

    write_text(OUTPUT_DIR / "interpretation_notes_v2.txt", "\n".join(interp))

    # Plots
    plot_actual_vs_predicted(model_df, OUTPUT_DIR / "actual_vs_predicted_price_v2.png")

    for feature in ["crime_events_per_capita", "Education_Per_Capita", "Public_Safety_Per_Capita", "amenities_index"]:
        if feature in model_df.columns:
            plot_binned_means(
                model_df,
                x_col=feature,
                y_col="price",
                output_path=OUTPUT_DIR / f"trend_{feature}_v2.png",
                q=12
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print("Starting v2 regression build...")
    df, diagnostics = build_model_dataset()
    print("Merged dataset ready.")
    print("Rows available after merge:", f"{len(df):,}")

    model, model_df, predictors, formula = run_regression(df)
    print("Regression finished.")
    print(f"Observations used in regression: {len(model_df):,}")
    print(f"R-squared: {model.rsquared:.4f}")

    save_outputs(model, model_df, predictors, formula, diagnostics)
    print(f"Done. Outputs saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
