"""
collinearity_diagnostics.py

Run this on your CleanedDataJune.csv to identify collinearity.

Outputs:
    - correlation_matrix.csv         (all pairwise Pearson correlations)
    - high_correlations.csv          (pairs with |r| > 0.8)
    - vif_table.csv                  (VIF per predictor, sorted)
    - collinearity_report.txt        (human-readable summary)

Install:
    pip install pandas numpy statsmodels scikit-learn
"""

from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("NewData/CleanedDataJune.csv")  # adjust if needed
OUTPUT_DIR = Path("collinearity_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# These should match the predictors used in your regression script.
# Edit this list to match exactly what you feed into OLS.
PREDICTORS = [
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
    "LAX_Median_Distance_June_(KM)",
    "Hollywood_Walk_of_Fame_Median_Distance_June_(KM)",
    "Griffith_Park_Median_Distance_June_(KM)",
    "Walt_Disney_Concert_Hall_Median_Distance_June_(KM)",
    "TCL_Chinese_Theater_Median_Distance_June_(KM)",
    "Madame_Tussauds_Median_Distance_June_(KM)",
    "Venice_Beach_Median_Distance_June_(KM)",
    "Crypto_Arena_Median_Distance_June_(KM)",
    "California_Science_Center_Median_Distance_June_(KM)",
    "Hollywood_Sign_Median_Distance_June_(KM)",
    "La_Brea_Median_Distance_June_(KM)",
    "Getty_Center_Median_Distance_June_(KM)",
    "Griffith_Observatory_Median_Distance_June_(KM)",
    "Universal_Studios_Hollywood_Median_Distance_June_(KM)",
    "Aquarium_of_the_Pacific_Median_Distance_June_(KM)",
    "Santa_Monica_Pier_Median_Distance_June_(KM)",
    "Queen_Mary_Median_Distance_June_(KM)",
]

# Thresholds
HIGH_CORR_THRESHOLD = 0.8
VIF_WARNING = 5.0
VIF_SEVERE = 10.0


# ============================================================
# FEATURE ENGINEERING (mirror your regression script)
# ============================================================

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the engineered features from your regression script."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "ZIP_Code" in df.columns:
        df = df.rename(columns={"ZIP_Code": "ZIP_CODE"})

    for col in df.columns:
        if col != "ZIP_CODE":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Median_Price_June"]).copy()
    df = df[(df["Median_Price_June"] > 0) & (df["Median_Price_June"] < 5000)].copy()

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

    parts = []
    for c in ["Count_Educational_Facility", "Count_Public_Safety",
              "Count_Fire_Stations", "Count_Police_Stations"]:
        if c in df.columns:
            parts.append(zscore(df[c]))
    if parts:
        df["external_amenities_index"] = sum(parts) / len(parts)

    distance_cols = [c for c in df.columns
                     if ("Distance" in c or "distance" in c) and c != "ZIP_CODE"]
    if distance_cols:
        zs = [zscore(df[c]) for c in distance_cols]
        df["avg_landmark_distance_z"] = sum(zs) / len(zs)
        df["centrality_index"] = -df["avg_landmark_distance_z"]

    return df


# ============================================================
# DIAGNOSTICS
# ============================================================

def correlation_diagnostics(X: pd.DataFrame):
    corr = X.corr(method="pearson")
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Extract upper triangle pairs with |r| > threshold
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= HIGH_CORR_THRESHOLD:
                pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "pearson_r": r,
                    "abs_r": abs(r),
                })
    high_corr = pd.DataFrame(pairs).sort_values("abs_r", ascending=False)
    high_corr.to_csv(OUTPUT_DIR / "high_correlations.csv", index=False)
    return corr, high_corr


def vif_diagnostics(X: pd.DataFrame):
    # Drop constant columns first; VIF is undefined for them
    X = X.loc[:, X.nunique(dropna=True) > 1].copy()
    X = X.fillna(X.median(numeric_only=True))

    # Standardize so VIF is scale-invariant in interpretation
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    Xs = Xs.assign(_intercept=1.0)  # statsmodels VIF expects an intercept column

    rows = []
    for i, col in enumerate(Xs.columns):
        if col == "_intercept":
            continue
        try:
            v = variance_inflation_factor(Xs.values, i)
        except Exception:
            v = np.inf
        rows.append({"feature": col, "VIF": v})

    vif_df = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(OUTPUT_DIR / "vif_table.csv", index=False)
    return vif_df


def write_report(high_corr: pd.DataFrame, vif_df: pd.DataFrame):
    lines = []
    lines.append("Collinearity diagnostic report")
    lines.append("=" * 70)
    lines.append("")

    lines.append(f"Pairs with |Pearson r| >= {HIGH_CORR_THRESHOLD}: {len(high_corr)}")
    if len(high_corr) > 0:
        lines.append("")
        for _, row in high_corr.head(40).iterrows():
            lines.append(f"  r = {row['pearson_r']:+.4f}   "
                         f"{row['feature_1']}   <->   {row['feature_2']}")
    lines.append("")

    severe = vif_df[vif_df["VIF"] > VIF_SEVERE]
    warn = vif_df[(vif_df["VIF"] > VIF_WARNING) & (vif_df["VIF"] <= VIF_SEVERE)]

    lines.append(f"Features with VIF > {VIF_SEVERE} (severe): {len(severe)}")
    for _, row in severe.iterrows():
        lines.append(f"  VIF = {row['VIF']:>12.2f}   {row['feature']}")
    lines.append("")
    lines.append(f"Features with {VIF_WARNING} < VIF <= {VIF_SEVERE} (warning): {len(warn)}")
    for _, row in warn.iterrows():
        lines.append(f"  VIF = {row['VIF']:>12.2f}   {row['feature']}")
    lines.append("")

    lines.append("Suggested action ordering:")
    lines.append("  1. Inspect any pairs with |r| > 0.99 -- these are likely structural duplicates.")
    lines.append("  2. For each VIF > 10 feature, decide whether to drop it (prefer raw")
    lines.append("     over derived; prefer interpretable units).")
    lines.append("  3. Re-run VIF after each removal -- they update non-trivially.")
    lines.append("  4. Stop when all remaining VIFs are below 5 (or below 10 if you")
    lines.append("     are willing to use Ridge).")

    (OUTPUT_DIR / "collinearity_report.txt").write_text("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main():
    df_raw = pd.read_csv(DATA_FILE)
    df = build_features(df_raw)

    available = [c for c in PREDICTORS if c in df.columns]
    missing = [c for c in PREDICTORS if c not in df.columns]
    if missing:
        print("Warning: predictors not found in data and skipped:")
        for m in missing:
            print(f"  - {m}")

    X = df[available].copy()
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    X = X.fillna(X.median(numeric_only=True))

    print(f"Running diagnostics on {len(X)} rows x {X.shape[1]} predictors")

    corr, high_corr = correlation_diagnostics(X)
    vif_df = vif_diagnostics(X)
    write_report(high_corr, vif_df)

    print(f"\nDone. See: {OUTPUT_DIR.resolve()}")
    print(f"  - correlation_matrix.csv  (full pairwise correlations)")
    print(f"  - high_correlations.csv   ({len(high_corr)} suspect pairs)")
    print(f"  - vif_table.csv           (VIF per predictor)")
    print(f"  - collinearity_report.txt (human-readable summary)")


if __name__ == "__main__":
    main()
