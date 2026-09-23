"""
cleaner.py
==========
Data cleaning pipeline for the Food Delivery dataset.

Steps performed:
  1. Remove duplicate Order_IDs
  2. Audit and report missing values
  3. Enforce correct data types
  4. Validate numeric value ranges and flag invalid rows
  5. Cap outliers using IQR method (cap, do not drop)
  6. Standardise string/categorical columns
  7. Validate date range
  8. Verify binary column values (Is_Weekend, Is_Festival)
  9. Write a data quality report to data/processed/quality_report.txt
"""

import os
import pandas as pd
import numpy as np


# ── Columns that must not be null after cleaning ────────────────────────────
CRITICAL_COLUMNS = [
    "Order_ID", "Order_Date", "Order_Hour", "Time_taken_min",
    "Road_Distance_km", "Preparation_Time_Min", "Average_Speed_kmph",
    "Rider_Rating", "Restaurant_Rating",
]

# ── Numeric columns to cap outliers on ─────────────────────────────────────
OUTLIER_COLUMNS = [
    "Time_taken_min",
    "Road_Distance_km",
    "Average_Speed_kmph",
    "Preparation_Time_Min",
]

# ── String columns to standardise ──────────────────────────────────────────
STRING_COLUMNS = [
    "Day_of_Week", "Weather", "Pickup_Zone", "Dropoff_Zone",
    "Vehicle_Type", "Cuisine_Type", "Restaurant_Load",
    "Delivery_Distance_Category", "Traffic_Level", "Delivery_Priority",
]


def _report_missing(df: pd.DataFrame) -> dict:
    """Return a dict of {column: (count, pct)} for columns with any nulls."""
    result = {}
    for col in df.columns:
        n_null = df[col].isna().sum()
        if n_null > 0:
            result[col] = (int(n_null), round(n_null / len(df) * 100, 2))
    return result


def _cap_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Cap values in `col` at [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    Values are capped (winsorised), NOT removed.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before = df[col].copy()
    df[col] = df[col].clip(lower=lower, upper=upper)
    n_capped = (before != df[col]).sum()
    print(f"  [outlier] {col}: capped {n_capped} values -> [{lower:.2f}, {upper:.2f}]")
    return df


def clean_data(df: pd.DataFrame, output_dir: str = None) -> pd.DataFrame:
    """
    Apply the full cleaning pipeline to the raw DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from data_loader.load_data().
    output_dir : str, optional
        Directory where quality_report.txt and cleaned_data.csv will be saved.
        Defaults to  data/processed/ relative to the project root.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    report_lines = []
    df = df.copy()
    original_rows = len(df)

    def log(msg: str):
        print(f"  [cleaner] {msg}")
        report_lines.append(msg)

    log(f"Starting cleaning - {original_rows:,} rows x {len(df.columns)} columns")

    # ── Step 1: Remove duplicate Order_IDs ──────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset="Order_ID")
    dropped = before - len(df)
    log(f"Step 1 - Duplicate removal: removed {dropped} duplicate Order_IDs")

    # ── Step 2: Missing value audit ─────────────────────────────────────────
    missing = _report_missing(df)
    if missing:
        log("Step 2 - Missing values detected:")
        for col, (cnt, pct) in missing.items():
            log(f"  {col}: {cnt} nulls ({pct}%)")
            # Impute only if <5% missing
            if pct < 5.0:
                if df[col].dtype in [np.float64, float] or str(df[col].dtype) in ["Int64", "int64"]:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    log(f"    -> Imputed {col} with median={median_val:.2f}")
                else:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    log(f"    -> Imputed {col} with mode='{mode_val}'")
            else:
                log(f"    -> {col} has >{pct}% missing - rows dropped")
                df = df.dropna(subset=[col])
    else:
        log("Step 2 - No missing values detected")

    # ── Step 3: Type enforcement ────────────────────────────────────────────
    int_cols = ["Order_Hour", "Order_Items", "Number_of_Signals", "Preparation_Time_Min", "Time_taken_min", "Is_Weekend", "Is_Festival"]
    float_cols = ["Rider_Experience_Years", "Rider_Rating", "Restaurant_Rating", "Road_Distance_km", "Average_Speed_kmph"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    log("Step 3 - Type enforcement applied")

    # ── Step 4: Range validation ─────────────────────────────────────────────
    invalid_masks = {
        "Time_taken_min < 1": df["Time_taken_min"] < 1,
        "Road_Distance_km <= 0": df["Road_Distance_km"] <= 0,
        "Average_Speed_kmph <= 0": df["Average_Speed_kmph"] <= 0,
        "Rider_Rating outside [1,5]": ~df["Rider_Rating"].between(1, 5),
        "Restaurant_Rating outside [1,5]": ~df["Restaurant_Rating"].between(1, 5),
        "Order_Hour outside [0,23]": ~df["Order_Hour"].between(0, 23),
        "Preparation_Time_Min < 0": df["Preparation_Time_Min"] < 0,
    }
    total_invalid = 0
    for reason, mask in invalid_masks.items():
        count = int(mask.sum())
        if count > 0:
            log(f"  Range check '{reason}': {count} rows -> dropped")
            df = df[~mask]
            total_invalid += count
    log(f"Step 4 - Range validation: dropped {total_invalid} invalid rows")

    # ── Step 5: Outlier capping (IQR) ────────────────────────────────────────
    log("Step 5 - Outlier capping (IQR 1.5x fence):")
    for col in OUTLIER_COLUMNS:
        if col in df.columns:
            df = _cap_outliers_iqr(df, col)

    # ── Step 6: Standardise string columns ──────────────────────────────────
    for col in STRING_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    log("Step 6 - String columns standardised (strip + title case)")

    # ── Step 7: Date integrity check ─────────────────────────────────────────
    bad_dates = (~df["Order_Date"].between(
        pd.Timestamp("2020-01-01"), pd.Timestamp("2030-12-31")
    )).sum()
    if bad_dates > 0:
        log(f"Step 7 - Dropped {bad_dates} rows with out-of-range Order_Date")
        df = df[df["Order_Date"].between(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2030-12-31")
        )]
    else:
        log("Step 7 - Date integrity: all dates in valid range")

    # ── Step 8: Binary column check ──────────────────────────────────────────
    for col in ["Is_Weekend", "Is_Festival"]:
        if col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            if not unique_vals.issubset({0, 1}):
                log(f"  WARNING: {col} contains unexpected values: {unique_vals}")
            else:
                log(f"Step 8 - {col}: only {{0, 1}} values - OK")

    # ── Final summary ────────────────────────────────────────────────────────
    final_rows = len(df)
    rows_removed = original_rows - final_rows
    log(f"Cleaning complete - {final_rows:,} rows remaining ({rows_removed} removed, {rows_removed/original_rows*100:.1f}%)")

    # ── Write quality report ──────────────────────────────────────────────────
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "quality_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATA QUALITY REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: Food Delivery Time Prediction\n")
        f.write(f"Original rows: {original_rows:,}\n")
        f.write(f"Final rows:    {final_rows:,}\n")
        f.write(f"Rows removed:  {rows_removed:,}\n\n")
        f.write("CLEANING LOG:\n")
        f.write("-" * 60 + "\n")
        for line in report_lines:
            f.write(line + "\n")
    print(f"  [cleaner] Quality report saved -> {report_path}")

    # ── Save cleaned CSV ──────────────────────────────────────────────────────
    cleaned_path = os.path.join(output_dir, "cleaned_data.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"  [cleaner] Cleaned data saved -> {cleaned_path}")

    return df
