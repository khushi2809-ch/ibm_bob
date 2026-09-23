"""
data_loader.py
==============
Loads and validates the raw Food Delivery dataset.

Responsibilities:
  - Read the CSV file with correct dtypes
  - Parse the Order_Date column
  - Assert that all 24 expected columns are present
  - Log basic dataset statistics (rows, columns, memory)
  - Return a clean, validated DataFrame ready for further processing
"""

import os
import pandas as pd


# ── Expected column names (24 columns confirmed from dataset) ───────────────
EXPECTED_COLUMNS = [
    "Order_ID",
    "Order_Date",
    "Order_Hour",
    "Day_of_Week",
    "Is_Weekend",
    "Is_Festival",
    "Weather",
    "Pickup_Zone",
    "Dropoff_Zone",
    "Vehicle_Type",
    "Rider_Experience_Years",
    "Rider_Rating",
    "Restaurant_Rating",
    "Cuisine_Type",
    "Order_Items",
    "Restaurant_Load",
    "Preparation_Time_Min",
    "Road_Distance_km",
    "Delivery_Distance_Category",
    "Traffic_Level",
    "Number_of_Signals",
    "Average_Speed_kmph",
    "Delivery_Priority",
    "Time_taken_min",
]

# ── Explicit dtype hints for string/category columns ────────────────────────
DTYPE_MAP = {
    "Order_ID": str,
    "Order_Hour": "Int64",
    "Day_of_Week": str,
    "Is_Weekend": "Int64",
    "Is_Festival": "Int64",
    "Weather": str,
    "Pickup_Zone": str,
    "Dropoff_Zone": str,
    "Vehicle_Type": str,
    "Cuisine_Type": str,
    "Restaurant_Load": str,
    "Delivery_Distance_Category": str,
    "Traffic_Level": str,
    "Delivery_Priority": str,
    "Order_Items": "Int64",
    "Number_of_Signals": "Int64",
    "Preparation_Time_Min": "Int64",
    "Time_taken_min": "Int64",
    "Rider_Experience_Years": float,
    "Rider_Rating": float,
    "Restaurant_Rating": float,
    "Road_Distance_km": float,
    "Average_Speed_kmph": float,
}


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the raw CSV dataset, validate its structure, and return a DataFrame.

    Parameters
    ----------
    filepath : str, optional
        Absolute or relative path to the CSV file.
        Defaults to  data/raw/Food_Delivery_Time_Prediction.csv
        relative to this file's parent directory.

    Returns
    -------
    pd.DataFrame
        Validated raw DataFrame with Order_Date parsed as datetime.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the given path.
    ValueError
        If any of the 24 expected columns are missing from the file.
    """
    # ── Resolve default path ─────────────────────────────────────────────────
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "raw", "Food_Delivery_Time_Prediction.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    # ── Read CSV ─────────────────────────────────────────────────────────────
    df = pd.read_csv(
        filepath,
        dtype=DTYPE_MAP,
        parse_dates=["Order_Date"],
        low_memory=False,
    )

    # ── Column validation ────────────────────────────────────────────────────
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing {len(missing_cols)} expected column(s): {missing_cols}"
        )

    # ── Basic statistics ─────────────────────────────────────────────────────
    mem_mb = df.memory_usage(deep=True).sum() / 1_048_576
    print(f"[data_loader] Loaded: {len(df):,} rows x {len(df.columns)} columns")
    print(f"[data_loader] Memory usage: {mem_mb:.2f} MB")
    print(f"[data_loader] Date range: {df['Order_Date'].min().date()} to {df['Order_Date'].max().date()}")

    return df
