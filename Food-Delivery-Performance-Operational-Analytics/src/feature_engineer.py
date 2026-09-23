"""
feature_engineer.py
===================
Creates derived columns used for analytics and visualisation.

New columns added:
  - Month               : integer month from Order_Date
  - Week_Number         : ISO week number from Order_Date
  - Time_of_Day         : binned from Order_Hour (Morning / Afternoon / Evening / Night / Late Night)
  - Prep_Category       : binned Preparation_Time_Min (Fast / Moderate / Slow)
  - Experience_Category : binned Rider_Experience_Years (Novice / Junior / Senior)
  - Speed_Category      : binned Average_Speed_kmph (Slow / Moderate / Fast)
  - Signals_Category    : binned Number_of_Signals (Low / Medium / High)
  - Day_Order           : integer sort key for Day_of_Week (Mon=0 → Sun=6)
  - Pickup_Dropoff_Pair : "Pickup_Zone → Dropoff_Zone" string
"""

import pandas as pd
import numpy as np


# ── Ordered day mapping ──────────────────────────────────────────────────────
DAY_ORDER_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered columns to the cleaned DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from cleaner.clean_data().

    Returns
    -------
    pd.DataFrame
        DataFrame with all new feature columns appended.
    """
    df = df.copy()

    # ── Month (1–12) ─────────────────────────────────────────────────────────
    df["Month"] = df["Order_Date"].dt.month

    # ── Week number (ISO, 1–53) ───────────────────────────────────────────────
    df["Week_Number"] = df["Order_Date"].dt.isocalendar().week.astype(int)

    # ── Time of Day ──────────────────────────────────────────────────────────
    # Bins:  Late Night (0–5), Morning (6–11), Afternoon (12–16),
    #        Evening (17–20), Night (21–23)
    hour = df["Order_Hour"].astype(float)
    conditions = [
        hour.between(6, 11),
        hour.between(12, 16),
        hour.between(17, 20),
        hour.between(21, 23),
        hour.between(0, 5),
    ]
    choices = ["Morning", "Afternoon", "Evening", "Night", "Late Night"]
    df["Time_of_Day"] = np.select(conditions, choices, default="Unknown")

    # ── Preparation Category ─────────────────────────────────────────────────
    prep = df["Preparation_Time_Min"].astype(float)
    df["Prep_Category"] = pd.cut(
        prep,
        bins=[-np.inf, 10, 20, np.inf],
        labels=["Fast", "Moderate", "Slow"],
        right=True,
    ).astype(str)

    # ── Rider Experience Category ─────────────────────────────────────────────
    exp = df["Rider_Experience_Years"].astype(float)
    df["Experience_Category"] = pd.cut(
        exp,
        bins=[-np.inf, 1, 3, np.inf],
        labels=["Novice", "Junior", "Senior"],
        right=True,
    ).astype(str)

    # ── Speed Category ───────────────────────────────────────────────────────
    speed = df["Average_Speed_kmph"].astype(float)
    df["Speed_Category"] = pd.cut(
        speed,
        bins=[-np.inf, 20, 35, np.inf],
        labels=["Slow", "Moderate", "Fast"],
        right=True,
    ).astype(str)

    # ── Signals Category ─────────────────────────────────────────────────────
    signals = df["Number_of_Signals"].astype(float)
    df["Signals_Category"] = pd.cut(
        signals,
        bins=[-np.inf, 5, 15, np.inf],
        labels=["Low", "Medium", "High"],
        right=True,
    ).astype(str)

    # ── Day sort order ────────────────────────────────────────────────────────
    # Title-case the Day_of_Week before mapping (cleaner already does this,
    # but we guard here for safety)
    df["Day_Order"] = (
        df["Day_of_Week"]
        .str.strip()
        .str.title()
        .map(DAY_ORDER_MAP)
        .fillna(-1)
        .astype(int)
    )

    # ── Pickup → Dropoff pair ─────────────────────────────────────────────────
    df["Pickup_Dropoff_Pair"] = df["Pickup_Zone"].str.strip() + " → " + df["Dropoff_Zone"].str.strip()

    print(f"[feature_engineer] Added 9 new feature columns. DataFrame now has {len(df.columns)} columns.")
    return df
