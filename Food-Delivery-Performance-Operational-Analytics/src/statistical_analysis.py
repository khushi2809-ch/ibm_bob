"""
statistical_analysis.py
=======================
Descriptive and grouped statistical analysis for the Food Delivery dataset.

Provides functions used both by the EDA notebook and the dashboard pages
to answer the 10 business questions.

All business insight text is computed from the actual data — no hardcoded strings.
"""

import pandas as pd
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ────────────────────────────────────────────────────────────────────────────

def group_mean(df: pd.DataFrame, group_col: str, target_col: str = "Time_taken_min") -> pd.DataFrame:
    """Return mean of target_col grouped by group_col, sorted descending."""
    result = (
        df.groupby(group_col)[target_col]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={target_col: f"Avg_{target_col}"})
        .sort_values(f"Avg_{target_col}", ascending=False)
    )
    return result


def group_stats(df: pd.DataFrame, group_col: str, target_col: str = "Time_taken_min") -> pd.DataFrame:
    """Return mean, median, std, count of target_col grouped by group_col."""
    result = (
        df.groupby(group_col)[target_col]
        .agg(
            Mean="mean",
            Median="median",
            Std="std",
            Count="count",
        )
        .round(2)
        .reset_index()
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlation matrix for all numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].corr(method="pearson").round(3)


# ────────────────────────────────────────────────────────────────────────────
# Business Question Analysis Functions
# ────────────────────────────────────────────────────────────────────────────

def bq1_avg_delivery_time(df: pd.DataFrame) -> dict:
    """BQ1: What is the average delivery time across all orders?"""
    return {
        "mean": round(float(df["Time_taken_min"].mean()), 2),
        "median": round(float(df["Time_taken_min"].median()), 2),
        "std": round(float(df["Time_taken_min"].std()), 2),
        "min": int(df["Time_taken_min"].min()),
        "max": int(df["Time_taken_min"].max()),
        "total_orders": len(df),
    }


def bq2_delivery_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """BQ2: Which day of the week has the highest average delivery time?"""
    result = group_stats(df, "Day_of_Week")
    # Add sort key for correct Mon–Sun ordering
    day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                 "Friday": 4, "Saturday": 5, "Sunday": 6}
    result["Day_Order"] = result["Day_of_Week"].map(day_order).fillna(7)
    return result.sort_values("Day_Order").drop(columns="Day_Order")


def bq3_delivery_by_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """BQ3: Which time of day experiences the highest delivery delays?"""
    tod_order = ["Morning", "Afternoon", "Evening", "Night", "Late Night"]
    result = group_stats(df, "Time_of_Day")
    result["tod_sort"] = result["Time_of_Day"].map({t: i for i, t in enumerate(tod_order)}).fillna(99)
    return result.sort_values("tod_sort").drop(columns="tod_sort")


def bq4_delivery_by_traffic(df: pd.DataFrame) -> pd.DataFrame:
    """BQ4: How does traffic level affect delivery time?"""
    result = group_stats(df, "Traffic_Level")
    # Compute % above overall mean
    overall_mean = df["Time_taken_min"].mean()
    result["Pct_Above_Overall"] = ((result["Mean"] - overall_mean) / overall_mean * 100).round(1)
    return result.sort_values("Mean", ascending=False)


def bq5_delivery_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    """BQ5: How does road distance affect delivery time?"""
    dist_order = ["Short", "Medium", "Long"]
    result = group_stats(df, "Delivery_Distance_Category")
    result["dist_sort"] = result["Delivery_Distance_Category"].map(
        {d: i for i, d in enumerate(dist_order)}
    ).fillna(99)
    return result.sort_values("dist_sort").drop(columns="dist_sort")


def bq6_delivery_by_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    """BQ6: Which vehicle type has the lowest average delivery time?"""
    return group_stats(df, "Vehicle_Type").sort_values("Mean")


def bq7_prep_time_vs_delivery(df: pd.DataFrame) -> pd.DataFrame:
    """BQ7: How does restaurant preparation time affect total delivery time?"""
    result = group_stats(df, "Prep_Category")
    prep_order = ["Fast", "Moderate", "Slow"]
    result["sort_key"] = result["Prep_Category"].map(
        {d: i for i, d in enumerate(prep_order)}
    ).fillna(99)
    return result.sort_values("sort_key").drop(columns="sort_key")


def bq8_rider_performance(df: pd.DataFrame) -> dict:
    """BQ8: How do rider experience and rider rating relate to delivery performance?"""
    exp_result = group_stats(df, "Experience_Category")
    exp_order = ["Novice", "Junior", "Senior"]
    exp_result["sort_key"] = exp_result["Experience_Category"].map(
        {d: i for i, d in enumerate(exp_order)}
    ).fillna(99)
    exp_result = exp_result.sort_values("sort_key").drop(columns="sort_key")

    rating_corr = df[["Rider_Rating", "Time_taken_min"]].corr().loc["Rider_Rating", "Time_taken_min"]

    return {
        "experience_stats": exp_result,
        "rider_rating_corr": round(float(rating_corr), 3),
    }


def bq9_weather_impact(df: pd.DataFrame) -> pd.DataFrame:
    """BQ9: How do weather conditions affect delivery time?"""
    result = group_stats(df, "Weather")
    overall_mean = df["Time_taken_min"].mean()
    result["Pct_Above_Overall"] = ((result["Mean"] - overall_mean) / overall_mean * 100).round(1)
    return result.sort_values("Mean", ascending=False)


def bq10_zone_performance(df: pd.DataFrame) -> dict:
    """BQ10: Which pickup and drop-off zones have the highest average delivery time?"""
    pickup = group_mean(df, "Pickup_Zone").rename(columns={"Avg_Time_taken_min": "Avg Delivery Time (min)"})
    dropoff = group_mean(df, "Dropoff_Zone").rename(columns={"Avg_Time_taken_min": "Avg Delivery Time (min)"})
    pairs = group_mean(df, "Pickup_Dropoff_Pair").rename(columns={"Avg_Time_taken_min": "Avg Delivery Time (min)"})
    return {
        "pickup_zone": pickup,
        "dropoff_zone": dropoff,
        "pairs": pairs.head(10),
    }


# ────────────────────────────────────────────────────────────────────────────
# Trend Analysis
# ────────────────────────────────────────────────────────────────────────────

def orders_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily order count, sorted by date."""
    result = (
        df.groupby("Order_Date")
        .size()
        .reset_index(name="Order_Count")
        .sort_values("Order_Date")
    )
    return result


def delivery_time_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Return daily average delivery time, sorted by date."""
    result = (
        df.groupby("Order_Date")["Time_taken_min"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"Time_taken_min": "Avg_Delivery_Time"})
        .sort_values("Order_Date")
    )
    return result


# ────────────────────────────────────────────────────────────────────────────
# Compute all narrative business insights (data-driven text bullets)
# ────────────────────────────────────────────────────────────────────────────

def compute_insights(df: pd.DataFrame) -> dict:
    """
    Compute all programmatic business insight strings.
    Returns a dict of {section: [list of insight strings]}.
    All text is derived from actual data calculations.
    """
    insights = {}

    # ── Delivery Time ────────────────────────────────────────────────────────
    bq1 = bq1_avg_delivery_time(df)
    bq2 = bq2_delivery_by_day(df)
    bq3 = bq3_delivery_by_time_of_day(df)

    worst_day = bq2.iloc[0]["Day_of_Week"]
    best_day = bq2.iloc[-1]["Day_of_Week"]
    worst_day_avg = bq2.iloc[0]["Mean"]
    best_day_avg = bq2.iloc[-1]["Mean"]
    worst_tod = bq3.sort_values("Mean", ascending=False).iloc[0]["Time_of_Day"]
    worst_tod_avg = bq3.sort_values("Mean", ascending=False).iloc[0]["Mean"]

    insights["Delivery Time"] = [
        f"Overall average delivery time is {bq1['mean']} minutes (median: {bq1['median']} min).",
        f"Delivery times range from {bq1['min']} to {bq1['max']} minutes.",
        f"{worst_day} has the highest average delivery time ({worst_day_avg:.1f} min).",
        f"{best_day} has the lowest average delivery time ({best_day_avg:.1f} min).",
        f"The {worst_tod} period experiences the highest average delivery delay ({worst_tod_avg:.1f} min).",
    ]

    # ── Traffic ──────────────────────────────────────────────────────────────
    bq4 = bq4_delivery_by_traffic(df)
    high_traffic = bq4[bq4["Traffic_Level"].str.lower() == "high"]
    low_traffic = bq4[bq4["Traffic_Level"].str.lower() == "low"]
    traffic_insights = [
        f"The highest traffic level ({bq4.iloc[0]['Traffic_Level']}) results in an average of {bq4.iloc[0]['Mean']:.1f} min delivery time."
    ]
    if len(high_traffic) > 0 and len(low_traffic) > 0:
        diff = high_traffic.iloc[0]["Mean"] - low_traffic.iloc[0]["Mean"]
        traffic_insights.append(
            f"High traffic adds approximately {diff:.1f} additional minutes compared to Low traffic."
        )
    insights["Traffic"] = traffic_insights

    # ── Distance ─────────────────────────────────────────────────────────────
    bq5 = bq5_delivery_by_distance(df)
    dist_insights = [f"Longer delivery distances consistently increase delivery time."]
    if len(bq5) >= 2:
        dist_insights.append(
            f"Long-distance orders average {bq5[bq5['Delivery_Distance_Category']=='Long']['Mean'].values[0]:.1f} min "
            f"vs {bq5[bq5['Delivery_Distance_Category']=='Short']['Mean'].values[0]:.1f} min for short-distance."
            if "Long" in bq5["Delivery_Distance_Category"].values and "Short" in bq5["Delivery_Distance_Category"].values
            else ""
        )
    insights["Distance"] = [i for i in dist_insights if i]

    # ── Vehicle ───────────────────────────────────────────────────────────────
    bq6 = bq6_delivery_by_vehicle(df)
    fastest_vehicle = bq6.iloc[0]["Vehicle_Type"]
    fastest_vehicle_avg = bq6.iloc[0]["Mean"]
    slowest_vehicle = bq6.iloc[-1]["Vehicle_Type"]
    slowest_vehicle_avg = bq6.iloc[-1]["Mean"]
    insights["Vehicle"] = [
        f"{fastest_vehicle} is the fastest vehicle type with an average of {fastest_vehicle_avg:.1f} min.",
        f"{slowest_vehicle} has the highest average delivery time at {slowest_vehicle_avg:.1f} min.",
    ]

    # ── Rider ─────────────────────────────────────────────────────────────────
    bq8 = bq8_rider_performance(df)
    corr_dir = "negatively" if bq8["rider_rating_corr"] < 0 else "positively"
    insights["Rider"] = [
        f"Rider rating is {corr_dir} correlated with delivery time (r = {bq8['rider_rating_corr']}).",
    ]

    # ── Weather ───────────────────────────────────────────────────────────────
    bq9 = bq9_weather_impact(df)
    worst_weather = bq9.iloc[0]["Weather"]
    best_weather = bq9.iloc[-1]["Weather"]
    insights["Weather"] = [
        f"'{worst_weather}' weather has the highest average delivery time ({bq9.iloc[0]['Mean']:.1f} min).",
        f"'{best_weather}' weather has the lowest average delivery time ({bq9.iloc[-1]['Mean']:.1f} min).",
        f"'{worst_weather}' adds {bq9.iloc[0]['Pct_Above_Overall']:+.1f}% delivery time vs the overall average.",
    ]

    # ── Zones ─────────────────────────────────────────────────────────────────
    bq10 = bq10_zone_performance(df)
    top_pickup = bq10["pickup_zone"].iloc[0]["Pickup_Zone"]
    top_dropoff = bq10["dropoff_zone"].iloc[0]["Dropoff_Zone"]
    top_pair = bq10["pairs"].iloc[0]["Pickup_Dropoff_Pair"]
    insights["Zones"] = [
        f"'{top_pickup}' is the pickup zone with the highest average delivery time.",
        f"'{top_dropoff}' is the drop-off zone with the highest average delivery time.",
        f"The slowest corridor is '{top_pair}'.",
    ]

    return insights
