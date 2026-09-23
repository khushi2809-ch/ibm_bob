"""
kpi_calculator.py
=================
Computes all 10 top-level KPIs for the Food Delivery analytics dashboard.

All values are derived programmatically from the actual dataset.
No hardcoded numbers.

KPIs:
  1. Total Orders
  2. Average Delivery Time (min)
  3. Median Delivery Time (min)
  4. Average Preparation Time (min)
  5. Average Road Distance (km)
  6. Average Rider Rating
  7. Average Restaurant Rating
  8. Average Speed (km/h)
  9. Weekend Orders Percentage (%)
 10. Festival Orders Percentage (%)
"""

import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute all 10 KPIs from the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and feature-engineered DataFrame.

    Returns
    -------
    dict
        Dictionary with KPI names as keys and computed values.
        Example:
        {
            "Total Orders": 49832,
            "Avg Delivery Time (min)": 52.4,
            ...
        }
    """
    if len(df) == 0:
        raise ValueError("Cannot compute KPIs on an empty DataFrame.")

    total_orders = len(df)
    weekend_pct = round(df["Is_Weekend"].sum() / total_orders * 100, 1)
    festival_pct = round(df["Is_Festival"].sum() / total_orders * 100, 1)

    kpis = {
        "Total Orders": int(total_orders),
        "Avg Delivery Time (min)": round(float(df["Time_taken_min"].mean()), 1),
        "Median Delivery Time (min)": round(float(df["Time_taken_min"].median()), 1),
        "Avg Preparation Time (min)": round(float(df["Preparation_Time_Min"].mean()), 1),
        "Avg Road Distance (km)": round(float(df["Road_Distance_km"].mean()), 2),
        "Avg Rider Rating": round(float(df["Rider_Rating"].mean()), 2),
        "Avg Restaurant Rating": round(float(df["Restaurant_Rating"].mean()), 2),
        "Avg Speed (km/h)": round(float(df["Average_Speed_kmph"].mean()), 1),
        "Weekend Orders (%)": float(weekend_pct),
        "Festival Orders (%)": float(festival_pct),
    }

    # ── Print summary ────────────────────────────────────────────────────────
    print("[kpi_calculator] Computed KPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    return kpis


def compute_delta_kpis(df: pd.DataFrame) -> dict:
    """
    Compute comparison KPIs — percentage difference between key sub-groups.
    Used for delta indicators in dashboard metric cards.

    Returns a dict of {kpi_name: delta_value} where delta is expressed as
    a percentage difference relative to the overall mean.

    Example: weekend avg delivery time vs overall avg delivery time.
    """
    if len(df) == 0:
        return {}

    overall_avg = df["Time_taken_min"].mean()

    def pct_diff(subset_mean: float) -> float:
        """Return % difference of subset_mean from overall_avg."""
        if overall_avg == 0:
            return 0.0
        return round((subset_mean - overall_avg) / overall_avg * 100, 1)

    weekend_df = df[df["Is_Weekend"] == 1]
    weekday_df = df[df["Is_Weekend"] == 0]
    festival_df = df[df["Is_Festival"] == 1]

    deltas = {}

    if len(weekend_df) > 0:
        deltas["Weekend vs Overall Delivery Time (%)"] = pct_diff(weekend_df["Time_taken_min"].mean())
    if len(weekday_df) > 0:
        deltas["Weekday vs Overall Delivery Time (%)"] = pct_diff(weekday_df["Time_taken_min"].mean())
    if len(festival_df) > 0:
        deltas["Festival vs Overall Delivery Time (%)"] = pct_diff(festival_df["Time_taken_min"].mean())

    return deltas
