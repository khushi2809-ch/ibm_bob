"""
02_Delivery_Performance.py
==========================
Page 2: Delivery Performance Analysis

Charts:
  - Delivery time by hour of day (bar chart)
  - Delivery time by day of week (bar chart, ordered Mon–Sun)
  - Delivery time by distance category (bar chart)
  - Delivery time by traffic level (bar chart)
  - Delivery time by delivery priority (box plot)

Business Questions answered: BQ2, BQ3, BQ4, BQ5
"""

import os, sys
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.statistical_analysis import (
    bq2_delivery_by_day,
    bq3_delivery_by_time_of_day,
    bq4_delivery_by_traffic,
    bq5_delivery_by_distance,
    group_mean,
    compute_insights,
)
from src.charts import bar_chart, box_plot, metric_comparison_bar

st.set_page_config(page_title="Delivery Performance", page_icon="🕐", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    from src.data_loader import load_data
    from src.cleaner import clean_data
    from src.feature_engineer import add_features
    st.session_state["df"] = add_features(clean_data(load_data()))

df = st.session_state["df"]
overall_avg = float(df["Time_taken_min"].mean())

st.title("🕐 Delivery Performance Analysis")
st.markdown("Analysis of delivery times across hours, days, distances, traffic levels, and priorities.")
st.markdown("---")

# ── Row 1: By Hour ────────────────────────────────────────────────────────────
st.subheader("BQ3 — Delivery Time by Hour of Day")
hourly = (
    df.groupby("Order_Hour")["Time_taken_min"]
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={"Time_taken_min": "Avg_Delivery_Time"})
    .sort_values("Order_Hour")
)
fig_hour = metric_comparison_bar(
    categories=[str(h) for h in hourly["Order_Hour"].tolist()],
    values=hourly["Avg_Delivery_Time"].tolist(),
    title="Average Delivery Time by Hour of Day",
    overall_avg=overall_avg,
)
fig_hour.update_layout(xaxis_title="Hour of Day (0–23)", yaxis_title="Avg Delivery Time (min)")
st.plotly_chart(fig_hour, use_container_width=True)

st.markdown("---")

# ── Row 2: By Day + By Distance ───────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("BQ2 — Delivery Time by Day of Week")
    day_stats = bq2_delivery_by_day(df)
    fig_day = metric_comparison_bar(
        categories=day_stats["Day_of_Week"].tolist(),
        values=day_stats["Mean"].tolist(),
        title="Average Delivery Time by Day of Week",
        overall_avg=overall_avg,
    )
    st.plotly_chart(fig_day, use_container_width=True)

with col_r:
    st.subheader("BQ5 — Delivery Time by Distance Category")
    dist_stats = bq5_delivery_by_distance(df)
    fig_dist = metric_comparison_bar(
        categories=dist_stats["Delivery_Distance_Category"].tolist(),
        values=dist_stats["Mean"].tolist(),
        title="Average Delivery Time by Distance Category",
        overall_avg=overall_avg,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# ── Row 3: By Traffic + By Priority ──────────────────────────────────────────
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.subheader("BQ4 — Delivery Time by Traffic Level")
    traffic_stats = bq4_delivery_by_traffic(df)
    fig_traffic = metric_comparison_bar(
        categories=traffic_stats["Traffic_Level"].tolist(),
        values=traffic_stats["Mean"].tolist(),
        title="Average Delivery Time by Traffic Level",
        overall_avg=overall_avg,
    )
    st.plotly_chart(fig_traffic, use_container_width=True)

with col_r2:
    st.subheader("Delivery Time by Delivery Priority")
    fig_priority = box_plot(
        df, x="Delivery_Priority", y="Time_taken_min",
        title="Delivery Time Distribution by Priority",
        x_label="Delivery Priority",
        y_label="Delivery Time (min)",
    )
    st.plotly_chart(fig_priority, use_container_width=True)

st.markdown("---")

# ── Insights expander ─────────────────────────────────────────────────────────
with st.expander("📌 Key Insights — Delivery Performance", expanded=False):
    insights = compute_insights(df)
    for item in insights.get("Delivery Time", []):
        st.markdown(f"- {item}")
    for item in insights.get("Traffic", []):
        st.markdown(f"- {item}")

st.caption("Data source: Food Delivery Time Prediction Dataset  |  IBM Data Analytics Internship")
