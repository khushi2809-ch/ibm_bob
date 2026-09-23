"""
03_Traffic_Route_Analysis.py
============================
Page 3: Traffic & Route Analysis

Charts:
  - Traffic level vs delivery time (box plot)
  - Distance vs delivery time (scatter, sampled)
  - Number of signals vs delivery time (scatter, sampled)
  - Pickup zone performance (horizontal bar)
  - Drop-off zone performance (horizontal bar)

Business Questions answered: BQ4, BQ5, BQ10
"""

import os, sys
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.statistical_analysis import bq4_delivery_by_traffic, bq10_zone_performance, compute_insights
from src.charts import box_plot, scatter_plot, hbar_chart

st.set_page_config(page_title="Traffic & Route Analysis", page_icon="🚦", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    from src.data_loader import load_data
    from src.cleaner import clean_data
    from src.feature_engineer import add_features
    st.session_state["df"] = add_features(clean_data(load_data()))

df = st.session_state["df"]

st.title("🚦 Traffic & Route Analysis")
st.markdown("Impact of traffic congestion, road distance, signals, and zone combinations on delivery time.")
st.markdown("---")

# ── Row 1: Traffic box + Distance scatter ─────────────────────────────────────
st.subheader("BQ4 — Traffic Level Impact on Delivery Time")
col_l, col_r = st.columns(2)

with col_l:
    traffic_order = ["Low", "Moderate", "High", "Jam"]
    # Keep only levels that exist in data
    existing_levels = df["Traffic_Level"].unique().tolist()
    traffic_order = [t for t in traffic_order if t in existing_levels]
    # Add any unexpected levels
    for t in existing_levels:
        if t not in traffic_order:
            traffic_order.append(t)

    fig_traffic_box = box_plot(
        df, x="Traffic_Level", y="Time_taken_min",
        title="Delivery Time Distribution by Traffic Level",
        x_label="Traffic Level",
        y_label="Delivery Time (min)",
        category_order=traffic_order,
    )
    st.plotly_chart(fig_traffic_box, use_container_width=True)

with col_r:
    st.subheader("BQ5 — Road Distance vs Delivery Time")
    fig_dist_scatter = scatter_plot(
        df, x="Road_Distance_km", y="Time_taken_min",
        title="Road Distance vs Delivery Time",
        x_label="Road Distance (km)",
        y_label="Delivery Time (min)",
        color_col="Traffic_Level",
        add_trendline=True,
    )
    st.plotly_chart(fig_dist_scatter, use_container_width=True)

st.markdown("---")

# ── Row 2: Signals scatter ────────────────────────────────────────────────────
st.subheader("Number of Traffic Signals vs Delivery Time")
fig_signals = scatter_plot(
    df, x="Number_of_Signals", y="Time_taken_min",
    title="Number of Traffic Signals vs Delivery Time",
    x_label="Number of Traffic Signals",
    y_label="Delivery Time (min)",
    color_col="Traffic_Level",
)
st.plotly_chart(fig_signals, use_container_width=True)

st.markdown("---")

# ── Row 3: Zone performance ───────────────────────────────────────────────────
st.subheader("BQ10 — Zone Performance Analysis")
zones = bq10_zone_performance(df)

col_l2, col_r2 = st.columns(2)

with col_l2:
    st.markdown("**Pickup Zone — Average Delivery Time**")
    pickup_df = zones["pickup_zone"].rename(columns={"Avg_Time_taken_min": "Avg Delivery Time (min)"})
    # Ensure column names are correct
    col_name = [c for c in pickup_df.columns if "Avg" in c][0]
    fig_pickup = hbar_chart(
        pickup_df.sort_values(col_name),
        x=col_name,
        y="Pickup_Zone",
        title="Average Delivery Time by Pickup Zone",
        x_label="Avg Delivery Time (min)",
        y_label="Pickup Zone",
    )
    st.plotly_chart(fig_pickup, use_container_width=True)

with col_r2:
    st.markdown("**Drop-off Zone — Average Delivery Time**")
    dropoff_df = zones["dropoff_zone"].rename(columns={"Avg_Time_taken_min": "Avg Delivery Time (min)"})
    col_name2 = [c for c in dropoff_df.columns if "Avg" in c][0]
    fig_dropoff = hbar_chart(
        dropoff_df.sort_values(col_name2),
        x=col_name2,
        y="Dropoff_Zone",
        title="Average Delivery Time by Drop-off Zone",
        x_label="Avg Delivery Time (min)",
        y_label="Drop-off Zone",
    )
    st.plotly_chart(fig_dropoff, use_container_width=True)

st.markdown("---")

# ── Top Pickup→Dropoff corridors ──────────────────────────────────────────────
st.subheader("Top 10 Slowest Pickup → Drop-off Corridors")
pairs_df = zones["pairs"]
pair_col = [c for c in pairs_df.columns if "Avg" in c][0]
fig_pairs = hbar_chart(
    pairs_df.sort_values(pair_col),
    x=pair_col,
    y="Pickup_Dropoff_Pair",
    title="Top 10 Slowest Delivery Corridors",
    x_label="Avg Delivery Time (min)",
    y_label="Corridor",
    height=420,
)
st.plotly_chart(fig_pairs, use_container_width=True)

st.markdown("---")

# ── Insights expander ─────────────────────────────────────────────────────────
with st.expander("📌 Key Insights — Traffic & Route", expanded=False):
    insights = compute_insights(df)
    for item in insights.get("Traffic", []):
        st.markdown(f"- {item}")
    for item in insights.get("Distance", []):
        st.markdown(f"- {item}")
    for item in insights.get("Zones", []):
        st.markdown(f"- {item}")

st.caption("Data source: Food Delivery Time Prediction Dataset  |  IBM Data Analytics Internship")
