"""
01_Executive_Overview.py
========================
Page 1: Executive Overview

Charts:
  - KPI metric cards (10 KPIs)
  - Orders over time (line chart)
  - Delivery time trend over time (line chart)
  - Orders by vehicle type (donut chart)
  - Delivery time distribution (histogram)
"""

import os, sys
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.kpi_calculator import compute_kpis
from src.statistical_analysis import orders_over_time, delivery_time_trend
from src.charts import line_chart, donut_chart, histogram

st.set_page_config(page_title="Executive Overview", page_icon="📊", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    from src.data_loader import load_data
    from src.cleaner import clean_data
    from src.feature_engineer import add_features
    st.session_state["df"] = add_features(clean_data(load_data()))

df = st.session_state["df"]
kpis = compute_kpis(df)

st.title("📊 Executive Overview")
st.markdown("High-level KPIs and overall delivery performance summary.")
st.markdown("---")

# ── Row 1: 5 KPI cards ────────────────────────────────────────────────────────
st.subheader("Key Performance Indicators")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Total Orders",          f"{kpis['Total Orders']:,}")
c2.metric("⏱️ Avg Delivery Time",     f"{kpis['Avg Delivery Time (min)']} min")
c3.metric("📊 Median Delivery Time",  f"{kpis['Median Delivery Time (min)']} min")
c4.metric("🍳 Avg Preparation Time",  f"{kpis['Avg Preparation Time (min)']} min")
c5.metric("📏 Avg Road Distance",     f"{kpis['Avg Road Distance (km)']} km")

# ── Row 2: 5 more KPI cards ───────────────────────────────────────────────────
c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("⭐ Avg Rider Rating",       str(kpis["Avg Rider Rating"]))
c7.metric("🍽️ Avg Restaurant Rating", str(kpis["Avg Restaurant Rating"]))
c8.metric("💨 Avg Speed",             f"{kpis['Avg Speed (km/h)']} km/h")
c9.metric("📅 Weekend Orders",        f"{kpis['Weekend Orders (%)']}%")
c10.metric("🎉 Festival Orders",       f"{kpis['Festival Orders (%)']}%")

st.markdown("---")

# ── Row 3: Orders over time + Delivery time trend ─────────────────────────────
st.subheader("Order & Delivery Time Trends")
col_l, col_r = st.columns(2)

with col_l:
    oot = orders_over_time(df)
    fig_orders = line_chart(
        oot, x="Order_Date", y="Order_Count",
        title="Daily Order Volume Over Time",
        x_label="Date", y_label="Number of Orders",
    )
    st.plotly_chart(fig_orders, use_container_width=True)

with col_r:
    dtt = delivery_time_trend(df)
    fig_trend = line_chart(
        dtt, x="Order_Date", y="Avg_Delivery_Time",
        title="Daily Average Delivery Time Trend",
        x_label="Date", y_label="Avg Delivery Time (min)",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ── Row 4: Vehicle type donut + Delivery time distribution ────────────────────
st.subheader("Orders by Vehicle Type & Delivery Time Distribution")
col_l2, col_r2 = st.columns(2)

with col_l2:
    vehicle_counts = (
        df.groupby("Vehicle_Type")
        .size()
        .reset_index(name="Order_Count")
    )
    fig_donut = donut_chart(
        vehicle_counts, names="Vehicle_Type", values="Order_Count",
        title="Orders by Vehicle Type",
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with col_r2:
    fig_hist = histogram(
        df, x="Time_taken_min",
        title="Delivery Time Distribution",
        x_label="Delivery Time (minutes)",
        nbins=50,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
st.caption("Data source: Food Delivery Time Prediction Dataset  |  IBM Data Analytics Internship")
