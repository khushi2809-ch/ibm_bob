"""
04_Rider_Restaurant_Performance.py
===================================
Page 4: Rider & Restaurant Performance

Charts:
  - Delivery time by vehicle type (box plot)
  - Rider experience category vs delivery time (bar chart)
  - Rider rating vs delivery time (scatter + trendline)
  - Restaurant rating vs delivery time (scatter + trendline)
  - Preparation time vs delivery time (scatter + trendline)

Business Questions answered: BQ6, BQ7, BQ8
"""

import os, sys
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.statistical_analysis import (
    bq6_delivery_by_vehicle,
    bq7_prep_time_vs_delivery,
    bq8_rider_performance,
    compute_insights,
)
from src.charts import box_plot, scatter_plot, bar_chart, metric_comparison_bar

st.set_page_config(page_title="Rider & Restaurant Performance", page_icon="🏍️", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    from src.data_loader import load_data
    from src.cleaner import clean_data
    from src.feature_engineer import add_features
    st.session_state["df"] = add_features(clean_data(load_data()))

df = st.session_state["df"]
overall_avg = float(df["Time_taken_min"].mean())

st.title("🏍️ Rider & Restaurant Performance")
st.markdown("Analysis of vehicle types, rider experience, ratings, and restaurant preparation impact.")
st.markdown("---")

# ── Row 1: Vehicle type box + Prep time bar ───────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("BQ6 — Delivery Time by Vehicle Type")
    fig_vehicle = box_plot(
        df, x="Vehicle_Type", y="Time_taken_min",
        title="Delivery Time Distribution by Vehicle Type",
        x_label="Vehicle Type",
        y_label="Delivery Time (min)",
    )
    st.plotly_chart(fig_vehicle, use_container_width=True)

with col_r:
    st.subheader("BQ7 — Preparation Time Category vs Delivery Time")
    prep_stats = bq7_prep_time_vs_delivery(df)
    fig_prep = metric_comparison_bar(
        categories=prep_stats["Prep_Category"].tolist(),
        values=prep_stats["Mean"].tolist(),
        title="Average Delivery Time by Preparation Speed",
        overall_avg=overall_avg,
    )
    st.plotly_chart(fig_prep, use_container_width=True)

st.markdown("---")

# ── Row 2: Rider experience bar ───────────────────────────────────────────────
st.subheader("BQ8 — Rider Experience vs Delivery Time")
rider_perf = bq8_rider_performance(df)
exp_stats = rider_perf["experience_stats"]
rider_corr = rider_perf["rider_rating_corr"]

col_l2, col_r2 = st.columns(2)

with col_l2:
    fig_exp = metric_comparison_bar(
        categories=exp_stats["Experience_Category"].tolist(),
        values=exp_stats["Mean"].tolist(),
        title="Average Delivery Time by Rider Experience",
        overall_avg=overall_avg,
    )
    st.plotly_chart(fig_exp, use_container_width=True)

with col_r2:
    st.subheader("Rider Rating vs Delivery Time")
    fig_rider_rating = scatter_plot(
        df, x="Rider_Rating", y="Time_taken_min",
        title="Rider Rating vs Delivery Time",
        x_label="Rider Rating",
        y_label="Delivery Time (min)",
        add_trendline=True,
    )
    st.plotly_chart(fig_rider_rating, use_container_width=True)
    st.info(f"📊 Pearson Correlation (Rider Rating vs Delivery Time): **{rider_corr}**")

st.markdown("---")

# ── Row 3: Restaurant rating + Prep time scatter ──────────────────────────────
col_l3, col_r3 = st.columns(2)

with col_l3:
    st.subheader("Restaurant Rating vs Delivery Time")
    fig_rest_rating = scatter_plot(
        df, x="Restaurant_Rating", y="Time_taken_min",
        title="Restaurant Rating vs Delivery Time",
        x_label="Restaurant Rating",
        y_label="Delivery Time (min)",
        add_trendline=True,
    )
    rest_corr = round(float(df[["Restaurant_Rating", "Time_taken_min"]].corr().loc["Restaurant_Rating", "Time_taken_min"]), 3)
    st.plotly_chart(fig_rest_rating, use_container_width=True)
    st.info(f"📊 Pearson Correlation (Restaurant Rating vs Delivery Time): **{rest_corr}**")

with col_r3:
    st.subheader("BQ7 — Preparation Time vs Delivery Time")
    fig_prep_scatter = scatter_plot(
        df, x="Preparation_Time_Min", y="Time_taken_min",
        title="Preparation Time vs Total Delivery Time",
        x_label="Preparation Time (min)",
        y_label="Delivery Time (min)",
        add_trendline=True,
    )
    prep_corr = round(float(df[["Preparation_Time_Min", "Time_taken_min"]].corr().loc["Preparation_Time_Min", "Time_taken_min"]), 3)
    st.plotly_chart(fig_prep_scatter, use_container_width=True)
    st.info(f"📊 Pearson Correlation (Prep Time vs Delivery Time): **{prep_corr}**")

st.markdown("---")

# ── Insights expander ─────────────────────────────────────────────────────────
with st.expander("📌 Key Insights — Rider & Restaurant", expanded=False):
    insights = compute_insights(df)
    for item in insights.get("Vehicle", []):
        st.markdown(f"- {item}")
    for item in insights.get("Rider", []):
        st.markdown(f"- {item}")

st.caption("Data source: Food Delivery Time Prediction Dataset  |  IBM Data Analytics Internship")
