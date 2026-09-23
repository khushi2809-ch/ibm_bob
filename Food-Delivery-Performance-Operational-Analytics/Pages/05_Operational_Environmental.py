"""
05_Operational_Environmental.py
================================
Page 5: Operational & Environmental Analysis

Charts:
  - Weather vs delivery time (box plot)
  - Cuisine type vs order count (horizontal bar)
  - Cuisine type vs preparation time (bar chart)
  - Weekend vs weekday delivery time (grouped bar)
  - Festival vs non-festival delivery time (grouped bar)
  - Delivery priority analysis (bar chart)

Business Questions answered: BQ9
"""

import os, sys
import streamlit as st
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.statistical_analysis import bq9_weather_impact, compute_insights
from src.charts import box_plot, hbar_chart, bar_chart, grouped_bar_chart, metric_comparison_bar

st.set_page_config(page_title="Operational & Environmental", page_icon="🌤️", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    from src.data_loader import load_data
    from src.cleaner import clean_data
    from src.feature_engineer import add_features
    st.session_state["df"] = add_features(clean_data(load_data()))

df = st.session_state["df"]
overall_avg = float(df["Time_taken_min"].mean())

st.title("🌤️ Operational & Environmental Analysis")
st.markdown("Weather impact, cuisine patterns, weekend/festival effects, and delivery priority analysis.")
st.markdown("---")

# ── Row 1: Weather ─────────────────────────────────────────────────────────────
st.subheader("BQ9 — Weather Conditions vs Delivery Time")
col_l, col_r = st.columns(2)

with col_l:
    fig_weather_box = box_plot(
        df, x="Weather", y="Time_taken_min",
        title="Delivery Time Distribution by Weather",
        x_label="Weather Condition",
        y_label="Delivery Time (min)",
    )
    st.plotly_chart(fig_weather_box, use_container_width=True)

with col_r:
    weather_stats = bq9_weather_impact(df)
    fig_weather_bar = metric_comparison_bar(
        categories=weather_stats["Weather"].tolist(),
        values=weather_stats["Mean"].tolist(),
        title="Average Delivery Time by Weather Condition",
        overall_avg=overall_avg,
    )
    st.plotly_chart(fig_weather_bar, use_container_width=True)

st.markdown("---")

# ── Row 2: Cuisine analysis ────────────────────────────────────────────────────
st.subheader("Cuisine Type Analysis")
col_l2, col_r2 = st.columns(2)

with col_l2:
    cuisine_orders = (
        df.groupby("Cuisine_Type")
        .size()
        .reset_index(name="Order_Count")
        .sort_values("Order_Count", ascending=False)
    )
    fig_cuisine_orders = hbar_chart(
        cuisine_orders.sort_values("Order_Count"),
        x="Order_Count",
        y="Cuisine_Type",
        title="Order Count by Cuisine Type",
        x_label="Number of Orders",
        y_label="Cuisine Type",
    )
    st.plotly_chart(fig_cuisine_orders, use_container_width=True)

with col_r2:
    cuisine_prep = (
        df.groupby("Cuisine_Type")["Preparation_Time_Min"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"Preparation_Time_Min": "Avg_Prep_Time"})
        .sort_values("Avg_Prep_Time", ascending=False)
    )
    fig_cuisine_prep = hbar_chart(
        cuisine_prep.sort_values("Avg_Prep_Time"),
        x="Avg_Prep_Time",
        y="Cuisine_Type",
        title="Average Preparation Time by Cuisine Type",
        x_label="Avg Prep Time (min)",
        y_label="Cuisine Type",
    )
    st.plotly_chart(fig_cuisine_prep, use_container_width=True)

st.markdown("---")

# ── Row 3: Weekend vs Weekday + Festival vs Non-Festival ──────────────────────
st.subheader("Weekend & Festival Impact")
col_l3, col_r3 = st.columns(2)

with col_l3:
    weekend_df = (
        df.groupby("Is_Weekend")["Time_taken_min"]
        .mean()
        .round(2)
        .reset_index()
    )
    weekend_df["Day_Type"] = weekend_df["Is_Weekend"].map({0: "Weekday", 1: "Weekend"})
    fig_weekend = bar_chart(
        weekend_df,
        x="Day_Type",
        y="Time_taken_min",
        title="Average Delivery Time: Weekend vs Weekday",
        x_label="Day Type",
        y_label="Avg Delivery Time (min)",
    )
    # Add overall average reference line
    import plotly.graph_objects as go
    fig_weekend.add_shape(
        type="line", x0=-0.5, x1=1.5,
        y0=overall_avg, y1=overall_avg,
        line=dict(color="#e05c32", width=2, dash="dash"),
    )
    fig_weekend.add_annotation(
        x=1, y=overall_avg,
        text=f"Overall avg: {overall_avg:.1f} min",
        showarrow=False, font=dict(color="#e05c32", size=11), yshift=10,
    )
    st.plotly_chart(fig_weekend, use_container_width=True)

with col_r3:
    festival_df = (
        df.groupby("Is_Festival")["Time_taken_min"]
        .mean()
        .round(2)
        .reset_index()
    )
    festival_df["Order_Type"] = festival_df["Is_Festival"].map({0: "Non-Festival", 1: "Festival"})
    fig_festival = bar_chart(
        festival_df,
        x="Order_Type",
        y="Time_taken_min",
        title="Average Delivery Time: Festival vs Non-Festival",
        x_label="Order Type",
        y_label="Avg Delivery Time (min)",
    )
    fig_festival.add_shape(
        type="line", x0=-0.5, x1=1.5,
        y0=overall_avg, y1=overall_avg,
        line=dict(color="#e05c32", width=2, dash="dash"),
    )
    fig_festival.add_annotation(
        x=1, y=overall_avg,
        text=f"Overall avg: {overall_avg:.1f} min",
        showarrow=False, font=dict(color="#e05c32", size=11), yshift=10,
    )
    st.plotly_chart(fig_festival, use_container_width=True)

st.markdown("---")

# ── Row 4: Delivery Priority + Restaurant Load ─────────────────────────────────
st.subheader("Delivery Priority & Restaurant Load")
col_l4, col_r4 = st.columns(2)

with col_l4:
    priority_stats = (
        df.groupby("Delivery_Priority")["Time_taken_min"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"Time_taken_min": "Avg_Delivery_Time"})
        .sort_values("Avg_Delivery_Time", ascending=False)
    )
    fig_priority = metric_comparison_bar(
        categories=priority_stats["Delivery_Priority"].tolist(),
        values=priority_stats["Avg_Delivery_Time"].tolist(),
        title="Average Delivery Time by Delivery Priority",
        overall_avg=overall_avg,
    )
    st.plotly_chart(fig_priority, use_container_width=True)

with col_r4:
    load_stats = (
        df.groupby("Restaurant_Load")["Time_taken_min"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"Time_taken_min": "Avg_Delivery_Time"})
        .sort_values("Avg_Delivery_Time", ascending=False)
    )
    fig_load = metric_comparison_bar(
        categories=load_stats["Restaurant_Load"].tolist(),
        values=load_stats["Avg_Delivery_Time"].tolist(),
        title="Average Delivery Time by Restaurant Load",
        overall_avg=overall_avg,
    )
    st.plotly_chart(fig_load, use_container_width=True)

st.markdown("---")

# ── Insights expander ─────────────────────────────────────────────────────────
with st.expander("📌 Key Insights — Operational & Environmental", expanded=False):
    insights = compute_insights(df)
    for item in insights.get("Weather", []):
        st.markdown(f"- {item}")

st.caption("Data source: Food Delivery Time Prediction Dataset  |  IBM Data Analytics Internship")
