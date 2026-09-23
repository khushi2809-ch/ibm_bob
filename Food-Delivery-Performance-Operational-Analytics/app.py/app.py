"""
app.py
======
Food Delivery Performance & Operational Analytics
Main Streamlit application entry point.

Responsibilities:
  - Configure the page layout and sidebar
  - Load and cache the dataset (run once for all pages)
  - Store the processed DataFrame in st.session_state
  - Display welcome / project overview on the home screen
"""

import os
import sys
import streamlit as st

# ── Make the src/ package importable ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data
from src.cleaner import clean_data
from src.feature_engineer import add_features
from src.kpi_calculator import compute_kpis

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: minimal professional styling ──────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2328;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #57606a;
        margin-bottom: 1.5rem;
    }
    /* KPI card style */
    .kpi-card {
        background-color: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f7f8fa;
    }
    /* Divider */
    hr { border-color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)


# ── Data loading with caching ─────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing dataset…")
def get_processed_data():
    """Load, clean, and engineer features — cached for performance."""
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    final_df = add_features(clean_df)
    return final_df


# ── Load data into session state ──────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = get_processed_data()

df = st.session_state["df"]
kpis = compute_kpis(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=80)
    st.markdown("### 🚀 Food Delivery Analytics")
    st.markdown("---")
    st.markdown("**Dataset Info**")
    st.markdown(f"- **Records:** {kpis['Total Orders']:,}")
    st.markdown(f"- **Columns:** {len(df.columns)}")
    st.markdown(f"- **Date Range:** {df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("""
    1. 📊 Executive Overview
    2. 🕐 Delivery Performance
    3. 🚦 Traffic & Route Analysis
    4. 🏍️ Rider & Restaurant
    5. 🌤️ Operational & Environmental
    """)
    st.markdown("---")
    st.caption("IBM Data Analytics Internship Project")
    st.caption("Food Delivery Performance & Operational Analytics")

# ── Home Page ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🚀 Food Delivery Performance & Operational Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">IBM Data Analytics Internship Project &nbsp;|&nbsp; Pure Data Analytics &nbsp;|&nbsp; No ML / No GenAI</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Top KPI row ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📦 Total Orders", f"{kpis['Total Orders']:,}")
col2.metric("⏱️ Avg Delivery Time", f"{kpis['Avg Delivery Time (min)']} min")
col3.metric("📏 Avg Distance", f"{kpis['Avg Road Distance (km)']} km")
col4.metric("⭐ Avg Rider Rating", str(kpis["Avg Rider Rating"]))
col5.metric("🍽️ Avg Restaurant Rating", str(kpis["Avg Restaurant Rating"]))

st.markdown("---")

# ── Project overview ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Project Overview")
    st.markdown("""
    This dashboard provides a comprehensive operational analysis of food delivery data,
    covering **50,000+ orders** across multiple zones, riders, vehicle types, and conditions.

    **Analysis Areas:**
    - 📈 Order Trends & Delivery Time Patterns
    - 🚦 Traffic Level & Route/Distance Impact
    - 🏍️ Rider Performance & Experience Analysis
    - 🍽️ Restaurant Preparation Time & Load Analysis
    - 🌤️ Weather & Environmental Impact
    - 📅 Weekend, Festival & Priority Order Analysis

    **Use the sidebar or page tabs above to navigate between dashboard sections.**
    """)

with col_right:
    st.subheader("10 Business Questions")
    st.markdown("""
    1. Overall average delivery time?
    2. Which day has highest delivery time?
    3. Which time of day has most delays?
    4. How does traffic affect delivery?
    5. How does distance affect delivery?
    6. Which vehicle type is fastest?
    7. Impact of preparation time?
    8. Rider experience & rating impact?
    9. Weather impact on delivery?
    10. Slowest pickup/drop-off zones?
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#57606a; font-size:12px; padding-top:10px;'>"
    "Food Delivery Performance & Operational Analytics &nbsp;|&nbsp; IBM Data Analytics Internship &nbsp;|&nbsp; Pure Data Analytics"
    "</div>",
    unsafe_allow_html=True,
)
