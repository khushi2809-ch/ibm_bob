"""
test_kpi_calculator.py
======================
Unit tests for src/kpi_calculator.py

Tests:
  - compute_kpis() returns a dictionary
  - All 10 required KPI keys are present
  - All values are of numeric types
  - Weekend_Pct is between 0 and 100
  - Festival_Pct is between 0 and 100
  - All average values are within plausible real-world ranges
  - Raises ValueError on empty DataFrame
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_data
from src.cleaner import clean_data
from src.feature_engineer import add_features
from src.kpi_calculator import compute_kpis, compute_delta_kpis


REQUIRED_KPI_KEYS = [
    "Total Orders",
    "Avg Delivery Time (min)",
    "Median Delivery Time (min)",
    "Avg Preparation Time (min)",
    "Avg Road Distance (km)",
    "Avg Rider Rating",
    "Avg Restaurant Rating",
    "Avg Speed (km/h)",
    "Weekend Orders (%)",
    "Festival Orders (%)",
]


@pytest.fixture(scope="module")
def df():
    """Full pipeline DataFrame."""
    raw = load_data()
    clean = clean_data(raw)
    return add_features(clean)


@pytest.fixture(scope="module")
def kpis(df):
    """Computed KPIs dictionary."""
    return compute_kpis(df)


class TestKPIStructure:
    """Test that compute_kpis returns the correct structure."""

    def test_returns_dict(self, kpis):
        assert isinstance(kpis, dict), "compute_kpis should return a dict"

    def test_all_10_keys_present(self, kpis):
        for key in REQUIRED_KPI_KEYS:
            assert key in kpis, f"KPI key '{key}' is missing from output"

    def test_exactly_10_keys(self, kpis):
        assert len(kpis) == 10, f"Expected 10 KPIs, got {len(kpis)}"


class TestKPITypes:
    """All KPI values must be numeric."""

    @pytest.mark.parametrize("key", REQUIRED_KPI_KEYS)
    def test_kpi_value_is_numeric(self, kpis, key):
        val = kpis[key]
        assert isinstance(val, (int, float, np.integer, np.floating)), \
            f"KPI '{key}' has non-numeric value: {type(val)}"

    def test_total_orders_is_int(self, kpis):
        assert isinstance(kpis["Total Orders"], int), \
            "Total Orders should be an integer"


class TestKPIValues:
    """Validate that KPI values are in plausible real-world ranges."""

    def test_total_orders_positive(self, kpis):
        assert kpis["Total Orders"] > 0

    def test_avg_delivery_time_positive(self, kpis):
        assert kpis["Avg Delivery Time (min)"] > 0

    def test_avg_delivery_time_plausible(self, kpis):
        # Real-world delivery: between 1 minute and 180 minutes
        assert 1 <= kpis["Avg Delivery Time (min)"] <= 180, \
            f"Avg Delivery Time = {kpis['Avg Delivery Time (min)']} seems implausible"

    def test_median_delivery_time_positive(self, kpis):
        assert kpis["Median Delivery Time (min)"] > 0

    def test_avg_prep_time_positive(self, kpis):
        assert kpis["Avg Preparation Time (min)"] > 0

    def test_avg_road_distance_positive(self, kpis):
        assert kpis["Avg Road Distance (km)"] > 0

    def test_avg_rider_rating_in_range(self, kpis):
        assert 1.0 <= kpis["Avg Rider Rating"] <= 5.0, \
            f"Avg Rider Rating = {kpis['Avg Rider Rating']} is outside [1, 5]"

    def test_avg_restaurant_rating_in_range(self, kpis):
        assert 1.0 <= kpis["Avg Restaurant Rating"] <= 5.0, \
            f"Avg Restaurant Rating = {kpis['Avg Restaurant Rating']} is outside [1, 5]"

    def test_avg_speed_positive(self, kpis):
        assert kpis["Avg Speed (km/h)"] > 0

    def test_weekend_pct_in_range(self, kpis):
        assert 0 <= kpis["Weekend Orders (%)"] <= 100, \
            f"Weekend Orders % = {kpis['Weekend Orders (%)']} is outside [0, 100]"

    def test_festival_pct_in_range(self, kpis):
        assert 0 <= kpis["Festival Orders (%)"] <= 100, \
            f"Festival Orders % = {kpis['Festival Orders (%)']} is outside [0, 100]"

    def test_median_le_max_delivery_time(self, df, kpis):
        max_val = float(df["Time_taken_min"].max())
        assert kpis["Median Delivery Time (min)"] <= max_val


class TestKPIEdgeCases:
    """Test edge cases and error handling."""

    def test_raises_on_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=[
            "Time_taken_min", "Preparation_Time_Min", "Road_Distance_km",
            "Rider_Rating", "Restaurant_Rating", "Average_Speed_kmph",
            "Is_Weekend", "Is_Festival",
        ])
        with pytest.raises(ValueError, match="empty"):
            compute_kpis(empty_df)


class TestDeltaKPIs:
    """Test the delta KPI helper."""

    def test_delta_kpis_returns_dict(self, df):
        result = compute_delta_kpis(df)
        assert isinstance(result, dict)

    def test_delta_kpis_not_empty(self, df):
        result = compute_delta_kpis(df)
        assert len(result) > 0

    def test_delta_values_are_floats(self, df):
        result = compute_delta_kpis(df)
        for k, v in result.items():
            assert isinstance(v, float), f"Delta '{k}' is not a float: {type(v)}"
