"""
test_cleaner.py
===============
Unit tests for src/cleaner.py

Tests:
  - No duplicate Order_IDs after cleaning
  - No nulls in critical columns
  - All numeric values within expected ranges
  - Binary columns contain only {0, 1}
  - Cleaned DataFrame has fewer or equal rows vs raw
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_data
from src.cleaner import clean_data


@pytest.fixture(scope="module")
def raw_df():
    """Load raw DataFrame once for all tests in this module."""
    return load_data()


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    """Run cleaning pipeline once for all tests."""
    return clean_data(raw_df)


class TestCleanerStructure:
    """Test the structure and shape of the cleaned DataFrame."""

    def test_cleaned_df_is_dataframe(self, cleaned_df):
        """Cleaned output must be a pandas DataFrame."""
        assert isinstance(cleaned_df, pd.DataFrame)

    def test_cleaned_df_not_empty(self, cleaned_df):
        """Cleaned DataFrame must have at least one row."""
        assert len(cleaned_df) > 0

    def test_cleaned_has_all_24_columns(self, cleaned_df):
        """All 24 original columns must still be present."""
        expected = [
            "Order_ID", "Order_Date", "Order_Hour", "Day_of_Week",
            "Is_Weekend", "Is_Festival", "Weather", "Pickup_Zone",
            "Dropoff_Zone", "Vehicle_Type", "Rider_Experience_Years",
            "Rider_Rating", "Restaurant_Rating", "Cuisine_Type",
            "Order_Items", "Restaurant_Load", "Preparation_Time_Min",
            "Road_Distance_km", "Delivery_Distance_Category",
            "Traffic_Level", "Number_of_Signals", "Average_Speed_kmph",
            "Delivery_Priority", "Time_taken_min",
        ]
        for col in expected:
            assert col in cleaned_df.columns, f"Column '{col}' missing from cleaned DataFrame"

    def test_fewer_or_equal_rows_after_cleaning(self, raw_df, cleaned_df):
        """Cleaning may remove rows but must not add rows."""
        assert len(cleaned_df) <= len(raw_df)


class TestDuplicates:
    """Test that duplicate records are removed."""

    def test_no_duplicate_order_ids(self, cleaned_df):
        """Each Order_ID must appear exactly once."""
        dup_count = cleaned_df["Order_ID"].duplicated().sum()
        assert dup_count == 0, f"Found {dup_count} duplicate Order_IDs"


class TestNullValues:
    """Test that critical columns have no null values after cleaning."""

    CRITICAL = [
        "Order_ID", "Order_Date", "Order_Hour", "Time_taken_min",
        "Road_Distance_km", "Preparation_Time_Min", "Average_Speed_kmph",
        "Rider_Rating", "Restaurant_Rating",
    ]

    @pytest.mark.parametrize("col", CRITICAL)
    def test_no_nulls_in_critical_columns(self, cleaned_df, col):
        """Critical columns must have no null values."""
        n_null = cleaned_df[col].isna().sum()
        assert n_null == 0, f"Column '{col}' has {n_null} null values after cleaning"


class TestValueRanges:
    """Test that numeric columns are within valid ranges."""

    def test_delivery_time_positive(self, cleaned_df):
        assert (cleaned_df["Time_taken_min"] >= 1).all(), \
            "Some Time_taken_min values are < 1"

    def test_road_distance_positive(self, cleaned_df):
        assert (cleaned_df["Road_Distance_km"] > 0).all(), \
            "Some Road_Distance_km values are <= 0"

    def test_speed_positive(self, cleaned_df):
        assert (cleaned_df["Average_Speed_kmph"] > 0).all(), \
            "Some Average_Speed_kmph values are <= 0"

    def test_rider_rating_in_range(self, cleaned_df):
        assert cleaned_df["Rider_Rating"].between(1, 5).all(), \
            "Some Rider_Rating values are outside [1, 5]"

    def test_restaurant_rating_in_range(self, cleaned_df):
        assert cleaned_df["Restaurant_Rating"].between(1, 5).all(), \
            "Some Restaurant_Rating values are outside [1, 5]"

    def test_order_hour_in_range(self, cleaned_df):
        assert cleaned_df["Order_Hour"].between(0, 23).all(), \
            "Some Order_Hour values are outside [0, 23]"

    def test_preparation_time_non_negative(self, cleaned_df):
        assert (cleaned_df["Preparation_Time_Min"] >= 0).all(), \
            "Some Preparation_Time_Min values are negative"


class TestBinaryColumns:
    """Test that binary flag columns contain only {0, 1}."""

    @pytest.mark.parametrize("col", ["Is_Weekend", "Is_Festival"])
    def test_binary_column_values(self, cleaned_df, col):
        unique_vals = set(cleaned_df[col].dropna().unique())
        assert unique_vals.issubset({0, 1}), \
            f"Column '{col}' contains non-binary values: {unique_vals}"


class TestDateRange:
    """Test that dates are within the valid range."""

    def test_order_dates_in_valid_range(self, cleaned_df):
        min_date = pd.Timestamp("2020-01-01")
        max_date = pd.Timestamp("2030-12-31")
        assert (cleaned_df["Order_Date"] >= min_date).all(), "Some dates are before 2020"
        assert (cleaned_df["Order_Date"] <= max_date).all(), "Some dates are after 2030"
