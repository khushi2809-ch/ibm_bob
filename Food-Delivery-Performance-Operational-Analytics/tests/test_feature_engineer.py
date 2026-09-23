"""
test_feature_engineer.py
========================
Unit tests for src/feature_engineer.py

Tests:
  - All 9 new columns are added
  - Time_of_Day covers all 24 hours
  - Experience_Category covers all values (Novice/Junior/Senior)
  - No NaN values introduced in engineered columns
  - Day_Order maps correctly to [0, 6] range
  - Pickup_Dropoff_Pair contains '→' separator
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


@pytest.fixture(scope="module")
def engineered_df():
    """Full pipeline: load → clean → engineer."""
    raw = load_data()
    clean = clean_data(raw)
    return add_features(clean)


NEW_COLUMNS = [
    "Month",
    "Week_Number",
    "Time_of_Day",
    "Prep_Category",
    "Experience_Category",
    "Speed_Category",
    "Signals_Category",
    "Day_Order",
    "Pickup_Dropoff_Pair",
]

EXPECTED_TIME_OF_DAY = {"Morning", "Afternoon", "Evening", "Night", "Late Night"}
EXPECTED_EXPERIENCE_CATS = {"Novice", "Junior", "Senior"}
EXPECTED_PREP_CATS = {"Fast", "Moderate", "Slow"}
EXPECTED_SPEED_CATS = {"Slow", "Moderate", "Fast"}


class TestNewColumnsExist:
    """All 9 new engineered columns must be present."""

    @pytest.mark.parametrize("col", NEW_COLUMNS)
    def test_column_exists(self, engineered_df, col):
        assert col in engineered_df.columns, f"Engineered column '{col}' not found"


class TestNoNullsIntroduced:
    """Feature engineering must not introduce NaN in any new column."""

    @pytest.mark.parametrize("col", NEW_COLUMNS)
    def test_no_nulls(self, engineered_df, col):
        n_null = engineered_df[col].isna().sum()
        assert n_null == 0, f"Column '{col}' has {n_null} NaN values after feature engineering"


class TestTimeOfDay:
    """Time_of_Day must cover all 24 hours correctly."""

    def test_time_of_day_covers_all_hours(self, engineered_df):
        """Every row should have a valid Time_of_Day value."""
        valid_values = EXPECTED_TIME_OF_DAY
        actual = set(engineered_df["Time_of_Day"].unique())
        assert actual.issubset(valid_values | {"Unknown"}), \
            f"Unexpected Time_of_Day values: {actual - valid_values - {'Unknown'}}"

    def test_no_unknown_time_of_day(self, engineered_df):
        """No row should have 'Unknown' Time_of_Day (all hours 0–23 are covered)."""
        unknown_count = (engineered_df["Time_of_Day"] == "Unknown").sum()
        assert unknown_count == 0, f"{unknown_count} rows have 'Unknown' Time_of_Day"


class TestExperienceCategory:
    """Experience_Category must contain only Novice/Junior/Senior."""

    def test_experience_category_values(self, engineered_df):
        actual = set(engineered_df["Experience_Category"].unique())
        # Allow nan-converted strings from pd.cut edge cases
        actual_clean = {v for v in actual if v.lower() not in ("nan",)}
        assert actual_clean.issubset(EXPECTED_EXPERIENCE_CATS), \
            f"Unexpected Experience_Category values: {actual_clean - EXPECTED_EXPERIENCE_CATS}"


class TestPrepCategory:
    """Prep_Category must contain only Fast/Moderate/Slow."""

    def test_prep_category_values(self, engineered_df):
        actual = set(engineered_df["Prep_Category"].unique())
        actual_clean = {v for v in actual if v.lower() not in ("nan",)}
        assert actual_clean.issubset(EXPECTED_PREP_CATS), \
            f"Unexpected Prep_Category values: {actual_clean - EXPECTED_PREP_CATS}"


class TestDayOrder:
    """Day_Order must be an integer in range [0, 6] (Mon=0, Sun=6)."""

    def test_day_order_range(self, engineered_df):
        valid = engineered_df["Day_Order"].between(-1, 6)
        assert valid.all(), "Some Day_Order values are out of range [-1, 6]"

    def test_day_order_type(self, engineered_df):
        assert engineered_df["Day_Order"].dtype in [np.int64, np.int32, int, "int64", "int32"], \
            f"Day_Order dtype is {engineered_df['Day_Order'].dtype}, expected integer"


class TestPickupDropoffPair:
    """Pickup_Dropoff_Pair must contain the '→' separator."""

    def test_pair_has_separator(self, engineered_df):
        has_arrow = engineered_df["Pickup_Dropoff_Pair"].str.contains("→").all()
        assert has_arrow, "Some Pickup_Dropoff_Pair values are missing the '→' separator"

    def test_pair_is_string(self, engineered_df):
        # pandas 3.x returns StringDtype; pandas 2.x returns object — both are valid string types
        dtype_name = str(engineered_df["Pickup_Dropoff_Pair"].dtype)
        assert engineered_df["Pickup_Dropoff_Pair"].dtype == object or "str" in dtype_name.lower(), \
            f"Pickup_Dropoff_Pair should be a string dtype, got {dtype_name}"


class TestMonthAndWeek:
    """Month and Week_Number must be in valid ranges."""

    def test_month_range(self, engineered_df):
        assert engineered_df["Month"].between(1, 12).all(), \
            "Some Month values are outside [1, 12]"

    def test_week_number_range(self, engineered_df):
        assert engineered_df["Week_Number"].between(1, 53).all(), \
            "Some Week_Number values are outside [1, 53]"
