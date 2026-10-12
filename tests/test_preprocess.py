"""Unit tests for data preprocessing logic."""

import pandas as pd
import pytest

from src.utils.preprocess import FEATURE_COLUMNS, TARGET_COLUMN, preprocess


def test_preprocess_duration_calculation_and_filtering():
    """Verify trip duration calculation and bounds filtering (1-60 minutes)."""
    raw_data = {
        "lpep_pickup_datetime": [
            "2024-01-01 00:00:00",  # 10 min trip (valid)
            "2024-01-01 00:00:00",  # 0.5 min trip (invalid, < 1 min)
            "2024-01-01 00:00:00",  # 75 min trip (invalid, > 60 min)
        ],
        "lpep_dropoff_datetime": [
            "2024-01-01 00:10:00",
            "2024-01-01 00:00:30",
            "2024-01-01 01:15:00",
        ],
        "PULocationID": [43, 43, 43],
        "DOLocationID": [151, 151, 151],
        "trip_distance": [2.5, 0.1, 15.0],
    }
    df = pd.DataFrame(raw_data)
    processed = preprocess(df)

    assert len(processed) == 1
    assert processed.iloc[0]["duration"] == 10.0
    assert processed.iloc[0]["PULocationID"] == "43"
    assert processed.iloc[0]["DOLocationID"] == "151"
    assert processed.iloc[0]["trip_distance"] == 2.5
