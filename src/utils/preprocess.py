"""Data preprocessing and feature engineering for NYC Green Taxi dataset."""

import pandas as pd

FEATURE_COLUMNS = ["PULocationID", "DOLocationID", "trip_distance"]
TARGET_COLUMN   = "duration"

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean taxi records, compute duration in minutes, and cast categories."""
    df                                   = df.copy()
    df["lpep_pickup_datetime"]           = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"]          = pd.to_datetime(df["lpep_dropoff_datetime"])
    df["duration"]                       = (df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]).dt.total_seconds() / 60
    df                                   = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
    df                                   = df[df["trip_distance"].between(0.1, 100)]
    df[["PULocationID", "DOLocationID"]] = df[["PULocationID", "DOLocationID"]].astype(str)
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]
