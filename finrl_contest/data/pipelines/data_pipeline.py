import pandas as pd
import numpy as np
from finrl_contest.features.engineering.fundamental_features import compute_fundamental_features
from finrl_contest.features.engineering.technical_features import compute_technical_features
from finrl_contest.features.engineering.sentiment_features import apply_sentiment_to_dataframe
from finrl_contest.data.pipelines.data_splitter import split_by_time_period
from finrl_contest.features.preprocessing.outlier_detector import remove_outliers_zscore
from finrl_contest.features.preprocessing.normalizer import impute_missing_values

def load_fnspid_csv(path: str) -> pd.DataFrame:
    """Load the FNSPID dataset from a local CSV file."""
    return pd.read_csv(path)


def run_preprocessing_pipeline(csv_path: str):
    """
    Runs the full data preprocessing pipeline.
    Steps: Load → Clean → Engineer Features → Sentiment → Split
    """
    df = pd.read_csv(csv_path)

    # Cleaning
    df = impute_missing_values(df)
    df = remove_outliers_zscore(df)

    # Feature Engineering
    df = compute_fundamental_features(df)
    df = compute_technical_features(df)
    df = apply_sentiment_to_dataframe(df)

    # Split
    train, val, test = split_by_time_period(df)

    # Optionally save
    train.to_csv("output/train.csv", index=False)
    val.to_csv("output/val.csv", index=False)
    test.to_csv("output/test.csv", index=False)

    return train, val, test