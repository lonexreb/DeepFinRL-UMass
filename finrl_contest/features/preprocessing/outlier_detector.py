import pandas as pd
import numpy as np
from scipy import stats

def remove_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Removes rows with z-score > threshold for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    mask = (z_scores < threshold).all(axis=1)
    return df[mask]