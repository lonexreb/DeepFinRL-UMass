import pandas as pd

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values using forward fill and backward fill."""
    return df.ffill().bfill()