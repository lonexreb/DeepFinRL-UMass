import pandas as pd

def compute_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute key financial ratios like P/E, debt ratio, ROE, etc."""
    df["pe_ratio"] = df["price"] / df["earnings_per_share"]
    df["debt_ratio"] = df["total_liabilities"] / df["total_assets"]
    df["roe"] = df["net_income"] / df["shareholder_equity"]
    return df

