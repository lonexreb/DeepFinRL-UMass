import pandas as pd

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators like MA, RSI, volatility, etc."""
    df["ma_10"] = df["price"].rolling(window=10).mean()
    df["rsi"] = 100 - (100 / (1 + df["price"].pct_change().rolling(14).mean()))
    df["volatility"] = df["price"].rolling(window=10).std()
    return df