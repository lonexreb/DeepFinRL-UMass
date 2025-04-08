import pandas as pd

def split_by_time_period(df: pd.DataFrame):
    """
    Splits the DataFrame into training (2015–2019), validation (2020), and test (2021–2022).
    Assumes the DataFrame has a 'date' column in string or datetime format.
    """
    df["date"] = pd.to_datetime(df["date"])
    train_df = df[(df["date"].dt.year >= 2015) & (df["date"].dt.year <= 2019)]
    val_df = df[df["date"].dt.year == 2020]
    test_df = df[(df["date"].dt.year >= 2021) & (df["date"].dt.year <= 2022)]
    return train_df, val_df, test_df
