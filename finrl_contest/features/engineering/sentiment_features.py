import numpy as np
import random

def compute_sentiment_score(headline: str) -> float:
    """
    Dummy sentiment analysis function that simulates output from an LLM.
    Replace this logic with actual model inference later if needed.
    """
    # Simulate sentiment score between -1 and 1
    return round(random.uniform(-1, 1), 3)

def apply_sentiment_to_dataframe(df):
    """
    Applies sentiment scoring to a DataFrame that has a 'headline' column.
    Adds a new 'sentiment_score' column with values from -1 to 1.
    """
    if "headline" not in df.columns:
        raise ValueError("DataFrame must contain a 'headline' column for sentiment analysis.")

    df["sentiment_score"] = df["headline"].apply(compute_sentiment_score)
    return df
