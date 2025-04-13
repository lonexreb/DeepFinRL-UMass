import numpy as np
import pandas as pd

def compute_annualized_return(df):
    """
    Calculate the annualized return from an equity curve.
    
    Parameters:
        df: DataFrame with a 'date' column (as datetime or string) and an 'equity' column.
        
    Returns:
        The annualized return as a float.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    total_return = df['equity'].iloc[-1] / df['equity'].iloc[0] - 1
    num_years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
    return annualized_return

def compute_sharpe_ratio(df, risk_free_rate=0.0):
    """
    Calculate the Sharpe ratio for an equity curve.
    
    Parameters:
        df: DataFrame with 'date' (datetime) and 'equity' columns.
        risk_free_rate: Annual risk-free rate, default is 0.
        
    Returns:
        The Sharpe ratio as a float.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['returns'] = df['equity'].pct_change()
    daily_excess = df['returns'] - risk_free_rate / 252  # assuming 252 trading days
    sharpe_ratio = np.sqrt(252) * daily_excess.mean() / daily_excess.std()
    return sharpe_ratio

def compute_max_drawdown(df):
    """
    Calculate the maximum drawdown from an equity curve.
    
    Parameters:
        df: DataFrame with 'date' and 'equity' columns.
        
    Returns:
        The maximum drawdown as a float (absolute value).
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    running_max = df['equity'].cummax()
    drawdown = (df['equity'] - running_max) / running_max
    max_drawdown = drawdown.min()
    return abs(max_drawdown)
