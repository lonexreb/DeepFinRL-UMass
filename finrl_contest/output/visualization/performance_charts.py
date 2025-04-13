import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(df, title="Equity Curve"):
    """
    Plot the equity curve for a given DataFrame.
    
    Parameters:
        df: DataFrame with 'date' and 'equity' columns.
        title: Plot title (optional).
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['equity'], label="Equity")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparative_performance(multi_agent_df, single_agent_df, benchmark_df):
    """
    Plot comparative performance for multi-agent system, single-agent PPO, 
    and market benchmark.
    
    Parameters:
        multi_agent_df: DataFrame for multi-agent performance (with 'date' and 'equity').
        single_agent_df: DataFrame for single-agent PPO performance.
        benchmark_df: DataFrame for market benchmark performance.
    """
    plt.figure(figsize=(12, 7))
    
    for df, label in zip([multi_agent_df, single_agent_df, benchmark_df],
                           ["Multi-Agent", "Single-Agent PPO", "Market Benchmark"]):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        plt.plot(df['date'], df['equity'], label=label)
    
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Comparative Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example: simulate data for demonstration.
    dates = pd.date_range(start="2021-01-01", periods=100, freq='B')
    multi_agent_df = pd.DataFrame({"date": dates, "equity": 100 * (1 + 0.001 * pd.Series(range(100))).cumprod()})
    single_agent_df = pd.DataFrame({"date": dates, "equity": 100 * (1 + 0.0008 * pd.Series(range(100))).cumprod()})
    benchmark_df = pd.DataFrame({"date": dates, "equity": 100 * (1 + 0.0005 * pd.Series(range(100))).cumprod()})
    
    plot_equity_curve(multi_agent_df, title="Multi-Agent Equity Curve")
    plot_comparative_performance(multi_agent_df, single_agent_df, benchmark_df)
