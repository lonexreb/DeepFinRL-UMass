def benchmark_models(multi_agent_results, single_agent_results, market_benchmark):
    """
    Compare performance across three systems:
      - Multi-agent system.
      - Single-agent PPO model.
      - Market benchmark.
    
    Parameters:
        multi_agent_results: dict with evaluation metrics for the multi-agent system.
        single_agent_results: dict with evaluation metrics for the single-agent PPO.
        market_benchmark: dict with evaluation metrics for the market benchmark.
    
    Returns:
        summary: A dictionary summarizing the three sets of results.
    """
    summary = {
        "multi_agent": multi_agent_results,
        "single_agent": single_agent_results,
        "market_benchmark": market_benchmark,
    }
    return summary

if __name__ == "__main__":
    # Example simulated benchmark results.
    multi_agent_results = {"annualized_return": 0.15, "sharpe_ratio": 1.5, "max_drawdown": 0.2}
    single_agent_results = {"annualized_return": 0.10, "sharpe_ratio": 1.2, "max_drawdown": 0.25}
    market_benchmark = {"annualized_return": 0.07, "sharpe_ratio": 1.0, "max_drawdown": 0.3}
    
    summary = benchmark_models(multi_agent_results, single_agent_results, market_benchmark)
    print("Benchmark Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
