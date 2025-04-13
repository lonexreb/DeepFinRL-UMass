# orchestration/workflows/multi_agent_workflow.py
import yaml
import sys
from agents.director.director_agent import DirectorAgent

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
def run_multi_agent_experiment(data, agents_config):
    """
    Placeholder function to simulate running the multi-agent experiment.
    Replace this simulation with your actual multi-agent experiment code.
    
    Parameters:
        data: Input market data.
        agents_config: Dictionary indicating which agents are active.
        
    Returns:
        A dictionary with simulated performance metrics.
    """
    import random
    results = {
        "annualized_return": round(random.uniform(0.0, 0.3), 3),
        "sharpe_ratio": round(random.uniform(0.0, 3.0), 3),
        "max_drawdown": round(random.uniform(0.0, 0.5), 3)
    }
    print("Simulated experiment results:", results)
    return results

def run_ablation_study(data, disable_agents=None):
    """
    Runs an ablation study on the multi-agent system by disabling selected agents.
    
    Parameters:
        data: Input market data.
        disable_agents: List of strings indicating the agents to disable (e.g., ['Fundamental', 'Technical']).
                        
    Returns:
        results: A dictionary containing simulated performance metrics.
    """
    # Default configuration: all agents enabled.
    agents_config = {
        "Fundamental": True,
        "Technical": True,
        "Contrarian": True,
        "PlanVerification": True
    }
    
    # Disable specified agents.
    if disable_agents is not None:
        for agent in disable_agents:
            if agent in agents_config:
                agents_config[agent] = False

    print("Running ablation study with the following agent configuration:")
    print(agents_config)
    
    results = run_multi_agent_experiment(data, agents_config)
    return results


def main():
    # Load configuration from the provided YAML file.
    config = load_config("config/default.yaml")
    
    # Initialize the DirectorAgent with the configuration.
    director = DirectorAgent(config)
    director.initialize_agents()
    
    # Simulated data input.
    data = {
        "fundamentals": [10, 20, 30],   # Example fundamental metrics.
        "prices": [100, 105, 102]        # Example price series for technical analysis.
    }
    
    # Choose mode based on command-line arguments.
    # If the script is run with the argument 'ablation', perform an ablation study.
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        # Example: disable the 'Technical' agent.
        results = run_ablation_study(data, disable_agents=["Technical"])
        print("Ablation Study Results:", results)
    else:
        # Default behavior: delegate tasks through the DirectorAgent.
        allocation = director.delegate_tasks(data)
        print("Final Portfolio Allocation Decision:", allocation)

if __name__ == "__main__":
    main()
