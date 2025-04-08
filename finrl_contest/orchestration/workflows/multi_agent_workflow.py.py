# orchestration/workflows/multi_agent_workflow.py
import yaml
from agents.director.director_agent import DirectorAgent

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
    
    # Delegate tasks through the DirectorAgent.
    allocation = director.delegate_tasks(data)
    print("Final Portfolio Allocation Decision:", allocation)

if __name__ == "__main__":
    main()
