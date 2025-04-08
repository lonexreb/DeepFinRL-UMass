#!/usr/bin/env python
# main.py

import argparse
import yaml
import os
from agents.director.director_agent import DirectorAgent

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_agent_workflow(config):
    """Run the multi-agent workflow with the given configuration."""
    # Initialize the director agent
    director = DirectorAgent(config)
    director.initialize_agents()
    
    # Sample data for demonstration
    sample_data = {
        "fundamentals": [0.8, 0.9, 0.7],
        "prices": [100, 105, 102, 110, 108]
    }
    
    # Run the workflow
    result = director.delegate_tasks(sample_data)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='DeepFinRL-UMass Multi-Agent Framework')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config(config_path)
    
    # Run the workflow
    result = run_agent_workflow(config)
    
    print("\nFinal allocation result:", result)

if __name__ == "__main__":
    main()
