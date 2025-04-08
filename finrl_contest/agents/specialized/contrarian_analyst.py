# agents/specialized/contrarian_agent.py
from agents.interfaces.base_agent import BaseAgent

class ContrarianAgent(BaseAgent):
    def initialize(self, config):
        self.config = config
        print("ContrarianAgent initialized with config:", config)

    def execute(self, data):
        # Using technical score as input and invert it (avoid division by zero).
        technical_score = data.get("technical_score", 1)
        result = {"contrarian_score": 1 / technical_score if technical_score != 0 else 0}
        print("ContrarianAgent output:", result)
        return result

    def update(self, feedback):
        print("ContrarianAgent updated with feedback:", feedback)
