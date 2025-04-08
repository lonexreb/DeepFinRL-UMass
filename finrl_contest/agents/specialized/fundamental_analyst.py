# agents/specialized/fundamental_agent.py
from agents.interfaces.base_agent import BaseAgent

class FundamentalAgent(BaseAgent):
    def initialize(self, config):
        self.config = config
        print("FundamentalAgent initialized with config:", config)
        # Initialize any required financial models or parameters here.

    def execute(self, data):
        # For demonstration, we compute a simple average of "fundamentals".
        fundamentals = data.get("fundamentals", [1])
        score = sum(fundamentals) / len(fundamentals)
        result = {"fundamental_score": score}
        print("FundamentalAgent output:", result)
        return result

    def update(self, feedback):
        print("FundamentalAgent updated with feedback:", feedback)
