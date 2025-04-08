# agents/specialized/technical_agent.py
from agents.interfaces.base_agent import BaseAgent

class TechnicalAgent(BaseAgent):
    def initialize(self, config):
        self.config = config
        print("TechnicalAgent initialized with config:", config)
        # Set up technical indicator parameters here.

    def execute(self, data):
        # For demonstration, we calculate a basic average of "prices".
        prices = data.get("prices", [1])
        score = sum(prices) / len(prices)
        result = {"technical_score": score}
        print("TechnicalAgent output:", result)
        return result

    def update(self, feedback):
        print("TechnicalAgent updated with feedback:", feedback)
