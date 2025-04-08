# agents/portfolio/portfolio_manager_agent.py
from agents.interfaces.base_agent import BaseAgent

class PortfolioManagerAgent(BaseAgent):
    def initialize(self, config):
        self.config = config
        print("PortfolioManagerAgent initialized with config:", config)

    def execute(self, verified_insights):
        # Extract scores from the verified plan.
        fundamental = verified_insights.get("fundamental", {}).get("fundamental_score", 1)
        technical = verified_insights.get("technical", {}).get("technical_score", 1)
        contrarian = verified_insights.get("contrarian", {}).get("contrarian_score", 1)
        # Simple aggregation: average the three scores.
        aggregated_score = (fundamental + technical + contrarian) / 3.0
        allocation = {"allocation": aggregated_score}
        print("PortfolioManagerAgent allocation decision:", allocation)
        return allocation

    def update(self, feedback):
        print("PortfolioManagerAgent updated with feedback:", feedback)
