# agents/director/director_agent.py
from agents.specialized.fundamental_agent import FundamentalAgent
from agents.specialized.technical_agent import TechnicalAgent
from agents.specialized.contrarian_agent import ContrarianAgent
from agents.portfolio.portfolio_manager_agent import PortfolioManagerAgent
from agents.verification.verification_agent import VerificationAgent

class DirectorAgent:
    def __init__(self, config):
        self.config = config
        self.fundamental_agent = FundamentalAgent()
        self.technical_agent = TechnicalAgent()
        self.contrarian_agent = ContrarianAgent()
        self.portfolio_manager = PortfolioManagerAgent()
        self.verification_agent = VerificationAgent()

    def initialize_agents(self):
        # Initialize all agents with the common configuration.
        for agent in [self.fundamental_agent, self.technical_agent,
                      self.contrarian_agent, self.portfolio_manager,
                      self.verification_agent]:
            agent.initialize(self.config)
        print("All agents initialized successfully.")

    def delegate_tasks(self, data):
        print("DirectorAgent delegating tasks...")
        # Execute specialized agents in parallel (or sequentially for simplicity).
        fundamental_result = self.fundamental_agent.execute(data)
        technical_result = self.technical_agent.execute(data)
        
        # Provide relevant outputs for contrarian analysis.
        technical_data = {"technical_score": technical_result.get("technical_score", 1)}
        contrarian_result = self.contrarian_agent.execute(technical_data)
        
        # Aggregate the results.
        aggregated_insights = {
            "fundamental": fundamental_result,
            "technical": technical_result,
            "contrarian": contrarian_result
        }
        print("Aggregated insights from specialized agents:", aggregated_insights)
        
        # Verify the aggregated insights.
        verified_plan = self.verification_agent.execute(aggregated_insights)
        
        # Use the verified plan to decide on portfolio allocation.
        allocation_decision = self.portfolio_manager.execute(verified_plan)
        print("DirectorAgent final decision:", allocation_decision)
        return allocation_decision
