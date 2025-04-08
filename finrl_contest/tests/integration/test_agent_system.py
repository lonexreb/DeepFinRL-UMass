# tests/integration/test_agent_system.py

import unittest
import yaml
import os
import numpy as np
from pathlib import Path

from agents.director.director_agent import DirectorAgent
from agents.specialized.fundamental_analyst import FundamentalAgent
from agents.specialized.technical_analyst import TechnicalAgent
from agents.specialized.contrarian_analyst import ContrarianAgent
from agents.portfolio.portfolio_manager import PortfolioManagerAgent
from agents.interfaces.verification_agent import VerificationAgent

class TestAgentSystem(unittest.TestCase):
    def setUp(self):
        # Create a simple test configuration
        self.config = {
            'fundamental': {'threshold': 0.5},
            'technical': {'window_size': 14},
            'contrarian': {'sensitivity': 1.0},
            'portfolio': {'risk_tolerance': 0.7},
            'verification': {'method': 'dummy'}
        }
        
        # Sample data for testing
        self.test_data = {
            'fundamentals': [0.8, 0.9, 0.7],
            'prices': [100, 105, 102, 110, 108]
        }
    
    def test_individual_agents(self):
        """Test that each agent can be initialized and executed individually."""
        # Test FundamentalAgent
        fundamental_agent = FundamentalAgent()
        fundamental_agent.initialize(self.config)
        fundamental_result = fundamental_agent.execute(self.test_data)
        self.assertIn('fundamental_score', fundamental_result)
        self.assertIsInstance(fundamental_result['fundamental_score'], (int, float))
        
        # Test TechnicalAgent
        technical_agent = TechnicalAgent()
        technical_agent.initialize(self.config)
        technical_result = technical_agent.execute(self.test_data)
        self.assertIn('technical_score', technical_result)
        self.assertIsInstance(technical_result['technical_score'], (int, float))
        
        # Test ContrarianAgent
        contrarian_agent = ContrarianAgent()
        contrarian_agent.initialize(self.config)
        # ContrarianAgent needs technical_score as input
        contrarian_data = {'technical_score': technical_result['technical_score']}
        contrarian_result = contrarian_agent.execute(contrarian_data)
        self.assertIn('contrarian_score', contrarian_result)
        self.assertIsInstance(contrarian_result['contrarian_score'], (int, float))
        
        # Test VerificationAgent
        verification_agent = VerificationAgent()
        verification_agent.initialize(self.config)
        aggregated_insights = {
            'fundamental': fundamental_result,
            'technical': technical_result,
            'contrarian': contrarian_result
        }
        verified_plan = verification_agent.execute(aggregated_insights)
        self.assertIn('verification', verified_plan)
        
        # Test PortfolioManagerAgent
        portfolio_manager = PortfolioManagerAgent()
        portfolio_manager.initialize(self.config)
        allocation = portfolio_manager.execute(verified_plan)
        self.assertIn('allocation', allocation)
        self.assertIsInstance(allocation['allocation'], (int, float))
    
    def test_director_agent_workflow(self):
        """Test the complete workflow through the DirectorAgent."""
        director = DirectorAgent(self.config)
        director.initialize_agents()
        
        # Run the complete workflow
        result = director.delegate_tasks(self.test_data)
        
        # Verify the result
        self.assertIn('allocation', result)
        self.assertIsInstance(result['allocation'], (int, float))
        
        # The allocation should be a value derived from the test data
        # (this is a simple check, the actual value depends on the agent implementations)
        self.assertGreater(result['allocation'], 0)

if __name__ == '__main__':
    unittest.main()
