# agents/interfaces/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def initialize(self, config):
        """Load configuration parameters, models, or other setups."""
        pass

    @abstractmethod
    def execute(self, data):
        """Perform the core logic of the agent."""
        pass

    @abstractmethod
    def update(self, feedback):
        """Update internal state or model parameters based on feedback."""
        pass
    
    def communicate(self, message):
        """(Optional) Send or process inter-agent messages."""
        return message
