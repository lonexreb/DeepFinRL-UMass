# DeepFinRL-UMass
# Multi-Agent Framework for Personalized ETF Trading

## Overview

This project implements a sophisticated multi-agent reinforcement learning (MARL) framework for automated stock trading and personalized ETF curation. The system combines reinforcement learning algorithms with large language models in a hierarchical architecture designed to attend to individual investors' complex financial objectives.

The framework integrates specialized agents that work cooperatively to analyze financial data, generate trading signals, and optimize portfolio allocations based on user preferences and market conditions.

*(Consider adding the Flow-diagram.jpg to a `docs/` directory in your project and uncommenting the line below)*
## Key Features

-   **Multi-Agent Architecture**: Director Agent coordinates specialized analyst agents using Financial Chain-of-Thought reasoning.
-   **Hierarchical MARL**: Meta-controller policy learns to delegate to specialized agent sub-policies.
-   **LLM Integration**: Uses LLMs for sentiment analysis from financial news and plan verification.
-   **Personalized Portfolio Construction**: Tailors ETF strategies to individual investor goals and preferences.
-   **Risk Management**: Integrated verification and contrarian analysis for robustness.
-   **CrewAI Orchestration**: Leverages CrewAI for managing agent collaboration and task execution.

## System Architecture

The system is organized into several key layers based on the provided diagram:

### 1. Data Layer

-   **Sources**:
    -   Financial Market Data
    -   Technical Indicators
    -   Financial News (LLM Sentiment)
    -   User Preferences/Goals
-   **Feature Engineering & State Representation**:
    -   Fundamental Features
    -   Technical Features
    -   Sentiment Features
    -   User Profile Features

### 2. Reinforcement Learning (RL) Implementation

-   **Components**:
    -   State Space (Market & Agent Signals)
    -   Action Space (Asset Allocation)
    -   Reward Function (Returns + Risk Metrics)
-   **Algorithms**:
    -   Proximal Policy Optimization (PPO)
    -   Continual Proximal Policy Optimization (CPPO)
    -   Group Relative Policy Optimization (GRPO)
-   **Hierarchical MARL**:
    -   Director Policy (Meta-Controller)
    -   Specialized Agent Sub-Policies
    -   Credit Assignment (e.g., TD Learning)

### 3. CrewAI Orchestration Layer

-   **CrewAI Task Manager**: Manages the workflow and task execution.
-   **Multi-Agent System**:
    -   **Director Agent**: Orchestrates specialized agents using Financial Chain-of-Thought Reasoning.
    -   **Specialized Agents**:
        -   Fundamental Analysis Agent
        -   Technical Analysis Agent
        -   Contrarian Analysis Agent
    -   **Portfolio Manager Agent**: Aggregates insights using Ensemble Learning.
    -   **Plan Verification Agent**: Performs LLM-based Scoring and verification.

### 4. Output Layer

-   **Results**:
    -   Personalized ETF (Optimal Asset Allocation)
    -   Risk-Managed Returns

## Installation

```bash
# Clone the repository (replace with your actual repo URL)
git clone [https://github.com/your-username/finrl-contest.git](https://www.google.com/search?q=https://github.com/your-username/finrl-contest.git)
cd finrl-contest

# Create and activate a virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
Configuration

Edit the configuration files located in the config/ directory:

default.yaml: General configuration parameters.
agents.yaml: Agent-specific configurations, roles, goals, and backstories for CrewAI.
environment.yaml: RL environment settings (state/action space, reward function params).
data.yaml: Data source endpoints, API keys, file paths.
Data Preparation

Run the script to download and preprocess the necessary data (based on data.yaml).

Bash
# Download and prepare data
python scripts/download_data.py
# This script should handle ingestion and feature engineering pipelines.
Training

Execute the training script, specifying the main configuration file. This script should initialize the agents, environment, and CrewAI, then run the MARL training loop.

Bash
# Run training
python scripts/train.py --config config/default.yaml
Backtesting

Evaluate the performance of a trained model using historical data.

```bash
# Replace 'path/to/your/trained_model' with the actual path
python scripts/backtest.py --model_path path/to/your/trained_model --config config/default.yaml
Project Structure
finrl_contest/
│
├── config/                 # Configuration files (default.yaml, agents.yaml, etc.)
├── data/                   # Data Layer
│   ├── sources/            # Raw data acquisition modules
│   ├── connectors/         # Connectors for different data sources (API, DB, CSV)
│   └── pipelines/          # Data ingestion and processing pipelines
├── features/               # Feature Engineering Layer
│   ├── engineering/        # Feature creation modules (technical, fundamental, etc.)
│   ├── preprocessing/      # Feature scaling, normalization, selection
│   └── state/              # State representation logic
├── agents/                 # Agent Layer
│   ├── interfaces/         # Abstract base classes for agents
│   ├── specialized/        # Implementations of analyst agents
│   ├── portfolio/          # Portfolio manager agent implementation
│   ├── verification/       # Plan verification agent implementation
│   └── director/           # Director agent implementation
├── orchestration/          # Orchestration Layer (CrewAI)
│   ├── crew/               # CrewAI setup (Agent/Task definitions, Crew creation)
│   ├── tools/              # Custom tools for CrewAI agents (analysis, RL interaction)
│   └── workflows/          # Defining specific agent workflows/processes
├── rl/                     # Reinforcement Learning Layer
│   ├── environment/        # Trading environment implementation (Gymnasium)
│   ├── algorithms/         # MARL algorithm implementations (PPO, CPPO, GRPO)
│   ├── hierarchical/       # Hierarchical RL components (Meta-Controller, Sub-Policies)
│   └── models/             # Neural network models (Policy, Value networks)
├── output/                 # Output Layer
│   ├── portfolio/          # Portfolio construction and rebalancing logic
│   ├── evaluation/         # Performance metrics and benchmarking tools
│   └── visualization/      # Tools for visualizing results
├── utils/                  # Utility modules (config manager, logger, helpers)
├── factories/              # Factory classes for creating components (agents, envs)
├── tests/                  # Unit and integration tests
│   ├── unit/
│   └── integration/
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── scripts/                # Utility scripts (download_data.py, train.py, backtest.py)
│
├── requirements.txt        # Project dependencies
├── setup.py                # Package setup file (optional)
├── README.md               # This file
├── LICENSE                 # Project license file
└── main.py                 # Main entry point for the application (optional)
```
```bash
## Dependencies
Python 3.8+
PyTorch
FinRL Library (finrl)
CrewAI (crewai, crewai[tools])
Langchain (core components used by CrewAI)
OpenAI API (openai) - If using OpenAI models for LLM tasks
Gymnasium (gymnasium)
Pandas, NumPy
TA-Lib (talib-binary) - For technical indicators
Matplotlib, Seaborn - For visualization
Other dependencies as listed in requirements.txt
```

## Contributing
- We welcome contributions! Please follow these steps:

## Fork the repository.
- Create your feature branch (git checkout -b feature/YourAmazingFeature).
- Commit your changes (git commit -m 'Add some AmazingFeature').
- Push to the branch (git push origin feature/YourAmazingFeature).
- Open a Pull Request.
- Please ensure your code adheres to the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
- This project is developed as part of the FinRL Contest 2025.
- Built upon the FinRL framework.
- Inspired by concepts from the FinRobot open-source AI agent platform.
- Utilizes the CrewAI framework for agent orchestration.
Contact
- (Optional: Add contact information or link to project issues page)
- Project Link: https://github.com/your-username/finrl-contest (Replace with your actual repository URL)