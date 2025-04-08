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