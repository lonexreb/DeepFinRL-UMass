#!/bin/bash

# Create base project directory
mkdir -p finrl_contest

# Create all directories
mkdir -p finrl_contest/config
mkdir -p finrl_contest/data/sources
mkdir -p finrl_contest/data/connectors
mkdir -p finrl_contest/data/pipelines
mkdir -p finrl_contest/features/engineering
mkdir -p finrl_contest/features/preprocessing
mkdir -p finrl_contest/features/state
mkdir -p finrl_contest/agents/interfaces
mkdir -p finrl_contest/agents/specialized
mkdir -p finrl_contest/agents/portfolio
mkdir -p finrl_contest/agents/verification
mkdir -p finrl_contest/agents/director
mkdir -p finrl_contest/orchestration/crew
mkdir -p finrl_contest/orchestration/tools
mkdir -p finrl_contest/orchestration/workflows
mkdir -p finrl_contest/rl/environment
mkdir -p finrl_contest/rl/algorithms
mkdir -p finrl_contest/rl/hierarchical
mkdir -p finrl_contest/rl/models
mkdir -p finrl_contest/output/portfolio
mkdir -p finrl_contest/output/evaluation
mkdir -p finrl_contest/output/visualization
mkdir -p finrl_contest/utils
mkdir -p finrl_contest/factories
mkdir -p finrl_contest/tests/unit
mkdir -p finrl_contest/tests/integration
mkdir -p finrl_contest/notebooks
mkdir -p finrl_contest/scripts

# Create configuration files
touch finrl_contest/config/default.yaml
touch finrl_contest/config/agents.yaml
touch finrl_contest/config/environment.yaml
touch finrl_contest/config/data.yaml

# Create data layer files
touch finrl_contest/data/__init__.py
touch finrl_contest/data/sources/__init__.py
touch finrl_contest/data/sources/base_source.py
touch finrl_contest/data/sources/market_data.py
touch finrl_contest/data/sources/news_data.py
touch finrl_contest/data/sources/technical_indicators.py
touch finrl_contest/data/sources/user_preferences.py
touch finrl_contest/data/connectors/__init__.py
touch finrl_contest/data/connectors/base_connector.py
touch finrl_contest/data/connectors/csv_connector.py
touch finrl_contest/data/connectors/api_connector.py
touch finrl_contest/data/connectors/database_connector.py
touch finrl_contest/data/pipelines/__init__.py
touch finrl_contest/data/pipelines/data_pipeline.py
touch finrl_contest/data/pipelines/data_transformers.py
touch finrl_contest/data/pipelines/data_splitter.py

# Create features layer files
touch finrl_contest/features/__init__.py
touch finrl_contest/features/engineering/__init__.py
touch finrl_contest/features/engineering/feature_engineer.py
touch finrl_contest/features/engineering/fundamental_features.py
touch finrl_contest/features/engineering/technical_features.py
touch finrl_contest/features/engineering/sentiment_features.py
touch finrl_contest/features/engineering/user_profile_features.py
touch finrl_contest/features/preprocessing/__init__.py
touch finrl_contest/features/preprocessing/normalizer.py
touch finrl_contest/features/preprocessing/outlier_detector.py
touch finrl_contest/features/preprocessing/feature_selector.py
touch finrl_contest/features/state/__init__.py
touch finrl_contest/features/state/state_representation.py

# Create agents layer files
touch finrl_contest/agents/__init__.py
touch finrl_contest/agents/interfaces/__init__.py
touch finrl_contest/agents/interfaces/base_agent.py
touch finrl_contest/agents/interfaces/analyst_agent.py
touch finrl_contest/agents/interfaces/manager_agent.py
touch finrl_contest/agents/interfaces/verification_agent.py
touch finrl_contest/agents/specialized/__init__.py
touch finrl_contest/agents/specialized/fundamental_analyst.py
touch finrl_contest/agents/specialized/technical_analyst.py
touch finrl_contest/agents/specialized/contrarian_analyst.py
touch finrl_contest/agents/portfolio/__init__.py
touch finrl_contest/agents/portfolio/portfolio_manager.py
touch finrl_contest/agents/verification/__init__.py
touch finrl_contest/agents/verification/plan_verifier.py
touch finrl_contest/agents/director/__init__.py
touch finrl_contest/agents/director/director_agent.py

# Create orchestration layer files
touch finrl_contest/orchestration/__init__.py
touch finrl_contest/orchestration/crew/__init__.py
touch finrl_contest/orchestration/crew/crew_factory.py
touch finrl_contest/orchestration/crew/tasks.py
touch finrl_contest/orchestration/crew/processes.py
touch finrl_contest/orchestration/tools/__init__.py
touch finrl_contest/orchestration/tools/financial_reasoning_tool.py
touch finrl_contest/orchestration/tools/analysis_tools.py
touch finrl_contest/orchestration/tools/verification_tools.py
touch finrl_contest/orchestration/tools/rl_tools.py
touch finrl_contest/orchestration/workflows/__init__.py
touch finrl_contest/orchestration/workflows/workflow_manager.py

# Create reinforcement learning layer files
touch finrl_contest/rl/__init__.py
touch finrl_contest/rl/environment/__init__.py
touch finrl_contest/rl/environment/trading_env.py
touch finrl_contest/rl/environment/state_space.py
touch finrl_contest/rl/environment/action_space.py
touch finrl_contest/rl/environment/reward_function.py
touch finrl_contest/rl/algorithms/__init__.py
touch finrl_contest/rl/algorithms/base_algorithm.py
touch finrl_contest/rl/algorithms/ppo.py
touch finrl_contest/rl/algorithms/cppo.py
touch finrl_contest/rl/algorithms/grpo.py
touch finrl_contest/rl/hierarchical/__init__.py
touch finrl_contest/rl/hierarchical/meta_controller.py
touch finrl_contest/rl/hierarchical/sub_policies.py
touch finrl_contest/rl/hierarchical/credit_assignment.py
touch finrl_contest/rl/models/__init__.py
touch finrl_contest/rl/models/policy_network.py
touch finrl_contest/rl/models/value_network.py
touch finrl_contest/rl/models/model_factory.py

# Create output layer files
touch finrl_contest/output/__init__.py
touch finrl_contest/output/portfolio/__init__.py
touch finrl_contest/output/portfolio/allocation_strategy.py
touch finrl_contest/output/portfolio/rebalancing.py
touch finrl_contest/output/evaluation/__init__.py
touch finrl_contest/output/evaluation/performance_metrics.py
touch finrl_contest/output/evaluation/risk_metrics.py
touch finrl_contest/output/evaluation/benchmarking.py
touch finrl_contest/output/visualization/__init__.py
touch finrl_contest/output/visualization/performance_charts.py
touch finrl_contest/output/visualization/allocation_visualization.py

# Create utility files
touch finrl_contest/utils/__init__.py
touch finrl_contest/utils/config_manager.py
touch finrl_contest/utils/logger.py
touch finrl_contest/utils/helpers.py

# Create factory files
touch finrl_contest/factories/__init__.py
touch finrl_contest/factories/agent_factory.py
touch finrl_contest/factories/environment_factory.py
touch finrl_contest/factories/model_factory.py

# Create test files
touch finrl_contest/tests/__init__.py
touch finrl_contest/tests/unit/__init__.py
touch finrl_contest/tests/unit/test_agents.py
touch finrl_contest/tests/unit/test_features.py
touch finrl_contest/tests/unit/test_rl.py
touch finrl_contest/tests/integration/__init__.py
touch finrl_contest/tests/integration/test_data_pipeline.py
touch finrl_contest/tests/integration/test_agent_system.py
touch finrl_contest/tests/integration/test_rl_environment.py

# Create notebook files
touch finrl_contest/notebooks/data_exploration.ipynb
touch finrl_contest/notebooks/feature_analysis.ipynb
touch finrl_contest/notebooks/agent_behavior.ipynb
touch finrl_contest/notebooks/performance_analysis.ipynb

# Create script files
touch finrl_contest/scripts/download_data.py
touch finrl_contest/scripts/train.py
touch finrl_contest/scripts/backtest.py

# Create root files
touch finrl_contest/requirements.txt
touch finrl_contest/setup.py
touch finrl_contest/README.md
touch finrl_contest/LICENSE
touch finrl_contest/main.py

echo "Project structure created successfully!"
