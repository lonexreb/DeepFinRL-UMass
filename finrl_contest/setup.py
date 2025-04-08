from setuptools import setup, find_packages

setup(
    name="finrl_contest",
    version="0.1.0",
    description="DeepFinRL-UMass Multi-Agent Framework for Personalized ETF Trading",
    author="UMass FinRL Team",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0",
        "torch>=1.10.0",
        "gymnasium>=0.28.0",
        "pytest>=7.0.0",
    ],
    python_requires=">=3.8",
)
