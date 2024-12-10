from setuptools import setup, find_packages

setup(
    name="eos_neural_nets",
    version="1.0.0",
    description="Neural Network Models for Equation of State",
    author="Semih Kacmaz",
    author_email="skacmaz2@illinois.edu",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
)
