# Quantum Optimal Control with Machine Learning

##  Overview
This repository explores the application of **Machine Learning (ML)** techniques in the field of **Quantum Optimal Control (QOC)**.

The primary objective is to prepare the Greenberger–Horne–Zeilinger (GHZ) state, a highly entangled quantum state, by optimising the fidelity of the quantum system.

This work is motivated by real experimental constraints: we aim to achieve high performance using a limited number of iterations, rather than relying on convergence criteria of the objective function. This makes the approach more viable for practical, resource-constrained quantum experiments.

The diagram below outlines the general optimisation framework. While the depicted optimiser is based on Bayesian Optimisation, the same workflow can accommodate other optimisation strategies.

![Diagram](https://github.com/babulab/QuantumOptimalControl-ML/blob/main/figures/diagram_exp.jpg?raw=true)


Currently, the repository includes implementations of the following optimisation techniques:
- Bayesian Optimisation
- Reinforcement Learning
- Simultaneous Perturbation Stochastic Approximation (SPSA)


A key design principle is modularity and flexibility. Components are loosely packaged to allow for easy customisation and experimentation, making this a useful tool for researchers and developers interested in hybrid quantum-classical optimisation.

---

If you're interested in the results, the good news is that they are summarised below. The figure demonstrates that Bayesian Optimization based models often outperform other approaches, achieving superior fidelity in GHZ state preparation. 

> ⚠️ Note: These findings are problem-specific and should not be generalised without considering the constraints, computational limitations, and further discussion in the accompanying [results notebook](https://github.com/babulab/QuantumOptimalControl-ML/blob/main/notebooks/results.ipynb).

![Figure](https://github.com/babulab/QuantumOptimalControl-ML/blob/main/figures/results_3_qubit_GHZ_infidelity.png?raw=true)

---


## 🚀 Getting Started

### Prerequisites

To install the required dependencies, it is recommended to use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Quick Start

To easily explore how the library works, run one of the basic training notebooks:

- `notebooks/train_bo.ipynb` – Run a simple Bayesian Optimisation example
- `notebooks/train_rl.ipynb` – Run a basic Reinforcement Learning example

These notebooks provide a hands-on way to understand the optimization process.

### Advanced Usage

If you want to include **hyperparameter optimization**, you can run the Python scripts directly:

- `src/bo/train_bo.py` – Bayesian Optimisation with hyperparameter tuning
- `src/rl/train_rl.py` – Reinforcement Learning with hyperparameter tuning
- `src/others/train_spsa.py` – SPSA with hyperparameter tuning

### New to Quantum Optimal Control?

Start here:
- `notebooks/Introduction_QOC.ipynb` – A beginner-friendly introduction to the principles of Quantum Optimal Control

---

## 🔮 Future Updates

Planned enhancements for the repository include:
- Implementation of new optimisation techniques (e.g., Genetic Programming, currently in progress)
- Comparative analysis with **GRAPE** (Gradient Ascent Pulse Engineering)

---

## Acknowledgments

This repository is based on my Master's research project:  
*Observable-Guided Bayesian Optimisation for Quantum Circuits Fidelity and Ground State Energy Estimation*  
conducted at **Imperial College London**.

Special thanks to **Florian M.** for his supervision and guidance throughout the project.


## Requirements

- qiskit    >=1.2 
- qiskit_algorithms    >=0.3
- gpytorch  >=1.12
- botorch   >= 0.11
- SMT: Surrogate Modeling Toolbox   >=2.6
- matplotlib    >=3.9
- seaborn   >=3.9
- gym   >=0.26
- stable_baselines3    >=2.3
- mlflow    >=2.16 
- optuna    >=4.2 
