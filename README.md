# Quantum Optimal Control with Machine Learning


##  Overview
This repository explores the application of Machine Learning (ML) techniques in the field of Quantum Optimal Control (QOC).

The main goal is to prepare the Greenberger–Horne–Zeilinger (GHZ) state, a highly entangled quantum state. The preparation process aims to optimise the fidelity.


The approach is designed taking into account experimental contexts, where good results are sought with few iterations, for the same reason, the algorithms have been designed taking into account a reduced number of iterations and not a convergence criterion of the objective function. 
The following flowchart presents how the optimisations have been carried out, in the case of the figure the optimiser corresponds to a system based on Bayesian Optimisation, but it could also be another type of optimiser.

![Diagram](https://github.com/babulab/QuantumOptimalControl-ML/blob/main/figures/diagram_exp.jpg?raw=true)



Currently, the repository implements optimisation techniques based on: Bayesian Optimization, Reinforcement Learning and SPSA.

A key goal of this repository is to provide a flexible and accessible platform for research. To achieve this, the components of the algorithm are lightly packaged, allowing for easy customization and modification.

## Future Updates 


Future improvements to the repository:
    - New optimisation techniques (Genetic Programing in progress)
    - A benchmark suite for comparing different optimisation techniques, including new target states
    - Improvements to the existing Bayesian Optimisation framework, including:
        - Parallelise training of the GPs associated with the observables
        - Enhancing the evaluation of GP performance
    


This repository is based on my research project,‘Observable-Guided Bayesian Optimisation for Quantum Circuits Fidelity and Ground State Energy Estimation’, conducted during my Master's degree at Imperial College London. I would like to extend my gratitude to Florian M. for his supervision throughout this project.


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
