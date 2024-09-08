# Quantum Optimal Control with Machine Learning


##  Overview
This repository explores the application of Machine Learning (ML) techniques in the field of Quantum Optimal Control (QOC).

Our primary focus is on preparing the Greenberger–Horne–Zeilinger (GHZ) state, a highly entangled quantum state. The preparation process revolves around optimising the fidelity, which is derived from the expectation values of certain observables.



The approach is designed with experimental contexts in mind, assuming prior knowledge of the quantum system (in this case, a quantum circuit) represented by a set of initial parameters $\theta_{0}$​. During the optimisation process, experiments are conducted on the system to iteratively maximise or minimise the figure of merit—fidelity. The workflow is illustrated in the following diagram:


![Diagram](https://github.com/babulab/QuantumOptimalControl-ML/blob/main/figures/diagram_exp.jpg?raw=true)



Currently, the repository implements optimisation techniques based on Bayesian Optimization.

A key goal of this repository is to provide a flexible and accessible platform for research. To achieve this, the components of the algorithm are lightly packaged, allowing for easy customization and modification.

## Future Updates 


We plan to enhance the repository with the following features:

    - New optimisation techniques (Reinforcement Learning is nearly complete)
    - A benchmark suite for comparing different optimisation techniques, including support for new target states
    - Improvements to the existing Bayesian Optimization framework, including:
        - Parallelize training of the GPs associated with the observables
        - Enhancing the evaluation of GP performance
    

This repository is based on my research project,‘Observable-Guided Bayesian Optimisation for Quantum Circuits Fidelity and Ground State Energy Estimation’, conducted during my Master's degree at Imperial College London. I would like to extend my gratitude to Florian M. for his supervision throughout this project.


## Requirements

- Qiskit
- GPyTorch
- BoTorch
- SMT: Surrogate Modeling Toolbox
- Matplotlib
- Seaborn
