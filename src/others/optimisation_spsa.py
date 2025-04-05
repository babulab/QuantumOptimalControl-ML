from qiskit_algorithms.optimizers import SPSA
import numpy as np
import sys
import os
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_path)
from functions import *

class SPSAOptimiser:
    def __init__(self, circuit, n_thetas, initial_state, target_state, num_qubits, hyperparams):
        """
        SPSA-based optimiser for quantum circuit optimisation.
        """
        self.circuit = circuit
        self.n_thetas = n_thetas
        self.initial_state = initial_state
        self.target_state = target_state
        self.num_qubits = num_qubits
        self.hyperparams = hyperparams.copy()
        self.model = SPSA(**self.hyperparams)
        self.info = {}
        self.X_train = []
        self.y_train = []
        self.best_fidelity = -float("inf")
        self.best_thetas = np.zeros(self.n_thetas)
        self.fidelity_history = []
        self.thetas_history = []

    def _compute_fidelity(self, thetas):
        """Compute the fidelity of the circuit with given parameters."""
        _, fidelity, _ = get_expectation_values(self.circuit, thetas, 
                                                self.initial_state, self.target_state, 
                                                self.num_qubits)
        return -fidelity 
    

    def _objective_function(self, thetas):
        """
        Wrapper function for the objective function that tracks fidelity and updates best parameters.
        """
        fidelity = self._compute_fidelity(thetas)
        
        self.X_train.append(thetas[0])
        self.y_train.append(fidelity)

        if np.abs(fidelity) > self.best_fidelity:
            self.best_fidelity = np.abs(fidelity)
            self.best_thetas = thetas

        self.fidelity_history.append(self.best_fidelity)
        self.thetas_history.append(self.best_thetas)

        return fidelity 
     

    def optimise(self):
        """
        Runs SPSA optimisation.
        """

        initial_thetas = np.random.uniform(0, 2 * np.pi, self.n_thetas)[np.newaxis,:]

        results = self.model.minimize(
                self._objective_function,
                x0=initial_thetas
                )
        optimal_params =  np.array(self.thetas_history)[self.hyperparams['maxiter']]
        optimal_fidelity =  np.abs(np.array(self.fidelity_history)[self.hyperparams['maxiter']])
        self.info = {'best_fidelity': optimal_fidelity,
                     'best_params': optimal_params,
                     'best_log':{'fidelity':np.abs(np.array(self.fidelity_history)[:self.hyperparams['maxiter']]),
                                 'thetas': np.array(self.thetas_history)[:self.hyperparams['maxiter']]}}
   
        return optimal_params, optimal_fidelity, self.info   