import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import numpy as np
import sys
sys.path.insert(1, '..')
import os
parallel_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parallel_folder_path)

from utils.functions import *
from models_bo import *


class BayesianOptimiserFidelity:
    def __init__(self, circuit, n_thetas, initial_state, target_state, param_bounds, num_qubits, hyperparams, num_initial_samples=50):
        self.circuit = circuit
        self.n_thetas = n_thetas
        self.initial_state = initial_state
        self.target_state = target_state
        self.param_bounds = torch.tensor(param_bounds, dtype=torch.double).T
        self.num_qubits = num_qubits
        self.num_initial_samples = num_initial_samples
        self.info =  {"best_fidelity": 0, "best_params": None, "best_log": {'fidelity':[], 'thetas':[]}}
        # Generate initial training data
        self.X_train, _, self.y_train = get_real_samples(circuit, num_initial_samples, n_thetas, initial_state, target_state, num_qubits)
        

        # Train GP Surrogate Model
        self.hyperparams = hyperparams.copy()
        self.num_iters = self.hyperparams.pop('num_iters', 100)

        self.model, self.likelihood, _, _ = train_surrogate_models_gp((self.X_train, self.y_train, [], [], self.hyperparams))
        
    def optimise(self):
        """Runs Bayesian optimisation to find optimal quantum parameters."""
        for i in range(self.num_iters):
            wrapped_gp_model = WrappedGPyTorchModelMonoGP(self.model) 
            # Define acquisition function (UCB)
            acqf = UpperConfidenceBound(wrapped_gp_model, beta=self.hyperparams['beta'])
            acqf.model.eval()
            # Optimise acquisition function to propose new sample
            candidate, _ = optimize_acqf(
                acq_function=acqf,
                bounds=self.param_bounds,
                q=1,
                num_restarts=self.hyperparams['num_restarts'],
                raw_samples=self.hyperparams['raw_samples'],
            )
            
            # Evaluate new sample
            new_x = np.array(candidate).squeeze(0)
            _, new_y, _ = get_expectation_values(self.circuit, new_x[np.newaxis,:], self.initial_state, self.target_state, self.num_qubits)
            
            # Update training data
            self.X_train = np.vstack((self.X_train, new_x[np.newaxis, :]))
            self.y_train = np.hstack((self.y_train, new_y))
            
            # Update best fidelity log
            if new_y > self.info["best_fidelity"]:
                self.info["best_fidelity"] = new_y
                self.info["best_params"] = new_x

            self.info["best_log"]['fidelity'].append(self.info["best_fidelity"])
            self.info["best_log"]['thetas'].append(self.info["best_params"])

            # Retrain GP Model
            self.model, self.likelihood, _, _ = train_surrogate_models_gp((self.X_train, self.y_train, [], [], self.hyperparams))

        # Return best parameters found
        best_idx = np.argmax(self.y_train)
        return self.X_train[best_idx], self.y_train[best_idx], self.info



class BayesianOptimiserObservables:
    def __init__(self, circuit, n_thetas, initial_state, target_state, param_bounds, num_qubits, hyperparams, num_initial_samples=50):
        self.circuit = circuit
        self.n_thetas = n_thetas
        self.initial_state = initial_state
        self.target_state = target_state
        self.param_bounds = torch.tensor(param_bounds, dtype=torch.double).T
        self.num_qubits = num_qubits
        self.num_initial_samples = num_initial_samples
        self.info =  {"best_fidelity": 0, "best_params": None, "best_log": {'fidelity':[], 'thetas':[]}}
        self.coefficients, _ = get_coefficients(target_state, self.num_qubits)
        self.coefficients_array = np.array([self.coefficients[k] for k in list(self.coefficients.keys())])
        self.n_surrogates_model = len(self.coefficients)
        # Generate initial training data
        self.X_train, self.y_train, self.y_train_fidelities = get_real_samples(circuit, num_initial_samples, n_thetas, initial_state, target_state, num_qubits)
        

        # Train GP Surrogate Model
        self.hyperparams = hyperparams.copy()
        self.num_iters = self.hyperparams.pop('num_iters', 100)

        self.surrogate_models = {'gp_model':[], 'likelihood':[], 'scaler_target':[]}
        for n_sm in range(self.n_surrogates_model):
            model, ll, _, scaler_y = train_surrogate_models_gp((self.X_train, self.y_train[:, n_sm], [], [], self.hyperparams))
            self.surrogate_models['gp_model'].append(model)
            self.surrogate_models['likelihood'].append(ll)
            self.surrogate_models['scaler_target'].append(scaler_y)


    def optimise(self):
        """Runs Bayesian optimisation to find optimal quantum parameters."""
        for i in range(self.num_iters):
            wrapped_gp_model = WrappedGPyTorchModelMultiGPFidelity(self.surrogate_models, self.coefficients_array, self.num_qubits) 
            # Define acquisition function (UCB)
            acqf = UpperConfidenceBound(wrapped_gp_model, beta=self.hyperparams['beta'])
            acqf.model.eval()
            # Optimise acquisition function to propose new sample
            candidate, _ = optimize_acqf(
                acq_function=acqf,
                bounds=self.param_bounds,
                q=1,
                num_restarts=self.hyperparams['num_restarts'],
                raw_samples=self.hyperparams['raw_samples'],
            )
            
            # Evaluate new sample
            new_x = np.array(candidate).squeeze(0)
            new_y, new_y_fid, _ = get_expectation_values(self.circuit, new_x[np.newaxis,:], self.initial_state, self.target_state, self.num_qubits)
            
            # Update training data
            self.X_train = np.vstack((self.X_train, new_x[np.newaxis, :]))
            self.y_train = np.vstack((self.y_train, new_y))
            self.y_train_fidelities = np.hstack((self.y_train_fidelities, new_y_fid))

            # Update best fidelity log
            if new_y_fid > self.info["best_fidelity"]:
                self.info["best_fidelity"] = new_y_fid
                self.info["best_params"] = new_x

            self.info["best_log"]['fidelity'].append(self.info["best_fidelity"])
            self.info["best_log"]['thetas'].append(self.info["best_params"])

            self.surrogate_models = {'gp_model':[], 'likelihood':[], 'scaler_target':[]}
            for n_sm in range(self.n_surrogates_model):
                model, ll, _, scaler_y = train_surrogate_models_gp((self.X_train, self.y_train[:, n_sm], [], [], self.hyperparams))
                self.surrogate_models['gp_model'].append(model)
                self.surrogate_models['likelihood'].append(ll)            
                self.surrogate_models['scaler_target'].append(scaler_y)

        # Return best parameters found
        best_idx = np.argmax(self.y_train_fidelities)
        return self.X_train[best_idx], self.y_train_fidelities[best_idx], self.info
