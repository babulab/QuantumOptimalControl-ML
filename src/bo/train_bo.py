import numpy as np
from bo.optimisation_bo import BayesianOptimiserFidelity, BayesianOptimiserObservables 
import yaml
import optuna
from optuna.samplers import TPESampler
import os
import json
import sys
sys.path.insert(1, '../')
from utils.functions import CustomEncoder
from utils.circuit_utils import create_parametrized_circuit



def load_yaml_config(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    # Convert lists to numpy arrays
    config["target_state"] = np.array(config["target_state"], dtype=np.complex128)
    config["initial_state"] = np.array(config["initial_state"], dtype=np.complex128)
 
    def normalize_state(state, name):
        norm = np.linalg.norm(state)
        if not np.isclose(norm, 1.0, atol=1e-6):
            print(f"WARNING: {name} is not normalized! Normalizing it now...")
            state = state / norm  # Normalize the state
            norm_after = np.linalg.norm(state)
            print(f"{name} has been normalized: ||{name}|| = {norm_after:.6f}")
        return state

    # Normalize if necessary
    config["initial_state"] = normalize_state(config["initial_state"], "initial_state")
    config["target_state"] = normalize_state(config["target_state"], "target_state")

    return config


def objective(trial):
    # Sample hyperparameters from Optuna
    hyperparams = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-1),
        "training_iter": trial.suggest_int("training_iter", 10, 50),
        "initial_lengthscale": trial.suggest_uniform("initial_lengthscale", 0.001, 5),
        "initial_noise": trial.suggest_loguniform("initial_noise", 1e-8, 1),
        "initial_output_scale": trial.suggest_uniform("initial_output_scale", 0.1, 5),
        "beta": trial.suggest_uniform("beta",0.1, 5),
        "raw_samples": 100,
        'num_restarts': 5,
        'scaler_target': None,
        "verbose":0,
        "num_iters": config["num_iters"],  # Read from YAML
    }

    param_bounds = [[0, 2 * np.pi] for _ in range(config["num_qubits"] * 2)]

    # Number of RL runs per hyperparameter set
    n_run = config["n_run"]  

    metrics = {
        "history_best_fidelity_runs": [],
        "history_best_thetas_runs": [],
        "best_fidelity_runs": [],
        "best_thetas_runs": [],
    }

    for iter_run in range(n_run):
        print(f"Run {iter_run+1}/{n_run}")

        # Initialize RL Optimiser
        if True:
            optimiser = BayesianOptimiserFidelity(
            circuit=create_parametrized_circuit,
            n_thetas=config["num_qubits"] * 2,
            initial_state=config["initial_state"],
            target_state=config["target_state"],
            param_bounds=param_bounds,
            num_qubits=config["num_qubits"],
            num_initial_samples=1,
            hyperparams=hyperparams,
            )
        if False:
            optimiser = BayesianOptimiserObservables(
                circuit=create_parametrized_circuit,
                n_thetas=config["num_qubits"] * 2,
                initial_state=config["initial_state"],
                target_state=config["target_state"],
                param_bounds=param_bounds,
                num_qubits=config["num_qubits"],
                num_initial_samples=1, 
                hyperparams=hyperparams
                )



        # Run optimisation
        optimal_params, fidelity, info = optimiser.optimise()

        print("Optimal parameters found:", optimal_params)
        print("Optimal fidelity found:", fidelity)

        log_best_thetas = np.array(info["best_log"]["thetas"])
        log_best_fidelity = np.array(info["best_log"]["fidelity"])

        metrics["history_best_thetas_runs"].append(log_best_thetas)
        metrics["history_best_fidelity_runs"].append(log_best_fidelity)
        metrics["best_thetas_runs"].append(optimal_params)
        metrics["best_fidelity_runs"].append(fidelity)

    # Compute the average best fidelity over n_run executions
    avg_fidelity = np.mean(metrics["best_fidelity_runs"])
    trial.set_user_attr("metrics", metrics)
    
    return avg_fidelity  # maximise this




if __name__ == "__main__":
    config = load_yaml_config("../../configs/config_train_bo.yaml")  # Load YAML config

    study = optuna.create_study(direction="maximize", sampler=TPESampler(multivariate=True))  # Maximize fidelity
    study.optimize(objective, n_trials=10)

    print("Best Hyperparameters:", study.best_params)
    print("Best Fidelity:", study.best_value)

    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_fidelity = best_trial.value
    best_metrics = best_trial.user_attrs["metrics"]


    print("Best Hyperparameters:", best_params)
    print("Best Average Fidelity:", best_fidelity)

    # Save best trial results
    output_path = config["output_path_results"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    best_results = {
        "best_hyperparameters": best_params,
        "best_fidelity": best_fidelity,
        "best_metrics": best_metrics
    }

    with open(output_path, "w") as f:
        json.dump(best_results, f, cls=CustomEncoder)
