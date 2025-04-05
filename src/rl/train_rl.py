import numpy as np
from rl.optimisation_rl import RLQuantumOptimiserPPO, RLQuantumOptimiserTD3
import yaml
import optuna
from optuna.samplers import TPESampler
import os
import json
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(parent_dir, 'src/utils/'))
sys.path.append(os.path.join(parent_dir, 'src/rl/'))
from functions import CustomEncoder
from circuit_utils import create_parametrized_circuit



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

    # Predefined valid (n_steps, batch_size) pairs where n_steps % batch_size == 0
    VALID_PPO_COMBINATIONS = [
        {"n_steps": 5,  "batch_size": 5},
        {"n_steps": 10, "batch_size": 5},
        {"n_steps": 10, "batch_size": 5},
        {"n_steps": 10, "batch_size": 10},
        {"n_steps": 20, "batch_size": 2},
        {"n_steps": 20, "batch_size": 5},
        {"n_steps": 20, "batch_size": 20},
        ]
    
    combo = trial.suggest_categorical("combo", VALID_PPO_COMBINATIONS)
    n_steps = combo["n_steps"]
    batch_size = combo["batch_size"]


    hyperparams = {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
        "gamma": trial.suggest_uniform("gamma", 0.95, 0.99),
        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.3),
        "n_epochs": trial.suggest_int("n_epochs",10, 30),
        "ent_coef": trial.suggest_uniform("ent_coef", 0.1, 0.3),
        "verbose":0,
        "total_timesteps": config["total_timesteps"],  # Read from YAML
        }

    hyperparams2 = {
        "batch_size": trial.suggest_categorical("batch_size", [64,128,256,512]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
        "tau": trial.suggest_uniform("tau", 0.001, 0.02),
        "noise_std": trial.suggest_uniform("noise_std", 0.001, 0.5),
        "policy_delay": trial.suggest_categorical("policy_delay", [2, 3, 4]),
        "verbose":0,
        "total_timesteps": config["total_timesteps"],  
        }


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
            optimiser = RLQuantumOptimiserPPO(
                circuit_fn=create_parametrized_circuit,
                n_thetas=config["num_qubits"] * 2,
                initial_state=config["initial_state"],
                target_state=config["target_state"],
                num_qubits=config["num_qubits"],
                hyperparams=hyperparams,
            )
        if False:
            optimiser = RLQuantumOptimiserTD3(
                circuit_fn=create_parametrized_circuit,
                n_thetas=config["num_qubits"] * 2,
                initial_state=config["initial_state"],
                target_state=config["target_state"],
                num_qubits=config["num_qubits"],
                hyperparams=hyperparams,)
        
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
    config = load_yaml_config("../../configs/config_train_rl.yaml")  # Load YAML config

    study = optuna.create_study(direction="maximize", sampler=TPESampler(multivariate=True))  # Maximize fidelity
    study.optimize(objective, n_trials=10)

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
