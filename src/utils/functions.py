from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import numpy as np
import qiskit.quantum_info as qi
from math import pi
import math
import matplotlib.pylab as plt
from itertools import product
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, state_fidelity, SparsePauliOp
import itertools
import warnings
from smt.sampling_methods import LHS
warnings.filterwarnings("ignore")
import json
import uuid


#%% QOC functions

# Function to calculate fidelity
def calculate_fidelity(real_state, pred_state):
    return qi.state_fidelity(real_state, pred_state)


def get_expectation_values(circuit, thetas, initial_state, target_state, num_qubits):
    # Create the circuit with given parameters
    circ = circuit(num_qubits=num_qubits, thetas=thetas, initial_state=initial_state)
    n_qubits = circ.num_qubits
    
    # Get coefficients and corresponding operators
    coefficients, _ = get_coefficients(target_state, n_qubits)
    operators = [Operator.from_label(i) for i in coefficients.keys()]
    
    # Run the quantum circuit on a statevector simulator
    simulator = Aer.get_backend('statevector_simulator')
    compiled_circuit = transpile(circ, simulator)
    result = simulator.run(compiled_circuit).result()
    state_vector = result.get_statevector()
    
    # Calculate fidelity
    real_fidelity = calculate_fidelity(target_state, state_vector)
    
    # Normalize state vector before calculating expectation values
    unnormalized_state_vector = state_vector.data*np.linalg.norm(state_vector)
    expectation_values = []
    for op in operators:
        expectation_values.append((np.dot(np.conj(unnormalized_state_vector), np.dot(op.data, unnormalized_state_vector))).real)   

    return expectation_values, real_fidelity, coefficients



def get_new_samples(n_samples, n_thetas):
    XNewOut = []
    for _ in range(n_samples):
        thetas_new = np.random.uniform(0,2*pi, n_thetas)
        XNewOut.append(thetas_new)
    return np.array(XNewOut)




def get_real_samples(circuit, n_samples, n_thetas, initial_state, target_state, num_qubits, type_sampling = 'Random_Uniform'):

    # Select sampling method
    if type_sampling == 'Random_Uniform':
        X_out = get_new_samples(n_samples, n_thetas)
    elif type_sampling == 'LHS':
        X_out = get_new_samples_lhs(n_samples, n_thetas)
    else:
        raise ValueError("Sampling method unavailable")
    
    # Compute expectation values and fidelities
    expectations_values = []
    fidelities_real = np.zeros(n_samples)
    
    for j in range(n_samples):
        expectation_values, fid_real, _ = get_expectation_values(circuit, X_out[j][np.newaxis, :], initial_state, target_state, num_qubits)
        expectations_values.append(expectation_values)
        fidelities_real[j] = fid_real
    
    return X_out, np.real(np.array(expectations_values)), fidelities_real



def expectation2fidelity(expectations,coefficients_array, num_qubits):
    #Calculate the Fidelity using the observables       
    fid = np.abs(np.sum((np.hstack((np.ones((np.shape(expectations)[0],1)), expectations *coefficients_array))/(2**num_qubits)), axis=1))    
    return fid



def get_coefficients(target_state, n_qubits):
    #Get the coefficients and observables, functions used to estimate the fidelity using observables
    str_list = [''.join(item) for item in product(['I','X','Y', 'Z'], repeat=n_qubits)]
    operators = [ Operator.from_label(i) for i in str_list]
    all_coef, coef = {}, {}
    for i in range(len(operators)):
        val = (np.dot(np.conj(target_state), np.dot(operators[i],target_state))).real
        all_coef[str_list[i]] = val
        if np.abs(val)>1e-8:
            coef[str_list[i]] = val
    del coef['I'*n_qubits] #Remove the Identity

    return coef, n_qubits



#%% General functions

# Generate samples using latin hypercube sampling
def get_new_samples_lhs(n_samples, n_thetas):
    xlimits = np.array([[0,2*pi]]*n_thetas )
    sampling = LHS(xlimits=xlimits)
    XNewOut = sampling(n_samples)

    return XNewOut


#Enconder json
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        elif isinstance(obj, uuid.UUID):
            return str(obj)  # Convert UUID to string
        elif isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.float32):
             return float(obj)
       
        return json.JSONEncoder.default(self, obj)





