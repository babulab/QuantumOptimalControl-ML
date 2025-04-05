from qiskit import QuantumCircuit
import numpy as np

def create_parametrized_circuitOLD(num_qubits, thetas):
    """
    Creates a quantum circuit with a flexible number of qubits and parameterized gates.
    
    Args:
        num_qubits (int): Number of qubits.
        thetas (np.ndarray): Parameter values for the gates.
        gate_set (list): List of gate types to apply. Default is ["ry", "rx"].
    
    Returns:
        QuantumCircuit: The constructed quantum circuit.
    """
    if gate_set is None:
        gate_set = ["ry", "rx"]
    
    circ = QuantumCircuit(num_qubits)
    circ.initialize([1] + [0] * (2**num_qubits - 1), list(range(num_qubits)))
    
    # Apply parameterized gates
    theta_idx = 0
    for i, gate in enumerate(gate_set):
        for qubit in range(num_qubits):
            if gate == "ry":
                circ.ry(thetas[theta_idx], qubit)
            elif gate == "rx":
                circ.rx(thetas[theta_idx], qubit)
            elif gate == "rz":
                circ.rz(thetas[theta_idx], qubit)
            theta_idx += 1
    
    # Apply entangling gates (CNOTs in a chain)
    for qubit in range(num_qubits - 1):
        circ.cx(qubit, qubit + 1)
    
    return circ

def create_parametrized_circuit(num_qubits, thetas, initial_state, entanglement='CX'):
    """
    Creates a generic parametrized quantum circuit.
    
    Parameters:
        - num_qubits: Number of qubits in the circuit.
        - thetas: Array of parameters for rotation gates.
        - initial_state: Initial state vector.
        - entanglement: Type of entanglement ('CX', 'CZ', 'SWAP').
    """
    circ = QuantumCircuit(num_qubits)
    
    # Initialize qubits with the specified state
    circ.initialize(initial_state, list(range(num_qubits)))
    
    # Apply parameterized Ry gates
    for i in range(num_qubits):
        circ.ry(thetas[0, i], i)
    
    # Apply chosen entanglement operation
    for i in range(1, num_qubits):
        if entanglement == 'CX':
            circ.cx(0, i)
        elif entanglement == 'CZ':
            circ.cz(0, i)
        elif entanglement == 'SWAP':
            circ.swap(0, i)
    
    # Apply parameterized Rx gates
    for i in range(num_qubits):
        circ.rx(thetas[0, i+num_qubits], i)
    
    return circ