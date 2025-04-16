from qiskit import QuantumCircuit
import numpy as np


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
