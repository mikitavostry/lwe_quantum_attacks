import numpy as np
from qrisp import QuantumVariable, h, rz, ry, cx, x
from qrisp.algorithms.vqe import VQEProblem
from qrisp.operators.qubit import Z
from qubo_utils import solutions_with_energy_from_counts


def hamiltonian_from_qubo_last_fixed(Q: np.ndarray):
    """
    Build a Qrisp Hamiltonian for a QUBO problem with the last binary variable fixed to 1.
    Acts on N-1 qubits (indices 0..N-2). Energy matches x^T Q x with x = [x_0..x_{N-2}, 1].
    Mapping: x_i = (1 - Z_i)/2.
    """
    n = Q.shape[0]
    hamiltonian = 0
    for i in range(n):
        for j in range(n):
            qij = Q[i, j]
            if i == n - 1 and j == n - 1:
                # Both indices point to the fixed qubit (always 1)
                hamiltonian += float(qij)
            elif i == n - 1:
                # i is the fixed qubit, j is free
                hamiltonian += qij * (1 - Z(j)) * 0.5
            elif j == n - 1:
                # j is the fixed qubit, i is free
                hamiltonian += qij * (1 - Z(i)) * 0.5
            elif i == j:
                # Diagonal term (excluding last qubit)
                hamiltonian += qij * (1 - Z(i)) * 0.5
            else:
                # General off-diagonal term
                hamiltonian += qij * (1 - Z(i)) * (1 - Z(j)) * 0.25

    return hamiltonian


def create_vqe_ansatz(n_qubits: int):
    """
    One layer = RZ on each qubit -> ring CX entanglers -> RY on each qubit.
    """
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")

    params_per_layer = 2 * n_qubits

    def ansatz(qv: QuantumVariable, theta: np.ndarray):
        for i in range(n_qubits):
            rz(theta[i], qv[i])
        for i in range(n_qubits):
            cx(qv[i], qv[(i + 1) % n_qubits])
        for i in range(n_qubits):
            ry(theta[n_qubits + i], qv[i])

        return qv

    return ansatz, params_per_layer


def init_function(qv: QuantumVariable):
    """
    Prepares the initial quantum state for the algorithm.
    """
    x(qv)
    h(qv)


def solve_qubo_with_vqe(
    Q: np.ndarray,
    depth: int = 5,
    max_iter: int = 300,
):
    """
    Optimize a QUBO (last bit fixed to 1) with VQE.
    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix (full size).
    depth : int
        Number of VQE layers (ansatz repetitions).
    max_iter : int, optional
        Maximum number of classical optimization iterations.

    Returns
    -------
    processed_res : list[dict]
        Each entry contains:
        - "solution": bitstring
        - "energy": evaluated QUBO cost
        - "prob": probability
    opt_costs : list[float]
        Optimization trajectory (objective values per iteration).
    """
    if Q.shape[1] != Q.shape[0]:
        raise ValueError("Q must be square.")
    if Q.shape[0] < 2:
        raise ValueError("Q must be at least 2x2 to fix the last variable.")

    H = hamiltonian_from_qubo_last_fixed(Q)
    n_qubits = Q.shape[0] - 1
    ansatz, params_per_layer = create_vqe_ansatz(n_qubits)

    vqe = VQEProblem(
        hamiltonian=H,
        ansatz_function=ansatz,
        num_params=params_per_layer,
        init_function=init_function,
        callback=True,
    )

    prep_opt = vqe.train_function(
        QuantumVariable(n_qubits), depth=depth, max_iter=max_iter
    )
    qv = QuantumVariable(n_qubits)
    prep_opt(qv)
    counts = qv.get_measurement()
    processed_res = solutions_with_energy_from_counts(counts, Q)

    return processed_res, vqe.optimization_costs
