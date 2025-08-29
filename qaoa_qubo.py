from typing import Dict
import numpy as np
from qrisp import QuantumArray, QuantumVariable, h, rx, x
from qrisp.algorithms.qaoa import QAOAProblem, create_QUBO_cost_operator
from qubo_utils import solutions_with_energy_from_counts, qubo_obj


def create_qubo_cl_cost_function(Q: np.ndarray, offset: float):
    """
    Build a classical cost function that maps measurement counts to expected energy.
    """

    def cl_cost(counts: Dict[str, int]):
        energy = 0.0
        for meas, cnt in counts.items():
            energy += qubo_obj(meas, Q, offset) * cnt
        return energy

    return cl_cost


def mixer(qv: QuantumArray, beta: float):
    """Standard RX mixer applied to all qubits."""
    for i in range(qv.size):
        rx(2 * beta, qv[i])

    return qv


def init_function(qv: QuantumArray):
    """
    Prepares the initial quantum state for the algorithm.
    """
    x(qv)
    h(qv)


def solve_qubo(
    Q: np.ndarray,
    depth: int,
    max_iter: int = 450,
    offset: float = 0.0,
):
    """
    Run QAOA on a QUBO instance.

    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix (already reduced if last bit is fixed).
    depth : int
        Number of QAOA layers.
    max_iter : int, optional
        Maximum number of classical optimization iterations (default: 450).
    offset : float, optional
        Energy offset from fixed last variable (default: 0.0).

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

    cost_op = create_QUBO_cost_operator(Q)
    cl_cost = create_qubo_cl_cost_function(Q, offset)

    problem = QAOAProblem(
        cost_op,
        mixer,
        cl_cost,
        callback=True,
        init_function=init_function,
    )

    results = problem.run(
        QuantumVariable(len(Q)),
        depth,
        max_iter=max_iter,
    )
    processed_res = solutions_with_energy_from_counts(
        results, Q, mode="qaoa", offset=offset
    )
    return processed_res, problem.optimization_costs
