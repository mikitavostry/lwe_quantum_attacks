from typing import Tuple, Dict
import numpy as np

# ---------- Validation --------------------------------------------------------


def _validate_lwe_inputs(A: np.ndarray, t: np.ndarray, p: int):
    """
    Ensure shapes and dtypes are consistent for building the lattice basis.
    A: n x n (or n x m with m=n if square), t: length n, p: modulus (>1).
    """
    A = np.asarray(A)
    t = np.asarray(t).reshape(-1)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix of shape (n, n).")
    n = A.shape[0]
    if t.shape[0] != n:
        raise ValueError(f"t must be a vector of length n={n}.")
    if not (isinstance(p, (int, np.integer)) and p > 1):
        raise ValueError("p must be an integer > 1.")
    return A, t, int(p)


# ---------- Lattice embedding (Bai–Galbraith style) --------------------------


def build_lattice_basis_B(A: np.ndarray, t: np.ndarray, p: int):
    """
    Construct the lattice basis B used to embed LWE:
        B = [  I_n   -A^T    0 ]
            [  0     p I_n   0 ]
            [  0      t^T    1 ]

    Shapes:
        A: (n, n)
        t: (n,)
        B: (2n+1, 2n+1)
    """
    A, t, p = _validate_lwe_inputs(A, t, p)
    n = A.shape[0]

    I_n = np.eye(n, dtype=float)

    B_upper = np.hstack([I_n, -A.T.astype(float), np.zeros((n, 1))])
    B_middle = np.hstack([np.zeros((n, n)), p * I_n, np.zeros((n, 1))])
    B_lower = np.hstack(
        [np.zeros((1, n)), t.reshape(1, -1).astype(float), np.array([[1.0]])]
    )

    B = np.vstack([B_upper, B_middle, B_lower])
    return B


# ---------- QUBO construction ------------------------------------------------


def qubo_from_lattice_B(B: np.ndarray):
    """
    Given lattice basis B, build the QUBO matrix.
    """
    B = np.asarray(B, dtype=float)
    return B @ B.T


def build_qubo_from_lwe(A: np.ndarray, t: np.ndarray, p: int):
    """
    Convenience wrapper: from provided LWE parameters (A, t, p) produce:
        - B : lattice basis (2n+1 x 2n+1)
        - Q : QUBO matrix = B B^T
    """
    B = build_lattice_basis_B(A, t, p)
    Q = qubo_from_lattice_B(B)
    return B, Q


def qubo_obj(bitstring: str, Q: np.ndarray, offset: float = 0) -> float:
    """Evaluate QUBO objective x^T Q x + offset for a bitstring."""
    x = np.fromiter(bitstring, dtype=int)
    return float(x.T @ Q @ x + offset)


def solutions_with_energy_from_counts(
    counts: Dict[str, int],
    Q: np.ndarray,
    mode: str = "vqe",
    offset: float = 0.0,
):
    """
    For each measured bitstring over (N-1) qubits, compute the energy of the full assignment (bitstring + '1').
    """
    if mode not in {"vqe", "qaoa"}:
        raise ValueError("mode must be 'vqe' or 'qaoa'")

    def as_str(key) -> str:
        if isinstance(key, (list, tuple, np.ndarray)):
            return "".join(map(str, key))
        return str(key)

    result = []
    for key, value in counts.items():
        s = as_str(key)
        x = np.fromiter(s, dtype=int)
        if mode == "vqe":
            x = np.append(x, 1)
        full_str = s + "1"
        energy = float(x.T @ Q @ x + offset)
        result.append({"solution": full_str, "energy": energy, "prob": value})

    return result


def fix_last_var(Q: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Eliminate the last binary variable by fixing it to 1.
    Returns the reduced Q' (without the last row/col) and the constant offset.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if Q.shape[1] != n:
        raise ValueError("Q must be square.")
    offset = float(Q[n - 1, n - 1])
    Qp = Q[: n - 1, : n - 1].copy()
    for i in range(n - 1):
        Qp[i, i] += 2.0 * Q[i, n - 1]
    return Qp, offset
