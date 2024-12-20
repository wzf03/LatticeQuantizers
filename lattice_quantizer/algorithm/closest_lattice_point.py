import numpy as np
import numba


def validate_input(G, r):
    """Validate the input matrix G and vector r."""
    if not isinstance(G, np.ndarray) or not isinstance(r, np.ndarray):
        raise ValueError("G and r must be NumPy arrays.")
    if G.shape[0] != G.shape[1]:
        raise ValueError("G must be a square (lower triangular) matrix.")
    if r.ndim != 1 or r.shape[0] != G.shape[0]:
        raise ValueError("r must be a row vector with the same dimension as G.")
    if not np.allclose(G, np.tril(G)):
        raise ValueError("G must be lower triangular.")


def new_G_based_lattices(G, r) -> np.ndarray:
    """
    Compute the closest lattice point to the received vector r.

    Args:
        r (numpy.ndarray): Input row vector.
        G (numpy.ndarray): Lower triangular generator matrix.

    Returns:
        numpy.ndarray: Indexes of the closest lattice vector.
    """
    validate_input(G, r)
    return _new_G_based_lattices(G.shape[0], G, r)


@numba.njit
def _new_G_based_lattices(n: int, G: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Args:
        n: dimension
        G: a lower triangular n x n generator matrix with positive diagonal entries
        r: received vector
    """

    d = np.full(n, n - 1)
    u = np.zeros(n, dtype=np.int64)
    u_hat = np.zeros(n, dtype=np.int64)
    p = np.zeros(n)
    delta = np.zeros(n, dtype=np.int64)
    lambda_ = np.zeros(n + 1)
    F = np.zeros((n, n))

    C = np.inf
    i = n
    F[n - 1, :] = r

    while True:
        while True:
            if i != 0:
                i = i - 1
                for j in range(d[i], i, -1):
                    F[j - 1, i] = F[j, i] - u[j] * G[j, i]
                p[i] = F[i, i] / G[i, i]
                u[i] = np.round(p[i])
                y = (p[i] - u[i]) * G[i, i]
                delta[i] = np.sign(y)
                lambda_[i] = lambda_[i + 1] + y * y
            else:
                u_hat[:] = u
                C = lambda_[0]

            if not (lambda_[i] < C):
                break

        m = i

        while True:
            if i == n - 1:
                return u_hat
            i = i + 1
            u[i] += delta[i]
            delta[i] = -delta[i] - np.sign(delta[i])
            y = (p[i] - u[i]) * G[i, i]
            lambda_[i] = lambda_[i + 1] + y * y

            if lambda_[i] < C:
                break

        d[m:i] = i
        for j in range(m - 1, -1, -1):
            if d[j] < i:
                d[j] = i
            else:
                break
