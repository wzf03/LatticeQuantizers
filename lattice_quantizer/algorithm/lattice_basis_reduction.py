from typing import Tuple

import numpy as np
from numba import boolean, float64, njit, types


@njit(
    [
        float64(float64[:], float64[:]),
    ]
)
def _inner_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the dot product of a vector with itself.
    """
    return np.sum(vec1 * vec2)


@njit(
    [
        float64(float64[:]),
    ]
)
def _norm_squared(vec: np.ndarray) -> float:
    """
    Computes the squared norm of a vector.
    """
    return _inner_product(vec, vec)


@njit(
    [
        boolean(float64[:, :], float64[:, :], float64),
    ]
)
def lovasz_condition(basis: np.ndarray, mu: np.ndarray, delta: float) -> bool:
    """
    Checks the Lovasz condition for LLL reduction.
    """
    n = basis.shape[0]
    for i in range(1, n):
        if delta * _norm_squared(basis[i - 1]) > (
            _norm_squared(basis[i]) + mu[i, i - 1] ** 2 * _norm_squared(basis[i - 1])
        ):
            return False
    return True


@njit(
    [
        types.Tuple((float64[:, :], float64[:, :]))(float64[:, :]),
    ]
)
def gramschmidt_process_unnormalized(
    basis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Gram-Schmidt orthogonalization without normalization.

    Args:
        basis: A basis matrix.

    Returns:
        A tuple containing the orthogonalized basis and the coefficients.
    """
    n = basis.shape[0]

    basis_star = np.zeros_like(basis)
    mu = np.zeros((n, n), dtype=basis.dtype)
    denominators = np.zeros(n, dtype=basis.dtype)

    basis_star[0, :] = basis[0, :]
    denominators[0] = _norm_squared(basis[0])

    for i in range(1, n):
        basis_star[i, :] = basis[i, :]
        for j in range(i):
            mu[i, j] = _inner_product(basis[i], basis_star[j]) / denominators[j]
            basis_star[i, :] -= mu[i, j] * basis_star[j]
        denominators[i] = _norm_squared(basis_star[i])

    return basis_star, mu


@njit(
    [
        float64[:, :](float64[:, :], float64),
        float64[:, :](float64[:, :], types.Omitted(0.75)),
    ],
    cache=True,
)
def lattice_basis_reduction(basis: np.ndarray, delta: float = 0.75) -> np.ndarray:
    """
    Computes LLL reduction algorithm on a given basis.
    Based on pseudo-code HPS 7.13.
    """

    n = basis.shape[0]

    basis_star, mu = gramschmidt_process_unnormalized(basis)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if np.abs(mu[k, j]) > 0.5:
                basis[k, :] -= np.round(mu[k, j]) * basis[j, :]
                basis_star, mu = gramschmidt_process_unnormalized(basis)

        if _norm_squared(basis_star[k]) > (
            (delta - mu[k, k - 1] ** 2) * _norm_squared(basis_star[k - 1])
        ):
            k += 1
        else:
            tmp = basis[k, :].copy()
            basis[k, :] = basis[k - 1, :]
            basis[k - 1, :] = tmp

            basis_star, mu = gramschmidt_process_unnormalized(basis)
            k = max(k - 1, 1)

    return basis
