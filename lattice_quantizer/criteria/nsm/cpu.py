import numpy as np
from numba import float64, int32, int64, njit, prange, types

from lattice_quantizer.algorithm import closest_lattice_point as clp


@njit(inline="always")
def _sample(
    basis: np.ndarray,
    z: np.ndarray,
) -> float:
    u_hat = clp.closest_lattice_point(basis, z @ basis)
    e = (z - u_hat) @ basis
    return np.sum(e**2)


@njit(
    [types.Tuple((float64, float64))(float64[:, :], int64, int32, types.npy_rng)],
    parallel=True,
    cache=True,
)
def _nsm_cpu_batched(
    basis: np.ndarray,
    num_samples: int,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    n = basis.shape[0]

    v = np.prod(np.diag(basis))
    coeffi = v ** (-2 / n) / n
    g_mean = 0.0
    g2_mean = 0.0

    for i in range(0, num_samples, batch_size):
        batch = min(batch_size, num_samples - i)
        g = np.zeros(batch_size)
        z = rng.random((batch, n))
        for j in prange(batch):
            g[j] = _sample(basis, z[j]) * coeffi
        g_mean += np.mean(g) * (batch / num_samples)
        g2_mean += np.mean(g**2) * (batch / num_samples)

    nsm = g_mean
    var = (g2_mean - nsm**2) / (num_samples - 1)
    return nsm, var


@njit(
    [types.Tuple((float64, float64))(float64[:, :], int64, types.npy_rng)],
    cache=True,
)
def _nsm_cpu(
    basis: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> float:
    n = basis.shape[0]
    g = np.zeros(num_samples)
    v = np.prod(np.diag(basis))
    coeffi = v ** (-2 / n) / n  # Note: V = det(B), not V = sqrt(det(B)) in the paper

    for i in prange(num_samples):
        z = rng.random(n)
        g[i] = _sample(basis, z) * coeffi

    nsm = np.mean(g)
    var = (np.mean(g**2) - nsm**2) / (num_samples - 1)
    return nsm, var


@njit(cache=True)
def nsm_cpu(
    basis: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
    batch_size: int = 0,
) -> float:
    if batch_size == 0:
        return _nsm_cpu(basis, num_samples, rng)
    return _nsm_cpu_batched(basis, num_samples, batch_size, rng)
