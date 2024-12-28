import numpy as np
from numba import float64, int32, njit, prange, types

from lattice_quantizer.algorithm import closest_lattice_point as clp


@njit(inline="always", parallel=False)
def _sample(
    basis: np.ndarray,
    z: np.ndarray,
) -> float:
    u_hat = clp.closest_lattice_point(basis, z @ basis)
    e = (z - u_hat) @ basis
    return np.sum(e**2)


@njit(
    [types.Tuple((float64, float64))(float64[:, :], int32, int32, types.npy_rng)],
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
    g = np.zeros(num_samples)
    v = np.prod(np.diag(basis))
    basis = v ** (-1 / n) * basis

    for i in range(0, num_samples, batch_size):
        batch = min(batch_size, num_samples - i)
        z = rng.random((batch, n))
        for j in prange(batch):
            g[i + j] = _sample(basis, z[j]) / n

    nsm = np.mean(g)
    var = (np.mean(g**2) - nsm**2) / (num_samples - 1)
    return nsm, var


@njit(
    [types.Tuple((float64, float64))(float64[:, :], int32, types.npy_rng)],
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
    basis = v ** (-1 / n) * basis

    for i in prange(num_samples):
        z = rng.random(n)
        g[i] = _sample(basis, z) / n

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
