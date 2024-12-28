import numpy as np
import pytest

from lattice_quantizer.criteria import baseline, nsm


@pytest.mark.parametrize("batch_size", [0, 1024])
def test_nsm_cpu_cubic(batch_size: int):
    basis = np.eye(2, dtype=float)
    result, _ = nsm.nsm_cpu(
        basis, 100000, np.random.default_rng(0), batch_size=batch_size
    )
    assert np.allclose(result, 1 / 12, atol=1e-2)

    basis = np.eye(3, dtype=float)
    result, _ = nsm.nsm_cpu(
        basis, 100000, np.random.default_rng(0), batch_size=batch_size
    )
    assert np.allclose(result, 1 / 12, atol=1e-2)

    basis = np.eye(4, dtype=float)
    result, _ = nsm.nsm_cpu(
        basis, 100000, np.random.default_rng(0), batch_size=batch_size
    )
    assert np.allclose(result, 1 / 12, atol=1e-2)


@pytest.mark.parametrize("batch_size", [0, 1024])
def test_nsm_cpu_known(batch_size: int):
    basis = baseline.B_Z
    result, _ = nsm.nsm_cpu(
        basis, 100000, np.random.default_rng(0), batch_size=batch_size
    )
    assert np.allclose(result, 0.0833, atol=1e-2)

    basis = baseline.B_A2
    result, _ = nsm.nsm_cpu(
        basis, 100000, np.random.default_rng(0), batch_size=batch_size
    )
    assert np.allclose(result, 0.0801, atol=1e-2)

    basis = baseline.B_D3
    result, _ = nsm.nsm_cpu(
        basis, 100000, np.random.default_rng(0), batch_size=batch_size
    )
    assert np.allclose(result, 0.0789, atol=1e-2)

    basis = baseline.B_D4
    result, _ = nsm.nsm_cpu(
        basis, 100000, np.random.default_rng(0), batch_size=batch_size
    )
    assert np.allclose(result, 0.0766, atol=1e-2)
