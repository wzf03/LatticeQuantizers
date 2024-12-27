import numpy as np

from lattice_quantizer.criteria import baseline, nsm


def test_nsm_cpu_cubic():
    basis = np.eye(2, dtype=float)
    result = nsm.nsm_cpu(basis, 100000, 1024, np.random.default_rng(0))
    assert np.allclose(result, 1 / 12, atol=1e-2)

    basis = np.eye(3, dtype=float)
    result = nsm.nsm_cpu(basis, 100000, 1024, np.random.default_rng(0))
    assert np.allclose(result, 1 / 12, atol=1e-2)

    basis = np.eye(4, dtype=float)
    result = nsm.nsm_cpu(basis, 100000, 1024, np.random.default_rng(0))
    assert np.allclose(result, 1 / 12, atol=1e-2)


def test_nsm_cpu_known():
    basis = baseline.B_Z
    result = nsm.nsm_cpu(basis, 100000, 1024, np.random.default_rng(0))
    assert np.allclose(result, 0.0833, atol=1e-2)

    basis = baseline.B_A2
    result = nsm.nsm_cpu(basis, 100000, 1024, np.random.default_rng(0))
    assert np.allclose(result, 0.0801, atol=1e-2)

    basis = baseline.B_D3
    result = nsm.nsm_cpu(basis, 100000, 1024, np.random.default_rng(0))
    assert np.allclose(result, 0.0789, atol=1e-2)

    basis = baseline.B_D4
    result = nsm.nsm_cpu(basis, 100000, 1024, np.random.default_rng(0))
    assert np.allclose(result, 0.0766, atol=1e-2)
