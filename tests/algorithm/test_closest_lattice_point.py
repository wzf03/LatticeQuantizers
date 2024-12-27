import numpy as np

import lattice_quantizer.algorithm.closest_lattice_point as clp


def test_clp_ngbl():
    G = np.array([[1, 0], [1 / 2, np.sqrt(3) / 2]])
    x = np.array([0.5, 0.5])
    assert (clp.new_G_based_lattices(G, x) == np.array([0, 1])).all()

    x = np.array([-0.5, 0.5])
    res = clp.new_G_based_lattices(G, x)
    assert (res == np.array([-1, 1])).all() or (res == np.array([0, -1])).all()
