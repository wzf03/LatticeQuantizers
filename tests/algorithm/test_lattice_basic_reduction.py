import numpy as np

import lattice_quantizer.algorithm.lattice_basis_reduction as lbr


def test_gramschmidt_process_unnormalized():
    basis = np.array([[1, 1, 1], [-1, 0, 2], [3, 5, 6]], dtype=float)

    basis_star, mu = lbr.gramschmidt_process_unnormalized(basis)

    t = basis_star @ basis_star.T
    np.allclose(t, t * np.eye(3))

    basis = np.array([[0, 1, 0], [1, 0, 1], [-1, 0, 2]], dtype=float)

    basis_star, mu = lbr.gramschmidt_process_unnormalized(basis)
    t = basis_star @ basis_star.T
    np.allclose(t, t * np.eye(3))
    assert np.all(np.abs(mu) <= 0.5)


def test_lovasz_condition():
    basis = np.array([[0, 1, -1], [1, 0, 0], [0, 1, 2]], dtype=float)
    _, mu = lbr.gramschmidt_process_unnormalized(basis)

    assert not lbr.lovasz_condition(basis, mu, 0.75)

    basis = np.array([[0, 1, 0], [1, 0, 1], [-1, 0, 2]], dtype=float)
    _, mu = lbr.gramschmidt_process_unnormalized(basis)

    assert lbr.lovasz_condition(basis, mu, 0.75)


def test_lattice_basis_reduction():
    basis = np.array([[1, 1, 1], [-1, 0, 2], [3, 5, 6]], dtype=float)

    reduced_basis = lbr.lattice_basis_reduction(basis)
    assert np.allclose(reduced_basis, np.array([[0, 1, 0], [1, 0, 1], [-1, 0, 2]]))
