import argparse
import numpy as np
from LLL_algo import red



def gran(x, y):
    """
    GRAN (n, m) returns an n * m matrix of random independent real numbers,
    each with a Gaussian zero-mean, unit-variance distribution. We use the
    Gaussian random number generator by Paley and Wiener
    """
    return np.random.randn(x, y)


def uran(x):
    """
    URAN (n) returns n random real numbers, which are uniformly distributed
    in [0, 1) and independent of each other. Interpreted as a vector, URAN (n)
    returns a random point uniformly distributed in an n-dimensional hypercube.
    We use the permuted congruential generator, which is well documented
    and fulfills advanced tests of randomness.
    """
    


def clp(x, y):
    """
    The closest lattice point function CLP (B, x) finds the point in the lattice
    generated by B that is closest to x. The output is not the lattice point
    itself but rather its integer coordinates $\\argmin_{u \in Z^n} || x - uB ||^2$.

    We use [22, Algorithm 5], which is the fastest general closest-point search
    algorithm known to us. It applies to square, lower-triangular generator matrices
    with positive diagonal elements, which is exactly how lattices are represented
    in Algorithm 1.
    """
    pass



def orth(x):
    """
    The orthogonal transformation function ORTH (B) rotates and reflects an
    arbitrary generator matrix into a square, lowertriangular form with positive
    diagonal elements. This corresponds to finding a new coordinate system for
    the lattice, in which the first i unit vectors span the subspace of the
    first i basis vectors, for i = 1, . . . , n. We implement this function by
    Cholesky-decomposing [27, Sec. 4.2] the Gram matrix A = BBT. In the context
    of (2), orthogonal transformation corresponds to right-multiplying the
    generator matrix by a semiorthogonal matrix R.
    """
    pass


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stochastic Gradient Descent method for lattice quantizers"
    )

    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        required=True,
        help="Dimension of the lattice quantizer",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arg()


if __name__ == "__main__":
    main()
