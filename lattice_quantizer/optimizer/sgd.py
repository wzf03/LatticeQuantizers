import numpy as np
from tqdm import tqdm

from lattice_quantizer.algorithm import closest_lattice_point as clp
from lattice_quantizer.algorithm import lattice_basis_reduction as lbr


class SGDLatticeQuantizerOptimizer:
    def __init__(
        self,
        dimension: int,
        initial_step_size: float = 0.001,
        radio: int = 500,
        steps: int = 10000000,
        reduction_interval: int = 100,
    ):
        self.dimension = dimension
        self.initial_step_size = initial_step_size
        self.radio = radio
        self.steps = steps
        self.reduction_interval = reduction_interval
        self.rng = np.random.default_rng()

    def gran(self):
        """
        GRAN returns an n * m matrix of random independent real numbers,
        each with a Gaussian zero-mean, unit-variance distribution. We use the
        Gaussian random number generator by Paley and Wiener
        """
        return self.rng.normal(0, 1, (self.dimension, self.dimension))

    def uran(self):
        """
        URAN (n) returns n random real numbers, which are uniformly distributed
        in [0, 1) and independent of each other. Interpreted as a vector, URAN (n)
        returns a random point uniformly distributed in an n-dimensional hypercube.
        We use the permuted congruential generator, which is well documented
        and fulfills advanced tests of randomness.
        """
        return self.rng.random(self.dimension)

    def clp(self, G, r):
        """
        The closest lattice point function CLP (B, x) finds the point in the lattice
        generated by B that is closest to x. The output is not the lattice point
        itself but rather its integer coordinates $\\argmin_{u \in Z^n} || x - uB ||^2$.

        We use [22, Algorithm 5], which is the fastest general closest-point search
        algorithm known to us. It applies to square, lower-triangular generator matrices
        with positive diagonal elements, which is exactly how lattices are represented
        in Algorithm 1.
        """
        return clp.closest_lattice_point(G, r)

    def orth(self, x):
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
        A = x @ x.T
        return np.linalg.cholesky(A)

    def red(self, basis):
        """
        The reduction function RED(B) returns another generator matrix for the
        lattice generated by B, in which the rows (basis vectors) are shorter and
        more orthogonal to each other than in B. If no improved generator matrix is
        found, B is returned unchanged. A fast and popular algorithm for the purpose
        is the Lenstra-Lenstra-Lov ́asz algorithm [24, Fig. 1], which we apply in
        this work. In the context of (2), reduction corresponds to finding a suitable U .
        """
        return lbr.lattice_basis_reduction(basis)

    def optimize(self) -> np.ndarray:
        n = self.dimension
        B = self.orth(self.red(self.gran()))
        V = np.prod(np.diag(B))
        B = (V ** (-1 / n)) * B

        mu0, v, T, Tr = (
            self.initial_step_size,
            self.radio,
            self.steps,
            self.reduction_interval,
        )

        for t in tqdm(range(T)):
            mu = mu0 * (v ** -(t / (T - 1)))
            z = self.uran()
            y = z - self.clp(B, z @ B)
            e = y @ B

            for i in range(n):
                for j in range(i):
                    B[i, j] = B[i, j] - mu * y[i] * e[j]
                B[i, i] = B[i, i] - mu * (y[i] * e[i] - np.sum(e**2) / (n * B[i, i]))

            if t % Tr == Tr - 1:
                B = self.orth(self.red(B))
                V = np.prod(np.diag(B))
                B = (V ** (-1 / n)) * B

        return B
