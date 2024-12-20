import argparse
import numpy as np
from LLL_algo import red

# from PCG32random import generate_random_reals
from tqdm import tqdm
from lattice_quantizer.algorithm import closest_lattice_point as clp


class SGDLatticeQuantizerOptimizer:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def gran(self, x, y):
        """
        GRAN (n, m) returns an n * m matrix of random independent real numbers,
        each with a Gaussian zero-mean, unit-variance distribution. We use the
        Gaussian random number generator by Paley and Wiener
        """
        return np.random.randn(x, y)

    def uran(self, x):
        """
        URAN (n) returns n random real numbers, which are uniformly distributed
        in [0, 1) and independent of each other. Interpreted as a vector, URAN (n)
        returns a random point uniformly distributed in an n-dimensional hypercube.
        We use the permuted congruential generator, which is well documented
        and fulfills advanced tests of randomness.
        """
        # return generate_random_reals(x)
        return np.random.rand(x)

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
        return clp.new_G_based_lattices(G, r)

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
        A = np.dot(x, x.T)
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
        return red(basis)


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

    parser.add_argument(
        "-mu0",
        "--initial_step_size",
        type=float,
        required=False,
        default=0.001,  # Medium:0.001; Fast: 0.005; Slow: 0.0005
        help="Initial step size for the stochastic gradient descent",
    )

    parser.add_argument(
        "-v",
        "--radio",
        type=int,
        required=False,
        default=500,  # Medium: 500; Fast: 200; Slow: 1000
        help="Ratio between initial and final step size",
    )

    parser.add_argument(
        "-T",
        "--steps",
        type=int,
        required=False,
        default=10000000,  # Medium: 10000000; Fast: 1000000; Slow: 100000000
        help="Number of steps for the stochastic gradient descent",
    )

    parser.add_argument(
        "-Tr",
        "--reduction_interval",
        type=int,
        required=False,
        default=100,
        help="Interval between consecutive reductions",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    dimen_n = args.dimension

    Opt = SGDLatticeQuantizerOptimizer(dimen_n)
    B = Opt.orth(Opt.red(Opt.gran(dimen_n, dimen_n)))

    # V = prod_{i=1}^n B_{i,i}
    V = np.prod(np.diag(B))

    # B = V^{-1/n} B, V is a constant here
    B = (V ** (-1 / dimen_n)) * B

    mu0, v, T, Tr = (
        args.initial_step_size,
        args.radio,
        args.steps,
        args.reduction_interval,
    )

    for t in tqdm(range(T)):
        # mu <- mu_0 v^(-t/(T-1))
        mu = mu0 * (v ** -(t / (T - 1)))

        # z <- URAN(n)
        z = Opt.uran(dimen_n)

        # y <- z - CLP(B, zB)
        zB = np.dot(z, B)
        y = z - Opt.clp(B, zB)

        # e <- yB
        e = np.dot(y, B)

        B -= mu * (
            np.outer(y, e) - np.diag(y * e - np.sum(e**2) / (dimen_n * np.diag(B)))
        )

        # for i in range(dimen_n):
        #     for j in range(i):
        #         B[i, j] = B[i, j] - mu * y[i] * e[j]
        #     B[i, i] = B[i, i] - mu * (y[i] * e[i] - np.sum(e**2) / (dimen_n * B[i, i]))

        # 计算矩阵的上三角部分更新
        indices = np.tril_indices(dimen_n, k=-1)  # 获取下三角部分的索引
        B[indices] -= mu * (y[:, None] @ e[None, :])[indices]

        # 计算对角线更新
        diag_indices = np.diag_indices(dimen_n)
        B[diag_indices] -= mu * (y * e - np.sum(e**2) / (dimen_n * B[diag_indices]))

        if t % Tr == Tr - 1:
            B = Opt.orth(Opt.red(B))
            V = np.prod(np.diag(B))
            B = (V ** (-1 / dimen_n)) * B

    # Save B to file
    np.savetxt(f"B_{dimen_n}.txt", B)


if __name__ == "__main__":
    main()
