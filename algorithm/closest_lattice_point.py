import numpy as np
import numba


@numba.njit
def new_G_based_lattices(n: int, G: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Args:
        n: dimension
        G: a lower triangular n x n generator matrix with positive diagonal entries
        r: received vector
    """
    C = np.inf

    i = n
    d = np.full(n, n - 1)
    lambda_ = np.zeros(n + 1)
    lambda_[n] = 0
    F = np.zeros((n, n))
    F[-1, :] = r

    u = np.zeros(n)
    u_hat = np.zeros(n)
    delta = np.zeros(n)

    loop_flag = True
    while loop_flag:
        loop_flag = False
        while True:
            if i != 0:
                i -= 1
                for j in range(d[i], i + 1):
                    F[j - 1, i] = F[j, i] - u[j] * G[j, i]
                p_i = F[i, i] / G[i, i]
                u[i] = np.round(p_i)
                y = (p_i - u[i]) * G[i, i]
                delta[i] = np.sign(y)
                lambda_[i] = lambda_[i + 1] + y * y
            else:
                u_hat = u
                C = lambda_[0]

            if not (lambda_[i] < C):
                break

        m = i

        while True:
            if i == n - 1:
                return u_hat
            else:
                i = i + 1
                u[i] = u[i] + delta[i]
                delta[i] = -delta[i] - np.sign(delta[i])
                y = (p_i - u[i]) * G[i, i]
                lambda_[i] = lambda_[i + 1] + y * y

            if not (lambda_[i] >= C):
                break

        for j in range(m, i):
            d[j] = i
        for j in reversed(range(1, m)):
            if d[j] < i:
                d[j] = i
            else:
                break
        loop_flag = True

