import multiprocessing as mp
from functools import partial
from itertools import product
from typing import Tuple, List

import numpy as np
from tqdm import tqdm


def find_neighboring_points(
    B: np.ndarray, max_coeff: int = 1
) -> Tuple[List[np.ndarray], float]:
    """
    Find the set of neighboring lattice points of the origin.
    """
    n = B.shape[1]
    neighbors = []
    max_norm = -np.inf

    # Generate all possible combinations of coefficients (excluding the zero vector)
    for coeff in product(range(-max_coeff, max_coeff + 1), repeat=n):
        if all(c == 0 for c in coeff):
            continue
        v = B @ np.array(coeff)
        neighbors.append(v)
        norm = np.linalg.norm(v)
        if norm > max_norm:
            max_norm = norm

    return neighbors, max_norm


def is_closest_to_origin(neighbors: List[np.ndarray], point: np.ndarray) -> bool:
    """
    Determine if a point is the closest to the origin among its neighbors.
    """
    ori_d = np.linalg.norm(point)
    judge = True
    for neighbor in neighbors:
        norm = np.linalg.norm(point - neighbor)
        if norm < ori_d:
            judge = False
            break
    return judge


def sample_batch(
    batch_size: int, d: int, max_norm: float, neighbors: List[np.ndarray]
) -> np.ndarray:
    points = np.random.uniform(-max_norm, max_norm, size=(batch_size, d))
    mask = np.apply_along_axis(lambda x: is_closest_to_origin(neighbors, x), 1, points)
    return points[mask]


def parallel_sampling(
    d: int,
    max_norm: float,
    neighbors: List[np.ndarray],
    sample_number: int,
    batch_size: int = 10000,
    num_workers: int = 60,
):
    manager = mp.Manager()
    points = manager.list()
    num_workers = num_workers or mp.cpu_count()
    pool = mp.Pool(processes=num_workers)
    worker = partial(sample_batch, d=d, max_norm=max_norm, neighbors=neighbors)

    pbar = tqdm(total=sample_number, desc="Sampling points", unit="points")

    while len(points) < sample_number:
        remaining = sample_number - len(points)
        current_batch_size = min(batch_size, remaining)

        # Submit asynchronous tasks and update the progress bar
        async_results = [
            pool.apply_async(worker, args=(current_batch_size,))
            for _ in range(num_workers)
        ]
        for res in async_results:
            batch_points = res.get()
            points.extend(batch_points)
            pbar.update(len(batch_points))  # Update the progress bar
            if len(points) >= sample_number:
                break

    pool.close()
    pool.join()
    pbar.close()
    return np.array(points[:sample_number])


# if __name__ == "__main__":
#     B_path = "/mnt/file2/changye/LatticeQuantizers/result/B_2.txt"
#     with open(B_path, "r") as f:
#         B = np.array([[float(num) for num in line.split()] for line in f])
#     d = B.shape[1]
#     volume = abs(np.linalg.det(B))
#     neighbors, max_norm = find_neighboring_points(B, max_coeff=1)

#     # Estimate the number of samples: nlogn rounded
#     sample_number = 500000 * d * int(np.log(d + 1))
#     points = parallel_sampling(
#         d, max_norm, neighbors, sample_number, batch_size=10000, num_workers=60
#     )

#     point_SUM = volume * np.mean(np.linalg.norm(points, axis=1) ** 2)

#     # Calculate NSM
#     NSM = point_SUM / (d * (volume ** (1 + (2 / d))))
#     print(NSM)
