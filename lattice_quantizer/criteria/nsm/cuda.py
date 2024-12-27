from itertools import product

import cupy as cp
import numpy as np
from tqdm import tqdm


# Construct the set of neighboring lattice points of the origin and precompute the squared norms of the neighbors
def find_neighboring_points(B, max_coeff=1):
    n = B.shape[1]
    coeffs = np.array(
        list(product(range(-max_coeff, max_coeff + 1), repeat=n)), dtype=np.int32
    )
    coeffs = coeffs[np.any(coeffs != 0, axis=1)]  # Exclude all-zero vectors
    neighbors = B @ coeffs.T  # (d, n_neighbors)
    neighbors = neighbors.T  # (n_neighbors, d)
    norms_sq = np.sum(neighbors**2, axis=1)  # Precompute squared norms
    max_norm = np.sqrt(np.max(norms_sq))  # Keep the calculation of max_norm
    return neighbors, norms_sq, max_norm


# Sample batch points on GPU using precomputed neighbor squared norms
def sample_batch_gpu(batch_size, d, max_norm, neighbors_gpu, _norms_sq_gpu, stream):
    with stream:
        # Generate random points on GPU
        points_gpu = cp.random.uniform(
            -max_norm, max_norm, size=(batch_size, d), dtype=cp.float32
        )

        # Compute the squared distance of each point to the origin
        ori_d_sq_gpu = cp.sum(points_gpu**2, axis=1).reshape(-1, 1)  # (batch_size, 1)

        # Compute the squared distance of each point to all neighbors
        diff = (
            points_gpu[:, cp.newaxis, :] - neighbors_gpu[cp.newaxis, :, :]
        )  # (batch_size, n_neighbors, d)
        distances_sq = cp.sum(diff**2, axis=2)  # (batch_size, n_neighbors)

        # Check if the squared distance to all neighbors is greater than or equal to the squared distance to the origin
        mask = cp.all(distances_sq >= ori_d_sq_gpu, axis=1)

        # Return points that meet the condition
        return points_gpu[mask].get()  # Convert back to CPU NumPy array


# Parallel sampling function using multiple GPU streams
def parallel_sampling_gpu(
    d,
    max_norm,
    neighbors,
    norms_sq,
    sample_number,
    batch_size=10000,
    device_id=0,
    num_streams=40,
):
    cp.cuda.Device(device_id).use()
    neighbors_gpu = cp.asarray(neighbors, dtype=cp.float32)
    norms_sq_gpu = cp.asarray(norms_sq, dtype=cp.float32)

    streams = [cp.cuda.Stream() for _ in range(num_streams)]

    points = []
    pbar = tqdm(total=sample_number, desc="Sampling points", unit="points")

    stream_idx = 0
    while len(points) < sample_number:
        current_batch_size = min(batch_size, sample_number - len(points))
        stream = streams[stream_idx % num_streams]
        batch_points = sample_batch_gpu(
            current_batch_size, d, max_norm, neighbors_gpu, norms_sq_gpu, stream
        )
        points.extend(batch_points)
        pbar.update(len(batch_points))
        stream_idx += 1

    pbar.close()
    return np.array(points[:sample_number])


# if __name__ == "__main__":
#     B_path = "/mnt/file2/changye/LatticeQuantizers/result/B_2.txt"
#     with open(B_path, "r") as f:
#         B = np.array(
#             [[float(num) for num in line.split()] for line in f], dtype=np.float32
#         )
#     B = B_D4
#     d = B.shape[1]
#     volume = abs(np.linalg.det(B))
#     neighbors, norms_sq, max_norm = find_neighboring_points(B, max_coeff=1)

#     # Estimate sample number: n log n rounded
#     sample_number = 1000000 * d * int(np.log(d + 1))

#     # Sample using GPU
#     points = parallel_sampling_gpu(
#         d, max_norm, neighbors, norms_sq, sample_number, batch_size=10000, num_streams=4
#     )

#     # Compute NSM
#     point_SUM = volume * np.mean(np.sum(points**2, axis=1))  # Already using squared distance
#     NSM = point_SUM / (d * (volume ** (1 + (2 / d))))
#     print(NSM)
