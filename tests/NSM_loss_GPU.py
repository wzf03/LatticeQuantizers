import numpy as np
import cupy as cp
from Baseline_Lattice import B_Z, B_A2, B_D3, B_D4
from itertools import product
from math import log
from tqdm import tqdm

# 构造原点的邻接格点集，并预计算邻居的平方范数
def find_neighboring_points(B, max_coeff=1):
    n = B.shape[1]
    coeffs = np.array(list(product(range(-max_coeff, max_coeff+1), repeat=n)), dtype=np.int32)
    coeffs = coeffs[np.any(coeffs != 0, axis=1)]  # 排除全零向量
    neighbors = B @ coeffs.T  # (d, n_neighbors)
    neighbors = neighbors.T  # (n_neighbors, d)
    norms_sq = np.sum(neighbors**2, axis=1)  # 预先计算平方范数
    max_norm = np.sqrt(np.max(norms_sq))  # 保持 max_norm 的计算
    return neighbors, norms_sq, max_norm

# 在GPU上采样批量点，使用预计算的邻居平方范数
def sample_batch_gpu(batch_size, d, max_norm, neighbors_gpu, norms_sq_gpu, stream):
    with stream:
        # 在GPU上生成随机点
        points_gpu = cp.random.uniform(-max_norm, max_norm, size=(batch_size, d), dtype=cp.float32)
        
        # 计算每个点到原点的平方距离
        ori_d_sq_gpu = cp.sum(points_gpu**2, axis=1).reshape(-1, 1)  # (batch_size, 1)
        
        # 计算每个点到所有邻居的平方距离
        diff = points_gpu[:, cp.newaxis, :] - neighbors_gpu[cp.newaxis, :, :]  # (batch_size, n_neighbors, d)
        distances_sq = cp.sum(diff**2, axis=2)  # (batch_size, n_neighbors)
        
        # 检查是否所有邻居的平方距离都大于或等于原点的平方距离
        mask = cp.all(distances_sq >= ori_d_sq_gpu, axis=1)
        
        # 返回满足条件的点
        return points_gpu[mask].get()  # 转回CPU的NumPy数组

# 并行采样函数，使用多个GPU流
def parallel_sampling_gpu(d, max_norm, neighbors, norms_sq, sample_number, batch_size=10000, device_id=0, num_streams=40):
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
        batch_points = sample_batch_gpu(current_batch_size, d, max_norm, neighbors_gpu, norms_sq_gpu, stream)
        points.extend(batch_points)
        pbar.update(len(batch_points))
        stream_idx += 1
    
    pbar.close()
    return np.array(points[:sample_number])

if __name__ == "__main__":
    B_path = "/mnt/file2/changye/LatticeQuantizers/result/B_2.txt"
    with open(B_path, "r") as f:
        B = np.array([[float(num) for num in line.split()] for line in f], dtype=np.float32)
    B=B_D4
    d = B.shape[1]
    volume = abs(np.linalg.det(B))
    neighbors, norms_sq, max_norm = find_neighboring_points(B, max_coeff=1)
    
    # 估计采样数: n log n 的取整
    sample_number = 1000000 * d * int(np.log(d + 1))
    
    # 使用GPU进行采样
    points = parallel_sampling_gpu(d, max_norm, neighbors, norms_sq, sample_number, batch_size=10000, num_streams=4)
    
    # 计算NSM
    point_SUM = volume * np.mean(np.sum(points**2, axis=1))  # 已经使用平方距离
    NSM = point_SUM / (d * (volume ** (1 + (2 / d))))
    print(NSM)
