import numpy as np
from Baseline_Lattice import B_Z, B_A2, B_D3, B_D4
from itertools import product
import random
import multiprocessing as mp
from math import log
from functools import partial
from tqdm import tqdm


#构造原点的邻接格点集
def find_neighboring_points(B, max_coeff=1):
    n = B.shape[1]
    neighbors = []
    max_norm = -np.inf
    
    # 生成所有可能的系数组合（排除全零向量）
    for coeff in product(range(-max_coeff, max_coeff+1), repeat=n):
        if all(c == 0 for c in coeff):
            continue
        v = B @ np.array(coeff)
        neighbors.append(v)
        norm=np.linalg.norm(v)
        if norm>max_norm:
            max_norm=norm
    
    return neighbors, max_norm

#判别一个点是否离原点最近
def is_closest_to_origin(neighbors, point):
    ori_d=np.linalg.norm(point)
    judge=True
    for neighbor in neighbors:
        norm = np.linalg.norm(point - neighbor)
        if norm < ori_d:
            judge=False
            break
    return judge




def sample_batch(batch_size, d, max_norm, neighbors):
    points = np.random.uniform(-max_norm, max_norm, size=(batch_size, d))
    mask = np.apply_along_axis(lambda x: is_closest_to_origin(neighbors, x), 1, points)
    return points[mask]
def parallel_sampling(d, max_norm, neighbors, sample_number, batch_size=10000, num_workers=60):
    manager = mp.Manager()
    points = manager.list()
    num_workers = num_workers or mp.cpu_count()
    pool = mp.Pool(processes=num_workers)
    worker = partial(sample_batch, d=d, max_norm=max_norm, neighbors=neighbors)

    pbar = tqdm(total=sample_number, desc="Sampling points", unit="points")

    while len(points) < sample_number:
        remaining = sample_number - len(points)
        current_batch_size = min(batch_size, remaining)
        
        # 提交异步任务，并更新进度条
        async_results = [pool.apply_async(worker, args=(current_batch_size,)) for _ in range(num_workers)]
        for res in async_results:
            batch_points = res.get()
            points.extend(batch_points)
            pbar.update(len(batch_points))  # 更新进度条
            if len(points) >= sample_number:
                break

    pool.close()
    pool.join()
    pbar.close()
    return np.array(points[:sample_number])

if __name__ == "__main__":
    B_path="/mnt/file2/changye/LatticeQuantizers/result/B_2.txt"
    with open(B_path, "r") as f:
        B = np.array([[float(num) for num in line.split()] for line in f])
    d=B.shape[1]
    volume=abs(np.linalg.det(B))
    neighbors,max_norm = find_neighboring_points(B, max_coeff=1)

    #给出估计采样数:nlogn的取整
    sample_number=500000*d*int(np.log(d+1))
    points=parallel_sampling(d, max_norm, neighbors, sample_number, batch_size=10000, num_workers=60)


    point_SUM=volume * np.mean(np.linalg.norm(points, axis=1) ** 2)

    #计算NSM
    NSM=point_SUM/(d*(volume**(1+(2/d))))
    print(NSM)
