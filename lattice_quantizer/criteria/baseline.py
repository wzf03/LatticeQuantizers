import numpy as np

# 1D: Integer lattice Z
B_Z = np.array([[1.0]])
    
# 2D: A2 lattice
B_A2 = np.array([
        [1.0, 0.0],
        [-0.5, np.sqrt(3)/2]
    ])
    
 # 3D: FCC (D3) lattice
B_D3 = np.array([
        [1.0, 0.0, 0.0],
        [-0.5, np.sqrt(3)/2, 0.0],
        [-0.5, -np.sqrt(3)/6, np.sqrt(2/3)]
    ])
    
# 4D: D4 lattice
B_D4 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [-0.5, np.sqrt(3)/2, 0.0, 0.0],
        [-0.5, -np.sqrt(3)/6, np.sqrt(2/3), 0.0],
        [-0.5, -np.sqrt(3)/6, -np.sqrt(2)/3, np.sqrt(5/6)]
    ])