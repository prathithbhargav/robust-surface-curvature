import numpy as np
def unit_vector(x):
    return x/np.linalg.norm(x)

def unit_normal(v1, v2, v3):
    v1, v2, v3 = np.array(v1), np.array(v2), np.array(v3)
    v31 = v3 - v1
    v21 = v2 - v1
    cross = np.cross(v31/np.linalg.norm(v31), v21/np.linalg.norm(v21))
    return cross/ np.linalg.norm(cross)