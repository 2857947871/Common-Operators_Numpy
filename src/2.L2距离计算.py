import numpy as np

# L2 = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
# cloud1: [m, d]   cloud2: [n, d]
# dist: 距离矩阵 [m, n]
# dist[i, j]: cloud1[i] 与 cloud2[j] 之间的 L2 距离
def L2_1(cloud1, cloud2):
    m, n = len(cloud1), len(cloud2)
    cloud1 = np.repeat(cloud1, n, axis=0)
    cloud1 = np.reshape(cloud1, (m, n, -1))
    
    # Calculate the L2 distance
    dist = np.sqrt(np.sum((cloud1 - cloud2)**2, axis=2))
    return dist

def L2_2(cloud1, cloud2):
    cloud1 = cloud1[:, None, :] # (m, n) -> (m, 1, n)
    
    # Calculate the L2 distance
    dist = np.sqrt(np.sum((cloud1 - cloud2)**2, axis=2))
    return dist

def L2_3(cloud1, cloud2):
    cloud1 = np.expand_dims(cloud1, axis=1)
    
    # Calculate the L2 distance
    dist = np.sqrt(np.sum((cloud1 - cloud2)**2, axis=2))
    return dist


if __name__ == "__main__":
    cloud1 = np.array([[1, 2], [3, 4], [5, 6]])
    cloud2 = np.array([[7, 8], [9, 10]])

    # 计算成对的L2距离
    distance_matrix_1 = L2_1(cloud1, cloud2)
    distance_matrix_2 = L2_2(cloud1, cloud2)
    distance_matrix_3 = L2_3(cloud1, cloud2)
    print("The pairwise L2 distance matrix is:")
    print(distance_matrix_1)
    print(distance_matrix_2)
    print(distance_matrix_3)
