import numpy as np


def lower_featuremap_resolution(map, rf):  # rf = reduction_factor
    """
    Reduces a matrix "resolution" by a reduction factor. If we have a 64x64 matrix and rf=4 the map
    will be reduced to 16x16 in which
    every new element of the matrix is an average from 4x4=16 elements from the original matrix
    """
    if rf == 1:
        return map

    N, M = map.shape
    N = N // rf
    M = M // rf

    reduced_map = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            # reduction_array = map[rf*i:rf*i+rf, rf*j:rf*j+rf].flatten()
            # reduced_map[i,j]  = Counter(reduction_array).most_common(1)[0][0]

            reduced_map[i, j] = (map[rf * i:rf * i + rf, rf * j:rf * j + rf].sum()) / (rf * rf)

    return reduced_map
