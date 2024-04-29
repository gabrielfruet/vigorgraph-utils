import utils
import numpy as np
from numba import njit

@njit
def find_neighbours(i: int,j: int, size: np.ndarray):
    pos = np.array([j,i])
    directions = np.array([[1,-1], [1,0], [1,1],[0,-1],[0,1],[-1,-1],[-1,0],[-1,1]])
    valid_neighbours = []
    for direc in directions:
        new_pos = direc + pos
        x,y = new_pos
        if x < size[1] and x >= 0 and y < size[0] and y >= 0:
            valid_neighbours.append(new_pos)
    
    return valid_neighbours

@utils.time_function
@njit
def edge_linking(A: np.ndarray):
    """

    A is a np.ndarray with 0s, and 1s

    """
    A = A.copy()
    links = []
    n,m =  A.shape
    size = np.array(A.shape)

    for i in range(n):
        for j in range(m):
            if A[i,j] == 255:
                A[i,j] = 0
                k,l = i,j
                new_link = [(j,i)]
                available_paths=True
                while available_paths:
                    available_paths = False
                    for q,p in find_neighbours(k,l,size):
                        if A[p,q] == 255:
                            A[p,q] = 0
                            new_link.append((q,p))
                            k,l = p,q
                            available_paths = True
                            break

                links.append(np.array(new_link))

    return links
