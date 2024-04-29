import utils
import numpy as np

def _ramer_douglas_peucker_rec(points: np.ndarray, epsilon: float) -> np.ndarray:
    if len(points) < 3: return points
    first, last = points[[0,-1]]
    y1,x1 = last

    delta_y, delta_x = last - first
    m = delta_y/(delta_x + 1e-10)
    a = m
    b = -1
    c = y1 - m*x1

    distances = np.abs(np.dot(points, [b,a]) + c) /  np.sqrt(a**2 + b**2)
    max_i = np.argmax(distances)
    max_dist = distances[max_i]

    if max_dist > epsilon:
        result1 = ramer_douglas_peucker(points[0:max_i+1], epsilon)
        result2 = ramer_douglas_peucker(points[max_i:-1], epsilon)
        return np.r_[result1, result2]

    return points[[0,-1]]

@utils.time_function
def ramer_douglas_peucker(points: np.ndarray, epsilon: float) -> np.ndarray:
    return _ramer_douglas_peucker_rec(points, epsilon)
