import logging
import time
import numpy as np
import cv2 as cv
from functools import wraps

logging.basicConfig(level=logging.INFO)

def time_function(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time() * 1000  # Convert start time to milliseconds
        result = f(*args, **kwargs)
        end_time = time.time() * 1000    # Convert end time to milliseconds
        duration = end_time - start_time
        logging.info(f"The function '{f.__name__}' took {duration:.4f} ms to execute.")
        return result
    return wrapper

def line_length(line):
    dist = 0
    for i in range(len(line) - 1):
        dist += np.linalg.norm(line[i] - line[i+1])

    return dist

def draw_line(img, line, color):
    for i in range(len(line) - 1):
        img = cv.line(img, line[i], line[i+1], color=color)
    return img
