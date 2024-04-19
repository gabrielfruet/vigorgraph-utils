import logging
import time
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

