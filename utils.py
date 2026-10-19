import time
import logging
import numpy as np
from scipy.special import stirling2

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logger.log', encoding='utf8', level=logging.INFO)

# decorator
def log_runtime(func):
    def wrapper(*arg,**kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        runtime = t2-t1
        logger.info(f"{func.__name__}: {runtime}")
        return res
    return wrapper


def print_runtime(func):
    def wrapper(*arg,**kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        runtime = t2-t1
        print(f"{func.__name__}: {runtime:.2f}s")
        return res
    return wrapper


def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)

def get_touchard_polynomial(sigma):
    def get_touchard_coefs(n):
        return np.array([stirling2(n, k) for k in range(n+1)])
    coefs = get_touchard_coefs(sigma)
    return np.polynomial.Polynomial(coefs)