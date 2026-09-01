import time
import logging

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