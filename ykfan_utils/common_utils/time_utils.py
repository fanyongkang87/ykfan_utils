import time
import functools
import numpy as np


def time_compute(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        tmp = fun(*args, **kwargs)
        end_time = time.time()
        print('{} cost {} s.'.format(fun.__name__, end_time-start_time))
        return tmp

    return wrapper


def time_avg_compute(fun):
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        times = []
        tmp = None
        for _ in range(20):
            start_time = time.time()
            tmp = fun(*args, **kwargs)
            end_time = time.time()
            times.append(end_time-start_time)

        times = np.array(times)
        times = times[5:]
        print('{} avg cost {} s.'.format(fun.__name__, times.mean()))
        return tmp

    return wrapper
