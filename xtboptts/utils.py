import os
import subprocess as sp
from threading import Thread
from typing import List, Tuple, Union

import numpy as np


def calc_limit_for_plot(value_list: Union[List[float], np.ndarray]) -> Tuple[float, float]:
    """
    # for pyplot return (min, max) in double
    :param value_list: list, of which element can be cast to float
    :return: (min, max)
    """
    min_v = float(np.nanmin(value_list))
    max_v = float(np.nanmax(value_list))
    buff = (max_v - min_v) * 0.05
    min_v = min_v - buff
    max_v = max_v + buff
    return min_v, max_v


def popen_bg(*args, **kwargs):
    if os.name == 'nt':
        startupinfo = sp.STARTUPINFO()
        startupinfo.dwFlags |= sp.STARTF_USESHOWWINDOW
        win_kwargs = {'startupinfo': startupinfo}
        return sp.Popen(*args, **kwargs, **win_kwargs)
    else:
        return sp.Popen(*args, **kwargs)


def async_func(func):
    def wrapper(*args, **kwargs):
        _func = Thread(target=func, args=args, kwargs=kwargs)
        _func.start()
        return _func

    return wrapper
