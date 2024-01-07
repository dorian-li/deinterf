from numbers import Integral

import numpy as np
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_consistent_length, column_or_1d

from .utils import filter_FOM


@validate_params(
    {
        "y": ["array-like"],
        "sampling_rate": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def noise_level_FOM(y, sampling_rate=10):
    """
    计算FOM机动飞行磁测信号的噪声水平。

    Parameters
    ----------
    y : array-like, shape of (n_samples,)
        待计算的信号。
    sampling_rate : int, default=10
        采样率。

    Returns
    -------
    noise_level : float
        噪声水平。
    """
    y = column_or_1d(y, dtype=np.float64)
    filtered = filter_FOM(y, sampling_rate=sampling_rate)
    noise_level = np.std(filtered)
    return noise_level


@validate_params(
    {
        "y_uncomp": ["array-like"],
        "y_comped": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def improve_rate_FOM(y_uncomp, y_comped, sampling_rate=10):
    """计算FOM机动飞行磁测数据补偿前后改善比

    Parameters
    ----------
    y_uncomp : array-like, shape of (n_samples,)
        未补偿的数据。
    y_comped : array-like, shape of (n_samples,)
        补偿后的数据。
    sampling_rate : int, default=10
        采样率。

    Returns
    -------
    ir : float
        改善比。
    """
    check_consistent_length(y_uncomp, y_comped)
    y_uncomp = column_or_1d(y_uncomp, dtype=np.float64)
    y_comped = column_or_1d(y_comped, dtype=np.float64)

    uncomped_noise_level = noise_level_FOM(y_uncomp, sampling_rate)
    comped_noise_level = noise_level_FOM(y_comped, sampling_rate)
    ir = uncomped_noise_level / comped_noise_level
    return ir
