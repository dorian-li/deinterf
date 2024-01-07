from numbers import Integral

import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import FLOAT_DTYPES, check_array


@validate_params(
    {
        "X": ["array-like"],
        "bandpass_range": [tuple],
        "sampling_rate": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def filter_FOM(X, bandpass_range=(0.1, 0.6), sampling_rate=10):
    """对输入的FOM飞行相关信号应用带通滤波器。

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        待滤波的输入信号。
    bandpass_range : tuple (min_freq, max_freq), default=(0.1, 0.6)
        滤波器的通带范围。
    sampling_rate : int, default=10
        采样频率，单位Hz。

    Returns
    -------
    filtered : array-like of shape (n_samples, n_features)
        滤波后的信号。
    """
    X = check_array(X, copy=True, ensure_2d=False, dtype=FLOAT_DTYPES)
    b, a = butter(
        4,
        Wn=list(bandpass_range),
        btype="bandpass",
        fs=sampling_rate,
        output="ba",
    )
    filtered = filtfilt(b, a, X, axis=0)
    return filtered


class EllipsoidFitter(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.x0 = None
        self.y0 = None
        self.z0 = None
        self.a = None
        self.b = None
        self.c = None

    def fit(self, X, y=None):
        def cost(params):
            x0, y0, z0, a, b, c = params
            residuals = (
                (X[:, 0] - x0) ** 2 / a**2
                + (X[:, 1] - y0) ** 2 / b**2
                + (X[:, 2] - z0) ** 2 / c**2
                - 1
            ) ** 2
            return np.sum(residuals)

        initial_guess = [0, 0, 0, 1, 1, 1]
        result = minimize(cost, initial_guess, method="L-BFGS-B")
        self.x0, self.y0, self.z0, self.a, self.b, self.c = result.x
        return self

    def predict(self, X):
        return (
            (X[:, 0] - self.x0) ** 2 / self.a**2
            + (X[:, 1] - self.y0) ** 2 / self.b**2
            + (X[:, 2] - self.z0) ** 2 / self.c**2
            - 1
        ) ** 2

    def score(self, X, y=None):
        # 这里的得分可以根据实际情况进行定义
        residuals = self.predict(X)
        return np.mean(residuals)
