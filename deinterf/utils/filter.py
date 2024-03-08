from numbers import Integral

from scipy.signal import butter, filtfilt
from sklearn.utils._param_validation import Interval, validate_params
from sklearn.utils.validation import check_array


@validate_params(
    {
        "X": ["array-like"],
        "bandpass_range": [tuple],
        "sampling_rate": [Interval(Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def fom_bpfilter(X, bandpass_range=(0.1, 0.6), sampling_rate=10):
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
    X = check_array(X, copy=True, ensure_2d=False)
    b, a = butter(
        4,
        Wn=list(bandpass_range),
        btype="bandpass",
        fs=sampling_rate,
        output="ba",
    )
    filtered = filtfilt(b, a, X, axis=0)
    return filtered
