from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy import ndarray
from scipy import linalg
from scipy.signal import detrend
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.linear_model import RidgeCV
from sklearn.utils import (
    FLOAT_DTYPES,
    Interval,
    Options,
    _num_features,
    check_array,
    check_consistent_length,
    check_is_fitted,
)

from .utils import filter_FOM


class TollesLawsonTMICompensation(
    OneToOneFeatureMixin, TransformerMixin, BaseEstimator
):
    """根据Tolles-Lawson磁干扰模型补偿磁总场数据

    Parameters
    ----------
    with_band_pass_filter : bool, default=True
        是否对输入数据进行带通滤波，滤波频率范围为[0.1, 0.5] Hz
    with_permanent_field : bool, default=True
        是否使用永久磁场项
    with_induced_field : bool, default=True
        是否使用感应磁场项
    with_eddy_field : bool, default=True
        是否使用涡流磁场项
    with_geomag_gradient : bool, default=False
        是否使用地磁梯度项
    base_tl_model_version : int, default=16
        指定基础T-L模型的版本，可选值为16或18，分别对应16项系数和18项系数模型
    with_scaling_factor_between_model_items : bool, default=True
        是否在模型项之间使用缩放系数
    sampling_rate : int, default=10
        采样频率，单位为Hz
    copy : bool, default=True
        是否复制输入数据

    Attributes
    ----------
    model_ : RidgeCV
        拟合后的模型

    Examples
    --------
    >>> from deinterf.attitude_estimation import MagVectorAttitudeEstimation
    >>> from deinterf.compensation import TollesLawsonTMICompensation
    >>> from deinterf.metrics import improve_rate_FOM
    >>> import numpy as np
    >>> mag_vec = np.tile(np.eye(3), (1000, 1))
    >>> attitude_features = MagVectorAttitudeEstimation().transform(mag_vec)
    >>> tmi_with_interf = np.linspace(5e4, 6e4, 3000)
    >>> compensation = TollesLawsonTMICompensation()
    >>> compensation.fit(attitude_features, tmi_with_interf)
    TollesLawsonTMICompensation()
    >>> tmi_pure = compensation.transform(attitude_features, tmi_with_interf)
    >>> print(tmi_pure)
    [-0.5  0.5  1.5]
    """

    _estimator_type = "regressor"
    _parameter_constraints: dict = {
        "with_band_pass_filter": ["boolean"],
        "with_permanent_field": ["boolean"],
        "with_induced_field": ["boolean"],
        "with_eddy_field": ["boolean"],
        "with_geomag_gradient": ["boolean"],
        "base_tl_model_version": [Options(Integral, {16, 18})],
        "with_scaling_factor_between_model_items": ["boolean"],
        "sampling_rate": [Interval(Integral, 1, None, closed="left")],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        with_band_pass_filter=True,
        with_permanent_field=True,
        with_induced_field=True,
        with_eddy_field=True,
        with_geomag_gradient=False,
        base_tl_model_version=16,
        with_scaling_factor_between_model_items=True,
        sampling_rate=10,
        copy=True,
    ) -> None:
        self.with_band_pass_filter = with_band_pass_filter
        self.with_permanent_field = with_permanent_field
        self.with_induced_field = with_induced_field
        self.with_eddy_field = with_eddy_field
        self.with_geomag_gradient = with_geomag_gradient
        self.base_tl_model_version = base_tl_model_version
        self.with_scaling_factor_between_model_items = (
            with_scaling_factor_between_model_items
        )
        self.sampling_rate = sampling_rate
        self.copy = copy

    def _reset(self):
        if hasattr(self, "model_"):
            del self.model_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        self._reset()
        return self.partial_fit(X, y)

    def _make_TollesLawson_features(self, X):
        def _merge_feature(permanent_items, induced_items, eddy_items):
            features = []
            if self.with_permanent_field:
                features.extend(permanent_items)
            if self.with_induced_field:
                features.extend(induced_items)
            if self.with_eddy_field:
                features.extend(eddy_items)
            return np.column_stack(features)

        (
            permanent_items,
            induced_items,
            eddy_items,
        ) = self._make_TollesLawson_directional_cosines(X)
        tl_features = _merge_feature(
            permanent_items,
            induced_items,
            eddy_items,
        )
        return tl_features

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y):
        def _check_scaling_factor(X):
            if self.with_scaling_factor_between_model_items and _num_features(X) != 4:
                raise ValueError(
                    "X should have 4 columns when with_scaling_factor_between_model_items is True."
                )

        check_consistent_length(X, y)
        X = self._validate_data(
            X, dtype=np.float64, copy=self.copy, ensure_min_samples=2
        )
        y = check_array(
            y, input_name="Y", dtype=np.float64, copy=self.copy, ensure_2d=False
        )
        _check_scaling_factor(X)

        tl_features = self._make_TollesLawson_features(X)

        tl_features = (
            filter_FOM(tl_features, sampling_rate=self.sampling_rate)
            if self.with_band_pass_filter
            else tl_features
        )

        tmi_with_interf = (
            filter_FOM(y, sampling_rate=self.sampling_rate)
            if self.with_band_pass_filter
            else y
        )

        self.model_ = RidgeCV(alphas=[0.1, 0.01, 0.001, 0.0001, 0.00001]).fit(
            tl_features, tmi_with_interf
        )

        return self

    def transform(self, X, y, copy=True):
        """对输入的磁总场数据进行补偿

        Parameters
        ----------
        X : array-like, shape of (n_samples, n_features)
            姿态特征数据，n_samples 为样本数，n_features 为特征数。

        y : array-like, shape of (n_samples,)
            待补偿的磁总场数据

        Returns
        -------
        comped : array-like, shape (n_samples,)
            补偿后的磁总场数据

        """
        check_is_fitted(self)
        check_consistent_length(X, y)
        X = self._validate_data(X, dtype=np.float64, copy=copy, ensure_min_samples=2)
        y = check_array(y, input_name="Y", dtype=np.float64, copy=copy, ensure_2d=False)
        interf = self.predict(X)
        comped = y - interf

        return comped

    def fit_transform(self, X, y) -> ndarray:
        return self.fit(X, y).transform(X, y)

    def predict(self, X):
        """根据姿态特征预测磁总场干扰

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            姿态特征，包含 n_samples 个样本和 n_features 个特征。

        Returns
        -------
        interf : array-like, shape (n_samples,)
            预测的磁总场干扰，与输入数据的样本数相同。

        """
        check_is_fitted(self)

        pred = self.model_.predict(self._make_TollesLawson_features(X))
        interf = detrend(pred, type="constant")

        return interf

    def fit_predict(self, X, y):
        self.fit(X, y).predict(X)

    def _make_TollesLawson_directional_cosines(self, X):
        if self.with_scaling_factor_between_model_items:
            cos_x, cos_y, cos_z, scaling_factor = X.T
        else:
            cos_x, cos_y, cos_z, _ = X.T
            scaling_factor = 1.0

        cos_x_dot = np.gradient(cos_x)
        cos_y_dot = np.gradient(cos_y)
        cos_z_dot = np.gradient(cos_z)

        permanent_items = [cos_x, cos_y, cos_z]

        induced_items = [
            scaling_factor * cos_x * cos_x,
            scaling_factor * cos_x * cos_y,
            scaling_factor * cos_x * cos_z,
            scaling_factor * cos_y * cos_y,  # removed in 16-item version
            scaling_factor * cos_y * cos_z,
            scaling_factor * cos_z * cos_z,
        ]

        eddy_items = [
            scaling_factor * cos_x * cos_x_dot,
            scaling_factor * cos_x * cos_y_dot,
            scaling_factor * cos_x * cos_z_dot,
            scaling_factor * cos_y * cos_x_dot,
            scaling_factor * cos_y * cos_y_dot,  # removed in 16-item version
            scaling_factor * cos_y * cos_z_dot,
            scaling_factor * cos_z * cos_x_dot,
            scaling_factor * cos_z * cos_y_dot,
            scaling_factor * cos_z * cos_z_dot,
        ]

        if self.base_tl_model_version == 16:
            del induced_items[len(induced_items) // 2]
            del eddy_items[len(eddy_items) // 2]

        return permanent_items, induced_items, eddy_items

    def _more_tags(self):
        return {
            "requires_y": True,
        }

