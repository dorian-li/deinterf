from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy import ndarray
from numpy.typing import ArrayLike
from scipy.signal import detrend
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    check_consistent_length,
    check_is_fitted,
    column_or_1d,
)
from typing_extensions import Literal, Self

from deinterf.data import DataGroup
from deinterf.utils.filter import fom_bpfilter
from deinterf.compensator.tmi.linear.kernel import Feature, TermsCollection
from deinterf.compensator.tmi.linear.terms import Terms


class TollesLawson(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """根据Tolles-Lawson磁干扰模型补偿磁总场数据

    Parameters
    ----------
    filter : {"bandpass", None}, default="bandpass"
        补偿系数计算时应用何种滤波器
    terms : Terms, default=Terms.Terms16
        采用的Tolles-Lawson模型项
    norm : bool, default=True
        是否执行特征标准化，见 :class:`sklearn.preprocessing.StandardScaler`
    sampling_rate : int, default=10
        采样频率，单位为Hz
    copy : bool, default=True
        是否复制输入数据

    Attributes
    ----------
    model_ : RidgeCV
        拟合后的模型
    scaler_ : StandardScaler
        如果norm为True，则存在拟合后的标准化器，model_进行拟合与推理前都进行输入特征的标准化

    Examples
    --------
    >>> from deinterf.attitude.estimator import MagVectorAttitudeEst
    >>> from deinterf.compensator.tmi.linear import TollesLawson
    >>> from deinterf.metrics.fom import improve_rate
    >>> import numpy as np
    >>> mag_vec = np.tile(np.eye(3), (1000, 1))
    >>> attitude_features = MagVector().transform(mag_vec)
    >>> tmi_with_interf = np.linspace(5e4, 6e4, 3000)
    >>> compensator = TollesLawson()
    >>> compensator.fit(attitude_features, tmi_with_interf)
    TollesLawson()
    >>> tmi_pure = compensator.transform(attitude_features, tmi_with_interf)
    >>> print(tmi_pure)
    """

    _estimator_type = "regressor"
    _parameter_constraints: dict = {
        "filter": [StrOptions({"bandpass"}), None],
        "terms": [Feature, TermsCollection],
        "norm": ["boolean"],
        "sampling_rate": [Interval(Integral, 1, None, closed="left")],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        filter: Literal["bandpass"] | None = "bandpass",
        terms=Terms.Terms_16,
        norm=True,
        sampling_rate=10,
        copy=True,
    ) -> None:
        self.filter = filter
        self.terms = terms
        self.norm = norm
        self.sampling_rate = sampling_rate
        self.copy = copy

    def _reset(self) -> None:
        """重置补偿器状态"""
        if hasattr(self, "model_"):
            del self.model_
        if hasattr(self, "scaler_"):
            del self.scaler_

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: DataGroup, y: ArrayLike) -> Self:
        """
        基于Tolles-Lawson磁干扰数学模型，根据FOM校准机动飞行测量的磁总场数据，建立飞行姿态-飞行平台干扰场的补偿模型。

        Parameters
        ----------
        X : DataGroup of shape (n_samples, n_features)
            计算Terms所需的FOM校准机动飞行相关数据，包括磁场数据、姿态数据等。
        y : array-like of shape (n_samples,)
            FOM校准飞行测量的磁总场数据，n_samples 为样本数。

        Returns
        -------
        self : object
            已拟合的补偿器。
        """
        self._reset()
        return self.partial_fit(X, y)

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X: DataGroup, y: ArrayLike) -> Self:
        """
        在线基于Tolles-Lawson磁干扰数学模型，根据FOM校准机动飞行测量的磁总场数据，建立飞行姿态-飞行平台干扰场的补偿模型。

        Parameters
        ----------
        X : DataGroup of shape (n_samples, n_features)
            计算Terms所需的FOM校准机动飞行相关数据，包括磁场数据、姿态数据等。
        y : array-like of shape (n_samples,)
            FOM校准飞行测量的磁总场数据，n_samples 为样本数。

        Returns
        -------
        self : object
            已拟合的补偿器。
        """
        y = column_or_1d(y, dtype=np.float64)
        tl_features = self.terms.make(X)
        check_consistent_length(tl_features, y)

        if self.norm:
            self.scaler_ = StandardScaler()
            tl_features = self.scaler_.fit_transform(tl_features)

        # 决定是计算Ax=b形式，还是bpf(Ax)=bpf(b)=>bpf(A)x=bpf(b)形式的模型
        tl_features = (
            fom_bpfilter(tl_features, sampling_rate=self.sampling_rate)
            if self.filter == "bandpass"
            else tl_features
        )

        interf_measured = (
            fom_bpfilter(y, sampling_rate=self.sampling_rate)
            if self.filter == "bandpass"
            else y
        )

        self.model_ = RidgeCV(
            fit_intercept=False,
            alphas=np.logspace(-6, 6, 13),
            cv=10,
        ).fit(tl_features, interf_measured)

        return self

    def transform(self, X: DataGroup, y: ArrayLike, copy=True) -> ndarray:
        """对输入的磁总场数据进行补偿

        Parameters
        ----------
        X : DataGroup of shape (n_samples, n_features)
            计算Terms所需的FOM校准机动飞行相关数据，包括磁场数据、姿态数据等。

        y : array-like of shape (n_samples,)
            待补偿的磁总场数据

        Returns
        -------
        comped : ndarray of shape (n_samples,)
            补偿后的磁总场数据

        """
        check_is_fitted(self)
        y = column_or_1d(y, dtype=np.float64)
        interf = self.predict(X)
        comped = y - interf

        return comped

    def fit_transform(self, X: DataGroup, y: ArrayLike) -> ndarray:
        return self.fit(X, y).transform(X, y)

    def predict(self, X: DataGroup) -> ndarray:
        """根据姿态特征预测磁总场干扰

        Parameters
        ----------
        X : DataGroup of shape (n_samples, n_features)
            计算Terms所需的FOM校准机动飞行相关数据，包括磁场数据、姿态数据等。

        Returns
        -------
        interf : ndarray of shape (n_samples,)
            预测的磁总场干扰，与输入数据的样本数相同。

        """
        check_is_fitted(self)

        tl_feats = self.terms.make(X)
        if self.norm:
            tl_feats = self.scaler_.transform(tl_feats)

        pred = self.model_.predict(tl_feats)
        interf = detrend(pred, type="constant")

        return interf

    def fit_predict(self, X: DataGroup, y: ArrayLike):
        """拟合FOM数据并预测当前FOM的干扰

        Parameters
        ----------
        X : DataGroup of shape (n_samples, n_features)
            计算Terms所需的FOM校准机动飞行相关数据，包括磁场数据、姿态数据等。
        y : array-like of shape (n_samples,)
            FOM飞行测量的磁总场数据
        """
        self.fit(X, y).predict(X)

    def _more_tags(self):
        return {
            "requires_y": True,
        }
