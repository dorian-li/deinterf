from __future__ import annotations

from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, detrend, filtfilt
from sklearn.base import BaseEstimator
from sklearn.linear_model import RidgeCV
from sklearn.utils._param_validation import Options
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self


class TollesLawson(BaseEstimator):

    _estimator_type = "regressor"
    _parameter_constraints: dict = {
        "coefficients_num": [Options(Integral, {16, 18})]
    }

    def __init__(self, coefficients_num: Literal[16, 18] = 16):
        if coefficients_num not in [16, 18]:
            raise ValueError("coefficients_num must be either 16 or 18.")

        self._bpf_enabled = True
        self._using_permanent = True
        self._using_induced = True
        self._using_eddy = True
        self.bt_scale = 50000
        self._coefficients_num = coefficients_num
        self.ridge_alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        self._sampling_rate = 10

        self.src_vector_ = None
        self.src_scalar_ = None
        self.model_ = None

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> Self:
        self.src_vector = X
        self.src_scalar = y
        self.model_ = RidgeCV(alphas=self.ridge_alphas).fit(self.X, self.y)

        return self

    def predict(
        self,
        X: ArrayLike
    ):
        check_is_fitted(self)

        pred = self.model_.predict(self.make_X(X))
        interf = detrend(pred, type="constant")

        return interf

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(self.src_vector)

    def __call__(self, X):
        return self.predict(X)

    def is_bpf_enabled(self):
        return self._bpf_enabled

    def enable_bpf(self, status: bool):
        self._bpf_enabled = status

    def current_sampling_rate(self):
        return self._sampling_rate

    def adjust_sampling_rate(self, fs):
        self._sampling_rate = fs

    def is_permanent_used(self):
        return self._using_permanent

    def use_permanent(self, status: bool = True):
        self._using_permanent = status

    def is_induced_used(self):
        return self._using_induced

    def use_induced(self, status: bool = True):
        self._using_induced = status

    def is_eddy_used(self):
        return self._using_eddy

    def use_eddy(self, status: bool = True):
        self._using_eddy = status
    
    @property
    def src_scalar(self):
        return self.src_scalar_

    @src_scalar.setter
    def src_scalar(self, scalar):
        self.src_scalar_ = np.asarray(scalar)

    @property
    def src_vector(self):
        return self.src_vector_.T

    @src_vector.setter
    def src_vector(self, vec):
        self.src_vector_ = np.ascontiguousarray(np.transpose(vec))

    @property
    def src_vec_x(self):
        return self.src_vector_[0]

    @property
    def src_vec_y(self):
        return self.src_vector_[1]

    @property
    def src_vec_z(self):
        return self.src_vector_[2]

    def make_X_3(self, vector_x, vector_y, vector_z):
        return self.make_X(np.c_[vector_x, vector_y, vector_z])

    def make_X(self, vectors: ArrayLike):
        (
            permanent_items,
            induced_items,
            eddy_items,
        ) = self._compute_directional_cosine_components(vectors)

        features = []
        if self.is_permanent_used():
            features.extend(permanent_items)
        if self.is_induced_used():
            features.extend(induced_items)
        if self.is_eddy_used():
            features.extend(eddy_items)

        return np.column_stack(features)

    @property
    def X(self):
        X = self.make_X(self.src_vector)
        return self._filter_data(X) if self.is_bpf_enabled() else X

    @property
    def y(self):
        return (
            self._filter_data(self.src_scalar)
            if self.is_bpf_enabled()
            else self.src_scalar
        )

    def _create_filter(self):
        b, a = butter(
            4,
            [0.1, 0.6],
            btype="bandpass",
            fs=self.current_sampling_rate(),
            output="ba",
        )
        return b, a

    def _filter_data(self, data):
        b, a = self._create_filter()
        return filtfilt(b, a, data, axis=0)

    def _compute_directional_cosine_components(self, vector: ArrayLike):
        vector_x, vector_y, vector_z = vector.T
        vector_t = np.linalg.norm(vector, axis=1)

        cos_x = vector_x / vector_t
        cos_y = vector_y / vector_t
        cos_z = vector_z / vector_t

        cos_x_dot = np.gradient(cos_x)
        cos_y_dot = np.gradient(cos_y)
        cos_z_dot = np.gradient(cos_z)

        scaling_factor = vector_t / self.bt_scale

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

        if self._coefficients_num == 16:
            del induced_items[len(induced_items) // 2]
            del eddy_items[len(eddy_items) // 2]

        return permanent_items, induced_items, eddy_items

    def _more_tags(self):
        return {
            'requires_y': True
        }
