from functools import cached_property
from typing import Literal, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import butter, detrend, filtfilt
from sklearn.linear_model import RidgeCV

ArrayLike = Union[pd.Series, np.ndarray]


class TollesLawsonCompensator:
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

    def fit(
        self,
        vector_x: ArrayLike,
        vector_y: ArrayLike,
        vector_z: ArrayLike,
        scalar: ArrayLike,
    ):
        self.src_vec_x = np.array(vector_x)
        self.src_vec_y = np.array(vector_y)
        self.src_vec_z = np.array(vector_z)
        self.src_scalar = np.array(scalar)

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

    def use_permanent(self, status: bool):
        self._using_permanent = status

    def is_induced_used(self):
        return self._using_induced

    def use_induced(self, status: bool):
        self._using_induced = status

    def is_eddy_used(self):
        return self._using_eddy

    def use_eddy(self, status: bool):
        self._using_eddy = status

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

    def _compute_directional_cosine_components(self, vector_x, vector_y, vector_z):
        vector_t = np.linalg.norm(np.c_[vector_x, vector_y, vector_z], axis=1)

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
            scaling_factor * cos_y * cos_y,
            scaling_factor * cos_y * cos_z,
            scaling_factor * cos_z * cos_z,
        ]

        eddy_items = [
            scaling_factor * cos_x * cos_x_dot,
            scaling_factor * cos_x * cos_y_dot,
            scaling_factor * cos_x * cos_z_dot,
            scaling_factor * cos_y * cos_x_dot,
            scaling_factor * cos_y * cos_y_dot,
            scaling_factor * cos_y * cos_z_dot,
            scaling_factor * cos_z * cos_x_dot,
            scaling_factor * cos_z * cos_y_dot,
            scaling_factor * cos_z * cos_z_dot,
        ]

        if self._coefficients_num == 16:
            induced_items = [
                item
                for item in induced_items
                if item is not scaling_factor * cos_y * cos_y
            ]
            eddy_items = [
                item
                for item in eddy_items
                if item is not scaling_factor * cos_y * cos_y_dot
            ]

        return permanent_items, induced_items, eddy_items

    def make_X(self, vector_x, vector_y, vector_z):
        (
            permanent_items,
            induced_items,
            eddy_items,
        ) = self._compute_directional_cosine_components(vector_x, vector_y, vector_z)

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
        X = self.make_X(self.src_vec_x, self.src_vec_y, self.src_vec_z)
        return self._filter_data(X) if self.is_bpf_enabled() else X

    @property
    def y(self):
        return (
            self._filter_data(self.src_scalar)
            if self.is_bpf_enabled()
            else self.src_scalar
        )

    @cached_property
    def model(self):
        model = RidgeCV(alphas=self.ridge_alphas)
        model.fit(self.X, self.y)
        return model

    def apply(
        self, flux_x: ArrayLike, flux_y: ArrayLike, flux_z: ArrayLike, op: ArrayLike
    ):
        X = self.make_X(flux_x, flux_y, flux_z)
        y = self.model.predict(X)
        interf = detrend(y, type="constant")
        comped = op - interf
        return comped, interf

    def evaluate(self, uncomped, comped):
        uncomped_bpf = self._filter_data(uncomped)
        comped_bpf = self._filter_data(comped)

        uncomped_noise_level = np.std(uncomped_bpf)
        comped_noise_level = np.std(comped_bpf)
        ir = uncomped_noise_level / comped_noise_level

        logger.info(f"{uncomped_noise_level=}, {comped_noise_level=}, {ir=}")

        return uncomped_noise_level, comped_noise_level, ir

    def evaluate_src(self):
        uncomped = self.src_scalar
        comped, _ = self.apply(
            self.src_vec_x, self.src_vec_y, self.src_vec_z, self.src_scalar
        )
        return self.evaluate(uncomped, comped)
