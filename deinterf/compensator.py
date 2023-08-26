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
        self._do_bpf = True
        self._using_permanent = True
        self._using_induced = True
        self._using_eddy = True
        self.bt_scale = 50000
        if coefficients_num in [16, 18]:
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

    def bpf(self, do_bpf=True):
        self._do_bpf = do_bpf

    def sampling_rate(self, fs):
        self._sampling_rate = fs

    def using_permanent(self, use=True):
        self._using_permanent = use

    def using_induced(self, use=True):
        self._using_induced = use

    def using_eddy(self, use=True):
        self._using_eddy = use

    def apply(
        self,
        flux_x: ArrayLike,
        flux_y: ArrayLike,
        flux_z: ArrayLike,
        op: ArrayLike,
    ):
        flux_x = np.array(flux_x)
        flux_y = np.array(flux_y)
        flux_z = np.array(flux_z)
        op = np.array(op)
        X = self.make_X(flux_x, flux_y, flux_z)
        y = self.model.predict(X)
        interf = detrend(y, type="constant")
        comped = op - interf
        return comped, interf

    def filter(self, x):
        b, a = butter(
            4, [0.1, 0.6], btype="bandpass", fs=self._sampling_rate, output="ba"
        )
        return filtfilt(b, a, x, axis=0)

    def make_X(self, vector_x, vector_y, vector_z):
        vector_t = np.linalg.norm(np.c_[vector_x, vector_y, vector_z], axis=1)
        nans = np.full_like(vector_t, np.nan)

        cos_x = vector_x / vector_t
        cos_y = vector_y / vector_t
        cos_z = vector_z / vector_t

        cos_x_dot = np.gradient(cos_x)
        cos_y_dot = np.gradient(cos_y)
        cos_z_dot = np.gradient(cos_z)

        cos_xx = vector_t * cos_x * cos_x / self.bt_scale
        cos_xy = vector_t * cos_x * cos_y / self.bt_scale
        cos_xz = vector_t * cos_x * cos_z / self.bt_scale
        cos_yy = vector_t * cos_y * cos_y / self.bt_scale
        cos_yz = vector_t * cos_y * cos_z / self.bt_scale
        cos_zz = vector_t * cos_z * cos_z / self.bt_scale

        cos_x_cos_x_dot = vector_t * cos_x * cos_x_dot / self.bt_scale
        cos_x_cos_y_dot = vector_t * cos_x * cos_y_dot / self.bt_scale
        cos_x_cos_z_dot = vector_t * cos_x * cos_z_dot / self.bt_scale
        cos_y_cos_x_dot = vector_t * cos_y * cos_x_dot / self.bt_scale
        cos_y_cos_y_dot = vector_t * cos_y * cos_y_dot / self.bt_scale
        cos_y_cos_z_dot = vector_t * cos_y * cos_z_dot / self.bt_scale
        cos_z_cos_x_dot = vector_t * cos_z * cos_x_dot / self.bt_scale
        cos_z_cos_y_dot = vector_t * cos_z * cos_y_dot / self.bt_scale
        cos_z_cos_z_dot = vector_t * cos_z * cos_z_dot / self.bt_scale

        permanent_items = [cos_x, cos_y, cos_z]
        induced_items = [cos_xx, cos_xy, cos_xz, cos_yy, cos_yz, cos_zz]
        eddy_items = [
            cos_x_cos_x_dot,
            cos_x_cos_y_dot,
            cos_x_cos_z_dot,
            cos_y_cos_x_dot,
            cos_y_cos_y_dot,
            cos_y_cos_z_dot,
            cos_z_cos_x_dot,
            cos_z_cos_y_dot,
            cos_z_cos_z_dot,
        ]
        if self._coefficients_num == 16:
            induced_items = [item for item in induced_items if item is not cos_yy]
            eddy_items = [item for item in eddy_items if item is not cos_y_cos_y_dot]

        ret_items = [
            permanent_items if self._using_permanent else [],
            induced_items if self._using_induced else [],
            eddy_items if self._using_eddy else [],
        ]

        ret = np.column_stack([item for components in ret_items for item in components])

        return ret

    @property
    def X(self):
        X = self.make_X(self.src_vec_x, self.src_vec_y, self.src_vec_z)
        if self._do_bpf:
            return self.filter(X)
        return X

    @property
    def y(self):
        if self._do_bpf:
            return self.filter(self.src_scalar)
        return self.src_scalar

    @cached_property
    def model(self):
        model = RidgeCV(alphas=self.ridge_alphas)
        model.fit(self.X, self.y)
        return model

    def evaluate(self, uncomped, comped):
        uncomped_bpf = self.filter(uncomped)
        comped_bpf = self.filter(comped)

        uncomped_noise_level = np.std(uncomped_bpf)
        comped_noise_level = np.std(comped_bpf)

        ir = uncomped_noise_level / comped_noise_level

        logger.info(f"{uncomped_noise_level=}, {comped_noise_level=}, {ir=}")

        return uncomped_noise_level, comped_noise_level, ir

    def evaluate_src(self):
        uncomped = self.src_scalar
        comped, _ = self.apply(
            self.src_vec_x,
            self.src_vec_y,
            self.src_vec_z,
            self.src_scalar,
        )
        return self.evaluate(uncomped, comped)
