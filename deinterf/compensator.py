from __future__ import annotations

from typing import Literal

import numpy as np
from loguru import logger
from numpy.typing import ArrayLike

from deinterf.tolles_lawson import TollesLawson


class TollesLawsonCompensator(TollesLawson):
    def __init__(self, coefficients_num: Literal[16, 18] = 16):
        super().__init__(coefficients_num)

    def build(
        self,
        scalar: ArrayLike,
        vector_x: ArrayLike,
        vector_y: ArrayLike,
        vector_z: ArrayLike,
    ):
        self.fit(np.c_[vector_x, vector_y, vector_z], scalar)

    def apply(
        self, flux_x: ArrayLike, flux_y: ArrayLike, flux_z: ArrayLike, op: ArrayLike
    ):
        interf = self.predict(np.c_[flux_x, flux_y, flux_z])
        comped = op - interf
        return comped, interf

    def transform(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ):
        interf = self.predict(X)
        comped = y - interf
        return comped

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(self.src_vector, self.src_scalar)

    def score(self, X, y):
        return self.evaluate(y, self.transform(X, y))

    def __call__(self, X, y=None):
        if y is None:
            return super().__call__(X)
        else:
            return self.transform(X, y)

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
