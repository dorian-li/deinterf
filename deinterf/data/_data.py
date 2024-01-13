from __future__ import annotations

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import ArrayLike
from pandas import DataFrame
from typing_extensions import Any, Literal, Self, Tuple, Union

from deinterf.attitude.estimator import AttitudeEstimator, MagVectorAttitudeEst


class Data:
    """数据基类"""

    _src = None

    @property
    def values(self) -> ndarray:
        """返回数据的值"""
        self.check_is_fitted()
        return self._src

    def fit(self, src_df: DataFrame, *columns: Tuple[str]) -> Self:
        """构造数据，基于输入的DataFrame和列名

        Parameters
        ----------
        src_df : DataFrame
            数据源的DataFrame
        columns : Tuple[str]
            所需数据项的列名

        Returns
        -------
        self : object
            已构造的数据
        """
        self._src = src_df[list(columns)].to_numpy()
        return self

    def check_is_fitted(self):
        if self._src is None:
            raise AttributeError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )


class MagVector(Data):
    """磁三分量数据"""

    def fit(self, src_df: DataFrame, bx: str, by: str, bz: str):
        """构造磁三分量数据

        Parameters
        ----------
        src_df : DataFrame
            数据源的DataFrame
        bx : str
            磁三分量x轴的列名
        by : str
            磁三分量y轴的列名
        bz : str
            磁三分量z轴的列名

        Returns
        -------
        self : object
            已构造的磁三分量数据
        """
        super().fit(src_df, bx, by, bz)
        return self

    def to_tmi(self):
        """磁三分量构造磁总场数据

        Returns
        -------
        tmi : Tmi
            磁总场数据
        """
        self.check_is_fitted()
        tmi_df = DataFrame(np.linalg.norm(self._src, axis=1), columns=["tmi"])
        return Tmi().fit(tmi_df, tmi="tmi")

    def to_dir_cosine(self) -> DirectionalCosine:
        """磁三分量构造姿态方向余弦

        Returns
        -------
        dir_cosine : DirectionalCosine
            姿态方向余弦
        """
        self.check_is_fitted()
        bx, by, bz = self._src.T
        tmi = self.to_tmi().values.squeeze()

        dir_conise_x = bx / tmi
        dir_conise_y = by / tmi
        dir_conise_z = bz / tmi

        dcos_df = DataFrame(
            np.column_stack((dir_conise_x, dir_conise_y, dir_conise_z)),
            columns=["dir_cosine_x", "dir_cosine_y", "dir_cosine_z"],
        )
        return DirectionalCosine().fit(
            dcos_df, "dir_cosine_x", "dir_cosine_y", "dir_cosine_z"
        )


class Tmi(Data):
    """磁总场数据"""

    def fit(self, src_df: DataFrame, tmi: str) -> Self:
        """根据磁三分量数据构造磁总场数据

        Parameters
        ----------
        mag_vec : MagVector
            磁三分量数据

        Returns
        -------
        self : object
            已构造的磁总场数据
        """
        super().fit(src_df, tmi)
        return self


class InsEuler(Data):
    """惯导输出的姿态角数据，以欧拉角表示"""

    def fit(self, src_df: DataFrame, roll: str, pitch: str, yaw: str):
        """根据惯导输出的姿态角数据构造姿态角数据

        Parameters
        ----------
        src_df : DataFrame
            数据源的DataFrame
        roll : str
            姿态角的滚转角列名
        pitch : str
            姿态角的俯仰角列名
        yaw : str
            姿态角的偏航角列名

        Returns
        -------
        self : object
            已构造的姿态角数据
        """
        super().fit(src_df, roll, pitch, yaw)
        return self


class DirectionalCosine(Data):
    """姿态的方向余弦"""

    def fit(
        self, src_df: DataFrame, dir_conise_x: str, dir_conise_y: str, dir_conise_z: str
    ):
        """构造姿态的方向余弦

        Parameters
        ----------
        src_df : DataFrame
            数据源的DataFrame
        dir_conise_x : str
            姿态方向余弦x轴的列名
        dir_conise_y : str
            姿态方向余弦y轴的列名
        dir_conise_z : str
            姿态方向余弦z轴的列名

        Returns
        -------
        self : object
            已构造的姿态方向余弦
        """
        super().fit(src_df, dir_conise_x, dir_conise_y, dir_conise_z)
        return self
