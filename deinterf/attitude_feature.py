import numpy as np
from sklearn.base import BaseEstimator


class MagVectorAttitudeEstimation(BaseEstimator):
    """通过磁三分量估计姿态

    Parameters
    ----------
    output_scaling_factor : bool, default=True
        是否输出缩放因子。

    copy : bool, default=True
        是否复制输入数据。

    Examples
    --------
    >>> from deinterf.attitude_estimation import MagVectorAttitudeEstimation
    >>> mag_vec = [[2e5, 3e5, 4e5], [3e5, 4e5, 5e5], [4e5, 5e5, 6e5]]
    >>> mva = MagVectorAttitudeEstimation()
    >>> mva.transform(mag_vec)
    array([[ 0.37139068  0.55708601  0.74278135 10.77032961]
           [ 0.42426407  0.56568542  0.70710678 14.14213562]
           [ 0.45584231  0.56980288  0.68376346 17.54992877]])
    """

    _parameter_constraints: dict = {
        "output_scaling_factor": ["boolean"],
        "copy": ["boolean"],
    }

    def __init__(self, output_scaling_factor=True, copy=True) -> None:
        self.output_scaling_factor = output_scaling_factor
        self.copy = copy

    def _compute_scaling_factor(self, X):
        """计算缩放因子。

        Parameters:
        ----------
        X : ndarray of shape (n_samples, 3)
            磁三分量，第二维度对应 x, y, z 三轴

        Returns:
        -------
        scaling_factor : ndarray of shape (n_samples,)
            缩放因子。

        Notes:
        ------
        缩放因子的计算方法为每个输入磁三分量的模除以 5e4。
        """
        tmi = np.linalg.norm(X, axis=1)
        scaling_factor = tmi / 5e4
        return scaling_factor

    def _mag_vector_to_directional_cosine(self, X):
        """使用磁三分量计算方向余弦

        Parameters
        ----------
        X : ndarray of shape (n_samples, 3)
            磁三分量，第二维度对应 x, y, z 三轴

        Returns
        -------
        directional_cosine : ndarray of shape (n_samples, 3)
            方向余弦，第二维度对应 x, y, z 三轴
        """
        bx, by, bz = X.T
        tmi = np.linalg.norm(X, axis=1)

        cos_x = bx / tmi
        cos_y = by / tmi
        cos_z = bz / tmi
        return np.column_stack((cos_x, cos_y, cos_z))

    def _merge_attitude_features(self, directional_cosines, scaling_factor):
        """根据 ``output_scaling_factor`` 参数决定是否将方向余弦和缩放因子合并为一个数组。

        Parameters:
        ----------
        directional_cosines : ndarray of shape (n_samples, 3)
            方向余弦数组，第二维度对应 x, y, z 三轴。
        scaling_factor : ndarray of shape (n_samples,)
            缩放因子。

        Returns:
        -------
        attitude_features : ndarray of shape (n_samples. 3) or (n_samples, 4)
            合并后的姿态特征，第二维度对应 x, y, z 三轴, 若 ``output_scaling_factor=True`` 则第四维度为缩放因子。
        """
        if self.output_scaling_factor:
            return np.column_stack((directional_cosines, scaling_factor))
        else:
            return directional_cosines

    def transform(self, X):
        """将输入数据进行变换。

        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            磁三分量，第二维度对应 x, y, z 三轴

        Returns
        -------
        attitude_features : ndarray of shape (n_samples, 3) or (n_samples, 4)
            姿态特征，第二维度对应 x, y, z 三轴, 若 ``output_scaling_factor=True`` 则第四维度为缩放因子
        """
        X = self._validate_data(X, dtype=np.float64, copy=self.copy)
        d_c = self._mag_vector_to_directional_cosine(X)
        s_f = self._compute_scaling_factor(X)
        attitude_features = self._merge_attitude_features(d_c, s_f)
        return attitude_features


class INSEulerAnglesAttitudeEstimation(BaseEstimator):
    ...
