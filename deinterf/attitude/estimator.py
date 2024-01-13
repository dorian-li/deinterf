import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator


class AttitudeEstimator(BaseEstimator):
    """姿态特征估计器

    Parameters
    ----------
    copy : bool, default=True
        是否复制输入数据。

    """

    _parameter_constraints: dict = {"copy": ["boolean"]}

    def __init__(self, copy=True) -> None:
        self.copy = copy

    def transform(self, X):
        """将输入数据进行变换。

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            输入数据。

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            变换后的数据。
        """
        X = self._validate_data(X, dtype=np.float64, copy=self.copy)
        return X


class MagVectorAttitudeEst(AttitudeEstimator):
    """通过磁三分量估计姿态特征

    Parameters
    ----------
    copy : bool, default=True
        是否复制输入数据。

    Examples
    --------
    >>> from deinterf.attitude.estimator import MagVectorAttitudeEst
    >>> mag_vec = [[2e5, 3e5, 4e5], [3e5, 4e5, 5e5], [4e5, 5e5, 6e5]]
    >>> MagVectorAttitudeEst().tansform(mag_vec)
    array([[ 0.37139068  0.55708601  0.74278135]
           [ 0.42426407  0.56568542  0.70710678]
           [ 0.45584231  0.56980288  0.68376346]])
    """

    def transform(self, X):
        """将输入数据进行变换。

        Parameters
        ----------
        X : array-like of shape (n_samples, 3)
            磁三分量，第二维度对应 bx, by, bz 三轴

        Returns
        -------
        attitude_features : ndarray of shape (n_samples, 3)
            方向余弦表示的姿态特征，第二维度对应 x, y, z 三轴
        """
        X = super().transform(X)
        d_c = self._mag_vector_to_dir_cosine(X)
        return d_c

    def _mag_vector_to_dir_cosine(self, X: ndarray) -> ndarray:
        """使用磁三分量计算方向余弦

        Parameters
        ----------
        X : ndarray of shape (n_samples, 3)
            磁三分量，第二维度对应 bx, by, bz 三轴

        Returns
        -------
        directional_cosine : ndarray of shape (n_samples, 3)
            方向余弦，第二维度对应 x, y, z 三轴
        """
        bx, by, bz = X.T
        tmi = np.linalg.norm(X, axis=1)

        dir_conise_x = bx / tmi
        dir_conise_y = by / tmi
        dir_conise_z = bz / tmi
        return np.column_stack((dir_conise_x, dir_conise_y, dir_conise_z))


class INSEulerAttitudeEst(AttitudeEstimator):
    ...
