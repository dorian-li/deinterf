import numpy as np
from deinterf.foundation.sensors import Tmi, MagVector
from deinterf.utils.data_ioc import DataIoC, DataNDArray
from deinterf.compensator.tmi.linear import TollesLawson, Terms
from numpy.typing import ArrayLike
from scipy.signal import butter, filtfilt
from deinterf.foundation import ComposableTerm
from deinterf.foundation.sensors import DirectionalCosine
from deinterf.foundation.sensors import DirectionalCosine, MagIntensity

class Current(DataNDArray):
    def __new__(cls, current: ArrayLike, fs=10):
        b, a = butter(4, Wn=0.4, btype="low", fs=fs)
        current_filtered = filtfilt(b, a, current)
        return super().__new__(cls, current_filtered)


class Wire(ComposableTerm):
    def __build__(self, container: DataIoC):
        current = container[Current]
        dcos = container[DirectionalCosine[0]]
        ret1 = current[:, None] * dcos
        current_dot = np.gradient(current)
        ret2 = current_dot[:, None] * dcos
        return np.column_stack((ret1, ret2))
    
class LoadVibration(ComposableTerm):
    def __build__(self, container: DataIoC) -> np.ndarray:
        intensity = container[MagIntensity]
        cos_x, cos_y, cos_z = container[DirectionalCosine].T
        cos_x_dot = np.gradient(cos_x)
        cos_y_dot = np.gradient(cos_y)
        cos_z_dot = np.gradient(cos_z)
        
        feats_triple_cos10 = intensity[:, None] * np.column_stack(
            (
                cos_x * cos_x * cos_x,
                cos_x * cos_x * cos_y,
                cos_x * cos_x * cos_z,
                cos_x * cos_y * cos_y,
                cos_x * cos_y * cos_z,
                cos_x * cos_z * cos_z,
                cos_y * cos_y * cos_y,
                cos_y * cos_y * cos_z,
                cos_y * cos_z * cos_z,
                cos_z * cos_z * cos_z,
            )
        )
        
        feats_double_cos_dcos18 = intensity[:, None] * np.column_stack(
            (
                cos_x * cos_x * cos_x_dot,
                cos_x * cos_x * cos_y_dot,
                cos_x * cos_x * cos_z_dot,
                cos_x * cos_y * cos_x_dot,
                cos_x * cos_y * cos_y_dot,
                cos_x * cos_y * cos_z_dot,
                cos_x * cos_z * cos_x_dot,
                cos_x * cos_z * cos_y_dot,
                cos_x * cos_z * cos_z_dot,
                cos_y * cos_y * cos_x_dot,
                cos_y * cos_y * cos_y_dot,
                cos_y * cos_y * cos_z_dot,
                cos_y * cos_z * cos_x_dot,
                cos_y * cos_z * cos_y_dot,
                cos_y * cos_z * cos_z_dot,
                cos_z * cos_z * cos_x_dot,
                cos_z * cos_z * cos_y_dot,
                cos_z * cos_z * cos_z_dot,
            )
        )
        
        feats_dcos3 = intensity[:, None] * np.column_stack(
            (
                cos_x_dot,
                cos_y_dot,
                cos_z_dot,
            )
        )
        
        return np.column_stack((feats_triple_cos10, feats_double_cos_dcos18, feats_dcos3))

