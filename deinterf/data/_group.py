from numpy import ndarray

from ._data import Data


class DataGroup:
    """数据组
    表示一组继承自Data的各种数据示例，供kenel在构造特征时，根据数据类型获取相应所需的数据输入

    Parameters
    ----------
    data : tuple[Data, ...]
        一组数据，包含继承自Data的各种数据实例。

    Attributes
    ----------
    _data : dict[str, Data]
        数据组，键为数据类型的名称，值为对应的数据实例。

    Examples
    --------
    >>> from deinterf.data import DataGroup, MagVector, Tmi, DirectionalCosine
    >>> import numpy as np
    >>> from pandas as pd
    >>> mag_vec_df = pd.DataFrame(np.eye(3), columns=["mag_x", "mag_y", "mag_z"])
    >>> mag_vec = MagVector().fit(mag_vec_df, bx="mag_x", by="mag_y", bz="mag_z")
    >>> data = DataGroup((mag_vec.to_tmi(), mag_vec.to_dir_cosine()))
    >>> data[DirectionalCosine]
    >>> data[Tmi]
    """

    __slots__ = ("_data",)

    def __init__(self, data: tuple[Data, ...]) -> None:
        # 目前不支持存储多个同类型的数据
        self._data = dict((type(d).__name__, d) for d in data)

    def __getitem__(self, dtype: type(Data)) -> ndarray:
        data = self._data.get(dtype.__name__)
        if data:
            return data.values
        raise KeyError(
            f"Need '{dtype.__name__}', but not in {self.__class__.__name__}:{list(self._data.keys())}"
        )

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._data})"

    def __add__(self, other: Data) -> "DataGroup":
        return DataGroup((*self._data, other))
