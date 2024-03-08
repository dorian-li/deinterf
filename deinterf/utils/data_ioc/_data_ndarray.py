from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import check_consistent_length

from ._data import IndexedData, DataIoC, IndexedDataIoC


def is_homogeneous(inp: np.ndarray, out: np.ndarray):
    if inp.shape != out.shape:
        # 不允许数组尺寸发生改变
        return False
    if out.dtype != np.promote_types(inp.dtype, out.dtype):
        # 输出只允许类型抬升
        return False

    return True


class DataNDArray(np.ndarray, IndexedData):
    def __new__(cls, *arrays: ArrayLike, force_column_stack=False, **kwargs):
        check_consistent_length(*arrays)
        if force_column_stack or len(arrays) > 1:
            return np.column_stack(arrays).view(cls)
        else:
            return np.asarray(arrays[0]).view(cls)

    def __array_finalize__(self, obj, **__):
        pass

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        typ = type(self)

        inputs = tuple(np.asarray(inp) if isinstance(inp, typ) else inp for inp in inputs)
        if out is not None:
            out = tuple(np.asarray(o) if isinstance(o, typ) else o for o in out)
        ret = super().__array_ufunc__(ufunc, method, *inputs, out=out, **kwargs)

        if ret is NotImplemented:
            return NotImplemented

        if is_homogeneous(self, ret):
            return ret.view(typ)
        else:
            return ret

    def reshape(self, *shape, **kwargs):
        return self.view(np.ndarray).reshape(*shape, **kwargs)

    def __getitem__(self, item):
        return super().__getitem__(item).view(np.ndarray)

    @classmethod
    def __build__(cls, container: DataIoC | IndexedDataIoC):
        id_str = f'[{container.id}]' if isinstance(container, IndexedDataIoC) else ''
        raise NotImplementedError(
            f'{cls.__name__}{id_str} must be provided.'
        )
