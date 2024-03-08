from __future__ import annotations

import contextlib
import copy
import inspect
from typing import Any, Dict, Protocol, Type, runtime_checkable, Callable, Generic, TypeVar, overload, Union

from typing_extensions import Self

DataT = TypeVar('DataT')


@runtime_checkable
class SupportsBuild(Protocol):
    def __build__(self, container: DataIoC) -> Any: ...


class DataDescriptor(Generic[DataT]):
    """用于在DataIoC中唯一标识一个数据和/或构造数据的数据类型描述符

    Parameters
    ----------
    id
        数据id

    Notes
    -----
    `DataDescriptor` 需要作为字典的键，因此子类需要保证可哈希与可比较，如果每个成员变量均满足条件，则子类自动满足该条件。

    相同类型的多个不同数据可以通过id来区分，不显式指定时默认为 `0` 。
    例如三个个同类型的传感器读数可以使用 `Sensor` 、 `Sensor[1]` 和 `Sensor[2]` 来标识，
    此时他们在 `DataIoC` 中可以映射到不同的数据上。

    Examples
    --------
    默认创建的数据类型描述符例如 `Sensor` 具有弱id，代表他们是用于指示数据类型而非具体的某个数据，
    如果在 __build__ 中使用这类描述符从 `DataIoC` 中提取数据，则会自动重映射到具体的id索引。
    >>> from deinterf.utils.data_ioc import DataNDArray
    >>>
    >>> class SensorData(DataNDArray):
    >>>     def __new__(cls, data, **kwargs):
    >>>         return super().__new__(cls, data, **kwargs)
    >>>
    >>> class Sum(DataDescriptor):
    >>>     def __build__(self, container: DataIoC):
    >>>        return container[SensorData].sum()  # 不指定id，则获取的 SensorData 和 Sum 具有相同的id
    >>>
    >>> container = DataIoC().with_data(SensorData([1, 1, 1]), SensorData[1]([2, 2, 2]))  # 不显式指定时默认为 `0` 号数据
    >>> print(container[Sum[0]], container[Sum[1]])
    3 6

    手动指定索引的数据类型描述符例如 `Sensor[1]` 具有强id，代表他们是用于指示某组具体的数据，
    在 __build__ 中不会被重新映射。

    >>> class OffsetSensor0(DataDescriptor):
    >>>     def __build__(self, container: DataIoC):
    >>>        base = container[SensorData[0]].sum()  # 指定id，显式获取 0 号 SensorData
    >>>        return base + container[SensorData]
    >>>
    >>> print(container[OffsetSensor0[0]], container[OffsetSensor0[1]])
    [4, 4, 4] [5, 5, 5]
    """
    __slots__ = ['_id']
    DefaultWeakID = -1
    DefaultID = -DefaultWeakID - 1

    def __init__(self, id=DefaultWeakID) -> None:
        self.id = id

    @property
    def id(self):
        if self.signed_id < 0:
            return -self.signed_id - 1
        else:
            return self.signed_id

    @id.setter
    def id(self, val):
        self._id = val

    @property
    def signed_id(self):
        return self._id

    @property
    def is_weak_id(self):
        return self.signed_id < 0

    def __build__(self, container: DataIoC) -> DataT: ...

    def index_implicit(self, new_index):
        return self.index(new_index, weak=True)

    def index_explicit(self, new_index):
        return self.index(new_index, weak=True)

    def index(self, new_index, weak=True):
        """将当前数据描述符绑定到新的id索引，以实现关联同类别的不同数据

        例如三个个同类型的传感器读数可以使用：

        * Sensor[0]
        * Sensor[1]
        * Sensor[2]

        标识

        Parameters
        ----------
        new_index
            新的数据索引
        weak
            是否为弱绑定，弱绑定下，如果目标为强索引，则索引失效

        Returns
        -------
        新的索引id的数据类型
        """
        ret = self
        if not weak or self.is_weak_id:
            # 如果为强索引，或者当前为弱索引，则可以覆盖，重映射到新索引位置下的数据
            ret = copy.copy(ret)
            ret.id = new_index

        return ret

    def __getitem__(self, index) -> DataDescriptor[DataT]:
        """直接强制绑定为新的id索引

        Notes
        -----
        内部代码应优先使用 `index_implicit` 以支持弱id的自动重绑定。
        """
        return self.index_explicit(index)

    def __class_getitem__(cls, id, *args):
        if isinstance(id, int):
            return cls(*args, id=id)
        else:
            return super().__class_getitem__(id, *args)

    def __hash__(self):
        return hash(tuple(getattr(self, k) for k in self.keys))

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        for k in self.keys:
            if getattr(self, k) != getattr(other, k):
                return False
        else:
            return True

    @property
    def keys(self):
        keys = []
        for c in reversed(inspect.getmro(type(self))):
            keys.extend(getattr(c, '__slots__', []))
        keys.extend(getattr(self, '__dict__', []))

        keys = set(keys)
        keys.remove('_id')
        keys.add('id')  # 保证 id 值恒为正

        return keys

    @property
    def params(self):
        return {k.strip("_"): getattr(self, k) for k in self.keys}

    def __repr__(self):
        params = self.params
        del params['id']

        id_str = f'[{self.id}]' if self.id > 0 else ''
        param_str = ', '.join([f'{k}={repr(v)}' for k, v in params.items()])

        return f'{type(self).__name__}{id_str}({param_str})'

    def __copy__(self):
        return type(self)(**self.params)


class IndexedDataTypeDescriptor(DataDescriptor[DataT]):
    __slots__ = ['_dtype']

    @classmethod
    def of(cls, dtype, id=DataDescriptor.DefaultWeakID):
        if isinstance(dtype, IndexedDataMeta):
            return type(dtype).__getitem__(dtype, id)
        elif issubclass(dtype, DataDescriptor):
            return dtype(id)
        else:
            # 常规类型，不允许进行索引，强制绑定为默认索引
            return cls(dtype, id=DataDescriptor.DefaultID)

    def __init__(self, dtype: Type, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __build__(self, container: DataIoC) -> DataT:
        ret = _extract_builder_with_context(self.dtype, self)(container)
        return ret

    def __call__(self, *args, **kwargs):
        return DescribedData(
            self,
            self.dtype(*args, **kwargs)
        )

    def __repr__(self):
        id_str = f'[{self.id}]' if self.id > 0 else ''
        return f'{self.dtype.__name__}{id_str}'


class DescribedData:
    def __init__(self, desc: DataDescriptor[DataT], data: DataT):
        self.desc = desc
        self.data = data


class IndexedDataMeta(type):
    """指示一个类型为可索引类型，支持在DataIoC容器中绑定多组相关数据，并通过索引来进行区分

    Notes
    -----
    派生类可以通过自己定义 `__class_index__` 方法来自定义索引行为

    派生类可以通过继承 `UniqueData` 来指示该类型唯一，不需要索引

    所有非 `IndexedData` 子类的类型均视为唯一类型，不会被索引
    """

    def __getitem__(self, id=DataDescriptor.DefaultWeakID, *args):
        class_getitem = getattr(self, '__class_index__', None)
        if class_getitem is not None:
            return class_getitem(id, *args)
        else:
            return IndexedDataTypeDescriptor(self, *args, id=id)

    def __repr__(self):
        return f'{self.__name__}'


class IndexedData(metaclass=IndexedDataMeta):
    pass


class UniqueData:
    @classmethod
    def __class_index__(cls, id, *args):
        # 强制绑定为默认索引来保证所有隐式访问下会得到同一组数据
        return IndexedDataTypeDescriptor(cls, *args, id=DataDescriptor.DefaultID)


class DataIoC:
    def __init__(self, allow_implicit_registering=True, record_all=False):
        """数据IoC容器

        Parameters
        ----------
        allow_implicit_registering
            允许隐式注册Data，若为False则只能获取/构建已注册的Data
        record_all
            是否保存所有IoC访问记录，若为False则只保留当次访问依赖用于调试输出，否则会保留完整访问树（可能导致额外开销）
        """
        self._collection: Dict[DataDescriptor | Type, Any] = {}
        self._lazy_collection: Dict[DataDescriptor | Type, Callable[[DataIoC], Any]] = {}
        self.allow_implicit_register = allow_implicit_registering
        self.record_all = record_all

        self._logger = _DataIoCAccessLogger(key=self)

    def with_data(self, *data: Any) -> Self:
        for d in data:
            if isinstance(d, DescribedData):
                self[d.desc] = d.data
            else:
                self[type(d)] = d

        return self

    def add(self, data_type: DataT | Type[DataT] | DataDescriptor[DataT], data: DataT | None = None) -> Self:
        if data is None:
            if isinstance(data_type, DataDescriptor) or isinstance(data_type, type):
                builder = _extract_builder_with_context(data_type)
                self._lazy_collection[data_type] = builder
            else:
                self.with_data(data_type)
        else:
            self[data_type] = data

        return self

    def add_provider(
            self,
            data_type: Type[DataT] | DataDescriptor[DataT],
            provider: SupportsBuild | Callable
    ) -> Self:
        initiator = None
        if isinstance(data_type, DataDescriptor):
            initiator = data_type

        self._lazy_collection[data_type] = Provider(
            initiator=initiator,
            builder=_extract_builder(provider),
            target=provider
        )

        return self

    @property
    def logger(self):
        return self._logger

    @overload
    def __getitem__(self, dtype: DataDescriptor[DataT]) -> DataT: ...

    @overload
    def __getitem__(self, dtype: Type[DataDescriptor[DataT]]) -> DataT: ...

    @overload
    def __getitem__(self, dtype: Type[DataT]) -> DataT: ...

    def __getitem__(self, dtype: Type[DataT] | DataDescriptor[DataT]) -> DataT:
        with self._logger.add(dtype):
            ret = self._collection.get(dtype, None)

            if ret is None:
                builder = self.find_builder(dtype)

                if builder is None:
                    if not self.allow_implicit_register:
                        raise RuntimeError(f'Builder for {repr(dtype)} not found in {DataIoC.__name__}.')
                    else:
                        self.add(dtype)
                        builder = self.find_builder(dtype)

                if builder is not None:
                    if isinstance(builder, Provider):
                        self._logger.mark_overwrite(builder.target)

                    try:
                        ret = builder(self)
                    except (Exception,):
                        self._logger.mark_failed()
                        if self._logger.at_level0:
                            print(self._logger)
                        raise

                    self._logger.mark_new()
                    self[dtype] = ret

        if self._logger.at_root and not self.record_all:
            self._logger.clear()

        return ret

    def __setitem__(self, data_type: Type[DataT] | DataDescriptor[DataT], data: DataT):
        self._collection[data_type] = data
        if isinstance(data_type, IndexedDataTypeDescriptor) and data_type.id == 0:
            self._collection[data_type.dtype] = data
        if isinstance(data_type, type):
            # 对于直接用类名绑定，则默认同时绑定对应的0号数据
            self._collection[IndexedDataTypeDescriptor.of(data_type)] = data

    def find_builder(self, dtype: Type[DataT] | DataDescriptor[DataT]):
        """查找构造器

        * 直接以类别指定的构造器：适用于所有id下的 IndexedDataTypeDescriptor ，可以通用

        * 以特定 DataDescriptor 指定的构造器：只适用于特定的 DataDescriptor
        """
        builder = self._lazy_collection.get(dtype, None)
        if builder is None:
            if isinstance(dtype, IndexedDataTypeDescriptor):
                # 如果是带索引的类型，则进一步搜索该类型的通用构造器
                builder = self._lazy_collection.get(dtype.dtype, None)
                if builder is not None:
                    builder = _bind_builder_context(initiator=dtype, builder=builder)

        return builder

    def __str__(self):
        return type(self).__name__


class IndexedDataIoC(DataIoC):
    def __init__(self, base_container: DataIoC, initiator=None):
        super().__init__()
        self._base_container = base_container
        self._initiator = initiator

    @property
    def id(self):
        if self._initiator is None:
            return DataDescriptor.DefaultWeakID
        else:
            return self._initiator.signed_id

    def __getattr__(self, item):
        return getattr(self._base_container, item)

    def __getitem__(self, item: Type[DataT] | DataDescriptor[DataT]) -> DataT:
        if isinstance(item, type):
            item = IndexedDataTypeDescriptor.of(item, id=self.id)
        elif isinstance(item, DataDescriptor):
            item = item.index_implicit(self.id)

        ret = self._base_container[item]

        return ret


class _DataIoCDependency(list[tuple[Union[DataDescriptor, Type], '_DataIoCDependency']]):
    def __init__(self, parent, key, new=False):
        super().__init__()
        self._parent = parent
        self._key = key
        self._new = new
        self._overwrite = None
        self._failed = False

    def mark_new(self):
        self._new = True

    def mark_overwrite(self, key):
        self._overwrite = key

    def mark_failed(self):
        self._failed = True

    def add(self, key):
        ret = self.get(key, None)
        if ret is None:
            ret = _DataIoCDependency(parent=self, key=key)
            self[key] = ret

        return ret

    @property
    def parent(self):
        return self._parent

    @property
    def last_child(self):
        if len(self) == 0:
            return None

        return self[-1][0]

    def __contains__(self, item):
        for k, v in self:
            if item == k:
                return True
        else:
            return False

    def __setitem__(self, key, value):
        self.append((key, value))

    def __getitem__(self, item):
        for k, v in self:
            if item == k:
                return v
        else:
            raise KeyError(item)

    def get(self, item, default=None):
        try:
            return self[item]
        except KeyError:
            return default

    def to_str(self, prefix=None, indent='', table_prefix='', align_first_non_blank=True):
        if prefix is None:
            prefix = ' * '

        this_prefix = prefix
        properties = []
        if self._failed:
            this_prefix = this_prefix.replace('*', 'X')
            if this_prefix == prefix:
                properties.append(' X <--- Failed here ')
        if self._new:
            properties.append('Created')

        if len(properties) == 0:
            props = ''
        else:
            props = ' /' + ', '.join(properties) + '/'

        key = str(self._key)
        if self._overwrite is not None:
            key = str(self._overwrite) + f' ( <- {key})'

        ret = table_prefix + this_prefix + key + props + '\n'

        if align_first_non_blank:
            for c in prefix:
                if c == ' ':
                    indent += ' '
                else:
                    break

        remaining = len(self)
        for k, v in self:
            if remaining > 1:
                ret += v.to_str(prefix=prefix, table_prefix=indent + '├─', indent=indent + '│ ')
            else:
                ret += v.to_str(prefix=prefix, table_prefix=indent + '└─', indent=indent + '  ')

            remaining -= 1

        return ret

    def __str__(self):
        return self.to_str()


class _DataIoCAccessLogger:
    def __init__(self, key: Any = 'root'):
        self.root = _DataIoCDependency(None, key)
        self.current = self.root

    @property
    def at_level0(self):
        """判断是否在某个子访问树的根节点
        """
        return self.current.parent is self.root

    @property
    def at_root(self):
        """判断是否在根节点
        """
        return self.current is self.root

    @contextlib.contextmanager
    def add(self, key):
        child = self.current.add(key)

        try:
            self.enter(child)
            yield
        finally:
            self.exit()

    def clear(self):
        self.root.clear()

    def mark_new(self):
        self.current.mark_new()

    def mark_overwrite(self, key):
        self.current.mark_overwrite(key)

    def mark_failed(self):
        self.current.mark_failed()

    def __enter__(self):
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit()

    def enter(self, node=None):
        if node is None:
            node = self.current.last_child
        self.current = node

    def exit(self):
        self.current = self.current.parent

    def to_str(self, prefix=None):
        return self.root.to_str(prefix=prefix)

    def __str__(self):
        return self.to_str()


def _extract_builder_with_context(dtype: DataDescriptor | SupportsBuild | Callable, initiator=None):
    if initiator is None:
        if isinstance(dtype, DataDescriptor):
            initiator = dtype

    builder = _extract_builder(dtype)

    if builder is None:
        return builder
    else:
        return _bind_builder_context(builder, initiator=initiator)


def _extract_builder(dtype: DataDescriptor | SupportsBuild | Callable):
    if isinstance(dtype, SupportsBuild):
        builder = dtype.__build__
    elif callable(dtype):
        # 类型的构造函数
        # 或者直接的构造器函数
        # TODO: 添加校验，验证可以用于DataIoC
        builder = dtype
    else:
        builder = None

    return builder


def _bind_builder_context(builder, initiator):
    if initiator is None:
        return builder
    elif isinstance(builder, BuilderWithContext):
        return builder.with_initiator(initiator)
    else:
        return BuilderWithContext(initiator, builder)


class BuilderWithContext:
    def __init__(self, initiator, builder):
        self.initiator = initiator
        self.builder = builder

    def __call__(self, container: DataIoC):
        return self.builder(IndexedDataIoC(container, initiator=self.initiator))

    def with_initiator(self, initiator):
        ret = copy.copy(self)
        ret.initiator = initiator

        return ret


class Provider(BuilderWithContext):
    def __init__(self, initiator, builder, target):
        super().__init__(initiator, builder)
        self._target = target

    @property
    def target(self):
        if isinstance(self._target, DataDescriptor):
            return self._target
        else:
            return self._target.__name__
