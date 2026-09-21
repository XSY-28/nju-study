"""在线学习函数共用的实数输入检查。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _real_array(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(values)
        # 必须在转 float 之前检查，否则 NumPy 会丢弃复数的虚部。
        if np.iscomplexobj(array):
            raise ValueError(f"{name} 必须只包含实数")
        return np.asarray(array, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须只包含实数") from error


def finite_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = _real_array(values, name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} 必须是非空的一维实数序列")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须只包含有限实数，不能包含 NaN 或无穷大")
    return array


def finite_scalar(value: float, name: str) -> float:
    array = _real_array(value, name)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f"{name} 必须是一个有限实数标量")
    return float(array)
