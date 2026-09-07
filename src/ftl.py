"""平方损失下的 Follow the Leader（FTL），保留原练习的均值递推思路。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def ftl_square_predictions(y: ArrayLike, x0: float = 0.5) -> NDArray[np.float64]:
    """返回每轮预测；预测 x[t] 时只使用 y[:t]。

    损失为 (x - y[t])**2，决策域为实数。除第一轮外，FTL 的预测
    是过去观测的平均值。第一轮没有历史数据，使用 x0（默认 0.5）。
    y 必须是一维、非空、只含有限实数的序列，x0 必须是有限实数。
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("y 必须是非空的一维序列")
    if not np.all(np.isfinite(y)):
        raise ValueError("y 不能包含 NaN 或无穷大")
    if not np.isfinite(x0):
        raise ValueError("x0 必须是有限实数")

    x = np.empty(len(y), dtype=float)
    x[0] = x0
    for i in range(len(y) - 1):
        # 观察到 y[i] 后，更新下一轮预测；i=0 时得到 x[1]=y[0]。
        x[i + 1] = x[i] + (y[i] - x[i]) / (i + 1)
    return x


def ftl_square(y: ArrayLike, x0: float = 0.5) -> float:
    """返回 FTL 相对于事后最佳固定决策的累计静态遗憾。

    保留原函数的返回方式：算法累计平方损失减去最佳固定决策的
    累计平方损失。全序列均值 u 仅用于事后评估，不参与在线预测。
    输入要求与 ftl_square_predictions 相同。
    """
    y = np.asarray(y, dtype=float)
    x = ftl_square_predictions(y, x0)
    u = y.mean()
    algorithm_loss = np.sum((x - y) ** 2)
    comparator_loss = np.sum((u - y) ** 2)
    return float(algorithm_loss - comparator_loss)
