"""决策域 [-1, 1]、线性损失 f_t(x)=g_t*x 下的 OGD 与 FTL。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite_scalar, finite_vector


def ogd_linear(g: ArrayLike, eta: float, x0: float = 0.0) -> NDArray[np.float64]:
    """返回每轮决策；先记录 x_t，再观察 g_t 并投影更新下一轮。

    g 为非空、有限的一维梯度序列，eta 为固定正步长，x0 在 [-1, 1]。
    返回值与 g 等长；算法不会修改输入。
    """
    g = finite_vector(g, "g")
    eta = finite_scalar(eta, "eta")
    x0 = finite_scalar(x0, "x0")
    if eta <= 0:
        raise ValueError("eta 必须是有限的正数")
    if not -1 <= x0 <= 1:
        raise ValueError("x0 必须在 [-1, 1] 内")

    decisions = np.empty(len(g), dtype=float)
    x = float(x0)
    for i in range(len(g)):
        decisions[i] = x
        x = np.clip(x - eta * g[i], -1, 1)
    return decisions


def ftl_linear(g: ArrayLike) -> NDArray[np.float64]:
    """最小化过去的累计线性损失；没有历史或累计梯度为零时取 0。"""
    g = finite_vector(g, "g")
    decisions = np.empty(len(g), dtype=float)
    past_gradient = 0.0
    for i in range(len(g)):
        decisions[i] = -np.sign(past_gradient)
        past_gradient += g[i]
    return decisions


def linear_regret(g: ArrayLike, decisions: ArrayLike) -> float:
    """计算静态遗憾；事后最佳固定决策的累计损失为 -abs(sum(g))。"""
    g = finite_vector(g, "g")
    decisions = finite_vector(decisions, "decisions")
    if decisions.shape != g.shape:
        raise ValueError("decisions 必须与 g 等长")
    if np.any(np.abs(decisions) > 1):
        raise ValueError("decisions 必须在 [-1, 1] 内")
    loss = np.sum(decisions * g)
    best_fixed_loss = -abs(np.sum(g))
    return float(loss - best_fixed_loss)
