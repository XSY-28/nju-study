"""区间约束、半平方损失下的在线梯度下降。"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite 1-D sequence")
    return array


def _interval(a: float, b: float) -> None:
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        raise ValueError("a and b must be finite, with a < b")
    if not np.isfinite(b - a):
        raise ValueError("interval width is too large")


def _step(eta: str | float, t: int) -> float:
    if isinstance(eta, str):
        if eta != "decreasing":
            raise ValueError("eta must be 'decreasing' or a positive finite number")
        return 1.0 / t
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite")
    return float(eta)


def ogd_square_predictions(
    y: ArrayLike,
    eta: str | float = "decreasing",
    a: float = 0.0,
    b: float = 1.0,
    x0: float | None = None,
) -> NDArray[np.float64]:
    """先预测，后观察；f_t(x) = (x-y_t)^2 / 2，梯度为 x-y_t。

    eta='decreasing' 表示第 t 轮观察后的步长为 1/t（t 从 1 开始）；
    数值 eta 表示固定步长。观测必须在 [a,b] 内，初值默认为中点。
    """
    y = _vector(y, "y")
    _interval(a, b)
    _step(eta, 1)  # 即使只有一轮，也检查步长。
    if np.any((y < a) | (y > b)):
        raise ValueError("observations must lie in [a, b]")
    current = a / 2 + b / 2 if x0 is None else float(x0)
    if not np.isfinite(current) or not a <= current <= b:
        raise ValueError("x0 must be finite and lie in [a, b]")

    x = np.empty(y.size, dtype=float)
    for i in range(y.size):
        x[i] = current  # 尚未使用本轮 y[i]。
        if i + 1 < y.size:
            gradient = current - y[i]
            current = float(np.clip(current - _step(eta, i + 1) * gradient, a, b))
    return x


def square_metrics(y: ArrayLike, x: ArrayLike) -> dict[str, float]:
    """事后评价：返回最佳固定决策、两方累计损失、累计和平均静态遗憾。

    在观测位于决策区间内时，均值也是区间内最佳固定决策。
    不把负遗憾截成 0：在线决策可以优于最佳固定决策。
    """
    y = _vector(y, "y")
    x = _vector(x, "x")
    if y.shape != x.shape:
        raise ValueError("x and y must have the same shape")
    u = float(np.mean(y))
    algorithm_loss = float(0.5 * np.sum((x - y) ** 2))
    comparator_loss = float(0.5 * np.sum((u - y) ** 2))
    regret = algorithm_loss - comparator_loss
    return {
        "comparator": u,
        "algorithm_loss": algorithm_loss,
        "comparator_loss": comparator_loss,
        "regret": regret,
        "average_regret": regret / y.size,
    }


def uniform_expected_regret(
    rounds: int, eta: str | float = "decreasing", a: float = 0.0, b: float = 1.0
) -> float:
    """i.i.d. Uniform[a,b]、初值为中点时的精确期望，不是单次实验值。

    仅支持 0 < eta <= 1 或 1/t，此时投影不生效。
    q_t = E[(x_t-m)^2]，q_{t+1} = (1-eta_t)^2 q_t + eta_t^2 sigma^2。
    """
    _interval(a, b)
    if not isinstance(rounds, (int, np.integer)) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if _step(eta, 1) > 1:
        raise ValueError("the expectation formula requires eta <= 1")
    variance = (b - a) ** 2 / 12
    q = 0.0
    # E[最佳固定决策累计损失] = (T-1)*sigma^2/2。
    # 因而 E[R_T] = sigma^2/2 + sum(q_t)/2；避免大数相减。
    expected_regret = variance / 2
    for t in range(1, rounds + 1):
        expected_regret += q / 2
        step = _step(eta, t)
        q = (1 - step) ** 2 * q + step**2 * variance
    return expected_regret
