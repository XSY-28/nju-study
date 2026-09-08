"""从仓库根目录运行：python -m experiments.run_ogd_linear。"""

import argparse

import numpy as np
from numpy.typing import NDArray

from src.ogd import ftl_linear, linear_regret, ogd_linear


def alternating_gradients(rounds: int) -> NDArray[np.float64]:
    """生成用于比较 FTL 与 OGD 的交替梯度序列：0.5, -1, 1, -1, ...。"""
    if isinstance(rounds, bool) or not isinstance(rounds, (int, np.integer)) or rounds < 1:
        raise ValueError("rounds 必须为正整数")
    t = np.arange(1, rounds + 1)
    g = np.where(t % 2 == 1, 1.0, -1.0)
    g[0] = 0.5
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description="线性损失下 FTL 与 OGD 的对比实验")
    parser.add_argument("--rounds", type=int, default=400, help="轮数（默认 400）")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds 必须为正整数")

    g = alternating_gradients(args.rounds)
    eta = 2 / np.sqrt(args.rounds)  # 决策域直径 D=2，梯度界 G=1。
    ogd = ogd_linear(g, eta)
    ftl = ftl_linear(g)

    print(f"轮数：{args.rounds}，OGD 步长：{eta:.6f}")
    print(f"前 5 轮梯度：{g[:5]}")
    print(f"OGD 前 5 轮决策：{ogd[:5]}")
    print(f"FTL 前 5 轮决策：{ftl[:5]}")
    print(f"最佳固定决策累计损失：{-abs(np.sum(g)):.6f}")
    print(f"OGD 累计损失：{np.sum(ogd * g):.6f}")
    print(f"OGD 累计静态遗憾：{linear_regret(g, ogd):.6f}")
    print(f"FTL 累计静态遗憾：{linear_regret(g, ftl):.6f}")
    print(f"OGD 理论上界 DG√T：{2 * np.sqrt(args.rounds):.6f}")


if __name__ == "__main__":
    main()
