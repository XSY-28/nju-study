"""从仓库根目录运行：python -m experiments.run_ftl_square。"""

import argparse

import numpy as np

from src.ftl import ftl_square, ftl_square_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="二元观测上的平方损失 FTL 实验")
    parser.add_argument("--rounds", type=int, default=400, help="轮数（默认 400）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds 必须为正整数")
    if args.seed < 0:
        parser.error("--seed 必须为非负整数")

    rng = np.random.default_rng(seed=args.seed)
    y = rng.integers(0, 2, size=args.rounds)
    x = ftl_square_predictions(y)
    u = y.mean()

    print(f"轮数：{args.rounds}，随机种子：{args.seed}")
    print(f"前 5 轮观测：{y[:5]}")
    print(f"前 5 轮预测：{x[:5]}")
    print(f"最佳固定决策：{u:.6f}")
    print(f"算法累计损失：{np.sum((x - y) ** 2):.6f}")
    print(f"最佳固定决策累计损失：{np.sum((u - y) ** 2):.6f}")
    print(f"累计静态遗憾：{ftl_square(y):.6f}")


if __name__ == "__main__":
    main()
