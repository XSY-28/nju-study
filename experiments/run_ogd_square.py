"""运行：python -m experiments.run_ogd_square --scenario all。"""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.ogd_square import (
    ogd_square_predictions,
    square_metrics,
    uniform_expected_regret,
)

SCENARIOS = ("uniform", "initial_outlier", "constant", "alternating", "switch")


def make_observations(
    scenario: str, rounds: int, a: float, b: float, seed: int
) -> np.ndarray:
    """先在 [0,1] 生成数据，再做仿射变换；每次使用独立的局部随机源。"""
    if scenario == "uniform":
        unit = np.random.default_rng(seed).uniform(size=rounds)
    elif scenario == "initial_outlier":
        unit = np.ones(rounds)
        unit[0] = 0
    elif scenario == "constant":
        unit = np.full(rounds, 0.7)
    elif scenario == "alternating":
        unit = np.arange(rounds) % 2
    elif scenario == "switch":
        unit = np.where(np.arange(rounds) < rounds // 2, 0.2, 0.8)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return np.clip(a + (b - a) * unit, a, b)


def main() -> None:
    parser = argparse.ArgumentParser(description="半平方损失 OGD：1/t 与固定步长比较")
    parser.add_argument("--a", type=float, default=0.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, nargs="+", default=[100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-eta", type=float, default=0.5)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="uniform")
    parser.add_argument("--csv", type=Path, help="可选：将汇总结果写入新 CSV 文件，不覆盖已有文件")
    args = parser.parse_args()
    if any(t < 1 for t in args.rounds):
        parser.error("--rounds 必须全部为正整数")
    if args.seed < 0:
        parser.error("--seed 必须为非负整数")
    # 在生成数据前验证区间、步长；与核心算法共用验证逻辑。
    try:
        ogd_square_predictions([args.a], args.fixed_eta, args.a, args.b)
    except ValueError as error:
        parser.error(str(error))

    rows = []
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    for scenario in scenarios:
        for rounds in args.rounds:
            y = make_observations(scenario, rounds, args.a, args.b, args.seed)
            print(f"\nscenario={scenario}, T={rounds}, interval=[{args.a}, {args.b}], seed={args.seed}")
            # 同一份数据比较两种算法；均先预测再观察。
            for label, eta in (("1/t", "decreasing"), (str(args.fixed_eta), args.fixed_eta)):
                x = ogd_square_predictions(y, eta, args.a, args.b)
                metrics = square_metrics(y, x)
                expectation = (
                    uniform_expected_regret(rounds, eta, args.a, args.b)
                    if scenario == "uniform" and (eta == "decreasing" or args.fixed_eta <= 1)
                    else None
                )
                row = dict(scenario=scenario, rounds=rounds, a=args.a, b=args.b,
                           seed=args.seed, eta=label, **metrics,
                           iid_expected_regret=expectation)
                rows.append(row)
                print(f"  eta={label}: u*={metrics['comparator']:.6f}, "
                      f"loss={metrics['algorithm_loss']:.6f}, "
                      f"best_fixed_loss={metrics['comparator_loss']:.6f}, "
                      f"R_T={metrics['regret']:.6f}, R_T/T={metrics['average_regret']:.6f}")
                if expectation is not None:
                    print(f"    i.i.d. 理论期望 E[R_T]={expectation:.6f}（不是本次样本的上界）")
    if args.csv:
        with args.csv.open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV: {args.csv}")


if __name__ == "__main__":
    main()
