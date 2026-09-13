"""检查更新时序、归一化、投影、理论期望和命令行复现。"""

import csv
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.run_ogd_square import make_observations
from src.ftl import ftl_square_predictions
from src.ogd_square import ogd_square_predictions, square_metrics, uniform_expected_regret

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("eta, predictions, regret", [
    ("decreasing", [0.5, 0, 0.5], 5 / 12),
    (0.5, [0.5, 0.25, 0.625], 0.14322916666666667),
])
def test_hand_calculation(eta, predictions, regret):
    y = [0, 1, 1]
    x = ogd_square_predictions(y, eta)
    np.testing.assert_allclose(x, predictions)
    result = square_metrics(y, x)
    assert result["comparator"] == pytest.approx(2 / 3)
    assert result["comparator_loss"] == pytest.approx(1 / 3)
    assert result["regret"] == pytest.approx(regret)
    assert result["average_regret"] == pytest.approx(regret / 3)


def test_single_round():
    x = ogd_square_predictions([1])
    np.testing.assert_array_equal(x, [0.5])
    assert square_metrics([1], x)["regret"] == pytest.approx(0.125)


def test_decreasing_equals_historical_mean_and_ftl():
    y = np.random.default_rng(3).uniform(size=100)
    x = ogd_square_predictions(y)
    oracle = np.r_[0.5, np.cumsum(y[:-1]) / np.arange(1, len(y))]
    np.testing.assert_allclose(x, oracle)
    np.testing.assert_allclose(x, ftl_square_predictions(y))


@pytest.mark.parametrize("eta", ["decreasing", 0.5, 3.0])
def test_no_future_information_and_no_input_mutation(eta):
    y = np.random.default_rng(2).uniform(size=20)
    original = y.copy()
    x = ogd_square_predictions(y, eta)
    np.testing.assert_array_equal(y, original)
    for t in range(len(y)):
        changed = y.copy()
        changed[t:] = 1
        np.testing.assert_allclose(ogd_square_predictions(changed, eta)[:t + 1], x[:t + 1])


def test_projection_can_be_active():
    np.testing.assert_allclose(ogd_square_predictions([1, 0, 1], eta=3), [0.5, 1, 0])


@pytest.mark.parametrize("eta", ["decreasing", 0.5])
def test_affine_scaling(eta):
    y = np.random.default_rng(42).uniform(size=50)
    x = ogd_square_predictions(y, eta)
    scaled_y = -2 + 5 * y
    scaled_x = ogd_square_predictions(scaled_y, eta, a=-2, b=3)
    np.testing.assert_allclose(scaled_x, -2 + 5 * x)
    assert square_metrics(scaled_y, scaled_x)["regret"] == pytest.approx(25 * square_metrics(y, x)["regret"])


def test_negative_regret_is_preserved():
    assert square_metrics([0, 1], [0, 1])["regret"] == pytest.approx(-0.25)


@pytest.mark.parametrize("y", [[], [np.nan], [np.inf], [[0, 1]], 0.5, [-0.1], [1.1]])
def test_bad_observations(y):
    with pytest.raises(ValueError):
        ogd_square_predictions(y)


@pytest.mark.parametrize("kwargs", [
    {"a": 1, "b": 1}, {"a": 2, "b": 1}, {"a": np.nan}, {"b": np.inf},
    {"eta": 0}, {"eta": -1}, {"eta": np.nan}, {"eta": np.inf}, {"eta": "unknown"},
    {"x0": -1}, {"x0": np.inf},
])
def test_bad_parameters_even_for_single_round(kwargs):
    with pytest.raises(ValueError):
        ogd_square_predictions([0.5], **kwargs)


@pytest.mark.parametrize("x", [[], [0], [0, np.nan], [[0, 1]]])
def test_bad_metric_inputs(x):
    with pytest.raises(ValueError):
        square_metrics([0, 1], x)


@pytest.mark.parametrize("eta", ["decreasing", 0.5, 1.0])
def test_expected_regret_by_exact_enumeration(eta):
    # 独立二元观测同均值、方差是 Uniform[0,1] 的 3 倍。
    # 无投影生效时，期望仅依赖前两阶矩，可穷举而不用随机近似。
    regrets = []
    for y in itertools.product([0.0, 1.0], repeat=5):
        regrets.append(square_metrics(y, ogd_square_predictions(y, eta))["regret"])
    assert np.mean(regrets) == pytest.approx(3 * uniform_expected_regret(5, eta))


@pytest.mark.parametrize("rounds", [1, 3, 100, 10000])
def test_expected_regret_closed_forms(rounds):
    harmonic = np.sum(1 / np.arange(1, rounds))
    assert uniform_expected_regret(rounds) == pytest.approx((1 + harmonic) / 24)
    fixed = rounds / 72 + 5 / 216 + (0.25 ** rounds) / 54
    assert uniform_expected_regret(rounds, 0.5) == pytest.approx(fixed)


@pytest.mark.parametrize("rounds, eta", [(0, 0.5), (1.5, 0.5), (3, 2), (3, 0)])
def test_bad_expectation_parameters(rounds, eta):
    with pytest.raises(ValueError):
        uniform_expected_regret(rounds, eta)


def test_strong_convexity_bound():
    for y in itertools.product([0.0, 1.0], repeat=6):
        regret = square_metrics(y, ogd_square_predictions(y))["regret"]
        assert regret <= 0.5 * np.sum(1 / np.arange(1, 7)) + 1e-12


@pytest.mark.parametrize("scenario", ["uniform", "initial_outlier", "constant", "alternating", "switch"])
def test_scenarios_reproducible_in_interval(scenario):
    y = make_observations(scenario, 20, -2, 3, 42)
    np.testing.assert_array_equal(y, make_observations(scenario, 20, -2, 3, 42))
    assert y.shape == (20,)
    assert np.all((-2 <= y) & (y <= 3))


def test_cli_csv_and_reproducibility(tmp_path):
    output = tmp_path / "summary.csv"
    command = [sys.executable, "-m", "experiments.run_ogd_square", "--scenario", "all", "--rounds", "3", "10"]
    first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert first.stdout == second.stdout
    subprocess.run([*command, "--csv", str(output)], cwd=ROOT, capture_output=True, check=True)
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 20
    assert float(rows[0]["iid_expected_regret"]) == pytest.approx(5 / 48)
    before = output.read_bytes()
    repeated = subprocess.run([*command, "--csv", str(output)], cwd=ROOT, capture_output=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == before


@pytest.mark.parametrize("arguments", [
    ["--rounds", "0"], ["--seed", "-1"], ["--a", "1", "--b", "0"],
    ["--fixed-eta", "nan"], ["--b", "inf"],
])
def test_cli_rejects_invalid_input(arguments):
    result = subprocess.run([sys.executable, "-m", "experiments.run_ogd_square", *arguments],
                            cwd=ROOT, capture_output=True)
    assert result.returncode == 2
