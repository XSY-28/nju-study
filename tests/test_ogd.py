"""核对讲义手算、投影、在线时序和输入边界。"""

import numpy as np
import pytest

from experiments.run_ogd_linear import alternating_gradients
from src.ogd import ftl_linear, linear_regret, ogd_linear


def test_hand_calculated_three_rounds():
    g = [0.5, -1, 1]
    ogd = ogd_linear(g, eta=0.1)
    np.testing.assert_allclose(ogd, [0, -0.05, 0.05])
    assert linear_regret(g, ogd) == pytest.approx(0.6)
    ftl = ftl_linear(g)
    np.testing.assert_allclose(ftl, [0, -1, 1])
    assert linear_regret(g, ftl) == pytest.approx(2.5)


def test_lecture_experiment():
    g = alternating_gradients(400)
    np.testing.assert_array_equal(g[:5], [0.5, -1, 1, -1, 1])
    assert linear_regret(g, ogd_linear(g, eta=0.1)) == pytest.approx(20.45)
    assert linear_regret(g, ftl_linear(g)) == pytest.approx(399.5)


def test_projection_reaches_both_endpoints():
    np.testing.assert_allclose(ogd_linear([1, -1, 1], eta=10), [0, -1, 1])


@pytest.mark.parametrize("algorithm", [ftl_linear, lambda g: ogd_linear(g, eta=0.1)])
def test_decisions_use_only_past_gradients(algorithm):
    g = alternating_gradients(6)
    original = g.copy()
    decisions = algorithm(g)
    for t in range(len(g)):
        changed = g.copy()
        changed[t:] = 0.3
        np.testing.assert_allclose(algorithm(changed)[:t + 1], decisions[:t + 1])
    np.testing.assert_array_equal(g, original)


def test_single_round_zero_gradients_and_ftl_ties():
    assert linear_regret([0.5], ogd_linear([0.5], eta=2)) == pytest.approx(0.5)
    np.testing.assert_array_equal(ftl_linear([1, -1, 1]), [0, -1, 0])
    np.testing.assert_array_equal(ogd_linear([0, 0], eta=1, x0=0.3), [0.3, 0.3])
    assert linear_regret([0, 0], [0.3, 0.3]) == 0


@pytest.mark.parametrize("g", [[], 1.0, [[1, -1]], [np.nan], [np.inf]])
def test_invalid_gradients(g):
    for algorithm in (ftl_linear, lambda g: ogd_linear(g, eta=0.1)):
        with pytest.raises(ValueError):
            algorithm(g)


@pytest.mark.parametrize("eta", [0, -1, np.nan, np.inf])
def test_invalid_step_size(eta):
    with pytest.raises(ValueError):
        ogd_linear([1], eta)


@pytest.mark.parametrize("x0", [-2, 2, np.nan, np.inf])
def test_invalid_initial_decision(x0):
    with pytest.raises(ValueError):
        ogd_linear([1], eta=0.1, x0=x0)


@pytest.mark.parametrize("decisions", [[0, 0], [2], [np.nan], []])
def test_invalid_regret_decisions(decisions):
    with pytest.raises(ValueError):
        linear_regret([1], decisions)


@pytest.mark.parametrize("rounds", [0, -1, 1.5, True])
def test_invalid_rounds(rounds):
    with pytest.raises(ValueError):
        alternating_gradients(rounds)
