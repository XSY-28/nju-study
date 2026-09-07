"""用手算、边界输入和历史信息约束检查 FTL。"""

import numpy as np
import pytest

from src.ftl import ftl_square, ftl_square_predictions


@pytest.mark.parametrize(
    "y, expected_predictions, expected_regret",
    [
        ([0, 1], [0.5, 0.0], 0.75),
        ([0, 1, 1], [0.5, 0.0, 0.5], 5 / 6),
        ([1], [0.5], 0.25),
        ([1, 1, 1], [0.5, 1.0, 1.0], 0.25),
    ],
)
def test_hand_calculated_examples(y, expected_predictions, expected_regret):
    np.testing.assert_allclose(ftl_square_predictions(y), expected_predictions)
    assert ftl_square(y) == pytest.approx(expected_regret)


def test_prediction_uses_only_past_observations():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    original = ftl_square_predictions(y)
    for t in range(len(y)):
        changed = y.copy()
        changed[t:] = 10.0  # 改变本轮及以后观测，本轮及此前预测不应改变。
        np.testing.assert_allclose(
            ftl_square_predictions(changed)[: t + 1], original[: t + 1]
        )


def test_initial_prediction_only_affects_first_round():
    np.testing.assert_allclose(ftl_square_predictions([0, 1, 1], x0=2), [2, 0, 0.5])
    assert ftl_square([0, 1], x0=2) == pytest.approx(4.5)


def test_input_is_not_modified():
    y = np.array([0.0, 1.0, 1.0])
    original = y.copy()
    ftl_square(y)
    np.testing.assert_array_equal(y, original)


@pytest.mark.parametrize("y", [[], 1.0, [[0, 1]], [0, np.nan], [np.inf], [-np.inf]])
@pytest.mark.parametrize("function", [ftl_square, ftl_square_predictions])
def test_invalid_observations_raise_value_error(y, function):
    with pytest.raises(ValueError):
        function(y)


@pytest.mark.parametrize("x0", [np.nan, np.inf, -np.inf])
def test_invalid_initial_prediction(x0):
    with pytest.raises(ValueError):
        ftl_square([0, 1], x0=x0)
