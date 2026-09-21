"""非法输入必须明确拒绝，不能丢弃虚部后继续计算。"""

import numpy as np
import pytest

from src.ftl import ftl_square, ftl_square_predictions
from src.ogd import ftl_linear, linear_regret, ogd_linear


@pytest.mark.parametrize(
    "algorithm",
    [
        ftl_square_predictions,
        ftl_square,
        ftl_linear,
        lambda values: ogd_linear(values, eta=0.1),
        lambda values: linear_regret(values, [0, 0]),
        lambda values: linear_regret([1, -1], values),
    ],
)
@pytest.mark.parametrize("values", [np.array([1 + 2j, 0]), [1 + 2j, 0]])
def test_complex_vectors_are_rejected(algorithm, values):
    with pytest.raises(ValueError, match="实数"):
        algorithm(values)


@pytest.mark.parametrize(
    "algorithm",
    [
        lambda value: ftl_square_predictions([0, 1], x0=value),
        lambda value: ftl_square([0, 1], x0=value),
        lambda value: ogd_linear([1, -1], eta=0.1, x0=value),
        lambda value: ogd_linear([1, -1], eta=value),
    ],
)
@pytest.mark.parametrize("value", [0.5 + 1j, [0.5], np.array([0.5]), None, "bad"])
def test_non_real_or_non_scalar_parameters_are_rejected(algorithm, value):
    with pytest.raises(ValueError, match="实数"):
        algorithm(value)


def test_numpy_real_scalars_remain_supported():
    np.testing.assert_allclose(
        ftl_square_predictions([0, 1], x0=np.float64(0.5)), [0.5, 0]
    )
    np.testing.assert_allclose(
        ogd_linear([1, -1], eta=np.array(0.1), x0=np.float64(0)), [0, -0.1]
    )
