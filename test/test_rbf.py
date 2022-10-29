import numpy as np
import pytest
from sympy import symbols

from source.implicators.rbf_implicator import RbfImplicator


@pytest.fixture
def rbf_implicator():
    return RbfImplicator()


class TestRbfImplicator:

    def test_rbf_simple_linear(self, rbf_implicator: RbfImplicator):
        path = np.asarray([
            [0, 0],
            [10, 10]
        ])

        inside = np.array([
            [10, 0]
        ])

        outside = np.array([
            [-10, 0]
        ])

        rbf_implicator.path_inside = inside
        rbf_implicator.path_outside = outside

        f = rbf_implicator.to_implicit(path_on=path)


        assert len(f.args) == 5

        assert f.args[0] == pytest.approx(0.0, abs=1e-5)
        assert str(f.args[1]) == '0.1*x'
        assert str(f.args[4]) == '-0.1*y'

    def test_rbf_simple_ellipse(self, rbf_implicator: RbfImplicator):

        path = np.asarray([
            [-3, 0],
            [0, 3],
            [5, 0],
            [0, -3]
        ])

        inside = np.array([
            [0, 0]
        ])

        outside = np.array([
            [10, 10]
        ])

        rbf_implicator.path_inside = inside
        rbf_implicator.path_outside = outside

        f = rbf_implicator.to_implicit(path_on=path)

        x, y = symbols("x, y")

        assert f.evalf(subs={x: -3, y: 0}) == pytest.approx(0.0, abs=1e-2)
        assert f.evalf(subs={x: 0, y: 3}) == pytest.approx(0.0, abs=1e-2)
        assert f.evalf(subs={x: 5, y: 0}) == pytest.approx(0.0, abs=1e-2)
        assert f.evalf(subs={x: 0, y: -3}) == pytest.approx(0.0, abs=1e-2)

        assert f.evalf(subs={x: 0, y: 0}) == pytest.approx(1.0, abs=1e-2)
        assert f.evalf(subs={x: 10, y: 10}) == pytest.approx(-1.0, abs=1e-2)

    def test_rbf_complex_path(self, rbf_implicator: RbfImplicator):

        path = np.asarray([
            [0, 1],
            [4, 3],
            [1, 3],
            [2, 4],
            [5, 0],
            [8, -1]
        ])

        inside = np.array([
            [0, 0]
        ])

        outside = np.array([
            [10, 10]
        ])

        rbf_implicator.path_inside = inside
        rbf_implicator.path_outside = outside

        f = rbf_implicator.to_implicit(path_on=path)

        x, y = symbols("x, y")

        assert f.evalf(subs={x: 0, y: 1}) == pytest.approx(0.0, abs=1e-1)
        assert f.evalf(subs={x: 4, y: 3}) == pytest.approx(0.0, abs=1e-1)
        assert f.evalf(subs={x: 1, y: 3}) == pytest.approx(0.0, abs=1e-1)
        assert f.evalf(subs={x: 2, y: 4}) == pytest.approx(0.0, abs=1e-1)
        assert f.evalf(subs={x: 5, y: 0}) == pytest.approx(0.0, abs=1e-1)
        assert f.evalf(subs={x: 8, y: -1}) == pytest.approx(0.0, abs=1e-1)






