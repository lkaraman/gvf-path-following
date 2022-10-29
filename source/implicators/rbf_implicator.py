import numpy as np
from matplotlib import pyplot as plt
from sympy import symbols, plot_implicit, simplify, lambdify, Expr


class RbfImplicator:
    """
    Convertion of points to implicit function
    Based on G.Turk J.F.O’Brien, 'Shape Transformation Using Variational Implicit Functions'
    """
    def __init__(self, rbf_degree: int = 3, regularization_factor: float = 0.1) -> None:
        self.rbf_degree = rbf_degree
        self.regularization_factor = regularization_factor

        self._path_inside = None
        self._path_outside = None

    @property
    def path_outside(self) -> np.ndarray:
        return self._path_outside

    @path_outside.setter
    def path_outside(self, path_outside: np.ndarray) -> None:
        if path_outside.shape[1] != 2 or path_outside.ndim != 2:
            raise ValueError

        self._path_outside = path_outside

    @property
    def path_inside(self) -> np.ndarray:
        return self._path_inside

    @path_inside.setter
    def path_inside(self, path_inside: np.ndarray) -> None:
        if path_inside.shape[1] != 2 or path_inside.ndim != 2:
            raise ValueError

        self._path_inside = path_inside

    def _base_function(self, a: np.ndarray, b: np.ndarray) -> float:
        assert a.shape == (2,)
        assert b.shape == (2,)

        x = (a[0], b[0])
        y = (a[1], b[1])

        return np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** self.rbf_degree

    def to_implicit(self, path_on: np.ndarray) -> Expr:
        """
        Converts path to implicit SymPy function
        For details please check the reference article
        :param path_on: array of trajectory points
        :return:
        """
        if path_on.shape[1] != 2 or path_on.ndim != 2:
            raise ValueError

        x, y, x0, y0 = symbols("x, y, x0, y0")

        # construct matrix containing all points
        P = np.vstack((path_on, self.path_inside, self.path_outside))

        V_path = np.zeros((path_on.shape[0], 1))
        V_inside = np.ones((self.path_inside.shape[0], 1))
        V_outside = -np.ones((self.path_outside.shape[0], 1))

        V = np.vstack((V_path, V_inside, V_outside))

        n = V.size
        H = np.zeros((n, n))

        # Evaluate the RBF on for all combinations of xy pairs in the path
        for i in range(n):
            for jj in range(n):
                H[i, jj] = self._base_function(a=P[i, :], b=P[jj, :])

        R = np.eye(n)  # regularization matrix
        H = H + self.regularization_factor * R
        C = np.ones((n, 3))

        for i in range(n):
            C[i, 1] = P[i, 0]
            C[i, 2] = P[i, 1]

        Z = np.zeros((3, 3))

        H = np.block([
            [H, C],
            [C.T, Z]
        ])

        temp = np.block([
            [V],
            [np.zeros((3, 1))]
        ])

        W = np.linalg.inv(H) @ temp
        d = (((x - x0) ** 2 + (y - y0) ** 2) ** 0.5) ** self.rbf_degree

        f = 0
        for i in range(n):
            f = f + W[i] * d.subs([(x0, P[i, 0]), (y0, P[i, 1])])

        f = f + W[n] + W[n + 1] * x + W[n + 2] * y

        return f[0]




if __name__ == '__main__':
    path = np.asarray([
        [0, 0],
        [2, 2],
        [4, 4],
        [10, 10]
    ])

    inside = np.array([
        [10, 0]
    ])

    outside = np.array([
        [-10, 0]
    ])

    rbf_imp = RbfImplicator(rbf_degree=3, regularization_factor=0.1)
    rbf_imp.path_inside = inside
    rbf_imp.path_outside = outside

    f = rbf_imp.to_implicit(path_on=path)

    f = simplify(f)

    # f = nsimplify(f, tolerance=0.00001)
    x, y = symbols("x, y")
    f1 = lambdify([x, y], f)
    X, Y = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
    U = np.asarray([f1(x1, y1) for x1, y1 in zip(X, Y)])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X.flatten(), Y.flatten(), U.flatten())
    plt.show()

    plot_implicit(f, x_var=(x, -10, 10), y_var=(y, -10, 10), adaptive=False)
