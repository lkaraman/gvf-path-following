"""

A Python implementation of the algorithm described in [#a] for control of nonholonomic robot using
guiding vector field

References
----------
.. [#a] Y. A. Kapitanyuk, A. V. Proskurnikov and M. Cao, "A guiding vector field algorithm for path
    following control of nonholonomic mobile robots"


"""

import numpy as np
from sympy import Matrix, symbols, derive_by_array, diff, cos, sin, sqrt


class GuidingVectorFieldControl:
    def __init__(self, kn=0.5, ur=1, kdelta: float = 0.5) -> None:
        """
        Guiding Vector Field control
        :param kn: compromise between convergence (k>>) and following (k<<) of the vector field
        :param ur: velocity control signal
        :param kdelta: proportional control parameter
        """
        self.kn = kn
        self.ur = ur
        self.kdelta = kdelta

        self.x, self.y, self.alpha = symbols("x, y, alpha")

        self.md = None
        self.omegad = None

    @staticmethod
    def gradient(f, vars):
        return Matrix(derive_by_array(f, vars))

    @staticmethod
    def hessian(f, vars):
        return Matrix(derive_by_array(derive_by_array(f, vars), vars))

    def initialize_gvf(self, f):
        """
        Initializes guiding vector field calculated from given implicit SymPy-like function
        :param f: SymPy symbolic function
        :return:
        """
        x = self.x
        y = self.y
        alpha = self.alpha

        phiSym = self.x

        I2 = Matrix([[1, 0], [0, 1]])
        E = Matrix([[0, 1], [-1, 0]])  # (8)

        # Equations (18)
        v = E * self.gradient(f, (x, y)) - self.kn * phiSym.subs(x, f) * self.gradient(f, (x, y))
        de = (self.ur * diff(phiSym, x).subs(x, f) * self.gradient(f, (x, y)).T * Matrix([cos(alpha), sin(alpha)]))[0]
        dv = self.ur * (E - self.kn * phiSym.subs(x, f) * I2) * self.hessian(f, (x, y)) * Matrix(
            [cos(alpha), sin(alpha)]) - self.kn * de * self.gradient(f, (x, y))

        v_norm = sqrt(v.dot(v))

        dmd = (I2 / v_norm - (v * v.T) / v_norm ** 3) * dv
        md = v / v_norm
        omegad = -dmd.T * E * md

        self.md = md
        self.omegad = omegad

        return v, omegad, md

    def step_gvf(self, current_pose) -> (float, float):
        """
        Calculates angle control signal based on the current pose
        :param current_pose:
        :return:
        """
        x_meas, y_meas, alpha_meas = current_pose

        omegad_num = float(self.omegad.subs([(self.x, x_meas), (self.y, y_meas), (self.alpha, alpha_meas)])[0])
        md_num = self.md.subs([(self.x, x_meas), (self.y, y_meas)])
        m1, m2 = float(md_num[0]), float(md_num[1])

        theta_d = np.arctan2(m2, m1)
        delta = alpha_meas - theta_d
        delta = (delta + np.pi) % (2 * np.pi) - np.pi

        omega = omegad_num - self.kdelta * delta
        print(f'omegad: {omegad_num}, delta: {self.kdelta * delta}')

        return omega, md_num
