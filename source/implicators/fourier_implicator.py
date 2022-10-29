from math import floor

import numpy as np
import scipy.linalg
from pyefd import elliptic_fourier_descriptors, calculate_dc_coefficients
from sympy import Expr, symbols


class FourierImplicator:

    def __init__(self, order: int = 3) -> None:
        self.order = order

    @staticmethod
    def myconv3(g, h, p, q):
        temp_g = 1
        for i in range(p):
            temp_g = np.convolve(temp_g, g)

        temp_h = temp_g

        for j in range(q):
            temp_h = np.convolve(temp_h, h)

        return np.atleast_2d(temp_h)

    @staticmethod
    def convert_to_implicit(coefficients, A0, C0, order):
        """
        Converts Fourier coefficients to implicit function using matrix anhilation
        :param coefficients:
        :param A0:
        :param C0:
        :param order:
        :return:
        """
        a = coefficients[:, 0]
        b = coefficients[:, 1]
        c = coefficients[:, 2]
        d = coefficients[:, 3]

        # Change the trigonometric coefficients to the exponential ones
        A = (a - 1j * b) / 2
        B = (a + 1j * b) / 2
        C = (c - 1j * d) / 2
        D = (c + 1j * d) / 2

        g = np.asarray(A0)
        h = np.asarray(C0)

        x, y = symbols("x, y")

        for i in range(order):
            g = np.hstack((B[i], g, A[i]))
            h = np.hstack((D[i], h, C[i]))

        d = 2 * order
        n = floor(d / 2)

        P = np.zeros((1, 2 * d * n + 1))
        P[0, d * n] = 1

        for i in range(1, d + 1):
            for jj in range(i + 1):
                c = FourierImplicator.myconv3(g, h, i - jj, jj)
                empty_side = round(((2 * d * n + 1) - np.shape(c)[1]) / 2)
                zeros_to_fill = np.zeros((1, empty_side))
                new_row = np.hstack((zeros_to_fill, c, zeros_to_fill))
                P = np.vstack((P, new_row))
                print(i, jj)

        P_hat = np.zeros((int((d + 1) * (d + 2) / 2), 2 * d ** 2 + 2))

        # Convolution matrix computed from g and h
        P_real = np.real(P)
        P_imag = np.imag(P)

        for i in range(1, P_hat.shape[1] + 1):
            if i % 2 == 1:
                P_hat[:, i - 1] = P_real[:, int((i + 1) / 2) - 1]
            else:
                P_hat[:, i - 1] = P_imag[:, int(i / 2) - 1]

        _, _, EE = scipy.linalg.qr(P_hat, pivoting=True)

        E = np.zeros((EE.shape[0], EE.shape[0]))

        for i in range(EE.shape[0]):
            E[i, EE[i]] = 1

        Ptild = P_hat @ E
        Ptild = Ptild[:, 0: (int(d * (d + 3) / 2))]

        Z = scipy.linalg.null_space(Ptild.T, 1e-10)
        X = [1]

        for i in range(1, d + 1):
            for jj in range(i + 1):
                new_row = x ** (i - jj) * y ** jj
                X = np.vstack((X, new_row))

        f = Z.T @ X

        return f[0][0]

    def to_implicit(self, path: np.ndarray) -> Expr:

        a0, c0 = calculate_dc_coefficients(path)
        coefficients = elliptic_fourier_descriptors(contour=path,
                                                    order=self.order,
                                                    normalize=False)

        f = FourierImplicator.convert_to_implicit(
            A0=a0, C0=c0, coefficients=coefficients, order=self.order)
        return f


if __name__ == '__main__':
    pth = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    fi = FourierImplicator()
    fi.to_implicit(path=pth)

