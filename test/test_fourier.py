import numpy as np
from matplotlib import pyplot as plt

from fourier.impl import convert_me


def test_fourier():
    x_compl = []
    y_compl = []

    for angle in np.linspace(0, 2*np.pi, 1000):
        x = np.cos(angle)
        y = np.sin(angle)
        plt.plot([x], [y], 'bo')
        x_compl.append(x)
        y_compl.append(y)


    plt.show()
    pth = np.asarray(list(zip(x_compl, y_compl)))

    pth = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
    # x, y = zip(*pth)

    f = convert_me(path=pth, degree=1)

    # plt.scatter(x, y)
    plt.show()

