import numpy as np
from matplotlib import pyplot as plt
from sympy import lambdify, plot_implicit, simplify
import matplotlib.patches

from source.implicators.fourier_implicator import FourierImplicator
from gvf import GuidingVectorFieldControl
# from rbf.interpolate import calculate_implicit
from vehicle import Vehicle

# Colors
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'

def plot(sym_x, sym_y, sym_md, vehicle):
    X, Y = np.meshgrid(np.linspace(-20, 20, 30), np.linspace(-20, 20, 30))

    f1 = lambdify([sym_x, sym_y], sym_md[0])
    f2 = lambdify([sym_x, sym_y], sym_md[1])

    U = [f1(x1, y1) for x1, y1 in zip(X, Y)]
    V = [f2(x1, y1) for x1, y1 in zip(X, Y)]

    p1 = plot_implicit(f, x_var=(sym_x, -10, 10), y_var=(sym_y, -10, 10), adaptive=False, show=False)

    fig, ax = plt.subplots()
    move_sympyplot_to_axes(p1, ax)

    plt.xlim([-20, 20])
    plt.ylim([-20, 20])
    ax.quiver(X, Y, U, V, linewidth=1)



def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['top'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    plt.close(backend.fig)


if __name__ == '__main__':
    # plt.ion()
    # ------------------------------------------------------------------------- #
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

    pth = np.asarray([[10, 0], [0, 10], [-10, 0], [0, -10], [10, 0]])
    fi = FourierImplicator(order=1)
    f = fi.to_implicit(path=pth)

    f = simplify(f)
    # f = nsimplify(f, tolerance=0.001)

    gvf = GuidingVectorFieldControl()
    gvf.initialize_gvf(f=f)

    vehicle = Vehicle(x_init=5, y_init=6, yaw_init=2)

    plot(sym_x=gvf.x, sym_y=gvf.y, sym_md=gvf.md, vehicle=vehicle)
    plt.ion()

    while True:
        current_pose = [vehicle.x, vehicle.y, vehicle.yaw]
        omega, _ = gvf.step_gvf(current_pose)
        print(omega)
        vehicle.drive(v=1.0, omega=omega)
        # vehicle.show()

        ax = plt.gca()

        yaw = np.rad2deg(vehicle.yaw)

        car_patch = matplotlib.patches.Rectangle((vehicle.x, vehicle.y), width=vehicle.length, height=vehicle.width,
                                                 angle=yaw, facecolor=CAR,
                                                 edgecolor=CAR_OUTLINE, zorder=20)

        # Shift center rectangle to match center of the car
        car_patch.set_x(vehicle.x - (vehicle.length / 2 *
                                     np.cos(vehicle.yaw) -
                                     vehicle.width / 2 *
                                     np.sin(vehicle.yaw)))
        car_patch.set_y(vehicle.y - (vehicle.width / 2 *
                                     np.cos(vehicle.width) +
                                     vehicle.length / 2 *
                                     np.sin(vehicle.yaw)))

        if len(ax.patches) > 0:
            ax.patches.pop()
        ax.add_patch(car_patch)
        # plt.scatter(vehicle.x, vehicle.y, s=0.1, c='r')

        # plt.gcf().canvas.draw()
        plt.pause(0.0001)

