import numpy as np

class Vehicle:
    def __init__(self, x_init: float, y_init: float, yaw_init: float):
        self.length = 2
        self.width = 0.5
        self.Ts = 0.1

        self.x = x_init
        self.y = y_init
        self.yaw = yaw_init

    def drive(self, v: float, omega: float):
        """

        :param v: velocity
        :param omega: angular rate
        :return:
        """

        self.x = self.x + v*np.cos(self.yaw)* self.Ts
        self.y = self.y + v*np.sin(self.yaw)* self.Ts
        self.yaw = self.yaw + omega*self.Ts

