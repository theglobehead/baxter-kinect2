import math

import numpy as np
import sys

class IkSolver:
    @staticmethod
    def rotation(theta):
        sin = np.sin(theta)
        cos = np.cos(theta)

        R = np.array([
            [cos, -sin],
            [sin, cos]
        ])

        return R

    @staticmethod
    def d_rotation(theta):
        sin = np.sin(theta)
        cos = np.cos(theta)

        R = np.array([
            [-sin, -cos],
            [cos, -sin]
        ])

        return R

    @classmethod
    def solve_ik_3dof(cls, target_point):
        anchor_point = np.array([0, 0])

        theta_1 = np.deg2rad(90)
        theta_2 = np.deg2rad(0)
        theta_3 = np.deg2rad(0)

        segment_1 = np.array([0.0, 1.0]) * 0.38
        segment_2 = np.array([0.0, 1.0]) * 0.37
        segment_3 = np.array([0.0, 1.0]) * 0.2

        for _ in range(5000):
            R1 = cls.rotation(theta_1)
            R2 = cls.rotation(theta_2)
            R3 = cls.rotation(theta_3)

            dR1 = cls.d_rotation(theta_1)
            dR2 = cls.d_rotation(theta_2)
            dR3 = cls.d_rotation(theta_3)

            point_1 = R1 @ segment_1
            point_2 = point_1 + R1 @ (R2 @ segment_2)
            point_3 = point_2 + R1 @ R2 @ (R3 @ segment_3)

            loss = np.sum((point_3 - target_point) ** 2)
            d_loss = -2 * (target_point - point_3)
            d_theta_1 = d_loss * (dR1 @ segment_1)
            d_theta_2 = d_loss * (R1 @ dR2 @ segment_2)
            d_theta_3 = d_loss * (R1 @ R2 @ dR3 @ segment_3)

            alpha = 1e-2
            theta_1 -= np.sum(d_theta_1 * alpha)
            theta_2 -= np.sum(d_theta_2 * alpha)
            theta_3 -= np.sum(d_theta_3 * alpha)

        s1 = theta_1 * -1
        e1 = theta_2 * -1
        w1 = theta_3 * -1

        print("#########################")
        print("loss:", loss)
        print("target_point: ", target_point)
        print("point_3:", point_3)
        print("#########################")

        return s1, e1, w1

if __name__ == "__main__":
    ee = np.array([0.20554575, 0.78545427])
    res = IkSolver.solve_ik_3dof(ee)
    print(np.degrees(res))

