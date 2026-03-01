#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def body_to_world_rotation(eta):
    """
    ZYX body-to-world rotation matrix.
    """
    phi = float(eta[3])
    theta = float(eta[4])
    psi = float(eta[5])

    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    return np.array([
        [cpsi * ctheta, cpsi * stheta * sphi - spsi * cphi, cpsi * stheta * cphi + spsi * sphi],
        [spsi * ctheta, spsi * stheta * sphi + cpsi * cphi, spsi * stheta * cphi - cpsi * sphi],
        [-stheta, ctheta * sphi, ctheta * cphi],
    ], dtype=float)


def euler_rate_transform(eta):
    phi = float(eta[3])
    theta = float(eta[4])
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    ctheta = np.sign(ctheta) * max(abs(ctheta), 1e-8)
    ttheta = stheta / ctheta

    return np.array([
        [1.0, sphi * ttheta, cphi * ttheta],
        [0.0, cphi, -sphi],
        [0.0, sphi / ctheta, cphi / ctheta],
    ])


def body_to_ned_velocity(eta_dot, eta):
    """
    Port of Simulink chart_328 `velocity transformations`.
    """
    Rb = body_to_world_rotation(eta)
    T_theta = euler_rate_transform(eta)

    J = np.block([
        [Rb, np.zeros((3, 3))],
        [np.zeros((3, 3)), T_theta],
    ])
    return J @ np.asarray(eta_dot, dtype=float)
