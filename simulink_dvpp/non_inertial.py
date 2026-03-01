#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .constants import REFERENCE_TOTAL_MASS, rigid_body_inertia_diag


def rigid_body_coriolis(eta_d, total_mass=REFERENCE_TOTAL_MASS, inertia_diag=None):
    """
    Port of Simulink chart_321 `non inertial effects`.
    """
    m = float(total_mass)
    x = y = z = 0.0

    u, v, w, p, q, r = np.asarray(eta_d, dtype=float)
    Iyz = Ixz = 0.0
    if inertia_diag is None:
        Ix, Iy, Iz = rigid_body_inertia_diag(m)
    else:
        Ix, Iy, Iz = np.asarray(inertia_diag, dtype=float)

    # Fossen (2011) Table 3.1 — full skew-symmetric C_RB.
    # Top-right block: -m * S(nu1) where S is the skew-symmetric cross-product matrix.
    # Bottom-right block: -S(I_b * omega).  With Iyz=Ixz=0 and CoG at origin (x=y=z=0):
    #   [3,5] = -Iy*q,  [4,5] = Ix*p,  [0,5] = -m*v,  [1,5] = m*u
    # Row 5 (yaw) is zeroed because this is a 5-DOF model.
    C_rb = -np.array([
        [0.0, 0.0, 0.0, m * (y * q + z * r), -m * (x * q - w), -m * (v + r * x - p * z)],
        [0.0, 0.0, 0.0, -m * (y * p + w), m * (z * r + x * p),  m * (u + q * z - r * y)],
        [0.0, 0.0, 0.0, -m * (z * p - v), -m * (z * q + u),      m * (x * p + y * q)    ],
        [-m * (y * q + z * r), m * (y * p + w), m * (z * p - v), 0.0, -Iyz * q - Ixz * p + Iz * r, -Iy * q + Iyz * r],
        [ m * (x * q - w), -m * (z * r + x * p), m * (z * q + u), Iyz * q + Ixz * p - Iz * r, 0.0,  Ix * p - Ixz * r],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    return C_rb
