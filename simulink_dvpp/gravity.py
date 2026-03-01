#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .constants import GRAVITY, REFERENCE_TOTAL_MASS, bare_hull_mass
from .kinematics import body_to_world_rotation


def gravity_vector_body(eta, mass):
    force_world = np.array([0.0, 0.0, -float(mass) * GRAVITY], dtype=float)
    return body_to_world_rotation(eta).T @ force_world


def gravity_force(eta, total_mass=REFERENCE_TOTAL_MASS):
    """
    Port of Simulink chart_155 `Gravity`.
    """
    hull_mass = bare_hull_mass(total_mass)
    return np.concatenate([gravity_vector_body(eta, hull_mass), np.zeros(3, dtype=float)])
