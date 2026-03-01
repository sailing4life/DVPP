#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .constants import REFERENCE_TOTAL_MASS, SEA_WATER_DENSITY, rigid_body_inertia_diag
from .types import MassBreakdown


def mass_and_added_mass(
    A44_sails,
    A22_sails,
    A_foil,
    wave,
    eta,
    zero,
    res_hs,
    a_inf_override=None,
    total_mass=REFERENCE_TOTAL_MASS,
):
    """
    Port of Simulink chart_131 `Mass and Added mass`.
    """
    z = float(eta[2] - zero)
    # wave[3] is encounter frequency (ω_e); res_hs is a buoyancy proxy.
    # A frequency-dependent added-mass correction A(ω_e) is not yet implemented;
    # A_inf (high-frequency asymptote) is used throughout instead.
    _ = res_hs

    total_mass = float(total_mass)
    inertia_diag = rigid_body_inertia_diag(total_mass)

    M_rb = np.array([
        [total_mass, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, total_mass, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, total_mass, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, inertia_diag[0], 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, inertia_diag[1], 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, inertia_diag[2]],
    ])

    rho = SEA_WATER_DENSITY
    Tc = 0.5
    bk = 3.806
    sk = 0.5 * (0.408 + 0.774) * bk
    cre_k = 0.408
    ct_k = 0.774

    aek = (2.0 * (bk + Tc)) / ((cre_k + ct_k) / 2.0)
    M22_k = (2.0 * rho * np.pi * bk * sk) / np.sqrt(aek**2 + 1.0)
    L_z = 2.2
    J44_k = M22_k * L_z**2

    if a_inf_override is None:
        a_inf_matrix = np.zeros((6, 6), dtype=float)
        a_inf_matrix[1, 1] = 3659.0
        a_inf_matrix[2, 2] = 90767.0
        a_inf_matrix[3, 3] = 86082.0
        a_inf_matrix[4, 4] = 1167779.0
        a_inf_matrix[1, 3] = -6854.0
        a_inf_matrix[3, 1] = -6854.0
        a_inf_matrix[2, 4] = 39028.0
        a_inf_matrix[4, 2] = 39028.0
    else:
        a_inf_matrix = np.asarray(a_inf_override, dtype=float).copy()

    a13 = 0.0
    a15 = 0.0
    a24 = a_inf_matrix[1, 3]
    a35 = a_inf_matrix[2, 4]

    M_a = np.array([
        [a_inf_matrix[0, 0], 0.0, a13, 0.0, a15, 0.0],
        [0.0, A22_sails + a_inf_matrix[1, 1], 0.0, a24, 0.0, 0.0],
        [a13, 0.0, A_foil + a_inf_matrix[2, 2], 0.0, a35, 0.0],
        [0.0, a24, 0.0, A44_sails + J44_k + a_inf_matrix[3, 3], 0.0, 0.0],
        [a15, 0.0, a35, 0.0, a_inf_matrix[4, 4], 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    M = M_rb + M_a
    return MassBreakdown(M=M, z=z, M_a=M_a, M_rb=M_rb)
