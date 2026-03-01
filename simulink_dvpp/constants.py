#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


KNOTS_PER_MPS = 1.943844
SEA_WATER_DENSITY = 1025.0
AIR_DENSITY = 1.225
GRAVITY = 9.81

REFERENCE_TOTAL_MASS = 12_500.0

KEEL_FIN_MASS  = 562.13100544
KEEL_BULB_MASS = 3500.0
KEEL_TOTAL_MASS = KEEL_FIN_MASS + KEEL_BULB_MASS
FOIL_TOTAL_MASS = 605.916447120677

# ── Geometry for parallel-axis inertia calculation ────────────────────────────
# Source: keel span from appendages._KEEL_POINTS (z: −0.260201 → −4.101993 +0.406 m)
#         bulb_arm from appendages.keel_gravity_force (= 4.1 m)
#         foil span from foils._FOIL_POINTS_ONSIDE (|ȳ| centroid ≈ 4.774 m, x ≈ 1.609 m)
_KEEL_FIN_COG_Z  = (-0.260201 + -4.101993) / 2.0 + 0.406  # ≈ −1.775 m below origin
_KEEL_BULB_COG_Z = -4.1                                     # m below origin
_FOIL_COG_Y      = 4.774    # |ȳ| of combined onside foil centroid, m from centreline
_FOIL_COG_X      = 1.609166 # foil quarter-chord x, m


def _reference_inertia_diag():
    """
    Total rigid-body inertia [Ixx, Iyy, Izz] about the body-frame origin at
    REFERENCE_TOTAL_MASS, computed via the parallel-axis theorem (Fossen 2011
    §3.1, Horel 2019 Table 1 methodology).

    Assumptions
    -----------
    - Hull CoG is at the body-frame origin (z = 0, x = 0, y = 0).
    - Hull Ixx_hull and Iyy_hull are about the body-frame origin.
    - Keel fin / bulb treated as point masses at their CoG.
    - Both foils treated as a single lumped mass at their average centroid
      (symmetric port/starboard, so net y-contribution to Iyy/Izz cancels,
      but Ixx contribution is 2 × each foil's y-offset squared).
    """
    m_hull = max(REFERENCE_TOTAL_MASS - KEEL_TOTAL_MASS - FOIL_TOTAL_MASS, 1.0)

    # System CoG z (hull CoG at origin, keel/bulb below):
    z_cog = (KEEL_FIN_MASS * _KEEL_FIN_COG_Z + KEEL_BULB_MASS * _KEEL_BULB_COG_Z) / REFERENCE_TOTAL_MASS

    # ── Hull: shift from origin (= hull CoG) to system CoG ───────────────────
    I_hull_xx = 15517.91   # hull-only values from NEMOH panel solver
    I_hull_yy = 189799.3
    I_hull_zz = 200553.5
    d_hull = z_cog          # displacement from hull CoG (origin) to system CoG
    dI_hull_xx = m_hull * d_hull**2   # Steiner term (roll: d in y-z plane, hull on centreline → d≈z_cog)
    dI_hull_yy = m_hull * d_hull**2   # Steiner term (pitch: same, keel on x-centreline)

    # ── Keel fin: point mass at _KEEL_FIN_COG_Z ──────────────────────────────
    dz_fin = _KEEL_FIN_COG_Z - z_cog
    dI_fin_xx = KEEL_FIN_MASS * dz_fin**2
    dI_fin_yy = KEEL_FIN_MASS * dz_fin**2   # keel on x=y=0 centreline

    # ── Keel bulb: point mass at _KEEL_BULB_COG_Z ────────────────────────────
    dz_bulb = _KEEL_BULB_COG_Z - z_cog
    dI_bulb_xx = KEEL_BULB_MASS * dz_bulb**2
    dI_bulb_yy = KEEL_BULB_MASS * dz_bulb**2

    # ── Foils: lumped at (x=_FOIL_COG_X, |y|=_FOIL_COG_Y, z≈0) ─────────────
    dz_foil = 0.0 - z_cog   # foil CoG near z=0, shifted by system CoG
    # Ixx (roll): d² = y_foil² + dz_foil²  (both foils contribute identically)
    dI_foil_xx = FOIL_TOTAL_MASS * (_FOIL_COG_Y**2 + dz_foil**2)
    # Iyy (pitch): d² = x_foil² + dz_foil²  (symmetric foils: y cancels for Iyy)
    dI_foil_yy = FOIL_TOTAL_MASS * (_FOIL_COG_X**2 + dz_foil**2)
    # Izz (yaw): d² = x_foil² + y_foil²  (foils spread wide)
    dI_foil_zz = FOIL_TOTAL_MASS * (_FOIL_COG_X**2 + _FOIL_COG_Y**2)

    Ixx = I_hull_xx + dI_hull_xx + dI_fin_xx + dI_bulb_xx + dI_foil_xx
    Iyy = I_hull_yy + dI_hull_yy + dI_fin_yy + dI_bulb_yy + dI_foil_yy
    Izz = I_hull_zz + dI_foil_zz   # keel on centreline → negligible yaw contribution
    return np.array([Ixx, Iyy, Izz], dtype=float)


REFERENCE_INERTIA_DIAG = _reference_inertia_diag()


def rigid_body_inertia_diag(total_mass):
    scale = float(total_mass) / REFERENCE_TOTAL_MASS
    return REFERENCE_INERTIA_DIAG * scale


def bare_hull_mass(total_mass):
    return max(float(total_mass) - KEEL_TOTAL_MASS - FOIL_TOTAL_MASS, 1.0)
