#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .constants import GRAVITY, KNOTS_PER_MPS, SEA_WATER_DENSITY


def resistance_force(eta, eta_d, hydrostat, LWL, WB, awa_deg, submerged_area, a_array, mode):
    """
    Port of Simulink chart_54 `Resistance`.
    """
    eta = np.asarray(eta, dtype=float)
    eta_d = np.asarray(eta_d, dtype=float)
    u = float(eta_d[0])
    v = float(eta_d[1])
    U = float(np.hypot(u, v))
    rho = SEA_WATER_DENSITY
    g = GRAVITY
    Sw = submerged_area
    Sw0 = 52.32
    displacement = hydrostat / g / rho
    Bwl = WB
    Beam = 5.346
    LCF = 7.47 + 0.53
    LCB = 7.47 + 0.23
    Aw0 = 50.32
    Draft = 0.275
    cp = displacement / max(LWL * Bwl * Draft, 1e-9)

    a0, a1, a2, a3, a4, a5, a6, a7 = np.asarray(a_array, dtype=float)[:8]

    tau = np.deg2rad(4.0)
    B = WB
    beta = np.deg2rad(0.6)
    Dis = hydrostat / GRAVITY
    Lk = 15.0
    Lc = 0.88 * LWL
    A = (Lk + Lc) / max(2.0 * B, 1e-9)
    fn = U / np.sqrt(max(g * max(LWL, 1e-9), 1e-9))
    D = 0.0

    V1 = U * (1.0 - (0.0120 * tau**1.1) / (np.sqrt(A) * np.cos(tau)))

    if LWL > 0.50:
        visc = 0.0000010034
        Re = max(2.0, U) * max(0.5, LWL) / visc
        Cf = 0.075 / ((np.log10(Re) - 2.0) ** 2)

        if mode == 1:
            D = Dis * np.tan(tau) + (rho * V1**2 * Cf * A * B**2) / (2.0 * np.cos(beta) * np.cos(tau))
        elif mode == 2:
            speeds = (1.0 / KNOTS_PER_MPS) * np.array([
                0.000, 0.875, 1.750, 2.625, 3.500, 4.375, 5.250, 6.125,
                7.000, 7.875, 8.750, 9.625, 10.500, 11.375, 12.250, 13.125,
                14.000, 14.875, 15.750, 16.625, 17.500, 18.375, 19.250, 20.125,
                21.000, 21.875, 22.750, 23.625, 24.500, 25.375, 26.250, 27.125,
                28.000, 28.875, 29.750, 30.625, 31.500, 32.375, 33.250, 34.125,
                35.000,
            ])
            res = np.array([
                0.000, 0.001, 0.005, 0.100, 0.200, 0.500, 1.000, 1.500, 2.200,
                3.200, 4.200, 5.600, 8.300, 6.200, 5.300, 5.900, 6.500, 7.000,
                7.600, 8.300, 8.900, 9.500, 10.100, 10.800, 11.400, 12.000,
                12.600, 13.200, 13.800, 14.400, 15.000, 15.600, 16.200, 16.800,
                17.400, 18.100, 18.700, 19.300, 20.000, 20.700, 21.300,
            ])
            # Enforce monotonicity: resistance must not decrease with speed on a
            # displacement hull.  The raw MaxSURF table has a planing-hull dip
            # from ~10.5 kn onward (artefact of hull/software settings) that
            # allows unrealistically low resistance above hull speed.
            D = np.interp(U, speeds, np.maximum.accumulate(res)) * 1000.0
        elif mode == 3:
            Tc = Draft
            Cm = 0.85
            Cr_dsyhs = a0 + (
                a1 * LCB / max(LWL, 1e-9) +
                a2 * cp +
                a3 * displacement**(2.0 / 3.0) / Aw0 +
                a4 * Bwl / max(LWL, 1e-9) +
                a5 * LCB / LCF +
                a6 * Bwl / Tc +
                a7 * Cm
            ) * (displacement**(1.0 / 3.0) / max(LWL, 1e-9))
            D = displacement * rho * g * Cr_dsyhs + (Sw / Sw0) * Cf * 0.5 * rho * U**2 * Sw
        elif mode == 4:
            # DSYHS (Delft Systematic Yacht Hull Series) displacement-mode model.
            # The original Savitsky blending above Fn=0.61 has been removed: Savitsky
            # assumes the hull is planing (lifted out of the water by dynamic pressure),
            # which never occurs on an IMOCA 60 sailing in displacement mode.  Using
            # Savitsky gave unrealistically LOW resistance above hull speed and allowed
            # the VPP to converge at >20 kn reaching — physically impossible without foils.
            # Pure DSYHS gives monotonically increasing D via the friction term (∝ U²)
            # and is the most appropriate displacement-mode model available here.
            Tc = Draft
            Cm = 0.85
            Cr_dsyhs = a0 + (
                a1 * LCB / max(LWL, 1e-9) +
                a2 * cp +
                a3 * displacement**(2.0 / 3.0) / Aw0 +
                a4 * Bwl / max(LWL, 1e-9) +
                a5 * LCB / LCF +
                a6 * Bwl / Tc +
                a7 * Cm
            ) * (displacement**(1.0 / 3.0) / max(LWL, 1e-9))

            if fn < 0.15:
                D = 0.0
            else:
                D = displacement * rho * g * Cr_dsyhs + (Sw / Sw0) * Cf * 0.5 * rho * U**2 * Sw

    # ── Heel resistance correction ────────────────────────────────────────────
    # An heeled hull presents a distorted waterplane that increases residuary
    # resistance.  From Horel (2019) Table 1 / ORC VPP §6 methodology:
    #   R_heeled = R_upright × (1 + β_h × sin²φ)   with β_h ≈ 0.9
    # At φ = 35° (typical upwind heel): factor ≈ 1.296 (+30%).
    phi = float(eta[3])   # heel angle [rad]; positive = heeled to starboard
    beta_h = 0.9
    D = D * (1.0 + beta_h * np.sin(phi) ** 2)

    # Hull aerodynamic windage is already handled in the sail subsystem.
    # Keep the resistance block hydrodynamic so loads are not double-counted.
    surge_drag = np.sign(u) * D if U > 1e-9 else 0.0
    Rt = np.array([-surge_drag, 0.0, 0.0, 0.0, 0.0, 0.0])
    return Rt, U, tau
