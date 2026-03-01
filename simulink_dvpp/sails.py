#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np

from .constants import AIR_DENSITY, GRAVITY
from .helpers import DelayLine, bilinear_interp, clamp_interp
from .types import SailInputs, SailOutputs


_AWA_LOOKUP = np.array([0.0, 7.0, 9.0, 12.0, 28.0, 60.0, 90.0, 120.0, 150.0, 180.0], dtype=float)
_MAIN_CD = np.array([0.03448, 0.01724, 0.01466, 0.01466, 0.02586, 0.11302, 0.38250, 0.96888, 1.31578, 1.34483], dtype=float)
_MAIN_CL = np.array([0.0, 0.94828, 1.13793, 1.25, 1.42681, 1.38319, 1.26724, 0.93103, 0.38793, -0.11207], dtype=float)
_JIB_CD = np.array([0.08, 0.05, 0.032, 0.031, 0.037, 0.25, 0.35, 0.73, 0.95, 0.9], dtype=float) * 1.15
_JIB_CL = np.array([0.0, 0.0, 1.1, 1.475, 1.5, 1.45, 1.25, 0.4, 0.0, -0.1], dtype=float)

_HEADSAIL_AWA = np.array([7, 15, 20, 27, 35, 41, 45, 50, 53, 60, 67, 70, 75, 80, 90, 100, 115, 120, 130, 150, 170, 180], dtype=float)
_HEADSAIL_RATIO = np.array([50.0, 55.0, 62.5, 73.0, 80.7, 85.0], dtype=float)
_HEADSAIL_CL = np.array([
    [0.0, 1.05, 1.425, 1.5, 1.505, 1.505, 1.498, 1.465, 1.418, 1.267, 1.1, 1.025, 0.9, 0.785, 0.57, 0.39, 0.205, 0.16, 0.09, 0.0, -0.07, -0.1],
    [0.0, 0.8, 1.21, 1.42, 1.485, 1.51, 1.515, 1.505, 1.49, 1.32, 1.16, 1.09, 0.98, 0.88, 0.69, 0.513, 0.281, 0.224, 0.136, 0.024, -0.024, -0.043],
    [0.0, 0.63, 0.96, 1.315, 1.44, 1.475, 1.48, 1.46, 1.45, 1.4, 1.265, 1.2, 1.09, 0.995, 0.8, 0.608, 0.357, 0.291, 0.176, 0.024, -0.012, -0.022],
    [0.0, 0.35, 0.62, 0.98, 1.15, 1.22, 1.265, 1.3, 1.31, 1.3, 1.27, 1.255, 1.222, 1.14, 0.96, 0.741, 0.451, 0.357, 0.2, 0.032, -0.006, -0.011],
    [-0.194, -0.078, 0.165, 0.601, 0.902, 1.028, 1.082, 1.14, 1.16, 1.185, 1.18, 1.17, 1.15, 1.125, 1.065, 0.92, 0.585, 0.45, 0.244, 0.035, 0.0, 0.0],
    [-0.38, -0.314, -0.247, -0.029, 0.41, 0.698, 0.805, 0.899, 0.945, 1.04, 1.072, 1.076, 1.075, 1.06, 1.03, 0.985, 0.805, 0.66, 0.372, 0.1, 0.02, 0.0],
], dtype=float)
_HEADSAIL_CD = np.array([
    [0.05, 0.032, 0.031, 0.037, 0.09, 0.145, 0.19, 0.245, 0.28, 0.355, 0.43, 0.46, 0.51, 0.56, 0.65, 0.664, 0.656, 0.652, 0.639, 0.614, 0.592, 0.585],
    [0.05, 0.037, 0.035, 0.045, 0.075, 0.105, 0.14, 0.185, 0.215, 0.28, 0.335, 0.36, 0.395, 0.43, 0.5, 0.522, 0.512, 0.507, 0.498, 0.474, 0.451, 0.444],
    [0.05, 0.044, 0.045, 0.06, 0.095, 0.135, 0.165, 0.205, 0.235, 0.295, 0.355, 0.38, 0.415, 0.455, 0.5, 0.522, 0.51, 0.504, 0.486, 0.451, 0.392, 0.37],
    [0.05, 0.05, 0.055, 0.077, 0.13, 0.17, 0.2, 0.24, 0.265, 0.32, 0.385, 0.41, 0.45, 0.475, 0.51, 0.525, 0.52, 0.51, 0.48, 0.422, 0.352, 0.322],
    [0.049, 0.068, 0.087, 0.126, 0.184, 0.223, 0.252, 0.291, 0.315, 0.369, 0.422, 0.441, 0.466, 0.483, 0.512, 0.53, 0.54, 0.527, 0.473, 0.39, 0.318, 0.29],
    [0.048, 0.081, 0.105, 0.147, 0.2, 0.239, 0.267, 0.309, 0.333, 0.389, 0.436, 0.455, 0.477, 0.492, 0.521, 0.545, 0.566, 0.545, 0.475, 0.352, 0.29, 0.262],
], dtype=float)


@dataclass
class _SailComponent:
    force: np.ndarray
    area: float
    a22: float
    a44: float
    awa_deg: float
    factor: float = 1.0


def _rotation_from_eta(eta):
    heel = float(eta[3])
    c = np.cos(heel)
    s = np.sin(heel)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=float)


def _apply_flutter_correction(cl, cd, delta_awa):
    delta_awa = abs(float(delta_awa))
    if delta_awa > 25.0:
        cl = cl * 0.50
        cd = cd * 1.40
    elif delta_awa > 10.0:
        cl = cl * 0.70
        cd = cd * 1.20
    return cl, cd


def _compose_sail_force(aws_body, ceh, area, rho, cl, cd, awa_deg):
    awa_rad = np.deg2rad(awa_deg)
    aws_norm = np.linalg.norm(aws_body)
    aerodynamic = 0.5 * rho * area * aws_norm**2 * np.array([
        -cd * np.cos(awa_rad) + cl * np.sin(awa_rad),
        -cl * np.cos(awa_rad) - cd * np.sin(awa_rad),
        0.0,
    ])
    aerodynamic_moment = np.cross(ceh, aerodynamic)

    sail_mass = 60.0
    gravity_force = np.array([0.0, 0.0, -sail_mass * GRAVITY], dtype=float)
    gravity_moment = np.cross(ceh, gravity_force)

    return np.array([
        aerodynamic[0],
        aerodynamic[1],
        0.0,
        aerodynamic_moment[0] + gravity_moment[0],
        aerodynamic_moment[1] + gravity_moment[1],
        0.0,
    ], dtype=float), aws_norm


def mainsail_force(aws, eta, nu, reef, flat, awa_old):
    mhb = 3.19
    muw = 3.77
    mtw = 4.85
    mhw = 6.54
    mqw = 7.56
    e = 7.636
    p = 26.522

    mhwh = p / 2.0 + (mhw - e / 2.0) / p * e
    mqwh = mhwh / 2.0 + (mqw - (e + mhw) / 2.0) / mhwh * (e - mhw)
    mtwh = (mhwh + p) / 2.0 + (mtw - mhw / 2.0) / (p - mhwh) * mhw
    muwh = (mtwh + p) / 2.0 + (muw - mtw / 2.0) / (p - mtwh) * mtw

    area_m = p / 8.0 * (e + 2.0 * mqw + 2.0 * mhw + 1.5 * mtw + muw + 0.5 * mhb) * reef

    area1 = 0.5 * (mhb + muw) * p / 8.0
    area2 = 0.5 * (muw + mtw) * p / 8.0
    area3 = 0.5 * (muw + mtw) * p / 4.0
    area4 = 0.5 * (mtw + mqw) * p / 4.0
    area5 = 0.5 * (mqw + e) * p / 4.0
    area_sum = area1 + area2 + area3 + area4 + area5

    ceh_z = (
        area1 * (p / 16.0 * 15.0) +
        area2 * (p / 16.0 * 13.0) +
        area3 * (p / 8.0 * 5.0) +
        area4 * (p / 8.0 * 3.0) +
        area5 * (p / 8.0)
    ) / area_sum + 0.024 * p
    ceh_x = (
        area1 * (e / 2.0) +
        area2 * (mqw / 2.0) +
        area3 * (mhw / 2.0) +
        area4 * (mtw / 2.0) +
        area5 * (muw / 2.0)
    ) / area_sum

    ceh = _rotation_from_eta(eta) @ np.array([-ceh_x, 0.0, ceh_z]) * reef
    angular = np.asarray(nu[3:6], dtype=float)
    aws_sail = np.asarray(aws, dtype=float) + np.cross(angular, ceh)
    awa_deg = np.rad2deg(np.arctan2(aws_sail[1], aws_sail[0]))

    factor = -1.0 if awa_deg < 0.0 else 1.0
    awa_lookup = abs(awa_deg)
    cl = clamp_interp(awa_lookup, _AWA_LOOKUP, _MAIN_CL)
    cd = clamp_interp(awa_lookup, _AWA_LOOKUP, _MAIN_CD)
    cl, cd = _apply_flutter_correction(cl, cd, awa_old - awa_lookup)

    cl = factor * reef**2 * flat * cl
    cd = reef**2 * cd

    force, _ = _compose_sail_force(aws_sail, ceh, area_m, AIR_DENSITY, cl, cd, awa_lookup)

    myy_ms_1 = np.pi / 4.0 * AIR_DENSITY * e**2 * mqwh
    myy_ms_2 = np.pi / 4.0 * AIR_DENSITY * mqw**2 * (mhwh - mqwh)
    myy_ms_3 = np.pi / 4.0 * AIR_DENSITY * mhw**2 * (mqwh - mtwh)
    myy_ms_4 = np.pi / 4.0 * AIR_DENSITY * mtw**2 * (mtwh - muwh)
    myy_ms_5 = np.pi / 4.0 * AIR_DENSITY * muw**2 * (muwh - p)
    a22_ms = myy_ms_1 + myy_ms_2 + myy_ms_3 + myy_ms_4 + myy_ms_5
    a44_ms = a22_ms * ceh[2] ** 2

    return _SailComponent(force=force, area=float(area_m), a22=float(a22_ms), a44=float(a44_ms), awa_deg=float(awa_lookup))


def jib_force(aws, eta, nu, reef, flat, c0_enabled):
    hhb = 0.14
    huw = 1.54
    htw = 2.94
    hhw = 5.86
    hqw = 8.91
    hlp = 12.17
    hlu = 24.72

    area_jib = 0.1125 * hlu * (1.445 * hlp + 2.0 * hqw + 2.0 * hhw + 1.5 * htw + huw + 0.5 * hhb) * reef
    ceh = _rotation_from_eta(eta) @ np.array([0.5 * hlp, 0.0, 0.5 * hlu]) * reef
    angular = np.asarray(nu[3:6], dtype=float)
    aws_sail = np.asarray(aws, dtype=float) + np.cross(angular, ceh)
    awa_deg = abs(np.rad2deg(np.arctan2(aws_sail[1], aws_sail[0])))

    cl = reef**2 * flat * clamp_interp(awa_deg, _AWA_LOOKUP, _JIB_CL)
    cd = reef**2 * clamp_interp(awa_deg, _AWA_LOOKUP, _JIB_CD)
    force, _ = _compose_sail_force(aws_sail, ceh, area_jib, AIR_DENSITY, cl, cd, awa_deg)
    if c0_enabled != 0:
        force = np.zeros(6, dtype=float)

    a22_jib = np.pi / 4.0 * AIR_DENSITY * hhw**2 * hlu
    a44_jib = a22_jib * ceh[2] ** 2
    return _SailComponent(force=force, area=float(area_jib), a22=float(a22_jib), a44=float(a44_jib), awa_deg=float(awa_deg))


def flying_headsail_force(aws, eta, nu, flat, c0_enabled, awa_old):
    hhb = 0.11
    huw = 2.80
    htw = 5.49
    hhw = 10.64
    hqw = 15.0
    hlp = 18.21
    hlu = 25.33

    ratio = hhw / hlp * 100.0
    area_c0 = 0.1125 * hlu * (1.445 * hlp + 2.0 * hqw + 2.0 * hhw + 1.5 * htw + huw + 0.5 * hhb)
    ceh = _rotation_from_eta(eta) @ np.array([0.5 * hlp, 0.0, 0.25 * hlu])

    angular = np.asarray(nu[3:6], dtype=float)
    aws_sail = np.asarray(aws, dtype=float) + np.cross(angular, ceh)
    awa_deg = np.rad2deg(np.arctan2(aws_sail[1], aws_sail[0]))
    factor = -1.0 if awa_deg < 0.0 else 1.0
    awa_lookup = max(7.0, abs(awa_deg))

    cl = factor * bilinear_interp(ratio, awa_lookup, _HEADSAIL_RATIO, _HEADSAIL_AWA, _HEADSAIL_CL)
    cd = bilinear_interp(ratio, awa_lookup, _HEADSAIL_RATIO, _HEADSAIL_AWA, _HEADSAIL_CD)
    cl, cd = _apply_flutter_correction(cl, cd, awa_old - awa_lookup)
    cl = flat * cl

    force, _ = _compose_sail_force(aws_sail, ceh, area_c0, AIR_DENSITY, cl, cd, awa_lookup)
    force = force * float(c0_enabled)

    a22_c0 = np.pi / 4.0 * AIR_DENSITY * hhw**2 * hlu
    a44_c0 = a22_c0 * ceh[2] ** 2
    return _SailComponent(force=force, area=float(area_c0), a22=float(a22_c0), a44=float(a44_c0), awa_deg=float(awa_lookup), factor=float(factor))


def spi_force(force):
    return np.asarray(force, dtype=float)


def windage_force(area_c0, area_m, area_jib, aws, reef, twa_deg):
    p = 26.522
    bas = 0.762
    i = 24.486
    isp = 27.23
    zm = 6.0

    tf = 0.16 * zm / p + 0.94
    ehm = max(p * tf + bas, i, isp)
    hbi = 1.393
    zce = hbi + ehm * reef / 2.0
    _ = (area_jib, zce)

    cd_ms_f = 0.4
    cd_ms_s = 0.6
    awa_deg = abs(np.rad2deg(np.arctan2(aws[1], aws[0])))
    aws_norm = np.linalg.norm(aws)
    rho_air = AIR_DENSITY

    def _windage_drag(area, front_reference, side_reference):
        if area <= 0.0:
            return 0.0
        cd = (cd_ms_f * front_reference * np.cos(np.deg2rad(awa_deg)) + cd_ms_s * side_reference * np.sin(np.deg2rad(awa_deg))) / area
        return 0.5 * rho_air * aws_norm**2 * area * cd

    drag_c0 = _windage_drag(area_c0, np.sin(np.deg2rad(40.0)) * area_c0, area_c0)
    drag_main = _windage_drag(area_m, np.sin(np.deg2rad(twa_deg / 2.0)) * area_m, np.cos(np.deg2rad(twa_deg / 2.0)) * area_m)
    return np.array([-(drag_main + drag_c0), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


def added_mass_sails(main, jib, flying, c0_enabled):
    if c0_enabled == 1:
        return float(main.a22 + flying.a22), float(main.a44 + flying.a44)
    return float(main.a22 + jib.a22), float(main.a44 + jib.a44)


class SailModel:
    """
    Port of the full Simulink sail-force subsystem, including 40-sample AWA delays.
    """

    def __init__(self, delay_length=40):
        self.main_awa_delay = DelayLine(delay_length, initial_value=0.0)
        self.headsail_awa_delay = DelayLine(delay_length, initial_value=0.0)

    def reset(self):
        self.main_awa_delay.reset()
        self.headsail_awa_delay.reset()

    def evaluate(self, inputs: SailInputs):
        main = mainsail_force(inputs.aws_body, inputs.eta, inputs.nu, inputs.reef, inputs.flat, self.main_awa_delay.read())
        jib = jib_force(inputs.aws_body, inputs.eta, inputs.nu, inputs.reef, inputs.flat, inputs.c0_enabled)
        flying = flying_headsail_force(inputs.aws_body, inputs.eta, inputs.nu, inputs.flat, inputs.c0_enabled, self.headsail_awa_delay.read())
        windage = windage_force(flying.area, main.area, jib.area, inputs.aws_body, inputs.reef, inputs.twa_deg)
        spi = spi_force(inputs.spi_force)
        total = main.force + jib.force + flying.force + windage + spi
        a22_sails, a44_sails = added_mass_sails(main, jib, flying, inputs.c0_enabled)

        return SailOutputs(
            total_force=total,
            mainsail_force=main.force,
            jib_force=jib.force,
            headsail_force=flying.force,
            spi_force=spi,
            windage_force=windage,
            a22_sails=a22_sails,
            a44_sails=a44_sails,
            area_main=main.area,
            area_jib=jib.area,
            area_headsail=flying.area,
            awa_main_deg=main.awa_deg,
            awa_headsail_deg=flying.awa_deg,
        )

    def commit(self, outputs: SailOutputs):
        self.main_awa_delay.push(outputs.awa_main_deg)
        self.headsail_awa_delay.push(outputs.awa_headsail_deg)
