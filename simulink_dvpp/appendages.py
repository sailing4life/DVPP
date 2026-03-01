#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np

from .constants import GRAVITY, SEA_WATER_DENSITY
from .helpers import (
    as_wave_array,
    bounded_section_coefficients,
    segment_intersection_at_plane,
    wave_velocity_components,
)
from .rotations import rotx, roty
from .types import AppendageInputs, AppendageOutputs


_KEEL_ALPHA = np.array([
    -40, -39, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -20,
    -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
], dtype=float)
_KEEL_CL = np.array([
    -1.1722, -1.1591, -1.1438, -1.1261, -1.107, -1.0866, -1.0675, -1.0462, -1.0205, -0.9948, -0.9723, -0.9502, -0.919, -0.8873,
    -0.8689, -0.8389, -0.801, -0.7846, -0.7516, -1.0985, -1.155, -1.1874, -1.1994, -1.1975, -1.182, -1.1652, -1.1316, -1.0915,
    -1.0541, -1.0378, -1.0246, -0.9803, -0.831, -0.681, -0.5324, -0.418, -0.3125, -0.2085, -0.1042, 0.0,
    0.1042, 0.2085, 0.3126, 0.418, 0.5324, 0.6811, 0.8312, 0.9806, 1.0245, 1.0381, 1.0549, 1.0928, 1.1332, 1.1676,
    1.1851, 1.2011, 1.2036, 1.1919, 1.1602, 1.1053, 0.787, 0.8037, 0.8418, 0.8711, 0.8898, 0.9215, 0.9521, 0.9739,
    0.9966, 1.022, 1.0476, 1.068, 1.0873, 1.1074, 1.1262, 1.1435, 1.1581, 1.1712,
], dtype=float)
_KEEL_CD = np.array([
    0.47313, 0.4621, 0.44509, 0.43537, 0.42539, 0.41304, 0.40008, 0.38269, 0.37305, 0.3621, 0.34955, 0.33645, 0.32065, 0.31082,
    0.29801, 0.28131, 0.27242, 0.25907, 0.24408, 0.11921, 0.09574, 0.07752, 0.06391, 0.05253, 0.04355, 0.03568, 0.02992, 0.02562,
    0.02214, 0.01998, 0.0186, 0.01711, 0.01557, 0.0141, 0.01283, 0.01171, 0.01085, 0.01029, 0.01001, 0.00993,
    0.01001, 0.01029, 0.01085, 0.0117, 0.01283, 0.0141, 0.01557, 0.0171, 0.01859, 0.01997, 0.02213, 0.02559, 0.0299, 0.03562,
    0.04348, 0.05243, 0.06378, 0.0774, 0.09561, 0.11887, 0.26127, 0.27522, 0.28448, 0.30184, 0.31559, 0.32603, 0.34263, 0.3568,
    0.37067, 0.38264, 0.39333, 0.41212, 0.42713, 0.44093, 0.45239, 0.46371, 0.48249, 0.49616,
], dtype=float)
_KEEL_POINTS = np.column_stack([
    np.full(151, -0.037595, dtype=float),
    np.zeros(151, dtype=float),
    np.linspace(-0.260201, -4.101993, 151, dtype=float) + 0.406,
])


@dataclass
class _KeelHydroResult:
    force: np.ndarray
    relative_speed: float
    relative_angle: float
    local_velocity: np.ndarray
    distance: float


def rudder_force(force):
    return np.asarray(force, dtype=float)


def keel_points():
    return _KEEL_POINTS.copy()


def keel_gravity_force(points, eta, keel_angle_deg):
    points = np.asarray(points, dtype=float)
    eta = np.asarray(eta, dtype=float)

    chord_length = 0.775
    foil_height = 0.05
    rho_foil = 3776.0
    rotation_x = rotx(-np.rad2deg(-eta[3]) - keel_angle_deg)

    rot_pts = (rotation_x @ points.T).T                         # (N, 3)
    rot_pts[:, 2] += eta[2]
    pts_a = rot_pts[:-1]                                         # (N-1, 3)
    lengths = np.linalg.norm(np.diff(rot_pts, axis=0), axis=1)  # (N-1,)
    strip_fz = -lengths * chord_length * rho_foil * foil_height * GRAVITY  # (N-1,)
    # cross([ax, ay, az], [0, 0, fz]) = [ay*fz, -ax*fz, 0]
    mx = pts_a[:, 1] * strip_fz
    my = -pts_a[:, 0] * strip_fz
    foil_gravity = np.array([0.0, 0.0, strip_fz.sum(), mx.sum(), my.sum(), 0.0], dtype=float)

    bulb_mass = 3500.0
    bulb_arm = 4.1
    bulb_force = np.array([0.0, 0.0, -bulb_mass * GRAVITY], dtype=float)
    # keel_rotation = roll + keel_angle; bulb centre of gravity in body frame.
    # z-component was using -eta[3] (wrong sign), corrected to +eta[3] (Fossen sign conv.).
    keel_rotation = eta[3] + np.deg2rad(keel_angle_deg)
    ceb = np.array([
        0.0,
        np.sin(keel_rotation) * bulb_arm,
        -np.cos(keel_rotation) * bulb_arm,
    ], dtype=float)
    bulb_moment = np.cross(ceb, bulb_force)
    bulb_moment[2] = 0.0

    return foil_gravity + np.array([bulb_force[0], bulb_force[1], bulb_force[2], bulb_moment[0], bulb_moment[1], bulb_moment[2]], dtype=float)


def distributed_keel_force(eta_dot, eta, points, keel_angle_deg, wave, time, zero):
    eta_dot = np.asarray(eta_dot, dtype=float)
    eta = np.asarray(eta, dtype=float)
    points = np.asarray(points, dtype=float).copy()
    wave_array = as_wave_array(wave)

    rho_water = SEA_WATER_DENSITY
    chord_length = 0.574
    rotation_x = rotx(-np.rad2deg(-eta[3]) - keel_angle_deg)
    rotation_y = roty(-np.rad2deg(eta[4]))

    points = points @ rotation_y @ rotation_x
    points = np.column_stack([points[:, 0], points[:, 1], points[:, 2] + eta[2]])
    submerged = points[:, 2] < zero
    true_indices = np.flatnonzero(submerged)

    seg_vecs = np.diff(points, axis=0)                          # (N-1, 3)
    segment_lengths = np.linalg.norm(seg_vecs, axis=1)         # (N-1,)
    aspect_ratio = np.sum(segment_lengths * submerged[:-1]) / chord_length if chord_length > 0.0 else 0.0

    if true_indices.size > 0 and points[0, 2] > zero and points[-1, 2] < zero:
        first = int(true_indices[0])
        if first > 0:
            points[first - 1, :] = segment_intersection_at_plane(points[first - 1, :], points[first, :], zero)

    # ── Local frame — keel is a straight line, so constant for all strips ─────
    # (one inv instead of one per segment)
    keel_dir = points[-1] - points[0]
    y_ax = keel_dir / np.linalg.norm(keel_dir)
    x_ax = np.array([1.0, 0.0, 0.0], dtype=float)
    z_ax = np.cross(x_ax, y_ax)
    z_ax = z_ax / np.linalg.norm(z_ax)
    local_frame = np.vstack([x_ax, y_ax, z_ax])        # (3, 3)
    local_frame_inv = np.linalg.inv(local_frame)

    # ── Slope correction — constant for all strips (hoisted from loop) ────────
    slope_cl = np.diff(_KEEL_CL[4:-4]).mean() / np.diff(_KEEL_ALPHA[4:-4]).mean() * (180.0 / np.pi)
    if aspect_ratio <= 4.0:
        corrected_slope = slope_cl / np.sqrt(1.0 + (slope_cl / (np.pi * 0.9 * max(aspect_ratio, 1e-9))) ** 2 + (slope_cl / (np.pi * 0.9 * max(aspect_ratio, 1e-9))))
    else:
        corrected_slope = slope_cl / (1.0 + slope_cl / (np.pi * 0.9 * aspect_ratio))
    cor3d = corrected_slope / slope_cl if abs(slope_cl) > 1e-12 else 1.0

    # ── Vectorised wave velocity at all segment start-points ──────────────────
    points_a = points[:-1]                              # (N-1, 3)
    seg_sub = points_a[:, 2] < zero                    # submerged mask, (N-1,)
    x_pts = points_a[:, 0] + eta[0]
    y_pts = points_a[:, 1] + eta[1]
    z_pts = points_a[:, 2]

    wave_vel = np.zeros((len(points_a), 3), dtype=float)
    for w_row in wave_array:
        zeta_a, k, nu_w, omega_w = w_row[0], w_row[1], w_row[2], w_row[3]
        if omega_w == 0.0:
            continue
        phase = -omega_w * time + k * x_pts * np.cos(nu_w) + k * y_pts * np.sin(nu_w)
        exp_t = np.exp(k * z_pts)
        wave_vel[:, 0] += zeta_a * omega_w * np.cos(nu_w) * exp_t * np.cos(phase)
        wave_vel[:, 1] += zeta_a * omega_w * np.sin(nu_w) * exp_t * np.cos(phase)
        wave_vel[:, 2] += zeta_a * omega_w * exp_t * np.sin(phase)

    # ── Relative velocity at each strip ───────────────────────────────────────
    cross_term = np.cross(eta_dot[3:6][np.newaxis, :], points_a)   # (N-1, 3)
    rel_vel = eta_dot[:3][np.newaxis, :] + cross_term - wave_vel    # (N-1, 3)
    local_vel = (local_frame @ rel_vel.T).T                         # (N-1, 3)

    rel_speed = np.hypot(local_vel[:, 0], local_vel[:, 2])          # (N-1,)
    rel_angle = np.rad2deg(np.arctan2(-local_vel[:, 2], local_vel[:, 0]))  # (N-1,)

    # ── Vectorised Cl/Cd (interp inside range, flat-plate outside) ────────────
    in_rng = (rel_angle >= _KEEL_ALPHA[0]) & (rel_angle <= _KEEL_ALPHA[-1])
    alpha_c = np.clip(rel_angle, _KEEL_ALPHA[0], _KEEL_ALPHA[-1])
    cl = np.where(in_rng, np.interp(alpha_c, _KEEL_ALPHA, _KEEL_CL) * cor3d,
                  np.sin(2.0 * np.deg2rad(rel_angle)))
    cd = np.where(in_rng, np.minimum(0.18, np.interp(alpha_c, _KEEL_ALPHA, _KEEL_CD)),
                  2.0 * np.sin(np.deg2rad(rel_angle)) ** 2)

    # ── Strip forces ──────────────────────────────────────────────────────────
    seg_len = segment_lengths                                        # (N-1,)
    dyn = 0.5 * rho_water * chord_length * rel_speed**2 * seg_len  # (N-1,)
    local_fs = np.column_stack([-dyn * cd, np.zeros(len(dyn)), dyn * cl])  # (N-1, 3)
    global_fs = (local_frame_inv @ local_fs.T).T                    # (N-1, 3)
    global_fs[~seg_sub] = 0.0

    # Moment lever arm in body frame (subtract eta[2] added for waterplane test)
    lever = points_a.copy()
    lever[:, 2] -= eta[2]
    moment_strips = np.cross(lever, global_fs)                      # (N-1, 3)

    force_sum = global_fs.sum(axis=0)
    moment_sum = moment_strips.sum(axis=0)

    # ── Diagnostics (last submerged strip, matching original convention) ───────
    sub_idx = np.flatnonzero(seg_sub)
    if sub_idx.size > 0:
        li = sub_idx[-1]
        _d = float(seg_len[li])
        _rs = float(rel_speed[li])
        _ra = float(rel_angle[li])
        _lv = local_vel[li].copy()
    else:
        _d, _rs, _ra, _lv = 0.0, 0.0, 0.0, np.zeros(3, dtype=float)

    return _KeelHydroResult(
        force=np.array([force_sum[0], force_sum[1], force_sum[2], moment_sum[0], moment_sum[1], 0.0], dtype=float),
        relative_speed=_rs,
        relative_angle=_ra,
        local_velocity=_lv,
        distance=_d,
    )


def appendage_forces(inputs: AppendageInputs):
    points = keel_points()
    keel_hydrodynamics = distributed_keel_force(
        eta_dot=inputs.eta_dot,
        eta=inputs.eta,
        points=points,
        keel_angle_deg=inputs.keel_angle_deg,
        wave=inputs.wave,
        time=inputs.time,
        zero=inputs.zero,
    )
    keel_gravity = keel_gravity_force(points, inputs.eta, inputs.keel_angle_deg)
    keel_total = keel_hydrodynamics.force + keel_gravity
    rudder = rudder_force(inputs.rudder_force)
    return AppendageOutputs(
        total_force=rudder + keel_total,
        keel_force=keel_total,
        rudder_force=rudder,
        keel_gravity_force=keel_gravity,
        keel_hydrodynamic_force=keel_hydrodynamics.force,
        keel_points=points,
        relative_speed=keel_hydrodynamics.relative_speed,
        relative_angle_deg=keel_hydrodynamics.relative_angle,
        relative_velocity_local=keel_hydrodynamics.local_velocity,
        segment_length=keel_hydrodynamics.distance,
    )
