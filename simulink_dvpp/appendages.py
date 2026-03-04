#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np

from .constants import GRAVITY, SEA_WATER_DENSITY
from .helpers import (
    as_wave_array,
    bounded_section_coefficients,
    segment_intersection_at_plane,
    wave_velocity_world,
)
from .kinematics import body_to_world_rotation
from .rotations import rotx
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


# ── Rudder geometry (IMOCA 60 straight rudder, approximate) ───────────────────
# Located at the stern (~LWL/2 aft of CoG), roughly mid-span of the blade.
# Span and chord are conservative estimates for a high-aspect composite rudder.
_RUDDER_SPAN  = 1.60   # m  (submerged blade)
_RUDDER_CHORD = 0.42   # m  (mean chord, area ≈ 0.67 m²)
# Position of mid-span in body frame [x, y, z]:
#   x ≈ −LWL/2 = −9.25 m  (aft of CoG at midship)
#   z ≈ −_RUDDER_SPAN/2    (mid-span depth below waterline)
_RUDDER_POS = np.array([-9.25, 0.0, -_RUDDER_SPAN / 2.0], dtype=float)


def _rudder_hydrodynamic_force(eta_dot):
    """
    Compute quasi-static rudder sideforce and drag from hull drift angle.

    The rudder is modelled as an unswept lifting surface fixed to the hull
    at zero deflection (VPP assumption: autopilot keeps heading).  It
    operates at the same incidence as the hull drift angle (β = atan2(−v, u)).
    Same NACA section data as the keel but with 3-D AR correction for the
    rudder's lower aspect ratio.

    Returns a (6,) array: [Fx, Fy, Fz, Mx, My, Mz] in body frame.
    """
    eta_dot = np.asarray(eta_dot, dtype=float)
    u = float(eta_dot[0])
    v = float(eta_dot[1])
    V = float(np.hypot(u, v))
    if V < 0.1:
        return np.zeros(6, dtype=float)

    # Drift incidence (positive when boat drifts to port, v < 0)
    alpha_deg = float(np.rad2deg(np.arctan2(-v, u)))

    # 2-D section CL/CD from the keel table (same NACA profile family)
    alpha_clip = float(np.clip(alpha_deg, _KEEL_ALPHA[0], _KEEL_ALPHA[-1]))
    cl_2d = float(np.interp(alpha_clip, _KEEL_ALPHA, _KEEL_CL))
    cd_2d = float(np.interp(abs(alpha_clip), _KEEL_ALPHA, _KEEL_CD))

    # 3-D lift-slope correction for finite aspect ratio
    AR = _RUDDER_SPAN / _RUDDER_CHORD
    slope_2d = float(np.diff(_KEEL_CL[4:-4]).mean() / np.diff(_KEEL_ALPHA[4:-4]).mean())
    cor3d = 1.0 / (1.0 + slope_2d / (np.pi * 0.9 * max(AR, 0.5)))
    cl = cl_2d * cor3d

    # Planform area and dynamic pressure
    A_plan = _RUDDER_SPAN * _RUDDER_CHORD
    q = 0.5 * SEA_WATER_DENSITY * V**2

    # Forces in body frame (upright rudder: lift → y-axis, drag → −x-axis)
    Fy = q * A_plan * cl
    Fx = -q * A_plan * cd_2d * (1.0 if u >= 0.0 else -1.0)

    F = np.array([Fx, Fy, 0.0], dtype=float)
    M = np.cross(_RUDDER_POS, F)
    return np.array([Fx, Fy, 0.0, M[0], M[1], 0.0], dtype=float)


def rudder_force(force):
    return np.asarray(force, dtype=float)


def keel_points():
    return _KEEL_POINTS.copy()


def _keel_points_body(points, keel_angle_deg):
    # Convention: positive keel_angle_deg = windward cant (bulb to port / -y)
    # for a port-tack boat (wind from +y).  Negate so that rotx(-angle) tilts
    # the bulb toward -y (windward) when the user passes a positive value.
    rotation_cant = rotx(-float(keel_angle_deg))
    return np.asarray(points, dtype=float) @ rotation_cant.T


def _intersection_weight(point_a, point_b, plane_z):
    dz = float(point_b[2] - point_a[2])
    if np.isclose(dz, 0.0):
        return 0.0
    return (float(plane_z) - float(point_a[2])) / dz


def _keel_gravity_force_body(points_body, gravity_dir_body, keel_angle_deg):
    points_body = np.asarray(points_body, dtype=float)
    gravity_dir_body = np.asarray(gravity_dir_body, dtype=float)
    chord_length = 0.775
    foil_height = 0.05
    rho_foil = 3776.0

    pts_a = points_body[:-1]
    lengths = np.linalg.norm(np.diff(points_body, axis=0), axis=1)
    strip_masses = lengths * chord_length * rho_foil * foil_height
    strip_forces = strip_masses[:, None] * GRAVITY * gravity_dir_body[None, :]
    strip_moments = np.cross(pts_a, strip_forces)
    foil_gravity = np.concatenate([strip_forces.sum(axis=0), strip_moments.sum(axis=0)])

    bulb_mass = 3500.0
    bulb_arm = 4.1
    bulb_force = bulb_mass * GRAVITY * gravity_dir_body
    ceb = rotx(-float(keel_angle_deg)) @ np.array([0.0, 0.0, -bulb_arm], dtype=float)  # positive = windward
    bulb_moment = np.cross(ceb, bulb_force)
    bulb_moment[2] = 0.0

    return foil_gravity + np.array([bulb_force[0], bulb_force[1], bulb_force[2], bulb_moment[0], bulb_moment[1], bulb_moment[2]], dtype=float)


def keel_gravity_force(points, eta, keel_angle_deg):
    eta = np.asarray(eta, dtype=float)
    points_body = _keel_points_body(points, keel_angle_deg)
    gravity_dir_body = body_to_world_rotation(eta).T @ np.array([0.0, 0.0, -1.0], dtype=float)
    return _keel_gravity_force_body(points_body, gravity_dir_body, keel_angle_deg)


def distributed_keel_force(
    eta_dot,
    eta,
    points,
    keel_angle_deg,
    wave,
    time,
    zero,
    wave_array=None,
    rotation_bw=None,
    points_body=None,
    points_world=None,
):
    eta_dot = np.asarray(eta_dot, dtype=float)
    eta = np.asarray(eta, dtype=float)
    points = np.asarray(points, dtype=float).copy()
    wave_array = as_wave_array(wave) if wave_array is None else np.asarray(wave_array, dtype=float)

    rho_water = SEA_WATER_DENSITY
    chord_length = 0.574
    rotation_bw = body_to_world_rotation(eta) if rotation_bw is None else np.asarray(rotation_bw, dtype=float)

    points_body = _keel_points_body(points, keel_angle_deg) if points_body is None else np.asarray(points_body, dtype=float).copy()
    points_world = (
        points_body @ rotation_bw.T + eta[:3]
        if points_world is None
        else np.asarray(points_world, dtype=float).copy()
    )
    submerged = points_world[:, 2] < zero
    true_indices = np.flatnonzero(submerged)

    if true_indices.size > 0 and points_world[0, 2] > zero and points_world[-1, 2] < zero:
        first = int(true_indices[0])
        if first > 0:
            weight = _intersection_weight(points_world[first - 1, :], points_world[first, :], zero)
            points_body[first - 1, :] = points_body[first - 1, :] + weight * (points_body[first, :] - points_body[first - 1, :])
            points_world[first - 1, :] = segment_intersection_at_plane(points_world[first - 1, :], points_world[first, :], zero)

    seg_vecs = np.diff(points_body, axis=0)                     # (N-1, 3)
    segment_lengths = np.linalg.norm(seg_vecs, axis=1)         # (N-1,)
    aspect_ratio = np.sum(segment_lengths * submerged[:-1]) / chord_length if chord_length > 0.0 else 0.0

    # ── Local frame — keel is a straight line, so constant for all strips ─────
    # (one inv instead of one per segment)
    keel_dir = points_body[-1] - points_body[0]
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
    points_a_body = points_body[:-1]
    points_a_world = points_world[:-1]
    seg_sub = points_a_world[:, 2] < zero
    wave_vel_world = wave_velocity_world(wave_array, points_a_world, time, shared_direction=True)
    wave_vel = wave_vel_world @ rotation_bw

    # ── Relative velocity at each strip ───────────────────────────────────────
    cross_term = np.cross(eta_dot[3:6][np.newaxis, :], points_a_body)   # (N-1, 3)
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

    moment_strips = np.cross(points_a_body, global_fs)              # (N-1, 3)

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
    points_body = _keel_points_body(points, inputs.keel_angle_deg)
    rotation_bw = body_to_world_rotation(inputs.eta)
    translation = np.asarray(inputs.eta[:3], dtype=float)
    points_world = points_body @ rotation_bw.T + translation
    gravity_dir_body = rotation_bw.T @ np.array([0.0, 0.0, -1.0], dtype=float)
    wave_array = as_wave_array(inputs.wave)
    keel_hydrodynamics = distributed_keel_force(
        eta_dot=inputs.eta_dot,
        eta=inputs.eta,
        points=points,
        keel_angle_deg=inputs.keel_angle_deg,
        wave=inputs.wave,
        time=inputs.time,
        zero=inputs.zero,
        wave_array=wave_array,
        rotation_bw=rotation_bw,
        points_body=points_body,
        points_world=points_world,
    )
    keel_gravity = _keel_gravity_force_body(points_body, gravity_dir_body, inputs.keel_angle_deg)
    keel_total = keel_hydrodynamics.force + keel_gravity
    rudder = rudder_force(inputs.rudder_force) + _rudder_hydrodynamic_force(inputs.eta_dot)
    return AppendageOutputs(
        total_force=rudder + keel_total,
        keel_force=keel_total,
        rudder_force=rudder,
        keel_gravity_force=keel_gravity,
        keel_hydrodynamic_force=keel_hydrodynamics.force,
        keel_points=points,
        keel_points_body=points_body,
        keel_points_world=points_world,
        relative_speed=keel_hydrodynamics.relative_speed,
        relative_angle_deg=keel_hydrodynamics.relative_angle,
        relative_velocity_local=keel_hydrodynamics.local_velocity,
        segment_length=keel_hydrodynamics.distance,
    )
