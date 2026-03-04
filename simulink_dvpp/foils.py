#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np

from .constants import GRAVITY, SEA_WATER_DENSITY
from .helpers import (
    as_wave_array,
    segment_intersection_at_plane,
    wave_velocity_world,
)
from .kinematics import body_to_world_rotation
from .types import FoilInputs, FoilOutputs


_FOIL_ALPHA = np.array([-20, -19, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], dtype=float)
_FOIL_CL = np.array([-0.7114, -0.667, -1.2092, -1.2157, -1.2463, -1.1856, -1.1033, -1.0087, -0.9061, -0.7992, -0.689, -0.5771, -0.4638, -0.3492, -0.2343, -0.1185, -0.0027, 0.1131, 0.3289, 0.4417, 0.556, 0.7782, 0.8866, 0.9923, 1.0958, 1.1951, 1.2898, 1.3773, 1.4531, 1.4855, 1.5154, 1.5422, 1.5643, 1.5835, 1.5964, 1.6035, 1.6036, 1.5805, 1.5595, 1.5268, 1.454, 1.3366, 1.1621], dtype=float)
_FOIL_CD = np.array([0.13106, 0.12534, 0.02603, 0.02063, 0.01721, 0.01507, 0.01335, 0.01194, 0.01069, 0.00968, 0.00881, 0.00809, 0.00749, 0.00696, 0.00662, 0.0062, 0.00593, 0.00572, 0.0079, 0.00813, 0.00738, 0.00794, 0.00851, 0.00912, 0.00979, 0.01067, 0.01162, 0.01268, 0.01381, 0.01503, 0.01677, 0.01945, 0.02313, 0.02771, 0.03366, 0.04111, 0.05034, 0.06406, 0.07943, 0.09846, 0.12723, 0.16703, 0.22301], dtype=float)
_FOIL_POINTS_ONSIDE = np.array([
    [1.609166, -2.635894, 0.229317],
    [1.609166, -2.825421, -0.008197],
    [1.609166, -2.9912, -0.263066],
    [1.609166, -3.201826, -0.479354],
    [1.609166, -3.473096, -0.613605],
    [1.609166, -3.774939, -0.63975],
    [1.609166, -4.077546, -0.611765],
    [1.609166, -4.377201, -0.560442],
    [1.609166, -4.672448, -0.488149],
    [1.609166, -4.958453, -0.385428],
    [1.609166, -5.231801, -0.252513],
    [1.609166, -5.493685, -0.098135],
    [1.609166, -5.743104, 0.075608],
    [1.609166, -5.970961, 0.276575],
    [1.609166, -6.173788, 0.502992],
    [1.609166, -6.365905, 0.738656],
    [1.609166, -6.547939, 0.982161],
    [1.609166, -6.70828, 1.240248],
    [1.609166, -6.825013, 1.520682],
    [1.609166, -6.911797, 1.812046],
], dtype=float)
_FOIL_POINTS_ONSIDE[:, 2] += 0.406


@dataclass
class _FoilHydroResult:
    force: np.ndarray
    local_velocity: np.ndarray
    relative_angle: float
    relative_velocity: np.ndarray
    cd: float
    sub_length: float


def foil_points_onside():
    return _FOIL_POINTS_ONSIDE.copy()


def foil_points_offside():
    points = _FOIL_POINTS_ONSIDE.copy()
    points[:, 1] = -points[:, 1]
    return points


def _foil_points_body(points, chord_length):
    points_body = np.asarray(points, dtype=float).copy()
    points_body[:, 0] += float(chord_length) / 2.0
    return points_body


def _intersection_weight(point_a, point_b, plane_z):
    dz = float(point_b[2] - point_a[2])
    if np.isclose(dz, 0.0):
        return 0.0
    return (float(plane_z) - float(point_a[2])) / dz


def _foil_gravity_force_body(points_body, gravity_dir_body, chord_length, rho_foil=1750.0):
    points_body = np.asarray(points_body, dtype=float)
    gravity_dir_body = np.asarray(gravity_dir_body, dtype=float)
    foil_height = 0.05

    points_a_body = points_body[:-1]
    strip_lengths = np.linalg.norm(np.diff(points_body, axis=0), axis=1)
    strip_masses = strip_lengths * chord_length * rho_foil * foil_height
    strip_forces = strip_masses[:, None] * GRAVITY * gravity_dir_body[None, :]
    strip_moments = np.cross(points_a_body, strip_forces)

    force_sum = strip_forces.sum(axis=0)
    moment_sum = strip_moments.sum(axis=0)
    return np.array([force_sum[0], force_sum[1], force_sum[2], moment_sum[0], moment_sum[1], 0.0], dtype=float)


def foil_gravity_force(points, eta, chord_length, rho_foil=1750.0):
    eta = np.asarray(eta, dtype=float)
    points_body = _foil_points_body(points, chord_length)
    gravity_dir_body = body_to_world_rotation(eta).T @ np.array([0.0, 0.0, -1.0], dtype=float)
    return _foil_gravity_force_body(points_body, gravity_dir_body, chord_length, rho_foil=rho_foil)


def _foil_lift_slope_correction(aspect_ratio):
    slope_cl = np.diff(_FOIL_CL[4:-4]).mean() / np.diff(_FOIL_ALPHA[4:-4]).mean() * (180.0 / np.pi)
    if aspect_ratio <= 4.0:
        ar_term = slope_cl / (np.pi * 0.9 * max(aspect_ratio, 1e-9))
        corrected_slope = slope_cl / np.sqrt(1.0 + ar_term ** 2 + ar_term)
    else:
        corrected_slope = slope_cl / (1.0 + slope_cl / (np.pi * 0.9 * aspect_ratio))
    return corrected_slope / slope_cl


def hydrofoil_force(
    eta_dot,
    eta,
    points,
    rake_foil_deg,
    chord_length,
    wave,
    time,
    zero,
    reverse_segment,
    free_surface_factor,
    wave_array=None,
    rotation_bw=None,
    points_body=None,
    points_world=None,
):
    eta_dot = np.asarray(eta_dot, dtype=float)
    eta = np.asarray(eta, dtype=float)
    wave_array = as_wave_array(wave) if wave_array is None else np.asarray(wave_array, dtype=float)

    rho_water = SEA_WATER_DENSITY
    rotation_bw = body_to_world_rotation(eta) if rotation_bw is None else np.asarray(rotation_bw, dtype=float)
    points_body = _foil_points_body(points, chord_length) if points_body is None else np.asarray(points_body, dtype=float).copy()
    points_world = (
        points_body @ rotation_bw.T + eta[:3]
        if points_world is None
        else np.asarray(points_world, dtype=float).copy()
    )
    submerged = points_world[:, 2] < zero
    true_indices = np.flatnonzero(submerged)

    if true_indices.size > 0 and points_world[0, 2] > zero and true_indices[0] > 0:
        first = int(true_indices[0])
        weight = _intersection_weight(points_world[first - 1, :], points_world[first, :], zero)
        points_body[first - 1, :] = points_body[first - 1, :] + weight * (points_body[first, :] - points_body[first - 1, :])
        points_world[first - 1, :] = segment_intersection_at_plane(points_world[first - 1, :], points_world[first, :], zero)
        submerged[first - 1] = True
    if true_indices.size > 0 and points_world[-1, 2] > zero and true_indices[-1] < points_world.shape[0] - 1:
        last = int(true_indices[-1])
        weight = _intersection_weight(points_world[last, :], points_world[last + 1, :], zero)
        points_body[last + 1, :] = points_body[last, :] + weight * (points_body[last + 1, :] - points_body[last, :])
        points_world[last + 1, :] = segment_intersection_at_plane(points_world[last, :], points_world[last + 1, :], zero)
        submerged[last + 1] = False

    segment_vectors = np.diff(points_body, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    aspect_ratio = np.sum(segment_lengths * submerged[:-1]) / chord_length if chord_length > 0.0 else 0.0

    total_force = np.zeros((points_body.shape[0] - 1, 3), dtype=float)
    total_moment = np.zeros((points_body.shape[0] - 1, 3), dtype=float)
    relative_angle = 0.0
    local_velocity = np.zeros(3, dtype=float)
    relative_velocity = np.zeros(3, dtype=float)
    last_cd = 0.0
    sub_length = 0.0
    correction_3d = _foil_lift_slope_correction(aspect_ratio) if aspect_ratio > 0.0 else 1.0
    points_a_body = points_body[:-1]
    points_a_world = points_world[:-1]
    wave_velocity = wave_velocity_world(wave_array, points_a_world, time, shared_direction=True) @ rotation_bw
    angular_velocity = eta_dot[3:6]
    translational_velocity = eta_dot[:3]
    relative_velocities = translational_velocity[None, :] + np.cross(angular_velocity[None, :], points_a_body) - wave_velocity
    oriented_segments = -segment_vectors if reverse_segment else segment_vectors
    valid_segments = segment_lengths > 1e-12
    if np.any(valid_segments):
        y_axes = np.zeros_like(oriented_segments)
        y_axes[valid_segments] = -oriented_segments[valid_segments] / segment_lengths[valid_segments, None]
        z_axes = np.cross(np.array([1.0, 0.0, 0.0], dtype=float), y_axes)
        z_norms = np.linalg.norm(z_axes, axis=1)
        valid_segments &= z_norms > 1e-12
        z_axes[valid_segments] = z_axes[valid_segments] / z_norms[valid_segments, None]

        local_u = relative_velocities[:, 0]
        local_w = np.sum(z_axes * relative_velocities, axis=1)
        relative_angles = np.rad2deg(np.arctan2(-local_w, local_u)) + rake_foil_deg

        alpha_min = _FOIL_ALPHA[0]
        alpha_max = _FOIL_ALPHA[-1]
        in_range = (relative_angles >= alpha_min) & (relative_angles <= alpha_max)
        alpha_clipped = np.clip(relative_angles, alpha_min, alpha_max)
        cl = np.where(
            in_range,
            np.interp(alpha_clipped, _FOIL_ALPHA, _FOIL_CL),
            np.sin(2.0 * np.deg2rad(relative_angles)),
        )
        cd = np.where(
            in_range,
            np.interp(alpha_clipped, _FOIL_ALPHA, _FOIL_CD),
            2.0 * np.sin(np.deg2rad(relative_angles)) ** 2,
        )
        cl[in_range] *= correction_3d
        cd[in_range] = np.minimum(0.15, cd[in_range])

        depth_ratio = np.abs(points_a_world[:, 2] - zero) / max(chord_length, 1e-12)
        cl_fs = cl * ((1.0 + 16.0 * depth_ratio**2) / (2.0 + 16.0 * depth_ratio**2)) * free_surface_factor
        relative_speeds = np.hypot(local_u, local_w)
        dynamic_strip = 0.5 * rho_water * chord_length * relative_speeds**2 * segment_lengths
        lift_force = dynamic_strip * cl_fs
        drag_force = dynamic_strip * cd

        active = submerged[:-1] & valid_segments
        total_force[active, 0] = -drag_force[active]
        total_force[active, :] += lift_force[active, None] * z_axes[active, :]
        total_moment[active, :] = np.cross(points_a_body[active, :], total_force[active, :])
        sub_length = float(np.sum(segment_lengths[active]))

        active_indices = np.flatnonzero(active)
        if active_indices.size > 0:
            li = int(active_indices[-1])
            local_velocity = np.array([local_u[li], 0.0, local_w[li]], dtype=float)
            relative_velocity = relative_velocities[li, :].copy()
            relative_angle = float(relative_angles[li])
            last_cd = float(cd[li])

    force_sum = total_force.sum(axis=0)
    moment_sum = total_moment.sum(axis=0)
    return _FoilHydroResult(
        force=np.array([force_sum[0], force_sum[1], force_sum[2], moment_sum[0], moment_sum[1], 0.0], dtype=float),
        local_velocity=local_velocity,
        relative_angle=relative_angle,
        relative_velocity=relative_velocity,
        cd=last_cd,
        sub_length=sub_length,
    )


def foil_added_mass(sub_length, chord):
    return float(sub_length * chord**2 * SEA_WATER_DENSITY * np.pi / 4.0)


def foil_forces(inputs: FoilInputs):
    points_on = foil_points_onside()
    points_off = foil_points_offside()
    # shift from raw STL coordinates to simulator body frame (waterline = z=0)
    points_on[:, 2] += inputs.wl_z_shift
    points_off[:, 2] += inputs.wl_z_shift
    points_on_body = _foil_points_body(points_on, inputs.chord_length)
    points_off_body = _foil_points_body(points_off, inputs.chord_length)
    rotation_bw = body_to_world_rotation(inputs.eta)
    translation = np.asarray(inputs.eta[:3], dtype=float)
    gravity_dir_body = rotation_bw.T @ np.array([0.0, 0.0, -1.0], dtype=float)
    wave_array = as_wave_array(inputs.wave)
    points_on_world = points_on_body @ rotation_bw.T + translation
    points_off_world = points_off_body @ rotation_bw.T + translation

    gravity_on = _foil_gravity_force_body(points_on_body, gravity_dir_body, inputs.chord_length)
    gravity_off = _foil_gravity_force_body(points_off_body, gravity_dir_body, inputs.chord_length)
    hydro_on = hydrofoil_force(
        inputs.eta_dot,
        inputs.eta,
        points_on,
        inputs.rake_foil_deg,
        inputs.chord_length,
        inputs.wave,
        inputs.time,
        inputs.zero,
        reverse_segment=False,
        free_surface_factor=0.75,
        wave_array=wave_array,
        rotation_bw=rotation_bw,
        points_body=points_on_body,
        points_world=points_on_world,
    )
    hydro_off = hydrofoil_force(
        inputs.eta_dot,
        inputs.eta,
        points_off,
        inputs.rake_foil_deg,
        inputs.chord_length,
        inputs.wave,
        inputs.time,
        inputs.zero,
        reverse_segment=True,
        free_surface_factor=1.0,
        wave_array=wave_array,
        rotation_bw=rotation_bw,
        points_body=points_off_body,
        points_world=points_off_world,
    )

    total_gravity = gravity_on + gravity_off
    total_hydrodynamics = hydro_on.force + hydro_off.force
    total_force = total_gravity + total_hydrodynamics
    a_foil = foil_added_mass(hydro_on.sub_length, inputs.chord_length)

    return FoilOutputs(
        total_force=total_force,
        hydrodynamic_force=total_hydrodynamics,
        gravity_force=total_gravity,
        added_mass=a_foil,
        aoa_deg=np.array([hydro_on.relative_angle, hydro_off.relative_angle], dtype=float),
        cd=np.array([hydro_on.cd, hydro_off.cd], dtype=float),
        onside_force=hydro_on.force + gravity_on,
        offside_force=hydro_off.force + gravity_off,
        onside_points=points_on,
        offside_points=points_off,
        onside_points_body=points_on_body,
        offside_points_body=points_off_body,
        onside_points_world=points_on_world,
        offside_points_world=points_off_world,
    )
