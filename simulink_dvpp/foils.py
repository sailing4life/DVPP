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
_FOIL_POINTS_ONSIDE[:, 2] = _FOIL_POINTS_ONSIDE[:, 2] + 0.406


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


def foil_gravity_force(points, eta, chord_length, rho_foil=1750.0):
    points = np.asarray(points, dtype=float)
    eta = np.asarray(eta, dtype=float)
    foil_height = 0.05
    rotation_x = rotx(-np.rad2deg(eta[3]))
    rotation_y = roty(-np.rad2deg(eta[4]))

    total_force = np.zeros((points.shape[0], 3), dtype=float)
    total_moment = np.zeros((points.shape[0], 3), dtype=float)
    for index in range(points.shape[0] - 1):
        point_a = points[index, :] @ rotation_x @ rotation_y + np.array([0.0, 0.0, eta[2]])
        point_b = points[index + 1, :] @ rotation_x @ rotation_y + np.array([0.0, 0.0, eta[2]])
        length_vector = np.linalg.norm(point_a - point_b)
        weight = length_vector * chord_length * rho_foil * foil_height
        force_z = -weight * GRAVITY
        total_force[index, :] = np.array([0.0, 0.0, force_z], dtype=float)
        total_moment[index, :] = np.cross(point_a, np.array([0.0, 0.0, force_z]))

    force_sum = total_force.sum(axis=0)
    moment_sum = total_moment.sum(axis=0)
    return np.array([force_sum[0], force_sum[1], force_sum[2], moment_sum[0], moment_sum[1], 0.0], dtype=float)


def _foil_lift_slope_correction(aspect_ratio):
    slope_cl = np.diff(_FOIL_CL[4:-4]).mean() / np.diff(_FOIL_ALPHA[4:-4]).mean() * (180.0 / np.pi)
    if aspect_ratio <= 4.0:
        ar_term = slope_cl / (np.pi * 0.9 * max(aspect_ratio, 1e-9))
        corrected_slope = slope_cl / np.sqrt(1.0 + ar_term ** 2 + ar_term)
    else:
        corrected_slope = slope_cl / (1.0 + slope_cl / (np.pi * 0.9 * aspect_ratio))
    return corrected_slope / slope_cl


def hydrofoil_force(eta_dot, eta, points, rake_foil_deg, chord_length, wave, time, zero, reverse_segment, free_surface_factor):
    eta_dot = np.asarray(eta_dot, dtype=float)
    eta = np.asarray(eta, dtype=float)
    points = np.asarray(points, dtype=float).copy()
    wave_array = as_wave_array(wave)

    rho_water = SEA_WATER_DENSITY
    rotation_x = rotx(-np.rad2deg(eta[3]))
    rotation_y = roty(-np.rad2deg(eta[4]))

    points = points @ rotation_y @ rotation_x
    points = np.column_stack([points[:, 0] + chord_length / 2.0, points[:, 1], points[:, 2] + eta[2]])
    submerged = points[:, 2] < zero
    true_indices = np.flatnonzero(submerged)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    aspect_ratio = np.sum(segment_lengths * submerged[:-1]) / chord_length if chord_length > 0.0 else 0.0

    if true_indices.size > 0 and points[0, 2] > zero and true_indices[0] > 0:
        first = int(true_indices[0])
        points[first - 1, :] = segment_intersection_at_plane(points[first - 1, :], points[first, :], zero)
        submerged[first - 1] = True
    if true_indices.size > 0 and points[-1, 2] > zero and true_indices[-1] < points.shape[0] - 1:
        last = int(true_indices[-1])
        points[last + 1, :] = segment_intersection_at_plane(points[last, :], points[last + 1, :], zero)
        submerged[last + 1] = False

    total_force = np.zeros((points.shape[0], 3), dtype=float)
    total_moment = np.zeros((points.shape[0], 3), dtype=float)
    relative_angle = 0.0
    local_velocity = np.zeros(3, dtype=float)
    relative_velocity = np.zeros(3, dtype=float)
    last_cd = 0.0
    sub_length = 0.0
    correction_3d = _foil_lift_slope_correction(aspect_ratio) if aspect_ratio > 0.0 else 1.0

    for index in range(points.shape[0] - 1):
        point_a = points[index, :]
        point_b = points[index + 1, :]
        segment = point_a - point_b if reverse_segment else point_b - point_a
        if np.allclose(segment, 0.0):
            continue

        wave_velocity = wave_velocity_components(wave_array, point_a, eta, time, shared_direction=True)
        relative_velocity = np.array([eta_dot[0], eta_dot[1], eta_dot[2]], dtype=float) + np.cross(
            np.array([eta_dot[3], eta_dot[4], eta_dot[5]], dtype=float),
            point_a,
        ) - wave_velocity

        y_axis = -segment / np.linalg.norm(segment)
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis = np.cross(x_axis, z_axis)
        local_frame = np.column_stack([x_axis, y_axis, z_axis])
        local_velocity = local_frame @ relative_velocity

        segment_length = float(np.linalg.norm(segment))
        relative_angle = float(np.rad2deg(np.arctan2(-local_velocity[2], local_velocity[0]) - eta[4]) + rake_foil_deg)
        if submerged[index]:
            cl, cd = bounded_section_coefficients(relative_angle, _FOIL_ALPHA, _FOIL_CL, _FOIL_CD)
            if _FOIL_ALPHA[0] <= relative_angle <= _FOIL_ALPHA[-1]:
                cl = cl * correction_3d
                cd = min(0.15, cd)
            depth = abs(point_a[2] - zero)
            cl_fs = cl * ((1.0 + 16.0 * (depth / chord_length) ** 2) / (2.0 + 16.0 * (depth / chord_length) ** 2)) * free_surface_factor
            relative_speed = float(np.hypot(local_velocity[0], local_velocity[2]))

            dynamic_strip = 0.5 * rho_water * chord_length * relative_speed**2 * segment_length
            lift_force = dynamic_strip * cl_fs
            drag_force = dynamic_strip * cd
            local_force = np.array([-drag_force, 0.0, lift_force], dtype=float)
            global_force = np.linalg.inv(local_frame) @ local_force
            shifted_point = np.array([point_a[0], point_a[1], point_a[2] - eta[2]], dtype=float)
            total_force[index, :] = global_force
            total_moment[index, :] = np.cross(shifted_point, global_force)
            last_cd = cd
            sub_length += segment_length

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

    gravity_on = foil_gravity_force(points_on, inputs.eta, inputs.chord_length)
    gravity_off = foil_gravity_force(points_off, inputs.eta, inputs.chord_length)
    hydro_on = hydrofoil_force(inputs.eta_dot, inputs.eta, points_on, inputs.rake_foil_deg, inputs.chord_length, inputs.wave, inputs.time, inputs.zero, reverse_segment=False, free_surface_factor=0.75)
    hydro_off = hydrofoil_force(inputs.eta_dot, inputs.eta, points_off, inputs.rake_foil_deg, inputs.chord_length, inputs.wave, inputs.time, inputs.zero, reverse_segment=True, free_surface_factor=1.0)

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
    )
