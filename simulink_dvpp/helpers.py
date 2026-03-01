#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque

import numpy as np


class DelayLine:
    """
    Fixed-length delay line matching the Simulink Delay block behavior.
    """

    def __init__(self, length, initial_value=0.0):
        self.length = int(length)
        self.initial_value = float(initial_value)
        self.buffer = deque([self.initial_value] * self.length, maxlen=self.length)

    def read(self):
        return float(self.buffer[0])

    def push(self, value):
        self.buffer.append(float(value))

    def reset(self):
        self.buffer = deque([self.initial_value] * self.length, maxlen=self.length)


def as_wave_array(wave):
    wave_array = np.asarray(wave, dtype=float)
    if wave_array.size == 0:
        return np.zeros((0, 4), dtype=float)
    if wave_array.ndim == 1:
        return wave_array.reshape(1, -1)
    return wave_array


def clamp_interp(x, xp, fp):
    return float(np.interp(float(x), np.asarray(xp, dtype=float), np.asarray(fp, dtype=float)))


def extrap_interp(x, xp, fp):
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    x = float(x)

    if x <= xp[0]:
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return float(fp[0] + slope * (x - xp[0]))
    if x >= xp[-1]:
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return float(fp[-1] + slope * (x - xp[-1]))
    return float(np.interp(x, xp, fp))


def flat_plate_coefficients(alpha_deg):
    """
    High-incidence fallback for lifting surfaces.

    For angles outside the tabulated pre-/post-stall range, use the classical
    flat-plate approximation:

        C_L = sin(2 alpha)
        C_D = 2 sin^2(alpha)

    This keeps lift bounded and drag positive at large incidence, unlike
    linear extrapolation of foil polars.
    """
    alpha = np.deg2rad(float(alpha_deg))
    return float(np.sin(2.0 * alpha)), float(2.0 * np.sin(alpha) ** 2)


def bounded_section_coefficients(alpha_deg, alpha_table, cl_table, cd_table):
    """
    Return section coefficients without unbounded linear extrapolation.

    The measured polars are used inside their tabulated incidence range.
    Outside that range, the model switches to a flat-plate approximation,
    which is qualitatively correct for deep stall and flow reversal.
    """
    alpha_deg = float(alpha_deg)
    alpha_table = np.asarray(alpha_table, dtype=float)
    cl_table = np.asarray(cl_table, dtype=float)
    cd_table = np.asarray(cd_table, dtype=float)

    if alpha_table[0] <= alpha_deg <= alpha_table[-1]:
        cl = np.interp(alpha_deg, alpha_table, cl_table)
        cd = np.interp(alpha_deg, alpha_table, cd_table)
        return float(cl), float(cd)

    return flat_plate_coefficients(alpha_deg)


def bilinear_interp(x, y, x_grid, y_grid, values):
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    values = np.asarray(values, dtype=float)

    x = float(np.clip(x, x_grid[0], x_grid[-1]))
    y = float(np.clip(y, y_grid[0], y_grid[-1]))

    ix = int(np.clip(np.searchsorted(x_grid, x, side="right") - 1, 0, len(x_grid) - 2))
    iy = int(np.clip(np.searchsorted(y_grid, y, side="right") - 1, 0, len(y_grid) - 2))

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    y0 = y_grid[iy]
    y1 = y_grid[iy + 1]

    q11 = values[ix, iy]
    q21 = values[ix + 1, iy]
    q12 = values[ix, iy + 1]
    q22 = values[ix + 1, iy + 1]

    tx = 0.0 if np.isclose(x1, x0) else (x - x0) / (x1 - x0)
    ty = 0.0 if np.isclose(y1, y0) else (y - y0) / (y1 - y0)

    return float(
        (1.0 - tx) * (1.0 - ty) * q11 +
        tx * (1.0 - ty) * q21 +
        (1.0 - tx) * ty * q12 +
        tx * ty * q22
    )


def segment_intersection_at_plane(point_a, point_b, plane_z):
    point_a = np.asarray(point_a, dtype=float)
    point_b = np.asarray(point_b, dtype=float)
    dz = point_b[2] - point_a[2]
    if np.isclose(dz, 0.0):
        return point_a.copy()
    weight = (plane_z - point_a[2]) / dz
    return point_a + weight * (point_b - point_a)


def wave_velocity_components(wave, point, eta, time, shared_direction=True):
    wave_array = as_wave_array(wave)
    point = np.asarray(point, dtype=float)
    eta = np.asarray(eta, dtype=float)

    if wave_array.shape[0] == 0:
        return np.zeros(3, dtype=float)

    x = point[0] + eta[0]
    y = point[1] + eta[1]
    z = point[2]

    common_nu = wave_array[0, 2]
    u_w = 0.0
    v_w = 0.0
    w_w = 0.0
    for row in wave_array:
        zeta_a = row[0]
        k = row[1]
        nu = common_nu if shared_direction else row[2]
        omega = row[3]
        phase = -omega * time + k * x * np.cos(nu) + k * y * np.sin(nu)
        exp_term = np.exp(k * z)
        u_w += zeta_a * omega * np.cos(nu) * exp_term * np.cos(phase)
        v_w += zeta_a * omega * np.sin(nu) * exp_term * np.cos(phase)
        w_w += zeta_a * omega * exp_term * np.sin(phase)
    return np.array([u_w, v_w, w_w], dtype=float)
