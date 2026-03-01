#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .constants import GRAVITY, KNOTS_PER_MPS


def wave_properties(ramp, wind_speed, eta, eta_dot, angle_deg, time, irregular):
    """
    Port of Simulink chart_37 `wave_properties`.

    Returns
    -------
    wave : ndarray, shape (10, 4)
        Columns [zeta_a, k, nu, omega].
    w_zero : float
        Wave elevation at the vessel origin.
    omega_out : float
        The single regular-wave omega when applicable, else 0.
    H_s : float
        Significant wave height.
    omega_e : ndarray, shape (10,)
        Encounter frequencies.
    """
    V = float(eta_dot[0])
    g = GRAVITY

    omega_e = np.zeros(10)
    wave = np.zeros((10, 4))
    nu = np.deg2rad(angle_deg)
    H_s = 0.0
    T_s = 0.0
    omega_out = 0.0

    if irregular == 1:
        w_ar = np.array([
            [10.46, 1.69, 171.1, 16.34],
            [6.92, 0.89, 74.8, 10.8],
            [4.15, 0.19, 26.9, 6.48],
        ], dtype=float)
        w_z = 0.0

        for i, data in enumerate(w_ar):
            omega = 2.0 * np.pi / data[0]
            zeta_a = data[1] * ramp
            k = omega**2 / g

            phase = -omega * time + k * eta[0] * np.cos(nu) + k * eta[1] * np.sin(nu)
            w_z += zeta_a * np.cos(phase) + ((k * zeta_a**2) / 2.0) * np.cos(2.0 * phase)

            omega_e[i] = omega - (omega**2 * V * np.cos(nu)) / g
            wave[i, :] = [zeta_a, k, nu, omega]

        w_zero = w_z

    elif irregular == 0:
        wind_speed = wind_speed / KNOTS_PER_MPS
        fetch = 500.0e3
        H_hat = 0.283 * np.tanh(0.0125 * ((fetch * g) / wind_speed**2) ** 0.42)
        T_hat = 7.54 * np.tanh(0.077 * ((fetch * g) / wind_speed**2) ** 0.25)

        H_s = H_hat * wind_speed**2 / g
        T_s = T_hat * wind_speed / g
        omega = 2.0 * np.pi / (T_s / 0.95)
        k = omega**2 / g
        zeta_a = (H_s / 2.0) * ramp
        wave[0, :] = [zeta_a, k, nu, omega]
        omega_out = omega
        phase = -omega * time + k * eta[0] * np.cos(nu) + k * eta[1] * np.sin(nu)
        w_zero = zeta_a * np.cos(phase)
        omega_e[0] = omega - (omega**2 * V * np.cos(nu)) / g

    else:
        wind_speed = wind_speed / KNOTS_PER_MPS
        fetch = 500.0e3
        H_hat = 0.283 * np.tanh(0.0125 * ((fetch * g) / wind_speed**2) ** 0.42)
        T_hat = 7.54 * np.tanh(0.077 * ((fetch * g) / wind_speed**2) ** 0.25)

        H_s = H_hat * wind_speed**2 / g
        T_s = T_hat * wind_speed / g
        omega_peak = 2.0 * np.pi / T_s

        n_freq = 10
        f_min = 0.15
        f_max = 1.50 * omega_peak
        frequencies = np.linspace(f_min, f_max, n_freq)

        S = (
            (5.0 / 16.0) * H_s**2 *
            (omega_peak**4 / frequencies**5) *
            np.exp(-5.0 * (omega_peak**4 / (4.0 * frequencies**4)))
        )
        df = frequencies[1] - frequencies[0]
        amplitude = np.sqrt(2.0 * S * df)

        w_z = 0.0
        for i in range(n_freq):
            omega = frequencies[i]
            zeta_a = amplitude[i] * ramp
            k = omega**2 / g

            phase = -omega * time + k * eta[0] * np.cos(nu) + k * eta[1] * np.sin(nu)
            w_z += zeta_a * np.cos(phase) + ((k * zeta_a**2) / 2.0) * np.cos(2.0 * phase)

            omega_e[i] = omega - (omega**2 * V * np.cos(nu)) / g
            wave[i, :] = [zeta_a, k, nu, omega]

        w_zero = w_z

    return wave, w_zero, omega_out, H_s, omega_e
