#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def symmetrize_matrix(matrix):
    """Return the symmetric part of a 2-D square matrix."""
    mat = np.asarray(matrix, dtype=float)
    return 0.5 * (mat + mat.T)


def symmetrize_matrix_slices(coeffs):
    """Return the symmetric part of a (n, n, n_freq) coefficient tensor."""
    arr = np.asarray(coeffs, dtype=float)
    return 0.5 * (arr + np.swapaxes(arr, 0, 1))


def project_psd_matrix(matrix, tol=0.0):
    """
    Project a symmetric matrix onto the positive semidefinite cone.

    Radiation damping and added-mass matrices should not inject energy, so
    slightly negative eigenvalues from numerical noise are clipped to zero.
    """
    sym = symmetrize_matrix(matrix)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.where(eigvals > tol, eigvals, 0.0)
    projected = (eigvecs * eigvals) @ eigvecs.T
    return symmetrize_matrix(projected)


def project_psd_slices(coeffs, tol=0.0):
    """Project each frequency slice of a coefficient tensor onto the PSD cone."""
    arr = symmetrize_matrix_slices(coeffs)
    out = np.empty_like(arr)
    for k in range(arr.shape[2]):
        out[:, :, k] = project_psd_matrix(arr[:, :, k], tol=tol)
    return out


def min_symmetric_eigenvalues(coeffs):
    """Return the minimum eigenvalue of each symmetric frequency slice."""
    arr = symmetrize_matrix_slices(coeffs)
    return np.array([
        np.linalg.eigvalsh(arr[:, :, k]).min()
        for k in range(arr.shape[2])
    ])


def estimate_ainf_from_added_mass(omega, a_omega, tail_points=6):
    """
    Estimate A_inf from the high-frequency added-mass asymptote.

    For free-surface radiation problems:
        A(omega) = A_inf + O(omega^-2)

    We fit the last few frequency points against 1/omega^2 and take the
    intercept as A_inf. This is more stable than integrating noisy B(omega)/omega
    over a truncated frequency band.
    """
    omega = np.asarray(omega, dtype=float)
    a_omega = np.asarray(a_omega, dtype=float)

    n_fit = min(max(3, tail_points), len(omega))
    w_tail = omega[-n_fit:]
    x = 1.0 / np.square(w_tail)
    design = np.column_stack([np.ones_like(x), x])

    a_inf = np.zeros(a_omega.shape[:2], dtype=float)
    for i in range(a_omega.shape[0]):
        for j in range(a_omega.shape[1]):
            y = a_omega[i, j, -n_fit:]
            if np.allclose(y, 0.0):
                continue
            coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
            intercept = coeffs[0]
            if not np.isfinite(intercept):
                intercept = y[-1]
            a_inf[i, j] = intercept

    return project_psd_matrix(a_inf)


def _extended_ogilvie_grid(omega, values, low_points=16, high_points=80,
                           tail_power=5.0):
    """Extend B(omega) to [0, infinity) with asymptotic low/high-frequency tails."""
    omega = np.asarray(omega, dtype=float)
    values = np.asarray(values, dtype=float)

    if np.allclose(values, 0.0):
        return omega, values

    omega_parts = []
    value_parts = []

    if omega[0] > 0.0 and low_points > 0:
        omega_low = np.linspace(0.0, omega[0], low_points, endpoint=False)
        slope = values[0] / max(omega[0], 1e-12)
        omega_parts.append(omega_low)
        value_parts.append(slope * omega_low)

    omega_parts.append(omega)
    value_parts.append(values)

    if high_points > 0 and len(omega) >= 2:
        n_fit = min(4, len(omega))
        tail_coeff = np.mean(values[-n_fit:] * omega[-n_fit:]**tail_power)
        omega_hi_start = omega[-1] * (1.0 + 1e-6)
        omega_hi_stop = max(10.0 * omega[-1], omega[-1] + 20.0)
        omega_high = np.geomspace(omega_hi_start, omega_hi_stop, high_points)
        omega_parts.append(omega_high)
        value_parts.append(tail_coeff / omega_high**tail_power)

    omega_ext = np.concatenate(omega_parts)
    value_ext = np.concatenate(value_parts)
    return omega_ext, value_ext


def ogilvie_added_mass_from_damping(omega, b_omega, a_inf):
    """
    Reconstruct A(omega) from B(omega) with the Ogilvie principal-value integral:

        A(omega) = A_inf + (2/pi) PV ∫ B(s) / (s^2 - omega^2) ds
    """
    omega = np.asarray(omega, dtype=float)
    b_omega = np.asarray(b_omega, dtype=float)
    a_inf = symmetrize_matrix(np.asarray(a_inf, dtype=float))

    a_omega = np.zeros_like(b_omega, dtype=float)
    for i in range(b_omega.shape[0]):
        for j in range(b_omega.shape[1]):
            omega_ext, b_ext = _extended_ogilvie_grid(omega, b_omega[i, j, :])
            for idx, w in enumerate(omega):
                denom = omega_ext**2 - w**2
                mask = np.abs(denom) > 1e-10 * max(1.0, w**2)
                integrand = np.zeros_like(omega_ext)
                integrand[mask] = b_ext[mask] / denom[mask]
                a_omega[i, j, idx] = (
                    a_inf[i, j] +
                    (2.0 / np.pi) * np.trapz(integrand, x=omega_ext)
                )

    return symmetrize_matrix_slices(a_omega)
