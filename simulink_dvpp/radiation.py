#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from .constants import GRAVITY
from radiation_utils import (
    project_psd_matrix,
    project_psd_slices,
    symmetrize_matrix_slices,
)

_OMEGA = np.array([
    0.1, 0.357894737, 0.615789474, 0.873684211, 1.131578947,
    1.389473684, 1.647368421, 1.905263158, 2.163157895, 2.421052632,
    2.678947368, 2.936842105, 3.194736842, 3.452631579, 3.710526316,
    3.968421053, 4.226315789, 4.484210526, 4.742105263, 5.0
], dtype=float)
_B22 = np.array([5.14737e-06, 0.03808079, 1.627771, 17.38488, 92.70217,
                 312.605, 739.908, 1319.029, 1907.884, 2421.43, 2843.844,
                 3172.293, 3419.419, 3599.881, 3725.52, 3809.361, 3858.969,
                 3883.483, 3885.75, 3873.631], dtype=float)
_B33 = np.array([160.5986, 6762.314, 28321.6, 60373.25, 92390.3, 117407.7,
                 132289.7, 137389.0, 137336.4, 135783.3, 133192.8, 129575.8,
                 125503.8, 121037.6, 116339.5, 111456.8, 106636.9, 101839.0,
                 97031.34, 92771.13], dtype=float)
_B44 = np.array([0.000157811, 1.215368, 51.68285, 547.7202, 2891.626,
                 9631.147, 22443.76, 39305.69, 55728.69, 69184.45, 79491.12,
                 86867.23, 91763.66, 94725.73, 96188.57, 96562.52, 96098.98,
                 95049.59, 93509.26, 91721.73], dtype=float)
_B55 = np.array([9.633365, 597.3659, 9738.507, 79331.76, 323523.7, 769297.1,
                 1263380.0, 1651269.0, 1862128.0, 1925453.0, 1936583.0,
                 1929139.0, 1896046.0, 1851460.0, 1798147.0, 1739053.0,
                 1677499.0, 1615577.0, 1555205.0, 1496252.0], dtype=float)
_B24 = np.array([-2.60e-05, -0.2127534, -9.072762, -96.46463, -511.0771,
                 -1707.897, -3992.239, -7008.802, -9976.949, -12475.93,
                 -14455.79, -15923.4, -16959.94, -17649.82, -18064.62,
                 -18278.16, -18331.82, -18272.19, -18113.64, -17900.02], dtype=float)
_B35 = np.array([41.37384, 1720.282, 7055.404, 14751.97, 22304.22, 28171.14,
                 32588.54, 36524.02, 39212.77, 39957.71, 39810.66, 39418.31,
                 38452.21, 36933.77, 34742.93, 32207.62, 29749.37, 27639.87,
                 25785.19, 24306.8], dtype=float)
_A22 = np.array([1847.62, 1865.516, 1905.946, 1980.74, 2091.774, 2203.595,
                 2250.847, 2192.23, 2051.373, 1879.338, 1706.763, 1546.331,
                 1406.116, 1285.556, 1183.836, 1098.635, 1027.511, 967.9418,
                 918.1838, 876.6674], dtype=float)
_A33 = np.array([169680.8, 178216.7, 175291.0, 158825.8, 136559.9, 114790.3,
                 96428.17, 83245.09, 74936.35, 69494.7, 65581.5, 62844.26,
                 60977.24, 59697.91, 58885.32, 58389.64, 58160.09, 58038.81,
                 58112.28, 57986.34], dtype=float)
_A44 = np.array([61792.0, 62250.95, 63466.08, 65620.88, 68805.18, 71905.16,
                 72915.98, 70551.15, 65846.98, 60478.3, 55342.73, 50746.14,
                 46844.27, 43607.52, 40977.52, 38854.86, 37150.99, 35787.41,
                 34699.09, 33847.45], dtype=float)
_A55 = np.array([1612879.0, 1639082.0, 1709144.0, 1827260.0, 1923098.0,
                 1891995.0, 1725236.0, 1500475.0, 1288806.0, 1136143.0,
                 1036763.0, 963747.2, 909664.9, 871112.4, 842676.6, 822650.1,
                 808761.4, 799198.8, 792968.6, 788193.4], dtype=float)
_A24 = np.array([-9898.775, -9986.147, -10202.04, -10591.44, -11168.44,
                 -11733.11, -11929.95, -11540.41, -10739.04, -9808.987,
                 -8900.736, -8075.32, -7367.192, -6770.16, -6277.969,
                 -5874.378, -5544.745, -5275.081, -5055.163, -4877.226], dtype=float)
_A35 = np.array([62195.04, 64322.77, 63525.0, 59614.1, 54534.48, 50027.25,
                 46720.94, 43931.11, 41092.59, 38722.02, 37048.71, 35744.22,
                 34695.63, 33901.64, 33389.35, 33182.74, 33233.8, 33456.82,
                 33745.38, 34417.65], dtype=float)

DEFAULT_OMEGA = _OMEGA
DEFAULT_B33 = _B33
# Infinite-frequency added-mass matrix from NEMOH panel solver (IMOCA 60).
# All entries in kg (translational) or kg·m² (rotational).
# A_INF[0,0] = 92 kg  — surge added mass (included here so mass.py can use
#                        a_inf_matrix[0,0] uniformly; updated by Capytaine).
A_INF = np.zeros((6, 6), dtype=float)
A_INF[0, 0] = 92.0        # surge
A_INF[1, 1] = 3659.394    # sway
A_INF[2, 2] = 90767.48    # heave
A_INF[3, 3] = 86082.77    # roll
A_INF[4, 4] = 1167779.0   # pitch
A_INF[2, 4] = 39028.0     # heave-pitch coupling
A_INF[4, 2] = 39028.0
A_INF[1, 3] = -6854.0     # sway-roll coupling
A_INF[3, 1] = -6854.0


def _build_ab_matrices():
    B = np.zeros((6, 6, len(_OMEGA)))
    A = np.zeros((6, 6, len(_OMEGA)))

    B[1, 1, :] = _B22
    B[2, 2, :] = _B33
    B[3, 3, :] = _B44
    B[4, 4, :] = _B55
    B[1, 3, :] = _B24
    B[3, 1, :] = _B24
    B[2, 4, :] = _B35
    B[4, 2, :] = _B35

    A[1, 1, :] = _A22
    A[2, 2, :] = _A33
    A[3, 3, :] = _A44
    A[4, 4, :] = _A55
    A[1, 3, :] = _A24
    A[3, 1, :] = _A24
    A[2, 4, :] = _A35
    A[4, 2, :] = _A35
    return A, B


class CumminsRadiationModel:
    """
    Faithful port of Simulink chart_468 `Cummins equation`, with optional
    panel-solver coefficient injection and explicit history updates for RK4.
    """

    def __init__(self, dt=0.10, end_t=10.0):
        self.dt = float(dt)
        self.end_t = float(end_t)
        self.timerange = np.arange(int(round(end_t / dt)) + 1, dtype=float) * self.dt
        a_matrix, b_matrix = _build_ab_matrices()
        self.A = symmetrize_matrix_slices(a_matrix)
        self.B = project_psd_slices(b_matrix)
        self.A_inf = project_psd_matrix(A_INF.copy())
        self.omega = _OMEGA

        self.Ls = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=float)
        self.velocity_history = np.zeros((6, len(self.timerange)))
        self.kernel_cache = {}

    def inject_solver_data(self, B_omega, A_omega, A_inf, omega):
        """Replace NEMOH coefficients with panel-solver results (strip theory or BEM)."""
        self.omega = np.asarray(omega, dtype=float)
        self.A = symmetrize_matrix_slices(np.asarray(A_omega, dtype=float))
        self.B = project_psd_slices(np.asarray(B_omega, dtype=float))
        self.A_inf = project_psd_matrix(np.asarray(A_inf, dtype=float))
        # Rebuild timerange to match the new dt (unchanged) but flush the kernel
        # cache so it is recomputed against the new B and A_inf values.
        self.kernel_cache.clear()

    def reset(self):
        self.velocity_history.fill(0.0)
        self.kernel_cache.clear()

    def get_A_inf(self):
        return self.A_inf.copy()

    def update(self, nu):
        self.velocity_history = np.roll(self.velocity_history, shift=1, axis=1)
        self.velocity_history[:, 0] = np.asarray(nu, dtype=float)

    def _compute_kernel(self, U):
        key = round(float(U), 2)
        if key in self.kernel_cache:
            return self.kernel_cache[key]

        # Forward-speed IRF correction (Delhommeau & Kobus 1987, thesis Eq. 4.35):
        #   K-integrand = B(ω) + U·(A_∞ − A(ω))·L_s
        # a_ls[i,j,k] = A(ω_k) @ L_s  (frequency-dependent added-mass projection)
        a_ls = np.matmul(self.A.transpose(2, 0, 1), self.Ls).transpose(1, 2, 0)
        b_corrected = self.B + U * (self.A_inf @ self.Ls)[:, :, None] - U * a_ls
        # Project each frequency slice to the PSD cone: the Delhommeau correction
        # can drive cross-coupling terms negative at high ω when A(ω) << A_∞,
        # which creates a growing IRF and crashes the integrator.  Projecting to
        # PSD clips unphysical energy-injecting components (same treatment applied
        # to the raw B matrix at initialisation).
        b_corrected = project_psd_slices(b_corrected)

        # Vectorised cosine transform: kernel shape (6, 6, n_tau)
        # integrand[i,j,k,t] = b_corrected[i,j,k] * cos(ω_k · t)
        cos_values = np.cos(np.outer(self.omega, self.timerange))   # (n_omega, n_tau)
        integrand = b_corrected[:, :, :, np.newaxis] * cos_values[np.newaxis, np.newaxis, :, :]
        kernel = (2.0 / np.pi) * np.trapz(integrand, x=self.omega, axis=2)

        self.kernel_cache[key] = kernel
        return kernel

    def get_force(self, nu, U=None, count_sum=12436.0 * GRAVITY):
        nu = np.asarray(nu, dtype=float)
        U = float(nu[0] if U is None else U)
        factor = 1.0 + ((count_sum / GRAVITY) - 12436.0) / 12436.0

        kernel = self._compute_kernel(U)
        taper = np.exp(-(3.0 * self.timerange / self.end_t) ** 2)
        velocity_history = self.velocity_history.copy()
        velocity_history[:, 0] = nu

        endpoints = (
            kernel[:, :, 0] @ velocity_history[:, 0] * taper[0] +
            kernel[:, :, -1] @ velocity_history[:, -1] * taper[-1]
        ) / 2.0
        interior = np.einsum(
            "ijk,jk->i",
            kernel[:, :, 1:-1],
            velocity_history[:, 1:-1] * taper[np.newaxis, 1:-1],
        )
        convolution = self.dt * (endpoints + interior)
        b_infinite = ((-U) * (self.A_inf @ self.Ls)) @ nu
        return (convolution + b_infinite) * factor

    def force(self, eta_dot, count_sum):
        return self.get_force(eta_dot, float(np.asarray(eta_dot, dtype=float)[0]), count_sum=count_sum)
