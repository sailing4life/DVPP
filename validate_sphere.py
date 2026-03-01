#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sphere drop-test validation for the DVPP panel solver.

Purpose
-------
Verify B(ω) and A_inf from the panel solver against exact analytical
results for a hemisphere at the free surface, then validate the damped
heave dynamics against an analytical damped-oscillator solution.

Tests
-----
1. A₃₃_inf:  panel solver vs. exact (2/3)πρR³
2. B₃₃(ω):   panel solver vs. exact  ρω(πR J₁(k₀R)/k₀)²
3. Drop test: 1-DOF Cummins heave ODE vs. analytical damped oscillator

Geometry
--------
Sphere, radius R, centred at origin, z-axis up.
Waterline at z = 0 (equator).  Mass = ρ_water·(2/3)πR³ → sphere
density = ρ_water/2 → half-submerged at equilibrium.

Analytical references
---------------------
Havelock T.H. (1942) Phil. Mag. 33, 666.
Gradshteyn & Ryzhik, Table of Integrals §3.771.1:
    ∫_{-R}^{R} √(R²–x²) e^{ikx} dx = πR J₁(kR) / k
"""

import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from mesh_utils import TriangleMesh
from radiation_utils import (
    min_symmetric_eigenvalues,
    ogilvie_added_mass_from_damping,
)

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

try:
    from capytaine_solver import CapytaineSolver, capytaine_available
except Exception:
    capytaine_available = lambda python_executable=None: False

from panel_solver import PanelSolver


# ── Sphere STL generator ────────────────────────────────────────────────────

def generate_sphere_stl(R=1.0, n_lat=40, n_lon=60):
    """
    Triangulated sphere centred at origin.

    Convention: x = forward (equatorial), y = starboard (equatorial),
    z = up (polar).  Waterline at z = 0 (equator).

    Parameters
    ----------
    R     : float  Radius [m].
    n_lat : int    Latitude bands (resolution in z).
    n_lon : int    Longitude points (resolution around z-axis).
    """
    phi   = np.linspace(0,   np.pi, n_lat + 1)   # colatitude: 0=top, π=bottom
    theta = np.linspace(0, 2*np.pi, n_lon + 1)   # longitude

    def vert(ph, th):
        return np.array([
            R * np.sin(ph) * np.cos(th),   # x  (forward)
            R * np.sin(ph) * np.sin(th),   # y  (starboard)
            R * np.cos(ph),                # z  (up)
        ])

    tris = []
    for i in range(n_lat):
        for j in range(n_lon):
            v00 = vert(phi[i],   theta[j])
            v10 = vert(phi[i+1], theta[j])
            v01 = vert(phi[i],   theta[j+1])
            v11 = vert(phi[i+1], theta[j+1])
            tris += [[v00, v10, v11], [v00, v11, v01]]

    arr = np.array(tris)
    return TriangleMesh.from_triangles(arr)


# ── Analytical reference ────────────────────────────────────────────────────

def sphere_analytical_B33(omega, R, rho=1025.0, g=9.81):
    """
    Exact strip-theory B₃₃ for a hemisphere (fac = ρω).

    For b(x) = √(R²–x²):
        ∫_{-R}^{R} √(R²–x²) e^{ik₀x} dx = πR J₁(k₀R) / k₀

    → B₃₃(ω) = ρω (πR J₁(k₀R)/k₀)²

    Parameters
    ----------
    omega : ndarray   Angular frequencies [rad/s].
    R     : float     Radius [m].
    """
    k0 = omega**2 / g
    I  = np.pi * R * j1(k0 * R) / k0    # ∫_{-R}^{R} b(x) e^{ik₀x} dx
    return rho * omega * I**2


# ── Retardation kernel ──────────────────────────────────────────────────────

def _retardation_kernel(B33, omega, tau_max=40.0, N_tau=400):
    """K(τ) = (2/π) ∫₀^∞ B₃₃(ω) cos(ωτ) dω  (trapezoidal rule)."""
    tau = np.linspace(0, tau_max, N_tau)
    K   = np.array([
        (2.0 / np.pi) * np.trapz(B33 * np.cos(omega * t), omega)
        for t in tau
    ])
    return tau, K


# ── 1-DOF Cummins heave simulator ───────────────────────────────────────────

def _simulate_cummins(M_tot, C33, tau_K, K, v0, t_end=30.0, dt=0.02,
                      B_visc=0.0):
    """
    Integrate the 1-DOF Cummins heave ODE:

        M_tot·ẇ + ∫₀ᵗ K(t–s)·w(s) ds + B_visc·w + C₃₃·z = 0

    z(0) = 0,  w(0) = –v0   (dropped onto the water surface).

    Parameters
    ----------
    B_visc : float  Optional additional linear viscous damping [N·s/m].
                    Represents skin friction, keel drag, bilge keels, etc.
                    not captured by radiation theory.  Default = 0.

    Uses RK4 with a running trapezoidal convolution on stored w history.
    """
    tau_disc = np.arange(int(round(tau_K[-1] / dt)) + 1, dtype=float) * dt
    K_disc = np.interp(tau_disc, tau_K, K, right=0.0)
    N_hist = len(K_disc)

    N       = int(round(t_end / dt))
    t_arr   = np.zeros(N + 1)
    z_arr   = np.zeros(N + 1)
    w_arr   = np.zeros(N + 1)
    w_arr[0] = -v0

    w_hist = np.zeros(N_hist)
    w_hist[0] = w_arr[0]

    for n in range(N):
        tn, zn, wn = t_arr[n], z_arr[n], w_arr[n]

        def conv_force(w):
            v_hist = w_hist.copy()
            v_hist[0] = w

            endpoint = 0.5 * (
                K_disc[0] * v_hist[0] +
                K_disc[-1] * v_hist[-1]
            )
            interior = np.dot(K_disc[1:-1], v_hist[1:-1])
            return dt * (endpoint + interior)

        def acc(z, w):
            return (-conv_force(w) - B_visc * w - C33 * z) / M_tot

        k1z = dt * wn;                  k1w = dt * acc(zn, wn)
        k2z = dt * (wn + .5*k1w);       k2w = dt * acc(zn + .5*k1z, wn + .5*k1w)
        k3z = dt * (wn + .5*k2w);       k3w = dt * acc(zn + .5*k2z, wn + .5*k2w)
        k4z = dt * (wn +    k3w);       k4w = dt * acc(zn +    k3z, wn +    k3w)

        z_arr[n+1] = zn + (k1z + 2*k2z + 2*k3z + k4z) / 6.0
        w_arr[n+1] = wn + (k1w + 2*k2w + 2*k3w + k4w) / 6.0
        t_arr[n+1] = tn + dt

        w_hist = np.roll(w_hist, shift=1)
        w_hist[0] = w_arr[n+1]

    return t_arr, z_arr


def _analytical_drop(M_tot, B_const, C33, v0, t):
    """
    Analytical damped-oscillator solution for constant damping B_const:
        M_tot·ẍ + B_const·ẋ + C33·x = 0,  x(0)=0, ẋ(0)=–v0
    """
    omega0 = np.sqrt(C33 / M_tot)
    zeta   = B_const / (2.0 * np.sqrt(C33 * M_tot))

    if zeta >= 1.0:                        # over-damped
        s1 = -omega0 * (zeta - np.sqrt(zeta**2 - 1))
        s2 = -omega0 * (zeta + np.sqrt(zeta**2 - 1))
        A  =  v0 / (s2 - s1)
        return A * (np.exp(s1*t) - np.exp(s2*t))

    omega_d = omega0 * np.sqrt(1.0 - zeta**2)
    return -(v0 / omega_d) * np.exp(-zeta * omega0 * t) * np.sin(omega_d * t)


# ── NEMOH data loader ─────────────────────────────────────────────────────────

def load_nemoh_data(excel_path):
    """
    Load NEMOH BEM added-mass and damping from the project Excel file.

    Expected layout
    ---------------
    Sheet 'A':  col 0 = omega,  col 1 = A22,  col 3 = omega,  col 4 = A33
    Sheet 'B':  col 0 = omega,  col 1 = B22,  col 5 = omega,  col 6 = B33

    Rows with non-numeric omega or value are silently dropped (header /
    blank rows).

    Returns
    -------
    dict with keys omega_B33, B33, omega_A33, A33  (all numpy arrays).
    None if pandas is not installed or the file cannot be read.
    """
    if not _PANDAS_OK:
        print("  [NEMOH] pandas not installed — skipping NEMOH comparison.")
        return None
    if not os.path.isfile(excel_path):
        print(f"  [NEMOH] file not found: {excel_path}")
        return None

    def _extract(df, omega_col, val_col):
        sub = df.iloc[:, [omega_col, val_col]].copy()
        sub.columns = ['omega', 'val']
        sub['omega'] = pd.to_numeric(sub['omega'], errors='coerce')
        sub['val']   = pd.to_numeric(sub['val'],   errors='coerce')
        sub = sub.dropna()
        return sub['omega'].values.astype(float), sub['val'].values.astype(float)

    dfA = pd.read_excel(excel_path, sheet_name='A', header=None)
    dfB = pd.read_excel(excel_path, sheet_name='B', header=None)

    omega_B33, B33 = _extract(dfB, 5, 6)   # heave damping
    omega_A33, A33 = _extract(dfA, 3, 4)   # heave added mass

    print(f"  [NEMOH] loaded {len(B33)} B33 points, {len(A33)} A33 points"
          f"  from {os.path.basename(excel_path)}")
    return {'omega_B33': omega_B33, 'B33': B33,
            'omega_A33': omega_A33, 'A33': A33}


# ── Main validation ──────────────────────────────────────────────────────────

def run_validation(R=5.0, rho=1025.0, g=9.81, h_drop=1.0,
                   n_lat=40, n_lon=60, n_sections=40,
                   nemoh_path='NEMOH old data/sphere_nemoh.xlsx',
                   use_capytaine=True, capytaine_python=None):
    """
    Run sphere panel-solver validation and drop test.

    Parameters
    ----------
    R         : float  Sphere radius [m].
    rho       : float  Water density [kg/m³].
    g         : float  Gravitational acceleration [m/s²].
    h_drop    : float  Drop height above still water [m].
    n_lat, n_lon : int  STL mesh resolution.
    n_sections   : int  Strip-theory cross-sections.
    nemoh_path   : str  Path to NEMOH Excel file (Sheet A / B).
    capytaine_python : str or None
        Optional path to a separate Python interpreter with a working
        Capytaine installation.

    Returns
    -------
    dict with A₃₃, B₃₃ errors and damping ratios.
    """
    # ── Load NEMOH reference data (optional) ─────────────────────────────────
    nemoh = load_nemoh_data(nemoh_path)

    # ── Analytical reference values ──────────────────────────────────────────
    A33_exact  = (2.0 / 3.0) * np.pi * rho * R**3   # exact for hemisphere
    C33_exact  = rho * g * np.pi * R**2               # ρg · Aw,  Aw = πR²
    mass_sphere = rho * (2.0 / 3.0) * np.pi * R**3   # density = ρ_water/2

    print(f"\n{'='*60}")
    print(f"  Sphere validation   R = {R} m")
    print(f"{'='*60}")
    print(f"  Exact A₃₃_inf  = {A33_exact:.1f} kg")
    print(f"  Exact C₃₃      = {C33_exact:.1f} N/m")
    print(f"  Sphere mass    = {mass_sphere:.1f} kg  (ρ_sphere = ρ_water/2)")

    # ── Generate sphere STL and run panel solver ──────────────────────────────
    sph = generate_sphere_stl(R=R, n_lat=n_lat, n_lon=n_lon)
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        stl_path = f.name
    sph.save(stl_path)

    omega  = np.linspace(0.2, 5.0, 200)
    # The sphere is centred at the origin with z-axis up, so the waterline
    # (equator) is analytically at z = 0.  Pass it directly so the panel
    # solver skips the hydrostatic bisection.
    solver = PanelSolver(stl_path, rho=rho, g=g, mass=mass_sphere,
                         waterline_z=0.0)
    solver.run(omega=omega, n_sections=n_sections)
    # Note: stl_path kept alive until after Capytaine also reads it (below).

    A33_num = solver.A_inf[2, 2]
    B33_num = solver.B_omega[2, 2]        # (N_omega,)
    A33_num_omega = solver.A_omega[2, 2]
    B33_ana = sphere_analytical_B33(omega, R, rho, g)
    B_psd_min = float(min_symmetric_eigenvalues(solver.B_omega).min())
    A_from_B = ogilvie_added_mass_from_damping(omega, solver.B_omega, solver.A_inf)
    A33_from_B = A_from_B[2, 2]
    kk_rel_rms = 100.0 * np.sqrt(np.mean((A33_num_omega - A33_from_B)**2)) / max(
        np.sqrt(np.mean(A33_num_omega**2)), 1.0
    )

    # ── Errors and damping ratios ─────────────────────────────────────────────
    err_A33 = 100.0 * (A33_num - A33_exact) / A33_exact
    M_tot   = mass_sphere + A33_num
    omega0  = np.sqrt(C33_exact / M_tot)

    B33_num_at_w0 = float(np.interp(omega0, omega, B33_num))
    B33_ana_at_w0 = float(np.interp(omega0, omega, B33_ana))
    B_crit        = 2.0 * np.sqrt(C33_exact * M_tot)
    zeta_num      = B33_num_at_w0 / B_crit
    zeta_ana      = B33_ana_at_w0 / B_crit
    err_B33       = 100.0 * (B33_num_at_w0 - B33_ana_at_w0) / max(abs(B33_ana_at_w0), 1.0)

    print(f"\n  [A₃₃_inf]  solver = {A33_num:.1f} kg  |  exact = {A33_exact:.1f} kg"
          f"  |  err = {err_A33:+.1f}%")
    print(f"  [Passivity] min λ(B(ω)) = {B_psd_min:.3e}")
    print(f"  [A(ω) mode] {solver.A_omega_method}")
    if solver.A_omega_method == 'Ogilvie':
        print(f"  [KK check]  RMS(A₃₃ - A₃₃[B]) / RMS(A₃₃) = {kk_rel_rms:.2f}%")
    print(f"  [ω₀]       {omega0:.3f} rad/s   (T₀ = {2*np.pi/omega0:.2f} s)")
    print(f"  [B₃₃(ω₀)]  solver = {B33_num_at_w0:.0f} N·s/m  (ζ = {zeta_num:.3f})")
    print(f"             exact  = {B33_ana_at_w0:.0f} N·s/m  (ζ = {zeta_ana:.3f})")
    print(f"             err    = {err_B33:+.1f}%")

    # NEMOH comparison stats
    zeta_nemoh = None
    if nemoh is not None:
        B33_nem_at_w0 = float(np.interp(omega0, nemoh['omega_B33'], nemoh['B33'],
                                         left=np.nan, right=np.nan))
        A33_nem_at_w0 = float(np.interp(omega0, nemoh['omega_A33'], nemoh['A33'],
                                         left=np.nan, right=np.nan))
        if np.isfinite(B33_nem_at_w0):
            zeta_nemoh = B33_nem_at_w0 / B_crit
            ratio_B    = B33_num_at_w0 / B33_nem_at_w0
            print(f"\n  [NEMOH B₃₃(ω₀)] = {B33_nem_at_w0:.0f} N·s/m  (ζ = {zeta_nemoh:.4f})")
            print(f"  Strip/NEMOH ratio at ω₀: {ratio_B:.1f}×  "
                  f"(expected >> 1 for non-slender sphere L/B=1)")

    # ── Drop speed (used in both Capytaine and drop test sections) ───────────
    v0 = np.sqrt(2.0 * g * h_drop)
    print(f"\n  Drop from {h_drop} m → impact speed {v0:.2f} m/s")

    # ── Optional: Capytaine 3D BEM on sphere ─────────────────────────────────
    cap_data = None
    if use_capytaine and capytaine_available(capytaine_python):
        try:
            print("\n  [Capytaine] Running 3D BEM on sphere …")
            cap_sol = CapytaineSolver(
                stl_path, rho=rho, g=g, waterline_z=0.0,
                python_executable=capytaine_python,
            )
            cap_sol.run(omega=omega)
            os.unlink(stl_path)   # delete temp file after Capytaine is done
            B33_cap    = cap_sol.B_omega[2, 2]       # (N_omega,)
            A33_cap    = cap_sol.A_omega[2, 2]
            A33inf_cap = float(cap_sol.A_inf[2, 2])
            C33_cap    = float(cap_sol.C[2, 2]) if cap_sol.C[2, 2] > 0 else C33_exact
            M_tot_cap  = mass_sphere + A33inf_cap
            B_cap_min  = float(min_symmetric_eigenvalues(cap_sol.B_omega).min())

            B33_cap_at_w0 = float(np.interp(omega0, omega, B33_cap))
            B_crit_cap    = 2.0 * np.sqrt(C33_cap * M_tot_cap)
            zeta_cap      = B33_cap_at_w0 / B_crit_cap
            print(f"  [Capytaine] A₃₃_inf = {A33inf_cap:.1f} kg  "
                  f"B₃₃(ω₀) = {B33_cap_at_w0:.0f} N·s/m  ζ = {zeta_cap:.4f}"
                  f"  min λ(B) = {B_cap_min:.3e}")

            tau_K_cap, K_cap   = _retardation_kernel(B33_cap, omega)
            t_sim_cap, z_sim_cap = _simulate_cummins(
                M_tot_cap, C33_cap, tau_K_cap, K_cap, v0, t_end=30.0, dt=0.02
            )
            cap_data = {
                'B33': B33_cap, 'A33': A33_cap,
                'A33inf': A33inf_cap, 'C33': C33_cap,
                'tau_K': tau_K_cap, 'K': K_cap,
                't_sim': t_sim_cap, 'z_sim': z_sim_cap,
                'B33_at_w0': B33_cap_at_w0, 'zeta': zeta_cap,
                'M_tot': M_tot_cap,
            }
        except Exception as e:
            print(f"  [Capytaine] Error: {e}")
            # Ensure temp file is cleaned up even if Capytaine fails
            if os.path.exists(stl_path):
                os.unlink(stl_path)

    elif use_capytaine and not capytaine_available(capytaine_python):
        print("\n  [Capytaine] not available in the current configuration.")
        if capytaine_python:
            print(f"  [Capytaine] external interpreter failed: {capytaine_python}")
        else:
            print("  [Capytaine] set --capytaine-python /path/to/python "
                  "or CAPYTAINE_PYTHON to use a separate environment.")
        os.unlink(stl_path)

    else:
        # Capytaine not requested
        os.unlink(stl_path)

    # ── Drop test ─────────────────────────────────────────────────────────────
    tau_K, K   = _retardation_kernel(B33_num, omega)
    t_sim, z_sim = _simulate_cummins(M_tot, C33_exact, tau_K, K, v0,
                                     t_end=30.0, dt=0.02)

    t_ana = np.linspace(0, 30.0, 1500)
    z_ana = _analytical_drop(M_tot, B33_num_at_w0, C33_exact, v0, t_ana)

    # ── Plots (2 × 2) ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── [0,0] B₃₃(ω) ─────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(omega, B33_ana / 1e3, 'k-',  lw=2.5, label='Exact (Bessel, analytical)')
    ax.plot(omega, B33_num / 1e3, 'r--', lw=2.0, label='Strip theory (panel solver)')
    if cap_data is not None:
        ax.plot(omega, cap_data['B33'] / 1e3, 'b-', lw=2.0,
                label=f"Capytaine 3D BEM  ζ={cap_data['zeta']:.4f}")
    if nemoh is not None:
        ax.plot(nemoh['omega_B33'], nemoh['B33'] / 1e3, 'g-o', lw=1.5,
                ms=4, label='NEMOH (3D BEM)')
    ax.axvline(omega0, ls=':', color='steelblue',
               label=f'ω₀ = {omega0:.2f} rad/s')
    ax.set_xlabel('ω [rad/s]')
    ax.set_ylabel('B₃₃ [kN·s/m]')
    ax.set_title('Heave radiation damping B₃₃(ω)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── [0,1] A₃₃(ω) ─────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(omega, A33_num_omega / 1e3, 'r--', lw=2.0, label='Strip theory (panel solver)')
    if solver.A_omega_method == 'Ogilvie':
        ax.plot(omega, A33_from_B / 1e3, color='darkorange', lw=1.8,
                label='Ogilvie reconstruction from B₃₃')
    if cap_data is not None:
        ax.plot(omega, cap_data['A33'] / 1e3, 'b-', lw=2.0, label='Capytaine 3D BEM')
    if nemoh is not None:
        ax.plot(nemoh['omega_A33'], nemoh['A33'] / 1e3, 'g-o', lw=1.5,
                ms=4, label='NEMOH (3D BEM)')
    ax.axhline(A33_exact / 1e3, ls=':', color='k', lw=1.5,
               label=f'Exact A₃₃_inf = {A33_exact/1e3:.1f} t')
    ax.axvline(omega0, ls=':', color='steelblue', label=f'ω₀ = {omega0:.2f} rad/s')
    ax.set_xlabel('ω [rad/s]')
    ax.set_ylabel('A₃₃ [t]')
    ax.set_title('Heave added mass A₃₃(ω)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── [1,0] Retardation kernel ──────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(tau_K, K / 1e3, 'r--', lw=1.8, label='Strip theory K(τ)')
    if cap_data is not None:
        ax.plot(cap_data['tau_K'], cap_data['K'] / 1e3, 'b-', lw=1.8,
                label='Capytaine K(τ)')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('τ [s]')
    ax.set_ylabel('K(τ) [kN/m]')
    ax.set_title('Retardation kernel K(τ)  — Ogilvie cosine transform of B₃₃(ω)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── [1,1] Drop test ───────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(t_ana, z_ana * 100, 'k-',  lw=2.5,
            label=f'Analytical (strip B₃₃)  ζ = {zeta_num:.3f}')
    ax.plot(t_sim, z_sim * 100, 'r--', lw=2.0,
            label='Cummins (strip theory)')
    if cap_data is not None:
        t_ana_cap = np.linspace(0, 30.0, 1500)
        z_ana_cap = _analytical_drop(
            cap_data['M_tot'], cap_data['B33_at_w0'],
            cap_data['C33'], v0, t_ana_cap
        )
        ax.plot(t_ana_cap, z_ana_cap * 100, 'b:', lw=2.0,
                label=f"Analytical (Capytaine B₃₃)  ζ = {cap_data['zeta']:.4f}")
        ax.plot(cap_data['t_sim'], cap_data['z_sim'] * 100, 'b-', lw=2.0,
                label='Cummins (Capytaine 3D BEM)')
    if zeta_nemoh is not None:
        z_nem = _analytical_drop(M_tot, B33_nem_at_w0, C33_exact, v0, t_ana)
        ax.plot(t_ana, z_nem * 100, 'g-', lw=1.5,
                label=f'Analytical (NEMOH B₃₃)  ζ = {zeta_nemoh:.4f}')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('z [cm]')
    ax.set_title(f'Heave drop test  (h = {h_drop} m, R = {R} m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Sphere validation  R={R} m  |  A₃₃ err={err_A33:+.1f}%'
        f'  |  B₃₃(strip) err={err_B33:+.1f}%',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    out = 'sphere_validation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n  Figure saved → {out}")
    print(f"{'='*60}\n")

    return {
        'A33_exact'      : A33_exact,
        'A33_num'        : A33_num,
        'err_A33_pct'    : err_A33,
        'min_B_eig'      : B_psd_min,
        'kk_rms_pct'     : kk_rel_rms,
        'B33_exact_at_w0': B33_ana_at_w0,
        'B33_num_at_w0'  : B33_num_at_w0,
        'err_B33_pct'    : err_B33,
        'zeta_num'       : zeta_num,
        'zeta_exact'     : zeta_ana,
        'zeta_nemoh'     : zeta_nemoh,
        'zeta_capytaine' : cap_data['zeta'] if cap_data else None,
        'cap_data'       : cap_data,
    }


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Sphere panel-solver validation')
    p.add_argument('--R',      type=float, default=5.0,  help='Sphere radius [m]')
    p.add_argument('--h',      type=float, default=1.0,  help='Drop height [m]')
    p.add_argument('--nlat',   type=int,   default=40,   help='Latitude bands in STL')
    p.add_argument('--nlon',   type=int,   default=60,   help='Longitude points in STL')
    p.add_argument('--nsec',   type=int,   default=40,   help='Strip-theory sections')
    p.add_argument('--nemoh',  type=str,
                   default='NEMOH old data/sphere_nemoh.xlsx',
                   help='Path to NEMOH Excel file (leave empty to skip)')
    p.add_argument('--no-capytaine', action='store_true',
                   help='Disable Capytaine 3D BEM comparison even if available')
    p.add_argument('--capytaine-python', type=str, default=None,
                   help='Path to a separate Python executable with a working '
                        'Capytaine installation')
    args = p.parse_args()
    run_validation(R=args.R, h_drop=args.h,
                   n_lat=args.nlat, n_lon=args.nlon, n_sections=args.nsec,
                   nemoh_path=args.nemoh,
                   use_capytaine=not args.no_capytaine,
                   capytaine_python=args.capytaine_python)
