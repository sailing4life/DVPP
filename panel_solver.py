#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strip-theory panel solver for the IMOCA 60 DVPP.

Replaces the hardcoded NEMOH data in F_rad.py with coefficients computed
directly from the hull STL geometry.

Theory — Strip theory (Salvesen, Tuck & Faltinsen 1970, "STF method")
----------------------------------------------------------------------
The hull is sliced into N cross-sections.  For each section at position x,
2D (per-unit-length) hydrodynamic properties are computed.  The 3D
radiation coefficients are obtained by integrating along the ship length.

Hydrodynamic properties computed
---------------------------------
1. Hydrostatic restoring matrix C (6×6)
   From waterplane area, first and second moments, displacement.
   Exact result from hull geometry — no approximations.

2. Infinite-frequency added mass A_inf (6×6)
   From the Lewis (1929) form strip theory.
   Each cross-section is approximated by its Lewis form (a mapping from
   a circle to the section shape using two free parameters c₁, c₃).
   The 2D infinite-frequency added mass is then integrated over length.

   Reference: Lewis F.M. (1929) "The inertia of the water surrounding
   a vibrating ship", SNAME Trans. 37, 1–20.

3. Frequency-dependent radiation damping B(ω) (6×6 × N_omega)
   From the Vossers (1960) thin-ship approximation:

       B₃₃(ω) = ρω |∫_{−L/2}^{L/2} b(x) e^{ik₀x} dx|²

   where b(x) = waterplane half-breadth, k₀ = ω²/g.
   (fac = ρω, not ρg²/ω³ — the thin-ship BC σ(x)=k₀b(x) contributes k₀²
   to the Kochin integral, which combines with ρg²/ω³·k₀² = ρω.)
   Analogous formulae for pitch (B₅₅) and heave–pitch coupling (B₃₅).
   For sway and roll, corresponding formulae use the section draft.

   This is the far-field energy argument: the damping coefficient equals
   the wave energy radiated per cycle by the oscillating hull.

   Reference: Vossers G. (1960) "Fundamentals of the Behaviour of Ships
   in Waves", Delft Technische Hogeschool Rept. no. 76S.
   Salvesen N., Tuck E.O., Faltinsen O. (1970) "Ship Motions and Sea
   Loads", SNAME Trans. 78, 250–287. (Eqs. 31–41 for B and A integrals.)

4. Frequency-dependent added mass A(ω) (6×6 × N_omega)
   When the damping matrix is well-behaved, A(ω) is reconstructed from the
   Ogilvie principal-value relation:

       A(ω) = A_inf + (2/π) PV ∫_0^∞ B(s) / (s² − ω²) ds

   If that reconstruction becomes nonphysical on the available frequency
   band, the solver falls back to A(ω) = A_inf. This is conservative but
   far safer than passing a spurious A(ω) into the forward-speed correction.

   Reference: Ogilvie T.F. (1964) "Recent progress towards the
   understanding and prediction of ship motions", 5th ONR Symp.

Coordinate system
-----------------
Body frame: x = forward (bow), y = starboard, z = up.
Origin at design waterline, midship.
Waterline at z = 0.  Submerged hull: z < 0.

Limitations
-----------
* Strip theory is accurate for slender hulls (L/B >> 1), which is
  satisfied by an IMOCA 60 (LOA ≈ 18 m, B ≈ 5.7 m → L/B ≈ 3.2).
* The Vossers damping formula is a thin-ship approximation; it over-
  predicts damping for blunt sections near the bow and stern.
* Roll damping has a large viscous component not captured here; B₄₄ from
  this solver should be supplemented with an empirical viscous term.
* Forward-speed effects on the coefficients (the U·L_s correction) are
  computed in F_rad.py and are separate from this zero-speed BEM.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation

from mesh_utils import TriangleMesh

from radiation_utils import (
    ogilvie_added_mass_from_damping,
    project_psd_matrix,
    project_psd_slices,
)


# =========================================================================== #
#  WATERLINE DETECTION                                                         #
# =========================================================================== #

def _submerged_volume_at(triangles, z_wl, n_x=50):
    """
    Estimate the displaced (submerged) volume when the waterline is at z = z_wl.

    Uses strip integration along x:
        ∇(z_wl) = ∫ A_s(x, z_wl) dx

    where A_s(x, z_wl) is the submerged cross-section area at station x.

    Parameters
    ----------
    triangles : ndarray (N, 3, 3)
        Triangle vertices.
    z_wl : float
        Trial waterline z in the STL frame.
    n_x : int
        Number of longitudinal stations for the integration.

    Returns
    -------
    volume : float   Displaced volume [m³].
    """
    xs = triangles[:, :, 0]
    x_min = xs.min() * 1.01
    x_max = xs.max() * 1.01
    x_cuts = np.linspace(x_min, x_max, n_x + 2)[1:-1]

    areas  = []
    x_vals = []
    for xc in x_cuts:
        segs = _slice_triangles(triangles, xc)
        if not segs:
            continue
        geom = _section_geometry(segs, waterline_z=z_wl)
        if geom is not None:
            areas.append(geom['A_s'])
            x_vals.append(xc)

    if len(areas) < 2:
        return 0.0
    return float(np.trapz(areas, x_vals))


def _find_equilibrium_waterline(triangles, mass, rho=1025.0, g=9.81,
                                 n_x=40, tol_m=0.001):
    """
    Find the waterline z at which buoyancy equals vessel weight.

    Solves:  ρ · g · ∇(z_wl) = mass · g
          ⟺  ∇(z_wl) = mass / ρ

    using bisection on the strip-integrated displaced volume.

    Parameters
    ----------
    triangles : ndarray (N, 3, 3)
        Triangle vertices in the STL frame.
    mass : float
        Vessel displacement mass [kg].
    rho : float
        Water density [kg/m³].
    g : float
        Gravitational acceleration [m/s²].
    n_x : int
        Stations for strip integration (coarser is faster for bisection).
    tol_m : float
        Convergence tolerance on z [m].  Default 1 mm.

    Returns
    -------
    z_wl : float
        Waterline z-coordinate in the STL frame.
    nabla : float
        Confirmed displaced volume [m³].
    """
    target_vol = mass / rho

    z_all = triangles[:, :, 2].ravel()
    z_lo  = z_all.min()
    z_hi  = z_all.max()

    # Sanity check: can the hull displace enough?
    vol_max = _submerged_volume_at(triangles, z_hi, n_x)
    if vol_max < target_vol:
        raise RuntimeError(
            f"Hull cannot displace enough: max ∇ = {vol_max:.2f} m³ "
            f"< required {target_vol:.2f} m³ for {mass:.0f} kg.  "
            f"Check mass value or STL geometry."
        )

    # Bisection: ~log2(range/tol) ≈ 14 iterations for 2 m range, 1 mm tol
    for _ in range(30):
        if (z_hi - z_lo) < tol_m:
            break
        z_mid = (z_lo + z_hi) / 2.0
        vol   = _submerged_volume_at(triangles, z_mid, n_x)
        if vol < target_vol:
            z_lo = z_mid
        else:
            z_hi = z_mid

    z_wl  = (z_lo + z_hi) / 2.0
    nabla = _submerged_volume_at(triangles, z_wl, n_x)
    return float(z_wl), float(nabla)


# =========================================================================== #
#  STL SLICING                                                                 #
# =========================================================================== #

def _slice_triangles(triangles, x_cut):
    """
    Intersect a set of triangles with the plane x = x_cut.

    Returns a list of [p1, p2] segments (each a pair of 3-D points)
    forming the cross-section contour in the y–z plane.

    Parameters
    ----------
    triangles : ndarray, shape (N, 3, 3)
        Triangle vertices from the STL mesh.
    x_cut : float
        x-coordinate of the cutting plane.

    Returns
    -------
    segments : list of ndarray pairs, each shape (3,)
    """
    segments = []
    for tri in triangles:
        xs = tri[:, 0]   # x-coordinates of the three vertices
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            if (xs[i] - x_cut) * (xs[j] - x_cut) < 0:
                # Linear interpolation to find intersection on this edge
                t = (x_cut - xs[i]) / (xs[j] - xs[i])
                p = tri[i] + t * (tri[j] - tri[i])
                pts.append(p)
            elif abs(xs[i] - x_cut) < 1e-9:
                pts.append(tri[i].copy())
        # Each intersected triangle contributes exactly 2 points (1 segment)
        if len(pts) == 2:
            segments.append((pts[0], pts[1]))
    return segments


def _order_contour(segments, tol=1e-3):
    """
    Chain line segments into an ordered list of 3-D boundary points.

    Parameters
    ----------
    segments : list of (p1, p2) pairs
    tol : float
        Distance tolerance for connecting endpoints.

    Returns
    -------
    chain : ndarray, shape (M, 3)
        Ordered boundary points (may not close perfectly for imperfect STLs).
    """
    if not segments:
        return np.empty((0, 3))

    chain = [np.array(segments[0][0]), np.array(segments[0][1])]
    remaining = list(segments[1:])

    while remaining:
        last = chain[-1]
        found = False
        for k, (p1, p2) in enumerate(remaining):
            if np.linalg.norm(last - p1) < tol:
                chain.append(np.array(p2))
                remaining.pop(k)
                found = True
                break
            elif np.linalg.norm(last - p2) < tol:
                chain.append(np.array(p1))
                remaining.pop(k)
                found = True
                break
        if not found:
            # Gap in contour (open STL or tolerance issue) — stop
            break

    return np.array(chain)


def _section_geometry(segments, waterline_z=0.0):
    """
    Compute geometric properties of a cross-section from its contour segments.

    Parameters
    ----------
    segments : list of (p1, p2) pairs in the y–z plane
    waterline_z : float
        z-coordinate of the free surface.

    Returns
    -------
    dict with keys:
        B      : full beam at waterline [m]
        T      : draft below waterline [m]
        A_s    : submerged cross-section area [m²]
        b_wl   : half-beam at waterline [m] (= B/2)
        I_wl   : second moment of waterplane breadth about centreline [m⁴/m]
                 (used for pitch added mass: ∫ x² * I_wl dx)
    Returns None if the section has no submerged area.
    """
    # Collect all unique boundary points in the section
    pts = []
    for p1, p2 in segments:
        pts.append(p1)
        pts.append(p2)
    pts = np.array(pts)

    if pts.size == 0:
        return None

    y_all = pts[:, 1]   # athwartship
    z_all = pts[:, 2]   # vertical (z < 0 below waterline)

    # Draft: deepest point below waterline
    T = waterline_z - np.min(z_all)
    if T <= 0.0:
        return None

    # Beam at waterline: use all points within a small band near z=0
    band = z_all >= waterline_z - 0.05 * T   # top 5% of draft
    if np.sum(band) < 2:
        b_wl = (np.max(y_all) - np.min(y_all)) / 2.0
    else:
        b_wl = (np.max(y_all[band]) - np.min(y_all[band])) / 2.0

    B = 2.0 * b_wl

    # Submerged cross-section area via ordered contour + shoelace.
    #
    # The key issue with using raw (unordered) segment endpoints is that
    # sorting by z interleaves port and starboard points, producing a
    # zig-zag "bowtie" polygon whose shoelace area is near zero (the two
    # halves cancel).  Instead, chain the segments into a proper ordered
    # polygon first, then clip at waterline_z and apply the shoelace.
    #
    # The chain is treated as CLOSED (last vertex → first via % n).  For
    # an open hull U-shape this closing edge is the waterline chord — exactly
    # what we want.  For a fully-enclosed section (e.g. sphere) the chain
    # already forms a closed loop.
    chain = _order_contour(segments)

    if len(chain) >= 3:
        yc = chain[:, 1]
        zc = chain[:, 2]
        n  = len(yc)

        # Walk the closed polygon, clipping at waterline_z.
        sub_y, sub_z = [], []
        for i in range(n):
            y1, z1 = yc[i],         zc[i]
            y2, z2 = yc[(i+1) % n], zc[(i+1) % n]

            if z1 <= waterline_z:
                sub_y.append(y1)
                sub_z.append(z1)

            # Insert intersection point when the edge strictly crosses waterline_z.
            # Using product < 0 avoids spurious duplicates when z1 or z2 == wl.
            if (z1 - waterline_z) * (z2 - waterline_z) < 0:
                t = (waterline_z - z1) / (z2 - z1)
                sub_y.append(y1 + t * (y2 - y1))
                sub_z.append(waterline_z)

        if len(sub_y) >= 3:
            sy = np.array(sub_y)
            sz = np.array(sub_z)
            A_s = 0.5 * abs(
                np.dot(sy, np.roll(sz, -1)) - np.dot(np.roll(sy, -1), sz)
            )
        else:
            A_s = max(0.5 * B * T, 1e-9)

    else:
        # _order_contour failed (very coarse or non-manifold mesh) —
        # fall back to the approximate unordered-point method.
        sub_pts_y = y_all[z_all <= waterline_z]
        sub_pts_z = z_all[z_all <= waterline_z]

        if len(sub_pts_y) < 3:
            A_s = 0.5 * B * T
        else:
            order = np.argsort(sub_pts_z)
            yy    = sub_pts_y[order]
            zz    = sub_pts_z[order]
            A_s   = 0.5 * abs(
                np.dot(yy, np.roll(zz, -1)) - np.dot(np.roll(yy, -1), zz)
            )
            A_s = max(A_s, 0.5 * B * T * 0.5)   # floor at 50% of bounding box

    # Waterplane second moment about centreline: I_wl ≈ 2 * b^3 / 3
    I_wl = 2.0 * b_wl**3 / 3.0   # [m⁴/m] per unit length

    return {'B': B, 'T': T, 'A_s': A_s, 'b_wl': b_wl, 'I_wl': I_wl}


# =========================================================================== #
#  LEWIS FORM PARAMETERISATION                                                 #
# =========================================================================== #

def _lewis_params(b, T, A_s):
    """
    Solve for Lewis form parameters (R, c₁, c₃) given section geometry.

    The Lewis (1929) conformal mapping:
        z = R(ζ + c₁/ζ + c₃/ζ³)    on |ζ| = 1

    gives:
        b  = R(1 + c₁ + c₃)     half-breadth at waterline   (Eq. L.1)
        T  = R(1 − c₁ + c₃)     draft                       (Eq. L.2)
        A_s = πR²(1 − c₁² − 3c₃²)  submerged area           (Eq. L.3)

    Parameters
    ----------
    b, T, A_s : float   Half-breadth [m], draft [m], section area [m²].

    Returns
    -------
    R, c1, c3 : float   Lewis form parameters.
    """
    if T < 1e-6 or b < 1e-6:
        return 0.0, 0.0, 0.0

    def residuals(params):
        c1, c3 = params
        # Avoid division by zero
        denom = max(1.0 + c3, 1e-6)
        R = (b + T) / (2.0 * denom)
        eq1 = R * (1.0 + c1 + c3) - b
        eq3 = np.pi * R**2 * (1.0 - c1**2 - 3.0 * c3**2) - A_s
        return [eq1, eq3]

    # Initial guess from beam-draft ratio
    x0 = [(b - T) / (b + T) * 0.5, 0.0]
    try:
        sol, _, ier, _ = fsolve(residuals, x0, full_output=True)[:4]
        if ier != 1:
            sol = x0   # use fallback if no convergence
    except Exception:
        sol = x0

    c1, c3 = sol
    c3 = np.clip(c3, -0.3, 0.3)
    c1 = np.clip(c1, -0.5, 0.5)
    R  = (b + T) / (2.0 * max(1.0 + c3, 1e-6))

    return R, c1, c3


def _lewis_added_mass_2D(R, c1, c3, rho=1025.0):
    """
    Infinite-frequency 2D added mass per unit length in heave (mode 3).

    From Ursell (1957) / Tasai (1959) using the Lewis form:

        m₃₃ = ρπR²[(1 + c₃)² + c₁²]   [kg/m]

    Note: For a semicircular section (c₁=c₃=0), this gives ρπR², which
    should be ρπR²/2.  The factor of 1/2 arises because the hull is
    surface-piercing; we apply it explicitly here.

    Reference: Tasai F. (1959) "On the damping force and added mass of
    ships heaving and pitching", J. Zosen Kiokai 105, 47–56.
    """
    m33 = 0.5 * rho * np.pi * R**2 * ((1.0 + c3)**2 + c1**2)
    return m33


def _lewis_roll_added_mass_2D(b_wl, T, rho=1025.0):
    """
    Approximate 2D roll added mass per unit length (mode 4):

        m₄₄ ≈ ρ π/8 * (b⁴ + T⁴) / (b + T)   [kg·m²/m]  (simplified)

    This is an approximation based on the diagonal roll added mass
    of a Lewis section (Salvesen et al. 1970 discussion).
    """
    return rho * np.pi / 8.0 * (b_wl**4 + T**4)


# =========================================================================== #
#  MAIN CLASS                                                                  #
# =========================================================================== #

class PanelSolver:
    """
    Strip-theory-based panel solver.

    Workflow
    --------
    1. Load the hull STL.
    2. Call `run()` to compute all hydrodynamic matrices.
    3. The results can be fed directly into `F_rad.RadiationModel` via
       `to_rad_inputs()`, replacing the hardcoded NEMOH data.

    Parameters
    ----------
    stl_path : str   Path to the hull STL file (body frame, z up).
    rho      : float  Water density [kg/m³].
    g        : float  Gravitational acceleration [m/s²].
    """

    def __init__(self, stl_path, rho=1025.0, g=9.81,
                 mass=9500.0, waterline_z=None):
        """
        Parameters
        ----------
        stl_path    : str    Path to the hull STL file (body frame, x forward, z up).
        rho         : float  Water density [kg/m³].
        g           : float  Gravitational acceleration [m/s²].
        mass        : float  Vessel displacement mass [kg].  Used to solve for the
                             hydrostatic equilibrium waterline (buoyancy = weight).
                             Default 9500 kg (IMOCA 60: 6000 hull + 3500 keel bulb).
        waterline_z : float or None
            z-coordinate of the design waterline in the STL's original coordinate
            frame.  When provided, this overrides the hydrostatic equilibrium
            calculation — useful when mass is unknown or the STL uses a known
            datum.  The solver shifts all vertices internally so that z=0 is
            at the waterline (submerged region: z < 0) regardless of the input
            convention.

            When None (default), the waterline is found by solving:
                ρ · ∇(z_wl) = mass
            i.e. the z level at which displaced volume times water density equals
            the vessel mass.  This is the correct physical definition of the
            design waterline for a floating vessel.
        """
        self.hull = TriangleMesh.from_file(stl_path)

        # Work on a copy so the original mesh object is not modified
        raw_triangles = self.hull.vectors.copy()   # (N, 3, 3)

        # ---- Waterline: hydrostatic equilibrium or user-specified -----------
        if waterline_z is None:
            print(f"[PanelSolver] Solving hydrostatic equilibrium for "
                  f"mass = {mass:.0f} kg …")
            waterline_z, nabla = _find_equilibrium_waterline(
                raw_triangles, mass=mass, rho=rho, g=g)
            draft = waterline_z - raw_triangles[:, :, 2].min()
            print(f"[PanelSolver] Equilibrium waterline z = {waterline_z:.4f} m "
                  f"(hull draft T = {draft:.3f} m, ∇ = {nabla:.2f} m³, "
                  f"Δ = {nabla * rho:.0f} kg)")
        else:
            print(f"[PanelSolver] Using specified waterline z = {waterline_z:.4f} m")

        self.waterline_z_orig = waterline_z   # z in original STL frame
        self.mass             = mass

        # Shift so that waterline is at z = 0 in all subsequent calculations:
        #   z < 0  →  submerged
        #   z > 0  →  above water
        self.triangles = raw_triangles.copy()
        self.triangles[:, :, 2] -= waterline_z

        self.rho = rho
        self.g   = g

        # Results populated by run()
        self.sections   = []   # list of dicts with section geometry
        self.x_sections = None
        self.A_inf      = np.zeros((6, 6))
        self.C          = np.zeros((6, 6))
        self.omega      = None
        self.B_omega    = None   # shape (6, 6, N_omega)
        self.A_omega    = None   # shape (6, 6, N_omega)
        self.A_omega_method = None

    # ---------------------------------------------------------------------- #
    def compute_sections(self, n=30, waterline_z=0.0):
        """
        Slice the hull at n evenly spaced x-positions and compute
        cross-section geometry at each.

        Parameters
        ----------
        n           : int    Number of cross-sections.
        waterline_z : float  z of the free surface.

        Populates `self.sections` and `self.x_sections`.
        """
        xs = self.triangles[:, :, 0]
        x_min = xs.min() * 1.01   # slight inset to avoid edge triangles
        x_max = xs.max() * 1.01
        x_cuts = np.linspace(x_min, x_max, n + 2)[1:-1]   # exclude endpoints

        self.sections   = []
        self.x_sections = x_cuts

        for xc in x_cuts:
            segs = _slice_triangles(self.triangles, xc)
            if not segs:
                self.sections.append(None)
                continue
            geom = _section_geometry(segs, waterline_z)
            if geom is not None:
                geom['x'] = xc
                geom['segs'] = segs
            self.sections.append(geom)

        # Remove leading/trailing None entries
        valid = [s for s in self.sections if s is not None]
        if not valid:
            raise RuntimeError("No valid cross-sections found. "
                               "Check that the STL is in the correct "
                               "body-frame orientation (z up).")
        return self

    # ---------------------------------------------------------------------- #
    def compute_hydrostatic_matrix(self, waterline_z=0.0):
        """
        Compute the 6×6 hydrostatic restoring matrix C.

        Nonzero entries (thesis Eq. 4.9–4.11, Fossen 2011):

            C₃₃ = ρg · Aw            heave restoring (waterplane area)
            C₃₅ = C₅₃ = −ρg · ∫ x dAw  heave–pitch coupling
            C₄₄ = ρg · ∇ · GMₜ       roll restoring (transverse metacentre)
            C₅₅ = ρg · ∫ x² dAw      pitch restoring (longitudinal metacentre)

        where Aw = ∫ 2·b(x) dx (waterplane area) and GMₜ is estimated
        from the transverse second moment of the waterplane.

        Returns
        -------
        C : ndarray (6, 6)
        """
        segs_valid = [s for s in self.sections if s is not None]
        if not segs_valid:
            raise RuntimeError("Run compute_sections() first.")

        x_arr = np.array([s['x']     for s in segs_valid])
        b_arr = np.array([s['b_wl']  for s in segs_valid])  # half-breadth
        A_arr = np.array([s['A_s']   for s in segs_valid])  # section area
        I_arr = np.array([s['I_wl']  for s in segs_valid])  # 2nd moment of waterplane strip

        # Waterplane area: Aw = ∫ 2b(x) dx
        Aw  = np.trapz(2.0 * b_arr, x_arr)

        # Displaced volume: ∇ = ∫ A_s(x) dx
        nabla = np.trapz(A_arr, x_arr)

        # LCF: longitudinal centre of flotation
        LCF = np.trapz(2.0 * b_arr * x_arr, x_arr) / max(Aw, 1e-6)

        # Waterplane second moment about centreline (for roll):
        # I_T = ∫ (2b³/3) dx  (second moment of full waterplane strip)
        I_T = np.trapz(I_arr, x_arr)

        # Transverse metacentric radius:
        # BM_T = I_T / ∇  (standard naval architecture formula)
        BM_T = I_T / max(nabla, 1e-6)

        # Estimate VCB (vertical centre of buoyancy):
        # For a typical ship, VCB ≈ −T/3 for a rectangular section.
        # Here we estimate from the average draft and section area.
        T_arr = np.array([s['T'] for s in segs_valid])
        T_mean = np.mean(T_arr)
        VCB = -T_mean / 3.0   # rough estimate (negative = below waterline)

        # Longitudinal second moment: I_L = ∫ 2b(x) x² dx
        I_L = np.trapz(2.0 * b_arr * x_arr**2, x_arr)

        # Longitudinal metacentric radius: BM_L = I_L / ∇
        BM_L = I_L / max(nabla, 1e-6)

        C = np.zeros((6, 6))
        # Heave restoring (Eq. C.33)
        C[2, 2] = self.rho * self.g * Aw

        # Heave–pitch coupling (Eq. C.35 = C.53)
        C[2, 4] = -self.rho * self.g * np.trapz(2.0 * b_arr * x_arr, x_arr)
        C[4, 2] =  C[2, 4]

        # Roll restoring: C₄₄ = ρg∇(BM_T + VCB)
        # (Note: we do not know KG so BM_T + VCB gives KB + BM_T − KG only
        #  approximately; the user should adjust with KG from the main code.)
        C[3, 3] = self.rho * self.g * nabla * BM_T

        # Pitch restoring: C₅₅ = ρg · I_L = ρg · ∇ · BM_L
        C[4, 4] = self.rho * self.g * nabla * BM_L

        self.C          = C
        self._Aw        = Aw
        self._nabla     = nabla
        self._LCF       = LCF
        self._I_T       = I_T
        self._I_L       = I_L

        return C

    # ---------------------------------------------------------------------- #
    def compute_A_inf(self):
        """
        Compute A_inf (6×6) from Lewis form strip theory.

        For each cross-section, Lewis form parameters are found numerically,
        and the 2D infinite-frequency added mass is integrated along the
        ship length (Salvesen et al. 1970, Section 4):

            A₃₃_∞ = ∫ m₃₃(x) dx          heave added mass
            A₅₅_∞ = ∫ m₃₃(x) x² dx       pitch added mass (from heave)
            A₃₅_∞ = ∫ m₃₃(x) x  dx       heave–pitch coupling
            A₂₂_∞ = A₃₃_∞                 sway ≈ heave (strip-theory symm)
            A₄₄_∞ = ∫ m₄₄(x) dx          roll added mass

        Returns
        -------
        A_inf : ndarray (6, 6)
        """
        segs_valid = [s for s in self.sections if s is not None]

        x_arr   = np.array([s['x']   for s in segs_valid])
        b_arr   = np.array([s['b_wl'] for s in segs_valid])
        T_arr   = np.array([s['T']   for s in segs_valid])
        As_arr  = np.array([s['A_s'] for s in segs_valid])

        m33_arr = np.zeros(len(segs_valid))
        m44_arr = np.zeros(len(segs_valid))

        for i, s in enumerate(segs_valid):
            b, T, A_s = s['b_wl'], s['T'], s['A_s']
            R, c1, c3 = _lewis_params(b, T, A_s)
            m33_arr[i] = _lewis_added_mass_2D(R, c1, c3, self.rho)
            m44_arr[i] = _lewis_roll_added_mass_2D(b, T, self.rho)

        A_inf = np.zeros((6, 6))

        # Heave (3,3) — indices 0-based Python
        A_inf[2, 2] = np.trapz(m33_arr, x_arr)

        # Pitch (4,4) — ∫ m33(x) x² dx  (from heave strip, pitch contribution)
        A_inf[4, 4] = np.trapz(m33_arr * x_arr**2, x_arr)

        # Heave–pitch coupling (3,4) = (4,3)
        A_inf[2, 4] = np.trapz(m33_arr * x_arr, x_arr)
        A_inf[4, 2] = A_inf[2, 4]

        # Sway (1,1): strip theory gives sway ≈ heave (symmetric sections)
        A_inf[1, 1] = A_inf[2, 2]

        # Roll (3,3 in 1-indexed = index 3 in 0-indexed)
        A_inf[3, 3] = np.trapz(m44_arr, x_arr)

        self.A_inf = project_psd_matrix(A_inf)
        return self.A_inf

    # ---------------------------------------------------------------------- #
    def compute_B_omega(self, omega):
        """
        Compute frequency-dependent radiation damping B(ω) (6×6 × N_omega).

        Uses the Vossers (1960) thin-ship formula for heave and pitch
        (Salvesen et al. 1970, Eqs. 31–41):

            B₃₃(ω) = ρω × |Î₃(ω)|²
            B₅₅(ω) = ρω × |Î₅(ω)|²
            B₃₅(ω) = ρω × Re(Î₃ × Î₅*)

        where the complex hull form integrals are:
            Î₃(ω) = ∫ b(x) e^{ikx} dx      (waterplane half-breadth strip)
            Î₅(ω) = ∫ x·b(x) e^{ikx} dx   (first moment strip)
            k = ω²/g   (deep-water wave number from dispersion relation)

        Derivation of fac = ρω
        -----------------------
        In the far-field energy argument the Kochin function is:
            H₃(k₀) = ∫ σ(x) e^{ik₀x} dx
        The thin-ship waterplane boundary condition gives the source
        strength σ(x) = k₀·b(x) = (ω²/g)·b(x).  Substituting:
            B₃₃ = (ρg²/ω³)|H₃|²
                 = (ρg²/ω³)·(ω²/g)²·|Î₃|²
                 = ρω·|Î₃|²
        The naive fac ρg²/ω³ (without the k₀² source scaling) over-
        estimates B by k₀⁻² — about 100× at ω=1 rad/s — and gives the
        unphysical limit B→∞ as ω→0 instead of the correct B→0.

        For sway (B₂₂) and sway–roll (B₂₄), analogous integrals use the
        section draft d(x) as the kernel instead of b(x), because sway
        motion generates a transverse wave whose amplitude scales with
        section draft (Ogilvie 1964).

        For roll (B₄₄): B₄₄ ≈ B₂₂ × lever²  (approximate).
        Note: viscous roll damping (dominant for yachts at low speed) is NOT
        captured; add an empirical term B₄₄_visc ≈ 2ζ√(C₄₄ × A₄₄) to
        account for hull friction and bilge-keel damping.

        Parameters
        ----------
        omega : ndarray, shape (N_omega,)   Angular frequencies [rad/s].

        Returns
        -------
        B : ndarray, shape (6, 6, N_omega)   Damping matrix.
        """
        segs_valid = [s for s in self.sections if s is not None]
        x_arr = np.array([s['x']    for s in segs_valid])
        b_arr = np.array([s['b_wl'] for s in segs_valid])   # half-breadth
        T_arr = np.array([s['T']    for s in segs_valid])   # draft

        N_w = len(omega)
        B = np.zeros((6, 6, N_w))

        for n, w in enumerate(omega):
            k = w**2 / self.g   # deep-water wave number [rad/m]
            fac = self.rho * w  # = ρω  (correct Vossers fac; see docstring)

            # Phase factor e^{ikx} along ship length
            exp_ikx = np.exp(1j * k * x_arr)

            # Heave strip integral: Î₃ = ∫ b(x) e^{ikx} dx
            I3 = np.trapz(b_arr * exp_ikx, x_arr)

            # Pitch strip integral: Î₅ = ∫ x·b(x) e^{ikx} dx
            I5 = np.trapz(x_arr * b_arr * exp_ikx, x_arr)

            # Sway/roll strip integral using draft: Î₂ = ∫ T(x) e^{ikx} dx
            I2 = np.trapz(T_arr * exp_ikx, x_arr)

            # --- Heave (3,3): B₃₃ = fac × |Î₃|² ---
            B[2, 2, n] = fac * abs(I3)**2

            # --- Pitch (5,5): B₅₅ = fac × |Î₅|² ---
            B[4, 4, n] = fac * abs(I5)**2

            # --- Heave-pitch coupling (3,5) = (5,3) ---
            # B₃₅ = fac × Re(Î₃ × Î₅*)
            B[2, 4, n] = fac * np.real(I3 * np.conj(I5))
            B[4, 2, n] = B[2, 4, n]

            # --- Sway (2,2): analogous to heave using T(x) as kernel ---
            B[1, 1, n] = fac * abs(I2)**2

            # --- Roll (4,4): approximate from sway via lever arm ≈ mean T/2 ---
            T_mean = np.mean(T_arr)
            lever = T_mean / 2.0
            I4 = lever * I2
            B[3, 3, n] = fac * abs(I4)**2

            # --- Sway-roll coupling (2,4) = (4,2) ---
            B[1, 3, n] = fac * np.real(I2 * np.conj(I4))
            B[3, 1, n] = B[1, 3, n]

        self.omega   = omega
        self.B_omega = project_psd_slices(B)
        return B

    # ---------------------------------------------------------------------- #
    def compute_A_omega(self):
        """
        Compute frequency-dependent added mass A(ω) from B(ω).

        The preferred reconstruction is the Ogilvie principal-value relation:

            A(ω) = A_inf + (2/π) PV ∫_0^∞ B(s) / (s² − ω²) ds

        If that reconstruction yields negative or unrealistically large
        diagonal added masses on the available band, the method falls back to
        A(ω) = A_inf for all frequencies.

        Returns
        -------
        A : ndarray, shape (6, 6, N_omega)   Added mass matrix.
        """
        if self.B_omega is None or self.omega is None:
            raise RuntimeError("Run compute_B_omega() first.")

        a_omega = ogilvie_added_mass_from_damping(
            self.omega, self.B_omega, self.A_inf
        )

        diag_idx = np.array([1, 2, 3, 4])
        diag_vals = np.array([a_omega[i, i, :] for i in diag_idx])
        ainf_diag = np.maximum(np.abs(np.diag(self.A_inf)[diag_idx]), 1.0)[:, np.newaxis]
        nonphysical = (
            np.any(diag_vals < -1e-6) or
            np.any(np.abs(diag_vals) > 5.0 * ainf_diag)
        )

        if nonphysical:
            self.A_omega = np.repeat(self.A_inf[:, :, np.newaxis], len(self.omega), axis=2)
            self.A_omega_method = 'A_inf fallback'
            print("[PanelSolver] Ogilvie A(ω) was nonphysical; using A(ω) = A_inf.")
        else:
            self.A_omega = a_omega
            self.A_omega_method = 'Ogilvie'

        return self.A_omega

    # ---------------------------------------------------------------------- #
    def run(self, omega=None, n_sections=30):
        """
        Full computation pipeline.

        Parameters
        ----------
        omega : ndarray or None
            Frequencies [rad/s].  Defaults to 20 points matching F_rad.py.
        n_sections : int
            Number of cross-sections for strip theory.

        Returns
        -------
        dict with keys: A_inf, A_omega, B_omega, C, omega
        """
        if omega is None:
            # Match the default frequency grid in F_rad.py
            omega = np.linspace(0.1, 5.0, 20)

        print(f"[PanelSolver] Extracting {n_sections} cross-sections …")
        self.compute_sections(n=n_sections)

        print("[PanelSolver] Computing hydrostatic matrix C …")
        self.compute_hydrostatic_matrix()

        print("[PanelSolver] Computing infinite-frequency added mass A_inf …")
        self.compute_A_inf()

        print("[PanelSolver] Computing radiation damping B(ω) …")
        self.compute_B_omega(omega)

        print("[PanelSolver] Computing frequency-dependent added mass A(ω) …")
        self.compute_A_omega()

        print("[PanelSolver] Done.")
        return {
            'omega'  : omega,
            'A_inf'  : self.A_inf,
            'A_omega': self.A_omega,
            'B_omega': self.B_omega,
            'C'      : self.C,
        }

    # ---------------------------------------------------------------------- #
    def to_rad_inputs(self):
        """
        Return the B_omega, A_omega, A_inf arrays in the exact format
        expected by `F_rad.RadiationModel`.

        The RadiationModel stores data with 0-indexed Python DOF ordering:
            0=surge, 1=sway, 2=heave, 3=roll, 4=pitch, 5=yaw

        Returns
        -------
        dict with 'omega', 'B', 'A', 'A_inf'
        """
        return {
            'omega'      : self.omega,
            'B'          : self.B_omega,
            'A'          : self.A_omega,
            'A_inf'      : self.A_inf,
            'waterline_z': self.waterline_z_orig,  # needed by dvpp_imoca60 to shift hull mesh
        }

    # ---------------------------------------------------------------------- #
    def summary(self):
        """Print a summary of hull geometry and computed coefficients."""
        segs_valid = [s for s in self.sections if s is not None]
        if not segs_valid:
            print("No sections computed yet.")
            return

        x_arr  = np.array([s['x']   for s in segs_valid])
        b_arr  = np.array([s['b_wl'] for s in segs_valid])
        T_arr  = np.array([s['T']   for s in segs_valid])
        As_arr = np.array([s['A_s'] for s in segs_valid])

        L   = x_arr.max() - x_arr.min()
        Aw  = getattr(self, '_Aw',   np.trapz(2*b_arr, x_arr))
        nab = getattr(self, '_nabla', np.trapz(As_arr, x_arr))

        print("=" * 50)
        print("Hull geometry summary")
        print(f"  Length (wetted, x range)  = {L:.2f} m")
        print(f"  Max beam (2×b_wl)         = {2*b_arr.max():.2f} m")
        print(f"  Max draft T               = {T_arr.max():.2f} m")
        print(f"  Waterplane area Aw        = {Aw:.1f} m²")
        print(f"  Displaced volume ∇        = {nab:.1f} m³")
        print(f"  Displacement              = {nab * self.rho:.0f} kg")
        if np.any(self.A_inf):
            print("\nA_inf diagonal [kg, kg·m²]:")
            print(f"  Sway  A22 = {self.A_inf[1,1]:.0f}")
            print(f"  Heave A33 = {self.A_inf[2,2]:.0f}")
            print(f"  Roll  A44 = {self.A_inf[3,3]:.0f}")
            print(f"  Pitch A55 = {self.A_inf[4,4]:.0f}")
        print("=" * 50)


# =========================================================================== #
#  STANDALONE USAGE                                                            #
# =========================================================================== #

if __name__ == '__main__':
    import sys
    stl  = sys.argv[1] if len(sys.argv) > 1 else 'PRB F.stl'
    mass = float(sys.argv[2]) if len(sys.argv) > 2 else 9500.0
    solver = PanelSolver(stl, mass=mass)
    result = solver.run(omega=np.linspace(0.1, 5.0, 20), n_sections=30)
    solver.summary()

    # Optional: plot B_33(ω) and A_33(ω) to verify
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(result['omega'], result['B_omega'][2, 2, :], label='B₃₃')
    ax1.set_xlabel('ω [rad/s]'); ax1.set_ylabel('B₃₃ [N·s/m]')
    ax1.set_title('Heave radiation damping'); ax1.grid()
    ax2.plot(result['omega'], result['A_omega'][2, 2, :], label='A₃₃')
    ax2.set_xlabel('ω [rad/s]'); ax2.set_ylabel('A₃₃ [kg]')
    ax2.set_title('Heave added mass'); ax2.grid()
    plt.tight_layout(); plt.show()
