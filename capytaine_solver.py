#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capytaine_solver.py — 3-D BEM radiation solver (drop-in for PanelSolver).

Uses the Capytaine library (from ECN / Ecole Centrale de Nantes) to solve the
full 3-D diffraction/radiation problem for all 6 rigid-body DOFs.

Public interface is identical to PanelSolver so the rest of the DVPP code
(F_rad.py, dvpp_imoca60.py, ui_dvpp.py) requires no changes.

Install:  pip install capytaine
Docs:     https://capytaine.github.io/

If the current interpreter cannot load Capytaine's Fortran extension, set
the environment variable CAPYTAINE_PYTHON to a different Python executable
that has a working Capytaine installation. The solver will run Capytaine in
that interpreter and return the coefficients to the main DVPP process.
"""

import numpy as np
import tempfile
import os
import subprocess
import sys
from functools import lru_cache

from mesh_utils import TriangleMesh, find_equilibrium_waterline
from radiation_utils import (
    estimate_ainf_from_added_mass,
    project_psd_slices,
    symmetrize_matrix_slices,
)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKER = os.path.join(_THIS_DIR, 'capytaine_worker.py')
_CAPYTAINE_PYTHON = os.environ.get('CAPYTAINE_PYTHON')


def _find_equilibrium_waterline(triangles, mass, rho):
    return find_equilibrium_waterline(triangles, mass=mass, rho=rho)

def _normalize_python_executable(python_executable):
    if not python_executable:
        return None
    return os.path.abspath(os.path.expanduser(python_executable))


@lru_cache(maxsize=8)
def _probe_external_capytaine(python_executable):
    """Return True if the given interpreter can actually solve a tiny BEM case."""
    python_executable = _normalize_python_executable(python_executable)
    if not python_executable:
        return False
    try:
        probe = subprocess.run(
            [python_executable, _WORKER, '--probe'],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except Exception:
        return False
    return probe.returncode == 0


def capytaine_available(python_executable=None):
    """
    Return True if Capytaine can be used either in-process or via an external
    interpreter provided explicitly or through CAPYTAINE_PYTHON.
    """
    python_executable = _normalize_python_executable(
        python_executable or os.environ.get('CAPYTAINE_PYTHON')
    )

    if python_executable:
        if (
            '_INPROCESS_CAPYTAINE_OK' in globals() and
            _INPROCESS_CAPYTAINE_OK and
            python_executable == os.path.abspath(sys.executable)
        ):
            return True
        return _probe_external_capytaine(python_executable)

    return '_INPROCESS_CAPYTAINE_OK' in globals() and _INPROCESS_CAPYTAINE_OK


try:
    import capytaine as cpt
    # Quick smoke-test: verify the Fortran extension actually works on this
    # numpy/Python combination.  Capytaine pre-built wheels are incompatible
    # with numpy >= 1.24 (Python 3.12) due to f2py intent(inplace/cache)
    # changes.  The workaround is: conda install -c conda-forge capytaine
    # or use a Python 3.10 / numpy < 1.24 environment.
    _body_test = cpt.FloatingBody(mesh=cpt.mesh_sphere(radius=1.0), name='_test')
    _body_test.add_all_rigid_body_dofs()
    if hasattr(_body_test, 'immersed_part'):
        _body_test = _body_test.immersed_part()
    elif hasattr(_body_test, 'keep_immersed_part'):
        _body_test.keep_immersed_part()
    _pb = cpt.RadiationProblem(body=_body_test, radiating_dof='Heave', omega=1.0)
    try:
        cpt.BEMSolver().solve(_pb, keep_details=False)
        _INPROCESS_CAPYTAINE_OK = True
    except TypeError:
        # f2py intent(inout|inplace|cache) error → numpy incompatibility
        _INPROCESS_CAPYTAINE_OK = False
        import warnings, numpy as np
        warnings.warn(
            f"Capytaine Fortran extension is incompatible with NumPy "
            f"{np.__version__} / Python {__import__('sys').version_info[:2]}. "
            "Fix: provide a working Capytaine interpreter via CAPYTAINE_PYTHON "
            "or use a dedicated Python/NumPy environment for Capytaine.",
            RuntimeWarning, stacklevel=1,
        )
    del _body_test, _pb
except ImportError:
    _INPROCESS_CAPYTAINE_OK = False

_CAPYTAINE_OK = capytaine_available(_CAPYTAINE_PYTHON)

# DOF name order used throughout DVPP (0-indexed)
_DOFS = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']


class CapytaineSolver:
    """
    3-D BEM radiation solver wrapping Capytaine.

    Parameters
    ----------
    stl_path    : str    Path to the hull STL (body frame, z = up).
    rho         : float  Water density [kg/m³].
    g           : float  Gravitational acceleration [m/s²].
    mass        : float  Vessel mass [kg].  Used to find the hydrostatic
                         equilibrium waterline when waterline_z is not given.
    waterline_z : float  Optional: waterline z in the STL frame [m].
                         If None it is found via hydrostatic bisection.
    python_executable : str or None
                         Optional path to a different Python interpreter with
                         a working Capytaine installation.

    After calling run(), exposes the same attributes as PanelSolver:
        .omega       (N_omega,)        angular frequencies [rad/s]
        .B_omega     (6, 6, N_omega)   radiation damping [N·s/m …]
        .A_omega     (6, 6, N_omega)   frequency-dependent added mass [kg …]
        .A_inf       (6, 6)            infinite-frequency added mass
        .C           (6, 6)            hydrostatic restoring matrix
        .sections    None              (not produced by 3-D BEM)
        .waterline_z_orig float        waterline z in original STL frame
    """

    def __init__(self, stl_path, rho=1025.0, g=9.81, mass=None,
                 waterline_z=None, python_executable=None):
        resolved_python = _normalize_python_executable(
            python_executable or os.environ.get('CAPYTAINE_PYTHON')
        )

        if not capytaine_available(resolved_python):
            raise ImportError(
                "Capytaine is not usable here. Provide a working interpreter "
                "via python_executable or CAPYTAINE_PYTHON."
            )

        self.stl_path = stl_path
        self.rho      = float(rho)
        self.g        = float(g)
        self.mass     = mass
        self.sections = None          # not applicable for 3-D BEM
        self.python_executable = resolved_python
        self._use_subprocess = (
            self.python_executable is not None and
            self.python_executable != os.path.abspath(sys.executable)
        )

        if waterline_z is not None:
            self.waterline_z_orig = float(waterline_z)
        else:
            m = TriangleMesh.from_file(stl_path)
            self.waterline_z_orig, _ = _find_equilibrium_waterline(
                m.vectors.copy(), mass=float(mass), rho=rho
            )

    def _build_shifted_stl(self):
        raw = TriangleMesh.from_file(self.stl_path)
        vecs = raw.vectors.copy()
        vecs[:, :, 2] -= self.waterline_z_orig

        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            tmp_path = f.name
        shifted = TriangleMesh.from_triangles(vecs)
        shifted.save(tmp_path)
        return tmp_path

    # ── Main solver ───────────────────────────────────────────────────────────

    def run(self, omega):
        """
        Solve the 3-D radiation problem for all 6 DOFs at each frequency.

        Parameters
        ----------
        omega : array-like  Angular frequencies [rad/s].

        Returns
        -------
        self
        """
        if self._use_subprocess:
            return self._run_subprocess(omega)
        return self._run_in_process(omega)

    def _run_in_process(self, omega):
        self.omega = np.asarray(omega, dtype=float)
        N_omega    = len(self.omega)

        # ── Build a shifted-mesh temp file (waterline → z = 0) ───────────────
        tmp_path = self._build_shifted_stl()

        # ── Create Capytaine floating body ────────────────────────────────────
        # load_mesh API: 2.x accepts `name`, 3.x does not
        try:
            cap_mesh = cpt.load_mesh(tmp_path, file_format='stl', name='hull')
        except TypeError:
            cap_mesh = cpt.load_mesh(tmp_path)
        os.unlink(tmp_path)

        body = cpt.FloatingBody(mesh=cap_mesh, name='hull')
        body.add_all_rigid_body_dofs()

        # API changed in Capytaine 3.x: keep_immersed_part → immersed_part (returns new body)
        if hasattr(body, 'immersed_part'):
            body = body.immersed_part()
        elif hasattr(body, 'keep_immersed_part'):
            body.keep_immersed_part()

        # Heal mesh (removed in Capytaine 3.x)
        try:
            body.mesh.heal_mesh()
        except AttributeError:
            pass   # Capytaine 3.x manages mesh quality internally

        print(f"  [Capytaine] {body.mesh.nb_faces} panels after clipping")

        # ── Radiation BEM ─────────────────────────────────────────────────────
        bem_solver = cpt.BEMSolver()
        problems   = self._make_problems(body)

        print(f"  [Capytaine] solving {len(problems)} radiation problems …")
        results = bem_solver.solve_all(problems, keep_details=False)

        # Assemble xarray Dataset (Capytaine 2.x uses assemble_dataset)
        try:
            ds = cpt.assemble_dataset(results)
        except AttributeError:
            ds = cpt.assembleDataset(results)   # fallback for 1.x

        # ── Fill 6×6×N_omega matrices ─────────────────────────────────────────
        self.A_omega = np.zeros((6, 6, N_omega))
        self.B_omega = np.zeros((6, 6, N_omega))

        for i, di in enumerate(_DOFS):
            for j, dj in enumerate(_DOFS):
                try:
                    self.A_omega[i, j] = (
                        ds['added_mass']
                        .sel(radiating_dof=di, influenced_dof=dj)
                        .values
                    )
                    self.B_omega[i, j] = (
                        ds['radiation_damping']
                        .sel(radiating_dof=di, influenced_dof=dj)
                        .values
                    )
                except Exception:
                    pass  # cross-coupled term might be absent for symmetric hulls

        self.B_omega = project_psd_slices(self.B_omega)
        self.A_omega = symmetrize_matrix_slices(self.A_omega)

        # ── A_inf from the high-frequency asymptote A(ω) = A_inf + O(ω^-2) ──
        self.A_inf = estimate_ainf_from_added_mass(self.omega, self.A_omega)

        # ── Hydrostatic restoring matrix ──────────────────────────────────────
        self.C = np.zeros((6, 6))
        self.C = self._compute_hydrostatics(body)

        return self

    def _run_subprocess(self, omega):
        """Run Capytaine in an external Python interpreter."""
        self.omega = np.asarray(omega, dtype=float)
        shifted_stl = self._build_shifted_stl()

        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f_omega:
            omega_path = f_omega.name
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f_out:
            out_path = f_out.name

        np.save(omega_path, self.omega)

        try:
            proc = subprocess.run(
                [
                    self.python_executable, _WORKER,
                    '--stl', shifted_stl,
                    '--omega', omega_path,
                    '--output', out_path,
                    '--rho', str(self.rho),
                    '--g', str(self.g),
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=1800,
            )
            if proc.returncode != 0:
                detail = proc.stderr.strip() or proc.stdout.strip()
                raise RuntimeError(
                    f"External Capytaine process failed via {self.python_executable}: {detail}"
                )

            data = np.load(out_path)
            self.omega = data['omega']
            self.A_omega = symmetrize_matrix_slices(data['A_omega'])
            self.B_omega = project_psd_slices(data['B_omega'])
            self.A_inf = estimate_ainf_from_added_mass(self.omega, self.A_omega)
            self.C = np.array(data['C'])
            return self
        finally:
            for path in [shifted_stl, omega_path, out_path]:
                if path and os.path.exists(path):
                    os.unlink(path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_problems(self, body):
        """Create RadiationProblem list, handling Capytaine 1.x and 2.x APIs."""
        problems = []
        for dof in _DOFS:
            for w in self.omega:
                try:
                    # Capytaine 2.x — rho/g are keyword args on the problem
                    pb = cpt.RadiationProblem(
                        body=body, radiating_dof=dof, omega=float(w),
                        rho=self.rho, g=self.g,
                    )
                except TypeError:
                    # Capytaine 1.x — used water_density keyword
                    pb = cpt.RadiationProblem(
                        body=body, radiating_dof=dof, omega=float(w),
                        water_density=self.rho, g=self.g,
                    )
                problems.append(pb)
        return problems

    def _compute_hydrostatics(self, body):
        """Compute 6×6 hydrostatic stiffness.  Falls back to C33 = ρgAw."""
        C = np.zeros((6, 6))
        try:
            # compute_hydrostatic_stiffness returns a 6×6 or 3×3 DataArray
            # (heave/roll/pitch in Capytaine convention).  Keyword args in 2.x.
            Craw = body.compute_hydrostatic_stiffness(
                rho=self.rho, g=self.g
            ).values
            if Craw.shape == (3, 3):
                # Map heave→2, roll→3, pitch→4
                for ii, gi in enumerate([2, 3, 4]):
                    for jj, gj in enumerate([2, 3, 4]):
                        C[gi, gj] = Craw[ii, jj]
            else:
                C = np.array(Craw)
            return C
        except Exception as e:
            print(f"  [Capytaine] hydrostatic stiffness fallback ({e})")

        # Fallback: estimate C33 from waterplane area
        try:
            from scipy.spatial import ConvexHull
            verts = body.mesh.vertices
            # Waterplane is near z ≈ 0 after our shift
            wl_mask  = np.abs(verts[:, 2]) < 0.05 * (verts[:, 2].max() -
                                                       verts[:, 2].min())
            wl_verts = verts[wl_mask]
            if len(wl_verts) >= 3:
                hull = ConvexHull(wl_verts[:, :2])
                C[2, 2] = self.rho * self.g * hull.volume  # Aw
        except Exception as e2:
            print(f"  [Capytaine] C33 fallback failed: {e2}")
        return C

    # ── Interface identical to PanelSolver ────────────────────────────────────

    def to_rad_inputs(self):
        """Return dict compatible with F_rad.py rad_inputs."""
        return {
            'omega'      : self.omega,
            'B'          : self.B_omega,
            'A'          : self.A_omega,
            'A_inf'      : self.A_inf,
            'waterline_z': self.waterline_z_orig,
        }
