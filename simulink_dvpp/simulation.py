#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from mesh_utils import TriangleMesh, find_equilibrium_waterline

from .appendages import appendage_forces
from .constants import GRAVITY, KNOTS_PER_MPS
from .diffraction import diffraction_force
from .gravity import gravity_force
from .hydrostatics import build_hydrostatic_mesh_cache, hydrostatic_forces_and_moments
from .kinematics import body_to_ned_velocity
from .foils import foil_forces
from .mass import mass_and_added_mass
from .non_inertial import rigid_body_coriolis
from .radiation import CumminsRadiationModel
from .resistance import resistance_force
from .sails import SailModel
from .types import AppendageInputs, DVPPState, FoilInputs, ForceBreakdown, SailInputs
from .waves import wave_properties


SIMULINK_DISPLACEMENT_KG = 12_500.0
CALM_WAVE = np.zeros((1, 4), dtype=float)
WAVE_MODE_FLAGS = {
    "calm": None,
    "regular": 0,
    "irregular_components": 1,
    "irregular_spectrum": 2,
}
RESISTANCE_BREAKPOINTS = np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], dtype=float)
RESISTANCE_TABLES = np.array([
    [-0.0005, -0.0003, -0.0002, -0.0009, -0.0026, -0.0064, -0.0218, -0.0388, -0.0347, -0.0361, 0.0008, 0.0108, 0.1023],
    [0.0023, 0.0059, -0.0156, 0.0016, -0.0567, -0.4034, -0.5261, -0.5986, -0.4764, 0.0037, 0.3728, -0.1238, 0.7726],
    [-0.0086, -0.0064, 0.0031, 0.0337, 0.0446, -0.125, -0.2945, -0.3038, -0.2361, -0.296, -0.3667, -0.2026, 0.504],
    [-0.0015, 0.007, -0.0021, -0.0285, -0.1091, 0.0273, 0.2485, 0.6033, 0.8726, 0.9661, 1.3957, 1.1282, 1.7867],
    [0.0061, 0.0014, -0.007, -0.0367, -0.0707, -0.1341, -0.2428, -0.043, 0.4219, 0.6123, 1.0343, 1.1836, 2.1934],
    [0.001, 0.0013, 0.0148, 0.0218, 0.0914, 0.3578, 0.6293, 0.8332, 0.899, 0.7534, 0.323, 0.4973, -1.5479],
    [0.0001, 0.0005, 0.001, 0.0015, 0.0021, 0.0045, 0.0081, 0.0106, 0.0096, 0.01, 0.0072, 0.0038, -0.0115],
    [0.0052, -0.002, -0.0043, -0.0172, -0.0078, 0.1115, 0.2086, 0.1336, -0.2272, -0.3352, -0.4632, -0.4477, -0.0977],
], dtype=float)


@dataclass
class SimulationOutputs:
    t_vals: np.ndarray
    y_vals: np.ndarray
    force_history: list[ForceBreakdown]
    waterline_z: float


def apparent_wind_body(TWS_kn, TWA_deg, nu):
    wind_body = (float(TWS_kn) / KNOTS_PER_MPS) * np.array([
        np.cos(np.deg2rad(TWA_deg)),
        np.sin(np.deg2rad(TWA_deg)),
        0.0,
    ], dtype=float)
    return wind_body + np.array([nu[0], nu[1], 0.0], dtype=float)


def resistance_coefficients(froude_number):
    fn = float(np.clip(froude_number, RESISTANCE_BREAKPOINTS[0], RESISTANCE_BREAKPOINTS[-1]))
    return np.array([np.interp(fn, RESISTANCE_BREAKPOINTS, row) for row in RESISTANCE_TABLES], dtype=float)


def initial_state_vector(
    eta0=None,
    nu0=None,
    state0=None,
):
    if state0 is not None:
        state = np.asarray(state0, dtype=float).reshape(12)
        return state.copy()

    eta = np.zeros(6, dtype=float) if eta0 is None else np.asarray(eta0, dtype=float).reshape(6)
    nu = np.zeros(6, dtype=float) if nu0 is None else np.asarray(nu0, dtype=float).reshape(6)
    return np.concatenate([eta, nu])


@lru_cache(maxsize=8)
def _load_hull_geometry_cached(hull_file, waterline_z, mass):
    hull = TriangleMesh.from_file(hull_file)
    if waterline_z is None:
        waterline_z, _ = find_equilibrium_waterline(hull.vectors.copy(), mass=mass)
    hull.vectors[:, :, 2] -= float(waterline_z)

    vertices, inverse = np.unique(hull.vectors.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return vertices, faces, float(waterline_z)


def load_hull_geometry(hull_file, waterline_z=None, mass=SIMULINK_DISPLACEMENT_KG):
    vertices, faces, cached_waterline = _load_hull_geometry_cached(
        str(hull_file),
        None if waterline_z is None else float(waterline_z),
        float(mass),
    )
    return vertices.copy(), faces.copy(), float(cached_waterline)


class SimulinkDVPP6DOF:
    def __init__(self, hull_file, dt=0.05, waterline_z=None, mass=SIMULINK_DISPLACEMENT_KG,
                 mainsail_geom=None, jib_geom=None, headsail_geom=None, radiation_data=None,
                 resistance_mode=4):
        self.dt = float(dt)
        self.vertices, self.faces, self.waterline_z = load_hull_geometry(hull_file, waterline_z=waterline_z, mass=mass)
        self.hydro_cache = build_hydrostatic_mesh_cache(self.vertices, self.faces)
        self.radiation = CumminsRadiationModel(dt=dt)
        if radiation_data is not None:
            self.radiation.inject_solver_data(**radiation_data)
        self.sails = SailModel(mainsail_geom=mainsail_geom, jib_geom=jib_geom, headsail_geom=headsail_geom)
        self.mass = float(mass)
        self.zero = 0.0
        self.resistance_mode = int(resistance_mode)

    @staticmethod
    def _check_finite(label, values):
        values = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{label} contains non-finite values")

    def reset(self):
        self.radiation.reset()
        self.sails.reset()

    def _wave_state(self, t, eta, nu, wave_mode, wave_ramp, wave_angle_deg, wave_wind_speed_kn):
        mode = str(wave_mode).lower()
        if mode not in WAVE_MODE_FLAGS:
            raise ValueError(f"Unknown wave_mode: {wave_mode}")
        if WAVE_MODE_FLAGS[mode] is None or float(wave_ramp) <= 0.0:
            return CALM_WAVE
        if mode in {"regular", "irregular_spectrum"} and float(wave_wind_speed_kn) <= 0.0:
            raise ValueError("wave_wind_speed_kn must be positive for the selected wave model")
        wave, _, _, _, _ = wave_properties(
            ramp=float(wave_ramp),
            wind_speed=float(wave_wind_speed_kn),
            eta=np.asarray(eta, dtype=float),
            eta_dot=np.asarray(nu, dtype=float),
            angle_deg=float(wave_angle_deg),
            time=float(t),
            irregular=WAVE_MODE_FLAGS[mode],
        )
        return np.asarray(wave, dtype=float)

    def _force_breakdown(
        self,
        t,
        eta,
        nu,
        twa_deg,
        tws_kn,
        reef,
        flat,
        c0_enabled,
        rake_foil_deg,
        use_foil,
        wave_mode,
        wave_ramp,
        wave_angle_deg,
        wave_wind_speed_kn,
        keel_angle_deg,
    ):
        wave = self._wave_state(t, eta, nu, wave_mode, wave_ramp, wave_angle_deg, wave_wind_speed_kn)
        hydro = hydrostatic_forces_and_moments(
            self.vertices,
            self.faces,
            eta,
            t,
            self.zero,
            wave,
            cache=self.hydro_cache,
        )
        aws_body = apparent_wind_body(tws_kn, twa_deg, nu)
        awa_deg = abs(np.rad2deg(np.arctan2(aws_body[1], aws_body[0])))

        sail_outputs = self.sails.evaluate(
            SailInputs(
                aws_body=aws_body,
                eta=eta,
                flat=float(flat),
                c0_enabled=float(c0_enabled),
                twa_deg=float(twa_deg),
                reef=float(reef),
                nu=nu,
            )
        )
        appendage_outputs = appendage_forces(
            AppendageInputs(
                eta_dot=nu,
                eta=eta,
                wave=wave,
                time=float(t),
                keel_angle_deg=float(keel_angle_deg),
                zero=self.zero,
                wl_z_shift=-self.waterline_z,
            )
        )
        foil_outputs = foil_forces(
            FoilInputs(
                eta_dot=nu,
                eta=eta,
                rake_foil_deg=float(rake_foil_deg),
                wave=wave,
                time=float(t),
                zero=self.zero,
                wl_z_shift=-self.waterline_z,
            )
        )
        if use_foil:
            foil_force = foil_outputs.total_force
            a_foil = foil_outputs.added_mass
        else:
            foil_force = foil_outputs.gravity_force
            a_foil = 0.0

        if hydro.LWL > 1e-9:
            froude_number = nu[0] / np.sqrt(GRAVITY * hydro.LWL)
        else:
            froude_number = 0.15
        resistance_coeffs = resistance_coefficients(froude_number)
        resistance_outputs, _, _ = resistance_force(
            eta=eta,
            eta_d=nu,
            hydrostat=hydro.res_hs,
            LWL=hydro.LWL,
            WB=hydro.WB,
            awa_deg=awa_deg,
            submerged_area=hydro.submerged_facet_areas,
            a_array=resistance_coeffs,
            mode=self.resistance_mode,
        )
        # res_hs is buoyancy force in N (≈ displacement × g ≈ 122 kN at design).
        # The radiation/diffraction scaling formula uses this to normalise to a
        # design-displacement factor close to 1.0.  count_sum (integer panel count)
        # must NOT be passed here — it would reduce both forces to ~10 % of correct.
        radiation_force = self.radiation.get_force(nu, U=nu[0], count_sum=hydro.res_hs)
        gravity = gravity_force(eta, total_mass=self.mass)
        diffraction = diffraction_force(wave, eta, nu, float(t), hydro.res_hs)

        forces = ForceBreakdown(
            gravity=gravity,
            hydrostatics=hydro.F_hs,
            resistance=resistance_outputs,
            radiation=radiation_force,
            diffraction=diffraction,
            sails=sail_outputs.total_force,
            appendages=appendage_outputs.total_force,
            foils=foil_force,
        )

        wave_mass_input = wave[0, :]
        mass_breakdown = mass_and_added_mass(
            sail_outputs.a44_sails,
            sail_outputs.a22_sails,
            a_foil,
            wave_mass_input,
            eta,
            self.zero,
            hydro.res_hs,
            a_inf_override=self.radiation.get_A_inf(),
            total_mass=self.mass,
        )
        self._check_finite("force vector", forces.total_without_radiation_sign())
        self._check_finite("mass matrix", mass_breakdown.M)
        return forces, mass_breakdown.M, sail_outputs

    def state_derivative(
        self,
        t,
        state_vec,
        twa_deg,
        tws_kn,
        reef,
        flat,
        c0_enabled,
        rake_foil_deg,
        use_foil,
        wave_mode,
        wave_ramp,
        wave_angle_deg,
        wave_wind_speed_kn,
        keel_angle_deg,
    ):
        eta = np.asarray(state_vec[:6], dtype=float)
        nu = np.asarray(state_vec[6:], dtype=float)
        forces, mass, sail_outputs = self._force_breakdown(
            t=t,
            eta=eta,
            nu=nu,
            twa_deg=twa_deg,
            tws_kn=tws_kn,
            reef=reef,
            flat=flat,
            c0_enabled=c0_enabled,
            rake_foil_deg=rake_foil_deg,
            use_foil=use_foil,
            wave_mode=wave_mode,
            wave_ramp=wave_ramp,
            wave_angle_deg=wave_angle_deg,
            wave_wind_speed_kn=wave_wind_speed_kn,
            keel_angle_deg=keel_angle_deg,
        )
        nu_dot = np.linalg.solve(
            mass,
            -rigid_body_coriolis(nu, total_mass=self.mass) @ nu - forces.radiation + forces.total_without_radiation_sign(),
        )
        nu_dot[5] = 0.0   # 5-DOF: no yaw acceleration
        eta_dot = body_to_ned_velocity(nu, eta)
        eta_dot[5] = 0.0  # 5-DOF: suppress kinematic psi_dot from roll/pitch coupling
        self._check_finite("state derivative", nu_dot)
        self._check_finite("kinematics", eta_dot)
        _ = sail_outputs
        return np.concatenate([eta_dot, nu_dot]), forces

    def simulate(
        self,
        twa_deg=70.0,
        tws_kn=15.0,
        reef=1.0,
        flat=1.0,
        c0_enabled=False,
        use_foil=True,
        rake_foil_deg=40.0,
        t_end=40.0,
        eta0=None,
        nu0=None,
        state0=None,
        wave_mode="calm",
        wave_ramp=0.0,
        wave_angle_deg=180.0,
        wave_wind_speed_kn=15.0,
        keel_angle_deg=0.0,
    ):
        self.reset()
        y0 = initial_state_vector(eta0=eta0, nu0=nu0, state0=state0)
        self.radiation.update(y0[6:12])

        t_vals = np.arange(0.0, float(t_end) + self.dt, self.dt)
        y_vals = np.zeros((len(t_vals), 12), dtype=float)
        force_history = []
        y_vals[0, :] = y0

        for index in range(1, len(t_vals)):
            t_n = t_vals[index - 1]
            y_n = y_vals[index - 1, :]

            def rhs(t_stage, y_stage):
                derivative, _forces = self.state_derivative(
                    t_stage,
                    y_stage,
                    twa_deg=twa_deg,
                    tws_kn=tws_kn,
                    reef=reef,
                    flat=flat,
                    c0_enabled=c0_enabled,
                    rake_foil_deg=rake_foil_deg,
                    use_foil=use_foil,
                    wave_mode=wave_mode,
                    wave_ramp=wave_ramp,
                    wave_angle_deg=wave_angle_deg,
                    wave_wind_speed_kn=wave_wind_speed_kn,
                    keel_angle_deg=keel_angle_deg,
                )
                return derivative

            k1 = rhs(t_n, y_n)
            k2 = rhs(t_n + self.dt / 2.0, y_n + self.dt * k1 / 2.0)
            k3 = rhs(t_n + self.dt / 2.0, y_n + self.dt * k2 / 2.0)
            k4 = rhs(t_n + self.dt, y_n + self.dt * k3)
            y_new = y_n + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            self._check_finite("state", y_new)

            # Commit radiation history with y_new BEFORE evaluating accepted
            # forces so that the logged forces and the next step's k1 both see
            # the same updated history (consistent with the accepted state).
            self.radiation.update(y_new[6:12])
            _, accepted_forces = self.state_derivative(
                t_n + self.dt,
                y_new,
                twa_deg=twa_deg,
                tws_kn=tws_kn,
                reef=reef,
                flat=flat,
                c0_enabled=c0_enabled,
                rake_foil_deg=rake_foil_deg,
                use_foil=use_foil,
                wave_mode=wave_mode,
                wave_ramp=wave_ramp,
                wave_angle_deg=wave_angle_deg,
                wave_wind_speed_kn=wave_wind_speed_kn,
                keel_angle_deg=keel_angle_deg,
            )
            accepted_sails = self.sails.evaluate(
                SailInputs(
                    aws_body=apparent_wind_body(tws_kn, twa_deg, y_new[6:12]),
                    eta=y_new[:6],
                    flat=float(flat),
                    c0_enabled=float(c0_enabled),
                    twa_deg=float(twa_deg),
                    reef=float(reef),
                    nu=y_new[6:12],
                )
            )
            self.sails.commit(accepted_sails)
            y_vals[index, :] = y_new
            force_history.append(accepted_forces)

        return SimulationOutputs(
            t_vals=t_vals,
            y_vals=y_vals,
            force_history=force_history,
            waterline_z=self.waterline_z,
        )


def run_simulation(**kwargs):
    dt = kwargs.pop("dt", 0.05)
    sim = SimulinkDVPP6DOF(
        hull_file=kwargs.pop("hull_file"),
        dt=dt,
        waterline_z=kwargs.pop("waterline_z", None),
        mass=kwargs.pop("mass", SIMULINK_DISPLACEMENT_KG),
        mainsail_geom=kwargs.pop("mainsail_geom", None),
        jib_geom=kwargs.pop("jib_geom", None),
        headsail_geom=kwargs.pop("headsail_geom", None),
        radiation_data=kwargs.pop("radiation_data", None),
        resistance_mode=kwargs.pop("resistance_mode", 4),
    )
    return sim.simulate(**kwargs)
