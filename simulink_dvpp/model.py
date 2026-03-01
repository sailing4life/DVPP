#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field

import numpy as np

from .appendages import appendage_forces
from .constants import REFERENCE_TOTAL_MASS
from .diffraction import diffraction_force
from .foils import foil_forces
from .gravity import gravity_force
from .kinematics import body_to_ned_velocity
from .mass import mass_and_added_mass
from .non_inertial import rigid_body_coriolis
from .radiation import CumminsRadiationModel
from .sails import SailModel
from .types import AppendageInputs, DVPPState, FoilInputs, ForceBreakdown, MassBreakdown, SailInputs


@dataclass
class SimulinkDVPPModel:
    """
    Clean Python shell around the original Simulink subsystem equations.

    This class intentionally mirrors the block-diagram layout rather than the
    current handwritten Python DVPP. Appendage, foil, sail, and hydrostatic
    subsystems can be plugged in progressively.
    """

    radiation: CumminsRadiationModel = field(default_factory=CumminsRadiationModel)
    sails: SailModel = field(default_factory=SailModel)
    total_mass: float = REFERENCE_TOTAL_MASS

    def mass_breakdown(self, A44_sails, A22_sails, A_foil, wave, eta, zero, res_hs):
        return mass_and_added_mass(A44_sails, A22_sails, A_foil, wave, eta, zero, res_hs, total_mass=self.total_mass)

    def translated_subsystems(
        self,
        state: DVPPState,
        time,
        wave,
        zero,
        sail_inputs: SailInputs | None = None,
        appendage_inputs: AppendageInputs | None = None,
        foil_inputs: FoilInputs | None = None,
    ):
        sail_output = None if sail_inputs is None else self.sails.evaluate(sail_inputs)
        appendage_output = None if appendage_inputs is None else appendage_forces(appendage_inputs)
        foil_output = None if foil_inputs is None else foil_forces(foil_inputs)
        _ = (state, time, wave, zero)
        return sail_output, appendage_output, foil_output

    def assembled_step(
        self,
        state: DVPPState,
        count_sum,
        wave,
        time,
        zero,
        res_hs=0.0,
        hydrostatics=None,
        resistance=None,
        sail_inputs: SailInputs | None = None,
        appendage_inputs: AppendageInputs | None = None,
        foil_inputs: FoilInputs | None = None,
    ):
        sail_output, appendage_output, foil_output = self.translated_subsystems(
            state=state,
            time=time,
            wave=wave,
            zero=zero,
            sail_inputs=sail_inputs,
            appendage_inputs=appendage_inputs,
            foil_inputs=foil_inputs,
        )

        sail_force = None if sail_output is None else sail_output.total_force
        appendage_force = None if appendage_output is None else appendage_output.total_force
        foil_force = None if foil_output is None else foil_output.total_force
        forces = self.force_breakdown(
            state=state,
            count_sum=count_sum,
            wave=wave,
            time=time,
            zero=zero,
            hydrostatics=hydrostatics,
            resistance=resistance,
            sails=sail_force,
            appendages=appendage_force,
            foils=foil_force,
        )

        wave_array = np.asarray(wave, dtype=float)
        wave_mass_input = wave_array[0, :] if wave_array.ndim == 2 else wave_array
        a22_sails = 0.0 if sail_output is None else sail_output.a22_sails
        a44_sails = 0.0 if sail_output is None else sail_output.a44_sails
        a_foil = 0.0 if foil_output is None else foil_output.added_mass
        mass = self.mass_breakdown(a44_sails, a22_sails, a_foil, wave_mass_input, state.eta, zero, res_hs)

        return mass, forces, sail_output, appendage_output, foil_output

    def force_breakdown(
        self,
        state: DVPPState,
        count_sum,
        wave,
        time,
        zero=0.0,
        sail_inputs: SailInputs | None = None,
        appendage_inputs: AppendageInputs | None = None,
        foil_inputs: FoilInputs | None = None,
        hydrostatics=None,
        resistance=None,
        sails=None,
        appendages=None,
        foils=None,
    ):
        eta = np.asarray(state.eta, dtype=float)
        nu = np.asarray(state.nu, dtype=float)

        gravity = gravity_force(eta, total_mass=self.total_mass)
        hydrostatics = np.zeros(6) if hydrostatics is None else np.asarray(hydrostatics, dtype=float)
        resistance = np.zeros(6) if resistance is None else np.asarray(resistance, dtype=float)
        if sails is None and sail_inputs is not None:
            sails = self.sails.evaluate(sail_inputs).total_force
        sails = np.zeros(6) if sails is None else np.asarray(sails, dtype=float)

        if appendages is None and appendage_inputs is not None:
            appendages = appendage_forces(appendage_inputs).total_force
        appendages = np.zeros(6) if appendages is None else np.asarray(appendages, dtype=float)

        if foils is None and foil_inputs is not None:
            foils = foil_forces(foil_inputs).total_force
        foils = np.zeros(6) if foils is None else np.asarray(foils, dtype=float)

        diffraction = diffraction_force(wave, eta, nu, time, count_sum)
        radiation = self.radiation.force(nu, count_sum)
        _ = zero

        return ForceBreakdown(
            gravity=gravity,
            hydrostatics=hydrostatics,
            resistance=resistance,
            radiation=radiation,
            diffraction=diffraction,
            sails=sails,
            appendages=appendages,
            foils=foils,
        )

    def state_derivative(
        self,
        state: DVPPState,
        mass: MassBreakdown,
        forces: ForceBreakdown,
    ):
        nu_dot = np.linalg.solve(
            mass.M,
            -rigid_body_coriolis(state.nu, total_mass=self.total_mass) @ state.nu
            - forces.radiation
            + forces.total_without_radiation_sign(),
        )
        eta_dot_ned = body_to_ned_velocity(state.nu, state.eta)
        return DVPPState(eta=eta_dot_ned, nu=nu_dot)
