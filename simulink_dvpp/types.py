#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WaveComponent:
    amplitude: float
    wavenumber: float
    direction: float
    omega: float

    def as_array(self):
        return np.array([self.amplitude, self.wavenumber, self.direction, self.omega], dtype=float)


@dataclass
class DVPPState:
    eta: np.ndarray
    nu: np.ndarray

    def as_vector(self):
        return np.concatenate([np.asarray(self.eta, dtype=float), np.asarray(self.nu, dtype=float)])


@dataclass
class MassBreakdown:
    M: np.ndarray
    z: float
    M_a: np.ndarray
    M_rb: np.ndarray


@dataclass
class SailInputs:
    aws_body: np.ndarray
    eta: np.ndarray
    flat: float
    c0_enabled: float
    twa_deg: float
    reef: float
    nu: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    spi_force: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))


@dataclass
class SailOutputs:
    total_force: np.ndarray
    mainsail_force: np.ndarray
    jib_force: np.ndarray
    headsail_force: np.ndarray
    spi_force: np.ndarray
    windage_force: np.ndarray
    a22_sails: float
    a44_sails: float
    area_main: float
    area_jib: float
    area_headsail: float
    awa_main_deg: float
    awa_headsail_deg: float


@dataclass
class AppendageInputs:
    eta_dot: np.ndarray
    eta: np.ndarray
    wave: np.ndarray
    time: float
    keel_angle_deg: float
    zero: float
    rudder_force: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    wl_z_shift: float = 0.0


@dataclass
class AppendageOutputs:
    total_force: np.ndarray
    keel_force: np.ndarray
    rudder_force: np.ndarray
    keel_gravity_force: np.ndarray
    keel_hydrodynamic_force: np.ndarray
    keel_points: np.ndarray
    keel_points_body: np.ndarray
    keel_points_world: np.ndarray
    relative_speed: float
    relative_angle_deg: float
    relative_velocity_local: np.ndarray
    segment_length: float


@dataclass
class FoilInputs:
    eta_dot: np.ndarray
    eta: np.ndarray
    rake_foil_deg: float
    wave: np.ndarray
    time: float
    zero: float
    chord_length: float = 0.6
    wl_z_shift: float = 0.0


@dataclass
class FoilOutputs:
    total_force: np.ndarray
    hydrodynamic_force: np.ndarray
    gravity_force: np.ndarray
    added_mass: float
    aoa_deg: np.ndarray
    cd: np.ndarray
    onside_force: np.ndarray
    offside_force: np.ndarray
    onside_points: np.ndarray
    offside_points: np.ndarray
    onside_points_body: np.ndarray
    offside_points_body: np.ndarray
    onside_points_world: np.ndarray
    offside_points_world: np.ndarray


@dataclass
class ForceBreakdown:
    gravity: np.ndarray
    hydrostatics: np.ndarray
    resistance: np.ndarray
    radiation: np.ndarray
    diffraction: np.ndarray
    sails: np.ndarray
    appendages: np.ndarray
    foils: np.ndarray

    def total_without_radiation_sign(self):
        return (
            self.gravity +
            self.hydrostatics +
            self.resistance +
            self.sails +
            self.appendages +
            self.foils +
            self.diffraction
        )
