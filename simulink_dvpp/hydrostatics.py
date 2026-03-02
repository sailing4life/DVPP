#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np

from .constants import GRAVITY, SEA_WATER_DENSITY
from .kinematics import body_to_world_rotation


@dataclass
class HydrostaticResult:
    F_hs: np.ndarray
    LWL: float
    WB: float
    submerged_facet_areas: float
    Draft: float
    LCF: float
    LCB: float
    count_sum: float
    res_hs: float


def hydrostatic_forces_and_moments(vertices, faces, eta, time, zero, wave):
    """
    Port of Simulink chart_170 `Hydrostatics`.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray, shape (M, 3)
        Zero-based triangle indices.
    """
    stl_vertices = np.asarray(vertices, dtype=float)
    stl_faces = np.asarray(faces, dtype=int)

    rotation_bw = body_to_world_rotation(eta)
    vertices_world = stl_vertices @ rotation_bw.T + np.asarray(eta[:3], dtype=float)

    facet_vectors_body_1 = stl_vertices[stl_faces[:, 1], :] - stl_vertices[stl_faces[:, 0], :]
    facet_vectors_body_2 = stl_vertices[stl_faces[:, 2], :] - stl_vertices[stl_faces[:, 0], :]
    facet_normals_body = np.cross(facet_vectors_body_1, facet_vectors_body_2)
    facet_areas = 0.5 * np.sqrt(np.sum(facet_normals_body**2, axis=1))
    facet_normals_body = facet_normals_body / np.maximum(np.linalg.norm(facet_normals_body, axis=1)[:, None], 1e-12)
    facet_normals_world = facet_normals_body @ rotation_bw.T

    facet_centers_body = (
        stl_vertices[stl_faces[:, 0], :] +
        stl_vertices[stl_faces[:, 1], :] +
        stl_vertices[stl_faces[:, 2], :]
    ) / 3.0
    facet_centers_world = (
        vertices_world[stl_faces[:, 0], :] +
        vertices_world[stl_faces[:, 1], :] +
        vertices_world[stl_faces[:, 2], :]
    ) / 3.0

    x = facet_centers_world[:, 0]
    y = facet_centers_world[:, 1]
    z = facet_centers_world[:, 2]

    z_wave = np.zeros(len(x))
    phi_dt = np.zeros(len(x))
    u_w = np.zeros(len(x))
    v_w = np.zeros(len(x))
    w_w = np.zeros(len(x))

    for data in np.asarray(wave, dtype=float):
        zeta_a, k, nu, omega = data
        if omega == 0.0:
            continue

        phase = -omega * time + k * x * np.cos(nu) + k * y * np.sin(nu)
        z_w = zeta_a * np.cos(phase) + ((k * zeta_a**2) / 2.0) * np.cos(2.0 * phase)
        z_wave = z_wave + z_w

        # Wheeler stretching applied consistently to both dynamic pressure
        # (phi_dt below) and orbital velocities (thesis §4.6.1):
        # exp(k·(z − η)) contracts the decay depth at crests, extends at troughs.
        exp_term = np.exp(k * (z - z_w))
        u_w = u_w + zeta_a * omega * np.cos(nu) * exp_term * np.cos(phase)
        v_w = v_w + zeta_a * omega * np.sin(nu) * exp_term * np.cos(phase)
        w_w = w_w + zeta_a * omega * exp_term * np.sin(phase)
        non_lin = 0.5 * (u_w**2 + v_w**2 + w_w**2)
        phi_dt = phi_dt + (-zeta_a * GRAVITY * np.exp(k * (z - z_w)) * np.cos(phase)) - non_lin

    water_density = SEA_WATER_DENSITY
    gravity = GRAVITY

    hydrostatic_pressure = water_density * gravity * (z_wave - z)
    displ_p = hydrostatic_pressure.copy()
    submerged = z < z_wave
    submerged_facet_areas = float(np.sum(facet_areas[submerged]))
    count_sum = float(np.sum(submerged))

    p_fk = -phi_dt * water_density

    if np.sum(submerged) > 20:
        submerged_centers_body = facet_centers_body[submerged, :]
        minl = np.min(submerged_centers_body[:, 0])
        maxl = np.max(submerged_centers_body[:, 0])
        LWL = float(maxl - minl)
        WB = float(np.max(submerged_centers_body[:, 1]) - np.min(submerged_centers_body[:, 1]))
        Draft = float(abs(np.min(submerged_centers_body[:, 2])))
        fpp = 7.47

        yz_projection = submerged_centers_body[:, 0:2]
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(yz_projection)
            hull_pts = yz_projection[hull.vertices]
            waterplane_centroid = np.array([np.mean(hull_pts[:, 0]), np.mean(hull_pts[:, 1])])
        except Exception:
            waterplane_centroid = np.array([np.mean(yz_projection[:, 0]), np.mean(yz_projection[:, 1])])

        # LCB = area-weighted centroid of submerged panel x-positions (volume
        # centroid approximation, consistent with DSYHS definition, Horel Eq. 22).
        # Previously used pressure-weighted x which biases toward the deepest panels.
        LCB = float(np.sum(facet_areas[submerged] * submerged_centers_body[:, 0]) / np.sum(facet_areas[submerged]) + fpp)
        LCF = float(waterplane_centroid[0] + fpp)
    else:
        # Fewer than 20 panels submerged — boat is nearly out of the water.
        # Use conservative minimum values so that the Delft DSYHS prismatic
        # coefficient  cp = Δ / (LWL · Bwl · T)  stays bounded.
        # Tiny LWL / WB (old: 0.2 / 0.05) drove cp → thousands, making the
        # resistance coefficient hugely negative → unphysical thrust → crash.
        LWL = 4.0
        WB  = 0.5
        LCB = 7.0
        LCF = 7.5
        Draft = 0.0

    hydrostatic_force_x = np.sum(hydrostatic_pressure[submerged] * facet_areas[submerged] * facet_normals_body[submerged, 0])
    hydrostatic_force_y = np.sum(hydrostatic_pressure[submerged] * facet_areas[submerged] * facet_normals_body[submerged, 1])
    hydrostatic_force_z = np.sum(hydrostatic_pressure[submerged] * facet_areas[submerged] * facet_normals_body[submerged, 2])

    lever_x = facet_centers_body[submerged, 0]
    lever_y = facet_centers_body[submerged, 1]
    lever_z = facet_centers_body[submerged, 2]
    hydrostatic_moment_x = np.sum(
        hydrostatic_pressure[submerged] * facet_areas[submerged] *
        (-lever_y * facet_normals_body[submerged, 2] - lever_z * -facet_normals_body[submerged, 1])
    )
    # Pitch moment: dM_y = p·A·(r × n̂)_y with force = -p·A·n̂
    # (r × n̂)_y = r_z·n_x − r_x·n_z → dM_y = p·A·(−r_z·n_x + r_x·n_z)
    hydrostatic_moment_y = np.sum(
        hydrostatic_pressure[submerged] * facet_areas[submerged] *
        (-lever_z * facet_normals_body[submerged, 0] + lever_x * facet_normals_body[submerged, 2])
    )

    hydrostatic_force = np.array([-hydrostatic_force_x, -hydrostatic_force_y, -hydrostatic_force_z])
    hydrostatic_moment = np.array([hydrostatic_moment_x, hydrostatic_moment_y, 0.0])

    submerged_fk = submerged
    fk_x = np.sum(p_fk[submerged_fk] * facet_areas[submerged_fk] * facet_normals_body[submerged_fk, 0])
    fk_y = np.sum(p_fk[submerged_fk] * facet_areas[submerged_fk] * facet_normals_body[submerged_fk, 1])
    fk_z = np.sum(p_fk[submerged_fk] * facet_areas[submerged_fk] * facet_normals_body[submerged_fk, 2])
    lever_x_fk = facet_centers_body[submerged_fk, 0]
    lever_y_fk = facet_centers_body[submerged_fk, 1]
    lever_z_fk = facet_centers_body[submerged_fk, 2]
    fk_mx = np.sum(
        p_fk[submerged_fk] * facet_areas[submerged_fk] *
        (-lever_y_fk * facet_normals_body[submerged_fk, 2] - lever_z_fk * -facet_normals_body[submerged_fk, 1])
    )
    fk_my = np.sum(
        p_fk[submerged_fk] * facet_areas[submerged_fk] *
        (-lever_z_fk * facet_normals_body[submerged_fk, 0] + lever_x_fk * facet_normals_body[submerged_fk, 2])
    )
    fk_force = np.array([-fk_x, -fk_y, -fk_z])
    fk_moment = np.array([fk_mx, fk_my, 0.0])

    F_hs = np.concatenate([hydrostatic_force + fk_force, hydrostatic_moment + fk_moment])
    # `res_hs` is used as a scalar displacement proxy downstream, so it must
    # reflect the vertical buoyancy component in the earth frame.
    res_hs = float(-np.sum(displ_p[submerged] * facet_areas[submerged] * facet_normals_world[submerged, 2]))

    return HydrostaticResult(
        F_hs=F_hs,
        LWL=LWL,
        WB=WB,
        submerged_facet_areas=submerged_facet_areas,
        Draft=Draft,
        LCF=LCF,
        LCB=LCB,
        count_sum=count_sum,
        res_hs=res_hs,
    )
