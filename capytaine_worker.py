#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys

import numpy as np


_DOFS = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']


def _compute_hydrostatics(body, rho, g):
    """Compute 6x6 hydrostatic stiffness, matching the main Capytaine wrapper."""
    C = np.zeros((6, 6))
    try:
        Craw = body.compute_hydrostatic_stiffness(rho=rho, g=g).values
        if Craw.shape == (3, 3):
            for ii, gi in enumerate([2, 3, 4]):
                for jj, gj in enumerate([2, 3, 4]):
                    C[gi, gj] = Craw[ii, jj]
        else:
            C = np.array(Craw)
        return C
    except Exception:
        return C


def _solve(stl_path, omega, rho, g):
    import capytaine as cpt

    try:
        cap_mesh = cpt.load_mesh(stl_path, file_format='stl', name='hull')
    except TypeError:
        cap_mesh = cpt.load_mesh(stl_path)

    body = cpt.FloatingBody(mesh=cap_mesh, name='hull')
    body.add_all_rigid_body_dofs()

    if hasattr(body, 'immersed_part'):
        body = body.immersed_part()
    elif hasattr(body, 'keep_immersed_part'):
        body.keep_immersed_part()

    try:
        body.mesh.heal_mesh()
    except AttributeError:
        pass

    problems = []
    for dof in _DOFS:
        for w in omega:
            try:
                pb = cpt.RadiationProblem(
                    body=body, radiating_dof=dof, omega=float(w), rho=rho, g=g
                )
            except TypeError:
                pb = cpt.RadiationProblem(
                    body=body, radiating_dof=dof, omega=float(w),
                    water_density=rho, g=g
                )
            problems.append(pb)

    results = cpt.BEMSolver().solve_all(problems, keep_details=False)
    try:
        ds = cpt.assemble_dataset(results)
    except AttributeError:
        ds = cpt.assembleDataset(results)

    n_omega = len(omega)
    A_omega = np.zeros((6, 6, n_omega))
    B_omega = np.zeros((6, 6, n_omega))

    for i, di in enumerate(_DOFS):
        for j, dj in enumerate(_DOFS):
            try:
                A_omega[i, j] = (
                    ds['added_mass']
                    .sel(radiating_dof=di, influenced_dof=dj)
                    .values
                )
                B_omega[i, j] = (
                    ds['radiation_damping']
                    .sel(radiating_dof=di, influenced_dof=dj)
                    .values
                )
            except Exception:
                pass

    C = _compute_hydrostatics(body, rho, g)
    return A_omega, B_omega, C


def _probe():
    import capytaine as cpt

    body = cpt.FloatingBody(mesh=cpt.mesh_sphere(radius=1.0), name='_probe')
    body.add_all_rigid_body_dofs()
    if hasattr(body, 'immersed_part'):
        body = body.immersed_part()
    elif hasattr(body, 'keep_immersed_part'):
        body.keep_immersed_part()

    pb = cpt.RadiationProblem(body=body, radiating_dof='Heave', omega=1.0)
    cpt.BEMSolver().solve(pb, keep_details=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--stl')
    parser.add_argument('--omega')
    parser.add_argument('--output')
    parser.add_argument('--rho', type=float, default=1025.0)
    parser.add_argument('--g', type=float, default=9.81)
    args = parser.parse_args()

    if args.probe:
        _probe()
        return 0

    omega = np.load(args.omega)
    A_omega, B_omega, C = _solve(args.stl, omega, args.rho, args.g)
    np.savez(args.output, omega=omega, A_omega=A_omega, B_omega=B_omega, C=C)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(json.dumps({
            'error_type': type(exc).__name__,
            'error': str(exc),
        }) + '\n')
        raise
