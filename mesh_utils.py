#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import meshio


class TriangleMesh:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=float)

    @classmethod
    def from_file(cls, path):
        mesh = meshio.read(path)
        triangles = [cell.data for cell in mesh.cells if cell.type == "triangle"]
        if not triangles:
            raise ValueError(f"No triangle cells found in STL: {path}")
        faces = np.concatenate(triangles, axis=0)
        vectors = mesh.points[faces]
        return cls(vectors)

    @classmethod
    def from_triangles(cls, triangles):
        return cls(triangles)

    def save(self, path):
        flat_points = self.vectors.reshape(-1, 3)
        points, inverse = np.unique(flat_points, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        mesh = meshio.Mesh(points=points, cells=[("triangle", faces)])
        meshio.write(path, mesh, file_format="stl")


def _slice_triangles(triangles, x_cut):
    segments = []
    for tri in triangles:
        xs = tri[:, 0]
        points = []
        for i in range(3):
            j = (i + 1) % 3
            if (xs[i] - x_cut) * (xs[j] - x_cut) < 0:
                weight = (x_cut - xs[i]) / (xs[j] - xs[i])
                points.append(tri[i] + weight * (tri[j] - tri[i]))
            elif abs(xs[i] - x_cut) < 1e-9:
                points.append(tri[i].copy())
        if len(points) == 2:
            segments.append((points[0], points[1]))
    return segments


def _order_contour(segments, tol=1e-3):
    if not segments:
        return np.empty((0, 3), dtype=float)

    chain = [np.asarray(segments[0][0], dtype=float), np.asarray(segments[0][1], dtype=float)]
    remaining = list(segments[1:])
    while remaining:
        last = chain[-1]
        found = False
        for index, (p1, p2) in enumerate(remaining):
            if np.linalg.norm(last - p1) < tol:
                chain.append(np.asarray(p2, dtype=float))
                remaining.pop(index)
                found = True
                break
            if np.linalg.norm(last - p2) < tol:
                chain.append(np.asarray(p1, dtype=float))
                remaining.pop(index)
                found = True
                break
        if not found:
            break
    return np.asarray(chain, dtype=float)


def _section_geometry(segments, waterline_z=0.0):
    points = []
    for p1, p2 in segments:
        points.append(p1)
        points.append(p2)
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return None

    y_all = points[:, 1]
    z_all = points[:, 2]
    draft = waterline_z - np.min(z_all)
    if draft <= 0.0:
        return None

    band = z_all >= waterline_z - 0.05 * draft
    if np.sum(band) < 2:
        b_wl = (np.max(y_all) - np.min(y_all)) / 2.0
    else:
        b_wl = (np.max(y_all[band]) - np.min(y_all[band])) / 2.0
    beam = 2.0 * b_wl

    chain = _order_contour(segments)
    if len(chain) >= 3:
        yc = chain[:, 1]
        zc = chain[:, 2]
        n_chain = len(yc)
        sub_y = []
        sub_z = []
        for i in range(n_chain):
            y1, z1 = yc[i], zc[i]
            y2, z2 = yc[(i + 1) % n_chain], zc[(i + 1) % n_chain]
            if z1 <= waterline_z:
                sub_y.append(y1)
                sub_z.append(z1)
            if (z1 - waterline_z) * (z2 - waterline_z) < 0:
                weight = (waterline_z - z1) / (z2 - z1)
                sub_y.append(y1 + weight * (y2 - y1))
                sub_z.append(waterline_z)
        if len(sub_y) >= 3:
            sub_y = np.asarray(sub_y, dtype=float)
            sub_z = np.asarray(sub_z, dtype=float)
            area = 0.5 * abs(
                np.dot(sub_y, np.roll(sub_z, -1)) - np.dot(np.roll(sub_y, -1), sub_z)
            )
        else:
            area = max(0.5 * beam * draft, 1e-9)
    else:
        sub_pts_y = y_all[z_all <= waterline_z]
        sub_pts_z = z_all[z_all <= waterline_z]
        if len(sub_pts_y) < 3:
            area = 0.5 * beam * draft
        else:
            order = np.argsort(sub_pts_z)
            yy = sub_pts_y[order]
            zz = sub_pts_z[order]
            area = 0.5 * abs(np.dot(yy, np.roll(zz, -1)) - np.dot(np.roll(yy, -1), zz))
            area = max(area, 0.25 * beam * draft)

    return {
        "B": beam,
        "T": draft,
        "A_s": area,
        "b_wl": b_wl,
        "I_wl": 2.0 * b_wl**3 / 3.0,
    }


def submerged_volume_at(triangles, z_wl, n_x=50):
    xs = triangles[:, :, 0]
    x_min = xs.min() * 1.01
    x_max = xs.max() * 1.01
    x_cuts = np.linspace(x_min, x_max, n_x + 2)[1:-1]

    areas = []
    x_vals = []
    for x_cut in x_cuts:
        segments = _slice_triangles(triangles, x_cut)
        if not segments:
            continue
        geom = _section_geometry(segments, waterline_z=z_wl)
        if geom is not None:
            areas.append(geom["A_s"])
            x_vals.append(x_cut)

    if len(areas) < 2:
        return 0.0
    return float(np.trapz(areas, x_vals))


def find_equilibrium_waterline(triangles, mass, rho=1025.0, n_x=40, tol_m=0.001):
    target_volume = float(mass) / float(rho)
    z_all = triangles[:, :, 2].ravel()
    z_lo = z_all.min()
    z_hi = z_all.max()

    if submerged_volume_at(triangles, z_hi, n_x=n_x) < target_volume:
        raise RuntimeError(
            f"Hull cannot displace required volume {target_volume:.2f} m^3 for mass {mass:.0f} kg."
        )

    for _ in range(30):
        if (z_hi - z_lo) < tol_m:
            break
        z_mid = (z_lo + z_hi) / 2.0
        volume = submerged_volume_at(triangles, z_mid, n_x=n_x)
        if volume < target_volume:
            z_lo = z_mid
        else:
            z_hi = z_mid

    z_wl = (z_lo + z_hi) / 2.0
    return float(z_wl), float(submerged_volume_at(triangles, z_wl, n_x=n_x))
