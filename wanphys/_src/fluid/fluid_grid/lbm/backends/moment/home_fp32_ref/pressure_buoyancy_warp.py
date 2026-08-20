# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU hydrostatic **pressure** buoyancy (surface integral), not displaced-volume.

Essence of buoyancy with uniform liquid density:

    p = ρ · |g| · depth_below_free_surface
    F = ∮ −p n dA    (wet surface only; p_atm = 0 reference)

ME (Eq.32) remains for hydrodynamic impact; this state force supplies the
hydrostatic pressure jump that Guo + ρ≈1 does not put into populations.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

CELL_GAS: int = 0
CELL_INTERFACE: int = 1
CELL_LIQUID: int = 2


def fibonacci_sphere_dirs(n: int = 96) -> tuple[tuple[float, float, float], ...]:
    """Unit outward normals (Fibonacci sphere)."""
    out: list[tuple[float, float, float]] = []
    n = max(1, int(n))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * i + 1.0) / float(n)
        r_xy = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * float(i)
        out.append((math.cos(theta) * r_xy, y, math.sin(theta) * r_xy))
    return tuple(out)


def _ensure_scratch(
    *,
    device: wp.context.Device | str,
    dirs_xyz: tuple[tuple[float, float, float], ...],
    body_ids: tuple[int, ...],
    nx: int,
    ny: int,
    nz: int,
    scratch: dict | None,
) -> dict:
    n_dirs = len(dirs_xyz)
    n_bodies = len(body_ids)
    if (
        scratch is not None
        and scratch.get("n_dirs") == n_dirs
        and scratch.get("n_bodies") == n_bodies
        and scratch.get("nx") == nx
        and scratch.get("ny") == ny
        and scratch.get("nz") == nz
        and tuple(scratch.get("body_ids_host", ())) == tuple(body_ids)
    ):
        return scratch
    vecs = [wp.vec3(float(d[0]), float(d[1]), float(d[2])) for d in dirs_xyz]
    return {
        "n_dirs": n_dirs,
        "n_bodies": n_bodies,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "body_ids_host": tuple(int(b) for b in body_ids),
        "dirs": wp.array(vecs, dtype=wp.vec3, device=device),
        "body_ids": wp.array([int(b) for b in body_ids], dtype=wp.int32, device=device),
        "fs_k": wp.zeros((nx, ny), dtype=wp.int32, device=device),
        "force_acc": wp.zeros(n_bodies, dtype=wp.vec3, device=device),
        "torque_acc": wp.zeros(n_bodies, dtype=wp.vec3, device=device),
        "wet": wp.zeros(n_bodies, dtype=wp.int32, device=device),
        "valid": wp.zeros(n_bodies, dtype=wp.int32, device=device),
        "forces": wp.zeros(n_bodies, dtype=wp.spatial_vector, device=device),
        "submerged": wp.zeros(n_bodies, dtype=float, device=device),
        "sub_ema": wp.full(n_bodies, -1.0, dtype=float, device=device),
    }


@wp.kernel
def buoy_mark_fs_k_kernel(
    cell: wp.array3d(dtype=wp.int32),
    phi: wp.array3d(dtype=float),
    solid: wp.array3d(dtype=float),
    fs_k: wp.array2d(dtype=wp.int32),
    phi_wet: float,
    nz: int,
) -> None:
    """Topmost liquid / wet IF per column → free-surface cell index."""
    i, j = wp.tid()
    fs_k[i, j] = -1
    for t in range(nz):
        k = nz - 1 - t
        if solid[i, j, k] < 0.0:
            continue
        ct = int(cell[i, j, k])
        if ct == CELL_LIQUID:
            fs_k[i, j] = k
            return
        if ct == CELL_INTERFACE and phi[i, j, k] > phi_wet:
            fs_k[i, j] = k
            return


@wp.kernel
def sample_pressure_surface_kernel(
    phi: wp.array3d(dtype=float),
    cell: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=float),
    fs_k: wp.array2d(dtype=wp.int32),
    dirs: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_ids: wp.array(dtype=wp.int32),
    radius: float,
    dh: float,
    nx: int,
    ny: int,
    nz: int,
    phi_wet: float,
    rho_liquid: float,
    gravity_abs: float,
    dA: float,
    force_acc: wp.array(dtype=wp.vec3),
    torque_acc: wp.array(dtype=wp.vec3),
    out_wet: wp.array(dtype=wp.int32),
    out_valid: wp.array(dtype=wp.int32),
) -> None:
    """Accumulate −p n dA on the sphere; p = ρ|g|·depth below column FS."""
    sid, bid = wp.tid()
    body = int(body_ids[bid])
    pos = wp.transform_get_translation(body_q[body])
    nrm = dirs[sid]
    # Surface sample just outside SDF (slightly beyond R).
    sx = pos[0] + nrm[0] * radius
    sy = pos[1] + nrm[1] * radius
    sz = pos[2] + nrm[2] * radius
    i = int(sx / dh)
    j = int(sy / dh)
    k = int(sz / dh)
    if i < 0:
        i = 0
    if j < 0:
        j = 0
    if k < 0:
        k = 0
    if i >= nx:
        i = nx - 1
    if j >= ny:
        j = ny - 1
    if k >= nz:
        k = nz - 1
    if solid[i, j, k] < 0.0:
        return
    wp.atomic_add(out_valid, bid, 1)

    k_fs = int(fs_k[i, j])
    if k_fs < 0:
        return
    # Atmospheric reference above / at free surface.
    z_fs = (float(k_fs) + 0.5) * dh
    depth = z_fs - sz
    if depth <= 0.0:
        return
    ct = int(cell[i, j, k])
    if ct == CELL_GAS:
        return
    pphi = phi[i, j, k]
    if pphi <= phi_wet:
        return

    # Hydrostatic gauge pressure (uniform ρ); wet weight by φ.
    p = rho_liquid * gravity_abs * depth * pphi
    # Fluid pushes solid: −p n (n outward from body).
    dF = wp.vec3(-p * nrm[0] * dA, -p * nrm[1] * dA, -p * nrm[2] * dA)
    r = wp.vec3(nrm[0] * radius, nrm[1] * radius, nrm[2] * radius)
    dtau = wp.cross(r, dF)
    wp.atomic_add(force_acc, bid, dF)
    wp.atomic_add(torque_acc, bid, dtau)
    wp.atomic_add(out_wet, bid, 1)


@wp.kernel
def assemble_pressure_buoyancy_kernel(
    force_acc: wp.array(dtype=wp.vec3),
    torque_acc: wp.array(dtype=wp.vec3),
    wet: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    buoyancy_scale: float,
    sub_ema: wp.array(dtype=float),
    ema_alpha: float,
    dsub_cap: float,
    out_forces: wp.array(dtype=wp.spatial_vector),
    out_submerged: wp.array(dtype=float),
) -> None:
    bid = wp.tid()
    n_valid = int(valid[bid])
    n_wet = int(wet[bid])
    submerged_raw = float(0.0)
    if n_valid > 0:
        submerged_raw = float(n_wet) / float(n_valid)
        if submerged_raw > 1.0:
            submerged_raw = 1.0

    prev = sub_ema[bid]
    submerged = submerged_raw
    if prev >= 0.0:
        blended = (1.0 - ema_alpha) * prev + ema_alpha * submerged_raw
        dsub = blended - prev
        if dsub > dsub_cap:
            blended = prev + dsub_cap
        elif dsub < -dsub_cap:
            blended = prev - dsub_cap
        submerged = blended
    sub_ema[bid] = submerged
    out_submerged[bid] = submerged

    f = force_acc[bid] * buoyancy_scale
    t = torque_acc[bid] * buoyancy_scale
    out_forces[bid] = wp.spatial_vector(f, t)


def apply_pressure_buoyancy_gpu(
    *,
    phi: wp.array,
    cell: wp.array,
    solid: wp.array,
    body_q: wp.array,
    body_f_apply,
    radius: float,
    dh: float,
    nx: int,
    ny: int,
    nz: int,
    scratch: dict,
    rho_liquid: float,
    gravity_abs: float,
    buoyancy_scale: float = 1.0,
    phi_wet: float = 0.05,
    ema_alpha: float = 0.08,
    dsub_cap: float = 0.04,
    sample_radius_scale: float = 1.02,
    sync_submerged: bool = False,
) -> dict[int, float]:
    """Integrate hydrostatic −p n dA on sphere samples; apply to rigid bodies."""
    n_dirs = int(scratch["n_dirs"])
    n_bodies = int(scratch["n_bodies"])
    r_samp = float(radius) * float(sample_radius_scale)
    dA = 4.0 * math.pi * r_samp * r_samp / float(max(n_dirs, 1))

    scratch["force_acc"].zero_()
    scratch["torque_acc"].zero_()
    scratch["wet"].zero_()
    scratch["valid"].zero_()

    wp.launch(
        buoy_mark_fs_k_kernel,
        dim=(nx, ny),
        inputs=[
            cell,
            phi,
            solid,
            scratch["fs_k"],
            float(phi_wet),
            int(nz),
        ],
    )
    wp.launch(
        sample_pressure_surface_kernel,
        dim=(n_dirs, n_bodies),
        inputs=[
            phi,
            cell,
            solid,
            scratch["fs_k"],
            scratch["dirs"],
            body_q,
            scratch["body_ids"],
            r_samp,
            float(dh),
            int(nx),
            int(ny),
            int(nz),
            float(phi_wet),
            float(rho_liquid),
            abs(float(gravity_abs)),
            float(dA),
            scratch["force_acc"],
            scratch["torque_acc"],
            scratch["wet"],
            scratch["valid"],
        ],
    )
    wp.launch(
        assemble_pressure_buoyancy_kernel,
        dim=n_bodies,
        inputs=[
            scratch["force_acc"],
            scratch["torque_acc"],
            scratch["wet"],
            scratch["valid"],
            float(buoyancy_scale),
            scratch["sub_ema"],
            float(ema_alpha),
            float(dsub_cap),
            scratch["forces"],
            scratch["submerged"],
        ],
    )
    body_f_apply(scratch["body_ids"], scratch["forces"])

    if not sync_submerged:
        return {}
    sub = scratch["submerged"].numpy()
    ids = scratch["body_ids"].numpy()
    return {int(ids[i]): float(sub[i]) for i in range(n_bodies)}


def ensure_pressure_buoyancy_scratch(
    *,
    device: wp.context.Device | str,
    dirs_xyz: tuple[tuple[float, float, float], ...],
    body_ids: tuple[int, ...],
    nx: int,
    ny: int,
    nz: int,
    scratch: dict | None,
) -> dict:
    return _ensure_scratch(
        device=device,
        dirs_xyz=dirs_xyz,
        body_ids=body_ids,
        nx=nx,
        ny=ny,
        nz=nz,
        scratch=scratch,
    )
