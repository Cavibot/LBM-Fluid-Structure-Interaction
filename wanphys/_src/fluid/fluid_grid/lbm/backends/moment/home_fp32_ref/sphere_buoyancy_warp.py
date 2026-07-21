# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU empirical sphere buoyancy / drag (HOME-FREE FSI helpers).

Samples fluid on device from ``body_q`` / macros and writes forces into a
persistent ``spatial_vector`` buffer for ``RigidState.apply_body_forces``.
Hot path avoids host sync; optional ``sync_submerged`` pulls scalars for logs.
"""

from __future__ import annotations

import numpy as np
import warp as wp

CELL_GAS: int = 0


@wp.kernel
def sample_spheres_shell_wet_kernel(
    phi: wp.array3d(dtype=float),
    cell: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
    offsets: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_ids: wp.array(dtype=wp.int32),
    radius: float,
    dh: float,
    nx: int,
    ny: int,
    nz: int,
    phi_wet: float,
    out_water: wp.array(dtype=wp.int32),
    out_valid: wp.array(dtype=wp.int32),
    out_vx: wp.array(dtype=float),
    out_vy: wp.array(dtype=float),
    out_vz: wp.array(dtype=float),
) -> None:
    sid, bid = wp.tid()
    body = int(body_ids[bid])
    pos = wp.transform_get_translation(body_q[body])
    o = offsets[sid]
    sx = pos[0] + o[0] * radius
    sy = pos[1] + o[1] * radius
    sz = pos[2] + o[2] * radius
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
    if int(cell[i, j, k]) == CELL_GAS:
        return
    if phi[i, j, k] <= phi_wet:
        return
    wp.atomic_add(out_water, bid, 1)
    wp.atomic_add(out_vx, bid, ux[i, j, k])
    wp.atomic_add(out_vy, bid, uy[i, j, k])
    wp.atomic_add(out_vz, bid, uz[i, j, k])


@wp.kernel
def assemble_sphere_buoyancy_forces_kernel(
    water: wp.array(dtype=wp.int32),
    valid: wp.array(dtype=wp.int32),
    vx: wp.array(dtype=float),
    vy: wp.array(dtype=float),
    vz: wp.array(dtype=float),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_ids: wp.array(dtype=wp.int32),
    density: wp.array(dtype=float),
    volume: float,
    rho_liquid: float,
    gravity_abs: float,
    buoyancy_scale: float,
    push_rate: float,
    drag_xy: float,
    drag_z: float,
    vel_scale: float,
    out_forces: wp.array(dtype=wp.spatial_vector),
    out_submerged: wp.array(dtype=float),
) -> None:
    bid = wp.tid()
    body = int(body_ids[bid])
    n_valid = int(valid[bid])
    n_water = int(water[bid])
    submerged = float(0.0)
    if n_valid > 0:
        submerged = float(n_water) / float(n_valid)
    out_submerged[bid] = submerged

    qd = body_qd[body]
    bv_x = qd[0]
    bv_y = qd[1]
    bv_z = qd[2]

    fv_x = float(0.0)
    fv_y = float(0.0)
    fv_z = float(0.0)
    if n_water > 0:
        inv_w = 1.0 / float(n_water)
        fv_x = vx[bid] * inv_w * vel_scale
        fv_y = vy[bid] * inv_w * vel_scale
        fv_z = vz[bid] * inv_w * vel_scale

    mass = density[bid] * volume
    buoyancy_z = buoyancy_scale * rho_liquid * volume * gravity_abs * submerged
    push = push_rate * mass * submerged
    fx = push * (fv_x - bv_x) - drag_xy * mass * submerged * bv_x
    fy = push * (fv_y - bv_y) - drag_xy * mass * submerged * bv_y
    fz = (
        buoyancy_z
        + 0.35 * push * (fv_z - bv_z)
        - drag_z * mass * submerged * bv_z
    )
    out_forces[bid] = wp.spatial_vector(
        wp.vec3(fx, fy, fz),
        wp.vec3(0.0, 0.0, 0.0),
    )


def ensure_buoyancy_scratch(
    *,
    device: wp.context.Device | str,
    offsets_xyz: tuple[tuple[float, float, float], ...],
    body_ids: tuple[int, ...],
    densities: tuple[float, ...],
    scratch: dict | None,
) -> dict:
    """Allocate / reuse persistent GPU buffers for multi-sphere buoyancy."""
    n_bodies = len(body_ids)
    if (
        scratch is not None
        and scratch.get("n_off") == len(offsets_xyz)
        and scratch.get("n_bodies") == n_bodies
        and tuple(scratch.get("body_ids_host", ())) == tuple(body_ids)
    ):
        return scratch
    off = np.asarray(offsets_xyz, dtype=np.float32)
    vecs = [wp.vec3(float(r[0]), float(r[1]), float(r[2])) for r in off]
    return {
        "n_off": len(offsets_xyz),
        "n_bodies": n_bodies,
        "body_ids_host": tuple(int(b) for b in body_ids),
        "offsets": wp.array(vecs, dtype=wp.vec3, device=device),
        "body_ids": wp.array([int(b) for b in body_ids], dtype=wp.int32, device=device),
        "density": wp.array([float(d) for d in densities], dtype=float, device=device),
        "water": wp.zeros(n_bodies, dtype=wp.int32, device=device),
        "valid": wp.zeros(n_bodies, dtype=wp.int32, device=device),
        "vx": wp.zeros(n_bodies, dtype=float, device=device),
        "vy": wp.zeros(n_bodies, dtype=float, device=device),
        "vz": wp.zeros(n_bodies, dtype=float, device=device),
        "forces": wp.zeros(n_bodies, dtype=wp.spatial_vector, device=device),
        "submerged": wp.zeros(n_bodies, dtype=float, device=device),
    }


def apply_sphere_buoyancy_forces_gpu(
    *,
    phi: wp.array,
    cell: wp.array,
    solid: wp.array,
    ux: wp.array,
    uy: wp.array,
    uz: wp.array,
    body_q: wp.array,
    body_qd: wp.array,
    body_f_apply,  # RigidState.apply_body_forces bound method or callable
    radius: float,
    dh: float,
    nx: int,
    ny: int,
    nz: int,
    scratch: dict,
    volume: float,
    rho_liquid: float,
    gravity_abs: float,
    buoyancy_scale: float,
    push_rate: float,
    drag_xy: float,
    drag_z: float,
    vel_scale: float,
    phi_wet: float = 0.25,
    sync_submerged: bool = False,
) -> dict[int, float]:
    """Sample shells + assemble forces on GPU; optionally sync submerged fractions.

    Returns ``{body_id: submerged}`` (empty unless ``sync_submerged``).
    """
    n_off = int(scratch["n_off"])
    n_bodies = int(scratch["n_bodies"])
    scratch["water"].zero_()
    scratch["valid"].zero_()
    scratch["vx"].zero_()
    scratch["vy"].zero_()
    scratch["vz"].zero_()

    wp.launch(
        sample_spheres_shell_wet_kernel,
        dim=(n_off, n_bodies),
        inputs=[
            phi,
            cell,
            solid,
            ux,
            uy,
            uz,
            scratch["offsets"],
            body_q,
            scratch["body_ids"],
            float(radius),
            float(dh),
            int(nx),
            int(ny),
            int(nz),
            float(phi_wet),
            scratch["water"],
            scratch["valid"],
            scratch["vx"],
            scratch["vy"],
            scratch["vz"],
        ],
    )
    wp.launch(
        assemble_sphere_buoyancy_forces_kernel,
        dim=n_bodies,
        inputs=[
            scratch["water"],
            scratch["valid"],
            scratch["vx"],
            scratch["vy"],
            scratch["vz"],
            body_qd,
            scratch["body_ids"],
            scratch["density"],
            float(volume),
            float(rho_liquid),
            float(gravity_abs),
            float(buoyancy_scale),
            float(push_rate),
            float(drag_xy),
            float(drag_z),
            float(vel_scale),
            scratch["forces"],
            scratch["submerged"],
        ],
    )
    body_f_apply(scratch["body_ids"], scratch["forces"])

    if not sync_submerged:
        return {}
    sub = scratch["submerged"].numpy()
    ids = scratch["body_ids_host"]
    return {int(ids[i]): float(sub[i]) for i in range(n_bodies)}
