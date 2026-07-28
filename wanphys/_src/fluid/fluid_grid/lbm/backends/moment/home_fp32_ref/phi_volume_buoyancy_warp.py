# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in φ-volume Archimedes buoyancy (example / plugin path).

Integrates liquid φ over a spherical shell just outside the rigid SDF (interior
cells are gas-masked).  Force is buoyancy-only by default (no push / drag).

Not part of the default HOME-FREE step or ``GridLbmRigidCoupling``.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

CELL_GAS: int = 0


def fibonacci_shell_offsets(
    n: int = 48,
    *,
    radii: tuple[float, ...] = (1.05, 1.15),
) -> tuple[tuple[float, float, float], ...]:
    """Unit-sphere Fibonacci directions scaled by each radius in ``radii``."""
    out: list[tuple[float, float, float]] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for radius in radii:
        for i in range(max(1, int(n))):
            y = 1.0 - (2.0 * i + 1.0) / float(n)
            r_xy = math.sqrt(max(0.0, 1.0 - y * y))
            theta = golden * float(i)
            x = math.cos(theta) * r_xy
            z = math.sin(theta) * r_xy
            out.append((radius * x, radius * y, radius * z))
    return tuple(out)


@wp.kernel
def sample_phi_shell_volume_kernel(
    phi: wp.array3d(dtype=float),
    cell: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=float),
    offsets: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_ids: wp.array(dtype=wp.int32),
    radius: float,
    dh: float,
    nx: int,
    ny: int,
    nz: int,
    phi_wet: float,
    out_phi_sum: wp.array(dtype=float),
    out_valid: wp.array(dtype=wp.int32),
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
    p = phi[i, j, k]
    if p <= phi_wet:
        return
    wp.atomic_add(out_phi_sum, bid, p)


@wp.kernel
def assemble_phi_volume_buoyancy_kernel(
    phi_sum: wp.array(dtype=float),
    valid: wp.array(dtype=wp.int32),
    volume: float,
    rho_liquid: float,
    gravity_abs: float,
    buoyancy_scale: float,
    sub_ema: wp.array(dtype=float),
    ema_alpha: float,
    dsub_cap: float,
    out_forces: wp.array(dtype=wp.spatial_vector),
    out_submerged: wp.array(dtype=float),
) -> None:
    bid = wp.tid()
    n_valid = int(valid[bid])
    submerged_raw = float(0.0)
    if n_valid > 0:
        submerged_raw = phi_sum[bid] / float(n_valid)
        if submerged_raw > 1.0:
            submerged_raw = 1.0
        if submerged_raw < 0.0:
            submerged_raw = 0.0

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

    fz = buoyancy_scale * submerged * rho_liquid * volume * gravity_abs
    out_forces[bid] = wp.spatial_vector(
        wp.vec3(0.0, 0.0, fz),
        wp.vec3(0.0, 0.0, 0.0),
    )


def ensure_phi_volume_scratch(
    *,
    device: wp.context.Device | str,
    offsets_xyz: tuple[tuple[float, float, float], ...],
    body_ids: tuple[int, ...],
    scratch: dict | None,
) -> dict:
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
        "phi_sum": wp.zeros(n_bodies, dtype=float, device=device),
        "valid": wp.zeros(n_bodies, dtype=wp.int32, device=device),
        "forces": wp.zeros(n_bodies, dtype=wp.spatial_vector, device=device),
        "submerged": wp.zeros(n_bodies, dtype=float, device=device),
        "sub_ema": wp.full(n_bodies, -1.0, dtype=float, device=device),
    }


def apply_phi_volume_buoyancy_gpu(
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
    volume: float,
    rho_liquid: float,
    gravity_abs: float,
    buoyancy_scale: float = 1.0,
    phi_wet: float = 0.05,
    ema_alpha: float = 0.08,
    dsub_cap: float = 0.04,
    sync_submerged: bool = False,
) -> dict[int, float]:
    """Sample φ on a dense shell and apply Archimedes force (no push/drag)."""
    n_off = int(scratch["n_off"])
    n_bodies = int(scratch["n_bodies"])
    scratch["phi_sum"].zero_()
    scratch["valid"].zero_()

    wp.launch(
        sample_phi_shell_volume_kernel,
        dim=(n_off, n_bodies),
        inputs=[
            phi,
            cell,
            solid,
            scratch["offsets"],
            body_q,
            scratch["body_ids"],
            float(radius),
            float(dh),
            int(nx),
            int(ny),
            int(nz),
            float(phi_wet),
            scratch["phi_sum"],
            scratch["valid"],
        ],
    )
    wp.launch(
        assemble_phi_volume_buoyancy_kernel,
        dim=n_bodies,
        inputs=[
            scratch["phi_sum"],
            scratch["valid"],
            float(volume),
            float(rho_liquid),
            float(gravity_abs),
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
