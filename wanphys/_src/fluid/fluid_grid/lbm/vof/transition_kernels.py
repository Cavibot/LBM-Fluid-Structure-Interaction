# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P4 deterministic VOF topology transition and mass redistribution kernels."""

from __future__ import annotations

import warp as wp

from ..encoding import direction_x, direction_y, direction_z


@wp.func
def _wrap_once(value: int, size: int) -> int:
    mapped = value
    if mapped < 0:
        mapped += size
    elif mapped >= size:
        mapped -= size
    return mapped


@wp.kernel
def propose_vof_type_kernel(
    mass_pre: wp.array3d(dtype=float),
    density_out: wp.array3d(dtype=float),
    cell_type_n: wp.array3d(dtype=wp.uint8),
    proposed_type: wp.array3d(dtype=wp.uint8),
    epsilon: float,
) -> None:
    """Classify only old INTERFACE cells using the paper thresholds."""

    i, j, k = wp.tid()
    kind = cell_type_n[i, j, k]
    proposal = kind
    if kind == wp.uint8(1):
        phi_pre = mass_pre[i, j, k] / density_out[i, j, k]
        if phi_pre >= 1.0 + epsilon:
            proposal = wp.uint8(2)
        elif phi_pre <= 0.0 - epsilon:
            proposal = wp.uint8(0)
    proposed_type[i, j, k] = proposal


@wp.kernel
def resolve_vof_topology_kernel(
    proposed_type: wp.array3d(dtype=wp.uint8),
    cell_type_n: wp.array3d(dtype=wp.uint8),
    final_type: wp.array3d(dtype=wp.uint8),
    new_interface: wp.array3d(dtype=wp.uint8),
    retired_active: wp.array3d(dtype=wp.uint8),
    changed: wp.array3d(dtype=wp.uint8),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Resolve topology with the reference I-to-L priority using gather only."""

    i, j, k = wp.tid()
    proposal = proposed_type[i, j, k]
    old_kind = cell_type_n[i, j, k]
    forced_interface = int(0)

    opposing = wp.uint8(1)
    checks_opposing = int(0)
    if old_kind == wp.uint8(0):
        opposing = wp.uint8(2)
        checks_opposing = 1
    elif old_kind == wp.uint8(2):
        opposing = wp.uint8(0)
        checks_opposing = 1
    elif old_kind == wp.uint8(1) and proposal == wp.uint8(0):
        opposing = wp.uint8(2)
        checks_opposing = 1

    if checks_opposing != 0:
        for q in range(1, 19):
            ni = i - direction_x(q)
            nj = j - direction_y(q)
            nk = k - direction_z(q)
            outside = bool(False)
            if ni < 0 or ni >= nx:
                if periodic_x != 0:
                    ni = _wrap_once(ni, nx)
                else:
                    outside = True
            if nj < 0 or nj >= ny:
                if periodic_y != 0:
                    nj = _wrap_once(nj, ny)
                else:
                    outside = True
            if nk < 0 or nk >= nz:
                if periodic_z != 0:
                    nk = _wrap_once(nk, nz)
                else:
                    outside = True
            if not outside and proposed_type[ni, nj, nk] == opposing:
                if (
                    old_kind != wp.uint8(1)
                    or cell_type_n[ni, nj, nk] == wp.uint8(1)
                ):
                    forced_interface = 1

    resolved = proposal
    if forced_interface != 0:
        resolved = wp.uint8(1)
    final_type[i, j, k] = resolved

    new_interface[i, j, k] = wp.uint8(
        int(old_kind == wp.uint8(0) and resolved == wp.uint8(1))
    )
    retired_active[i, j, k] = wp.uint8(
        int(old_kind != wp.uint8(0) and resolved == wp.uint8(0))
    )
    changed[i, j, k] = wp.uint8(int(old_kind != resolved))


@wp.kernel
def prepare_vof_redistribution_kernel(
    mass_pre: wp.array3d(dtype=float),
    density_out: wp.array3d(dtype=float),
    cell_type_n: wp.array3d(dtype=wp.uint8),
    final_type: wp.array3d(dtype=wp.uint8),
    mass_base: wp.array3d(dtype=float),
    excess: wp.array3d(dtype=float),
    share: wp.array3d(dtype=float),
    receiver_count: wp.array3d(dtype=wp.uint8),
    unresolved_excess: wp.array3d(dtype=float),
    mass_tolerance: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Canonicalize by final type and prepare equal per-interface shares."""

    i, j, k = wp.tid()
    kind = final_type[i, j, k]
    old_kind = cell_type_n[i, j, k]
    rho = density_out[i, j, k]
    mass = mass_pre[i, j, k]

    base = float(0.0)
    if kind == wp.uint8(2):
        base = mass
        if old_kind == wp.uint8(1) and wp.abs(mass - rho) > mass_tolerance:
            base = rho
    elif kind == wp.uint8(1):
        base = wp.min(wp.max(mass, 0.0), rho)
    local_excess = mass - base

    count = int(0)
    for q in range(1, 19):
        ni = i - direction_x(q)
        nj = j - direction_y(q)
        nk = k - direction_z(q)
        outside = bool(False)
        if ni < 0 or ni >= nx:
            if periodic_x != 0:
                ni = _wrap_once(ni, nx)
            else:
                outside = True
        if nj < 0 or nj >= ny:
            if periodic_y != 0:
                nj = _wrap_once(nj, ny)
            else:
                outside = True
        if nk < 0 or nk >= nz:
            if periodic_z != 0:
                nk = _wrap_once(nk, nz)
            else:
                outside = True
        if not outside and final_type[ni, nj, nk] == wp.uint8(1):
            count += 1

    local_share = float(0.0)
    unresolved = float(0.0)
    if count > 0:
        local_share = local_excess / float(count)
    elif wp.abs(local_excess) > mass_tolerance:
        unresolved = local_excess

    mass_base[i, j, k] = base
    excess[i, j, k] = local_excess
    share[i, j, k] = local_share
    receiver_count[i, j, k] = wp.uint8(count)
    unresolved_excess[i, j, k] = unresolved


@wp.kernel
def finalize_bounded_vof_kernel(
    density_out: wp.array3d(dtype=float),
    final_type: wp.array3d(dtype=wp.uint8),
    mass_base: wp.array3d(dtype=float),
    mass_final: wp.array3d(dtype=float),
    phi_final: wp.array3d(dtype=float),
) -> None:
    """Commit bounded geometry while leaving excess in the side channel."""

    i, j, k = wp.tid()
    kind = final_type[i, j, k]
    mass = mass_base[i, j, k]

    mass_final[i, j, k] = mass
    if kind == wp.uint8(0):
        phi_final[i, j, k] = 0.0
    elif kind == wp.uint8(2):
        phi_final[i, j, k] = 1.0
    else:
        phi_final[i, j, k] = mass / density_out[i, j, k]


__all__ = [
    "finalize_bounded_vof_kernel",
    "prepare_vof_redistribution_kernel",
    "propose_vof_type_kernel",
    "resolve_vof_topology_kernel",
]
