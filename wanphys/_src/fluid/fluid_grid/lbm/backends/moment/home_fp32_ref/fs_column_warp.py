# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared column free-surface helpers for HOME-FREE VOF buoyancy / Path A/B.

Marks the topmost liquid or wet interface index ``k`` per ``(i,j)`` column.
Used by pressure Archimedes, hydrostatic ρ(z), and Path B F_α,H.
"""

from __future__ import annotations

import warp as wp

CELL_GAS: int = 0
CELL_INTERFACE: int = 1
CELL_LIQUID: int = 2


@wp.kernel
def mark_column_fs_k_kernel(
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


def launch_mark_column_fs_k(
    *,
    cell: wp.array,
    phi: wp.array,
    solid: wp.array,
    fs_k: wp.array,
    phi_wet: float,
    nx: int,
    ny: int,
    nz: int,
    device: wp.context.Device | str | None = None,
) -> None:
    """Launch :func:`mark_column_fs_k_kernel` over the XY plane."""
    kw: dict = {}
    if device is not None:
        kw["device"] = device
    wp.launch(
        mark_column_fs_k_kernel,
        dim=(int(nx), int(ny)),
        inputs=[
            cell,
            phi,
            solid,
            fs_k,
            float(phi_wet),
            int(nz),
        ],
        **kw,
    )


@wp.kernel
def sum_wet_mass_kernel(
    cell: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    acc_idx: int,
    acc: wp.array(dtype=float),
) -> None:
    """Atomic-add Σmass over liquid + interface cells (excludes gas/solid)."""
    i, j, k = wp.tid()
    if solid[i, j, k] < 0.0:
        return
    ct = int(cell[i, j, k])
    if ct != CELL_LIQUID and ct != CELL_INTERFACE:
        return
    wp.atomic_add(acc, acc_idx, mass[i, j, k])


def sum_wet_mass_gpu(
    *,
    cell: wp.array,
    solid: wp.array,
    mass: wp.array,
    shape: tuple[int, int, int],
    acc: wp.array | None = None,
    device: wp.context.Device | str | None = None,
) -> float:
    """Return wet ``Σmass`` (liquid + interface, non-solid) via one GPU reduction."""
    nx, ny, nz = (int(shape[0]), int(shape[1]), int(shape[2]))
    dev = device if device is not None else cell.device
    if acc is None:
        acc = wp.zeros(1, dtype=float, device=dev)
    else:
        acc.zero_()
    wp.launch(
        sum_wet_mass_kernel,
        dim=(nx, ny, nz),
        inputs=[cell, solid, mass, 0, acc],
        device=dev,
    )
    return float(acc.numpy()[0])
