# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME IC factory (host ``HomeVofState``).

Shared by numpy steppers, GPU ``upload_home_vof_state``, and
``HomeFp32Bridge.seed_*``. Showcase examples should call these instead of
hand-rolled φ masks.
"""

from __future__ import annotations

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.step import (
    HomeMomentArrays,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_step import (
    CELL_GAS,
    CELL_INTERFACE,
    CELL_LIQUID,
    HomeVofState,
)

__all__ = [
    "mark_liquid_interfaces",
    "seed_dam_break_column",
    "seed_droplet",
    "seed_full_liquid",
    "seed_pool",
]


def _empty_fields(shape: tuple[int, int, int]) -> tuple[HomeMomentArrays, np.ndarray, np.ndarray]:
    moments = HomeMomentArrays(
        rho=np.zeros(shape, dtype=np.float64),
        ux=np.zeros(shape, dtype=np.float64),
        uy=np.zeros(shape, dtype=np.float64),
        uz=np.zeros(shape, dtype=np.float64),
        sxx=np.zeros(shape, dtype=np.float64),
        syy=np.zeros(shape, dtype=np.float64),
        szz=np.zeros(shape, dtype=np.float64),
        sxy=np.zeros(shape, dtype=np.float64),
        sxz=np.zeros(shape, dtype=np.float64),
        syz=np.zeros(shape, dtype=np.float64),
    )
    phi = np.zeros(shape, dtype=np.float64)
    cell_type = np.zeros(shape, dtype=np.int32)
    return moments, phi, cell_type


def mark_liquid_interfaces(
    cell_type: np.ndarray,
    phi: np.ndarray,
    *,
    interface_phi: float = 0.5,
) -> None:
    """In-place: liquid cells touching gas → INTERFACE with ``φ=interface_phi``."""
    liq = cell_type == CELL_LIQUID
    touch = np.zeros_like(liq, dtype=bool)
    touch[1:, :, :] |= liq[1:, :, :] & (cell_type[:-1, :, :] == CELL_GAS)
    touch[:-1, :, :] |= liq[:-1, :, :] & (cell_type[1:, :, :] == CELL_GAS)
    touch[:, 1:, :] |= liq[:, 1:, :] & (cell_type[:, :-1, :] == CELL_GAS)
    touch[:, :-1, :] |= liq[:, :-1, :] & (cell_type[:, 1:, :] == CELL_GAS)
    touch[:, :, 1:] |= liq[:, :, 1:] & (cell_type[:, :, :-1] == CELL_GAS)
    touch[:, :, :-1] |= liq[:, :, :-1] & (cell_type[:, :, 1:] == CELL_GAS)
    convert = touch & liq
    cell_type[convert] = CELL_INTERFACE
    phi[convert] = float(interface_phi)


def seed_full_liquid(
    shape: tuple[int, int, int],
    rho_liquid: float = 1.0,
    *,
    ux0: float = 0.0,
    uy0: float = 0.0,
    uz0: float = 0.0,
) -> HomeVofState:
    """Fill the domain with liquid only (HOME base / ``phase_mode=none``)."""
    moments, phi, cell_type = _empty_fields(shape)
    moments.rho[:, :, :] = float(rho_liquid)
    moments.ux[:, :, :] = float(ux0)
    moments.uy[:, :, :] = float(uy0)
    moments.uz[:, :, :] = float(uz0)
    moments.sxx[:, :, :] = float(ux0) * float(ux0)
    moments.syy[:, :, :] = float(uy0) * float(uy0)
    moments.szz[:, :, :] = float(uz0) * float(uz0)
    moments.sxy[:, :, :] = float(ux0) * float(uy0)
    moments.sxz[:, :, :] = float(ux0) * float(uz0)
    moments.syz[:, :, :] = float(uy0) * float(uz0)
    phi[:, :, :] = 1.0
    cell_type[:, :, :] = CELL_LIQUID
    return HomeVofState(moments=moments, phi=phi, cell_type=cell_type)


def seed_dam_break_column(
    shape: tuple[int, int, int],
    dam_x: int,
    fill_z: int,
    rho_liquid: float = 1.0,
    *,
    interface_phi: float = 0.5,
) -> HomeVofState:
    """Liquid column ``x < dam_x`` and ``z < fill_z``; mark free-surface IF."""
    moments, phi, cell_type = _empty_fields(shape)
    nx, _ny, nz = shape
    dx = max(0, min(int(dam_x), nx))
    fz = max(0, min(int(fill_z), nz))
    if dx > 0 and fz > 0:
        moments.rho[:dx, :, :fz] = float(rho_liquid)
        phi[:dx, :, :fz] = 1.0
        cell_type[:dx, :, :fz] = CELL_LIQUID
    mark_liquid_interfaces(cell_type, phi, interface_phi=interface_phi)
    return HomeVofState(moments=moments, phi=phi, cell_type=cell_type)


def seed_pool(
    shape: tuple[int, int, int],
    fill_z: int,
    rho_liquid: float = 1.0,
    *,
    interface_phi: float = 0.5,
) -> HomeVofState:
    """Still pool: liquid for all ``z < fill_z`` (open free surface on top)."""
    moments, phi, cell_type = _empty_fields(shape)
    nz = shape[2]
    fill = max(0, min(int(fill_z), nz))
    if fill > 0:
        moments.rho[:, :, :fill] = float(rho_liquid)
        phi[:, :, :fill] = 1.0
        cell_type[:, :, :fill] = CELL_LIQUID
    mark_liquid_interfaces(cell_type, phi, interface_phi=interface_phi)
    return HomeVofState(moments=moments, phi=phi, cell_type=cell_type)


def seed_droplet(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    rho_liquid: float = 1.0,
    *,
    interface_phi: float = 0.5,
) -> HomeVofState:
    """Spherical droplet (cell-center lattice coords)."""
    moments, phi, cell_type = _empty_fields(shape)
    nx, ny, nz = shape
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    r2 = float(radius) * float(radius)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dx = (i + 0.5) - cx
                dy = (j + 0.5) - cy
                dz = (k + 0.5) - cz
                if dx * dx + dy * dy + dz * dz <= r2:
                    moments.rho[i, j, k] = float(rho_liquid)
                    phi[i, j, k] = 1.0
                    cell_type[i, j, k] = CELL_LIQUID
    mark_liquid_interfaces(cell_type, phi, interface_phi=interface_phi)
    return HomeVofState(moments=moments, phi=phi, cell_type=cell_type)
