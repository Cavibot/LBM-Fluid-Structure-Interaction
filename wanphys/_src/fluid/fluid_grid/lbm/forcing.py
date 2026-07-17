# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unified physical-force computation and hydrodynamic closure kernels."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .contracts import ForceModel


@dataclass(frozen=True)
class ForceProvider:
    """Host-side description of a provider that returns force density ``F``."""

    model: ForceModel

    @property
    def includes_gravity(self) -> bool:
        return self.model in (ForceModel.GRAVITY, ForceModel.GRAVITY_SHAN_CHEN)

    @property
    def includes_shan_chen(self) -> bool:
        return self.model in (ForceModel.SHAN_CHEN, ForceModel.GRAVITY_SHAN_CHEN)


@wp.kernel
def compose_force_density_kernel(
    rho: wp.array3d(dtype=float),
    sc_fx: wp.array3d(dtype=float),
    sc_fy: wp.array3d(dtype=float),
    sc_fz: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    gx: float,
    gy: float,
    gz: float,
    include_shan_chen: int,
) -> None:
    """Compose total force density ``F = rho*g + F_sc`` exactly once."""

    i, j, k = wp.tid()
    r = rho[i, j, k]
    # Keep conditional scalar updates separate.  Warp 1.12 can otherwise
    # emit an unterminated CPU kernel for tuple assignment in a branch.
    sx = 0.0
    sy = 0.0
    sz = 0.0
    if include_shan_chen != 0:
        sx = sc_fx[i, j, k]
        sy = sc_fy[i, j, k]
        sz = sc_fz[i, j, k]
    fx[i, j, k] = r * gx + sx
    fy[i, j, k] = r * gy + sy
    fz[i, j, k] = r * gz + sz


@wp.kernel
def hydro_closure_kernel(
    rho: wp.array3d(dtype=float),
    jx: wp.array3d(dtype=float),
    jy: wp.array3d(dtype=float),
    jz: wp.array3d(dtype=float),
    fx: wp.array3d(dtype=float),
    fy: wp.array3d(dtype=float),
    fz: wp.array3d(dtype=float),
    ux: wp.array3d(dtype=float),
    uy: wp.array3d(dtype=float),
    uz: wp.array3d(dtype=float),
) -> None:
    """Compute physical velocity ``u = (j + F/2) / rho``."""

    i, j, k = wp.tid()
    inverse_rho = 1.0 / wp.max(rho[i, j, k], 1.0e-12)
    ux[i, j, k] = (jx[i, j, k] + 0.5 * fx[i, j, k]) * inverse_rho
    uy[i, j, k] = (jy[i, j, k] + 0.5 * fy[i, j, k]) * inverse_rho
    uz[i, j, k] = (jz[i, j, k] + 0.5 * fz[i, j, k]) * inverse_rho
