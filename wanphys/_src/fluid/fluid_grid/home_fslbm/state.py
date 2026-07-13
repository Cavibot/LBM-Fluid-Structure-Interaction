# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM GPU-resident time-varying state (10-moment SoA layout).

Reference:
    [REF] Home-FSLBM inc/3D/cpu/mrFlow3D.h — mrFlow3D data layout.
    [P4] Wang et al. 2025, Eq.(16) — 10-moment encoding.

Layout per cell (stored in SoA 1D arrays of length ``total_num``):
    Offset 0: rho   — density
    Offset 1: u_x   — x-velocity
    Offset 2: u_y   — y-velocity
    Offset 3: u_z   — z-velocity
    Offset 4: S_xx  — xx stress (stored as S_xx = pixx/rho - cs2)
    Offset 5: S_xy  — xy stress
    Offset 6: S_xz  — xz stress
    Offset 7: S_yy  — yy stress
    Offset 8: S_yz  — yz stress
    Offset 9: S_zz  — zz stress

Additional 3D fields per cell:
    mass        — actual fluid mass
    massex      — excess mass pending redistribution
    phi         — VOF fill fraction [0, 1]
    flag        — cell type bitmask (TYPE_F / TYPE_I / TYPE_G / TYPE_S / ...)
    force_x/y/z — body force components (= rho * g)
    disjoin_force — separation force (initialised to zero)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from wanphys._src.core.domain import DomainState

if TYPE_CHECKING:
    from .model import HomeFSLbmModel


class HomeFSLbmState(DomainState):
    """GPU-resident state for a HOME-FSLBM simulation.

    Stores 10 velocity moments per cell in SoA layout plus VOF and
    boundary metadata.  Memory footprint: ~12 scalars per cell.

    Parameters
    ----------
    model:
        Static HOME-FSLBM configuration.
    requires_grad:
        If ``True``, allocate arrays with gradient tracking.
    """

    def __init__(
        self, model: HomeFSLbmModel, requires_grad: bool = False
    ) -> None:
        self.model: HomeFSLbmModel = model
        nx: int = int(model.nx)
        ny: int = int(model.ny)
        nz: int = int(model.nz)
        self.res: tuple[int, int, int] = (nx, ny, nz)
        self.device: wp.Device = model._device
        self.requires_grad: bool = requires_grad

        # Total number of cells
        self._total_num: int = nx * ny * nz

        # ---- 10 velocity moments (SoA: 10 * total_num flat array) -----------
        self.f_mom: wp.array = wp.zeros(
            10 * self._total_num,
            dtype=float,
            device=self.device,
            requires_grad=requires_grad,
        )
        self.f_mom_post: wp.array = wp.zeros(
            10 * self._total_num,
            dtype=float,
            device=self.device,
            requires_grad=requires_grad,
        )

        # ---- Per-cell scalar fields -----------------------------------------
        self.mass: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.massex: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.phi: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        # flag defaults to TYPE_G for all cells (empty domain)
        self.flag: wp.array3d = wp.full(
            (nx, ny, nz),
            0b00000100,  # TYPE_G
            dtype=wp.int32,
            device=self.device,
        )
        self.force_x: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.force_y: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.force_z: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.disjoin_force: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )

    # ------------------------------------------------------------------
    # DomainState protocol
    # ------------------------------------------------------------------

    def clear_forces(self) -> None:
        """Reset accumulated force arrays to zero."""
        self.force_x.zero_()
        self.force_y.zero_()
        self.force_z.zero_()
        self.disjoin_force.zero_()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def total_num(self) -> int:
        """Total cell count ``nx * ny * nz``."""
        return self._total_num

    def _offset(self, moment_index: int) -> int:
        """Byte-safe offset into the flat SoA array."""
        return moment_index * self._total_num
