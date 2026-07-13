# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM solver pipeline (skeleton for stage 1 — MomSwap only).

Reference:
    [REF] Home-FSLBM inc/3D/gpu/mrLbmSolverGpu3D.cu — mrSolver3DGpu.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ..base import FluidGridSolverBase
from .model import HomeFSLbmModel
from .state import HomeFSLbmState


class HomeFSLbmSolver(FluidGridSolverBase):
    """HOME-FSLBM solver with double-buffered moment swap.

    Stage 1: only performs MomSwap (pointer exchange) on the two
    flat moment arrays so that the domain double-buffer cycle
    runs without error.  The full stream+collide pipeline is
    added in stage 2.

    Parameters
    ----------
    model:
        Static HOME-FSLBM configuration.
    """

    def __init__(self, model: HomeFSLbmModel) -> None:
        self.model: HomeFSLbmModel = model
        self.nx: int = int(model.nx)
        self.ny: int = int(model.ny)
        self.nz: int = int(model.nz)
        self.device: wp.Device = model._device
        self._total_num: int = self.nx * self.ny * self.nz

    # ------------------------------------------------------------------
    # DomainSolver protocol
    # ------------------------------------------------------------------

    def step(
        self,
        state_in: HomeFSLbmState,
        state_out: HomeFSLbmState,
        dt: float,
        **kwargs: Any,
    ) -> None:
        """Advance simulation by *dt*.

        Stage 1: copies state_in.f_mom to state_out.f_mom_post, then
        performs MomSwap so the domain's double-buffer cycle completes.
        This enables the domain step()/state access to function correctly
        before the full LBM pipeline is implemented.

        Args:
            state_in: Current state (read from ``.f_mom``).
            state_out: Next state (write target ``.f_mom_post``).
            dt: Timestep in seconds (ignored, LBM uses lattice dt=1).
            **kwargs: Ignored.
        """
        # Copy current moments → post array
        wp.copy(state_out.f_mom_post, state_in.f_mom)

        # Copy mass / phi / flag / force fields (pass-through for now)
        wp.copy(state_out.mass, state_in.mass)
        wp.copy(state_out.massex, state_in.massex)
        wp.copy(state_out.phi, state_in.phi)
        wp.copy(state_out.flag, state_in.flag)
        wp.copy(state_out.force_x, state_in.force_x)
        wp.copy(state_out.force_y, state_in.force_y)
        wp.copy(state_out.force_z, state_in.force_z)
        wp.copy(state_out.disjoin_force, state_in.disjoin_force)

        # MomSwap: f_mom ↔ f_mom_post (pointer exchange)
        # Done at the domain level in HomeFSLbmDomain.step().
        # Here we just ensure out == in so the swap produces no net change.
