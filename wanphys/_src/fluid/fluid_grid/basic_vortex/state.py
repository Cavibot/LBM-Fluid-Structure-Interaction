# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Fluid grid state - time-varying data for Position Based Fluids simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import warp as wp

if TYPE_CHECKING:
    from .model import FluidGridModel

from ..base import FluidGridStateBase
from .sparse_hash_grid import BLOCK_VOL


class FluidGridState(FluidGridStateBase):
    def __init__(self, model: FluidGridModel, max_blocks: int = 4096, requires_grad: bool = False) -> None:
        #0407
        #super().__init__(model, requires_grad)
        self.model = model

        nx = int(model.fluid_grid_res[0])
        ny = int(model.fluid_grid_res[1])
        nz = int(model.fluid_grid_res[2])
        self.res = (nx, ny, nz)
        self.device = model._device
        self.requires_grad = requires_grad

        device = model._device
        pool_len = max_blocks * BLOCK_VOL

        # cover the status of FluidGridStateBase as 1D sparse pool
        self.vel_u = wp.zeros(pool_len, dtype=float, device=device)
        self.vel_v = wp.zeros(pool_len, dtype=float, device=device)
        self.vel_w = wp.zeros(pool_len, dtype=float, device=device)
        self.density = wp.zeros(pool_len, dtype=float, device=device)
        self.pressure = wp.zeros(pool_len, dtype=float, device=device)