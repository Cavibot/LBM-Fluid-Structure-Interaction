# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent HOME-LBM state."""

from __future__ import annotations

import warp as wp

from .constants import MOMENT_COUNT
from .model import HomeLbmModel


class HomeLbmState:
    """GPU state containing ten moment scalars per cell and no persistent ``f_i``."""

    def __init__(self, model: HomeLbmModel, requires_grad: bool = False) -> None:
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.cell_count = self.res[0] * self.res[1] * self.res[2]
        self.device = model._device
        self.requires_grad = requires_grad

        self.moments = wp.zeros(
            MOMENT_COUNT * self.cell_count,
            dtype=float,
            device=self.device,
            requires_grad=requires_grad,
        )
        self.solid_phi = wp.full(self.res, 1000.0, dtype=float, device=self.device)
        self.solid_body_id = wp.full(self.res, -1, dtype=wp.int32, device=self.device)

    @property
    def persistent_scalar_count(self) -> int:
        return MOMENT_COUNT * self.cell_count

    def clear(self) -> None:
        self.moments.zero_()
        self.solid_phi.fill_(1000.0)
        self.solid_body_id.fill_(-1)

    def clear_forces(self) -> None:
        """Satisfy the DomainState protocol; fluid forces are solver temporaries."""

    def copy_from(self, other: HomeLbmState) -> None:
        if self.res != other.res:
            raise ValueError(f"cannot copy HOME states with resolutions {other.res} and {self.res}")
        wp.copy(self.moments, other.moments)
        wp.copy(self.solid_phi, other.solid_phi)
        wp.copy(self.solid_body_id, other.solid_body_id)
