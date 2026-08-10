# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Ten persistent scalars per cell, with no boundary or interface state."""

from __future__ import annotations

import warp as wp

from .constants import MOMENT_COUNT
from .model import HomeCoreModel


class HomeCoreState:
    def __init__(self, model: HomeCoreModel, requires_grad: bool = False) -> None:
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

    @property
    def persistent_scalar_count(self) -> int:
        return MOMENT_COUNT * self.cell_count

    def clear(self) -> None:
        self.moments.zero_()

    def clear_forces(self) -> None:
        """The core stores no accumulated force field."""

    def copy_from(self, other: HomeCoreState) -> None:
        if self.res != other.res:
            raise ValueError(f"cannot copy HOME core states with resolutions {other.res} and {self.res}")
        wp.copy(self.moments, other.moments)
