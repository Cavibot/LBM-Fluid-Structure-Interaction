# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent sharp-interface state layered on a HOME moment state."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..model import HomeLbmModel
from ..state import HomeLbmState
from .reference import HomeFreeStateDiagnostics, classify_fill_levels, validate_state_fields


class HomeFreeState:
    """Committed liquid state plus per-recipient excess mass and momentum."""

    def __init__(self, model: HomeLbmModel) -> None:
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.cell_count = int(np.prod(self.res))
        self.device = model._device
        self.mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.fill_level = wp.zeros(self.res, dtype=float, device=self.device)
        self.excess_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.excess_momentum = wp.zeros(
            3 * self.cell_count, dtype=float, device=self.device
        )
        self.flags = wp.zeros(self.res, dtype=wp.int32, device=self.device)

    @property
    def persistent_scalar_count(self) -> int:
        return 7 * self.cell_count

    def initialize_from_fill_level(
        self,
        fluid_state: HomeLbmState,
        fill_level: np.ndarray,
        *,
        validate_topology: bool = True,
    ) -> HomeFreeStateDiagnostics:
        if fluid_state.res != self.res:
            raise ValueError("fluid and HOME-FREE state resolutions must match")
        fill = np.asarray(fill_level, dtype=np.float32)
        if fill.shape != self.res:
            raise ValueError(f"fill_level shape {fill.shape} does not match grid {self.res}")
        solid = fluid_state.solid_phi.numpy() < 0.0
        fill = fill.copy()
        fill[solid] = 0.0
        flags = classify_fill_levels(fill, solid)
        density = fluid_state.moments.numpy()[: self.cell_count].reshape(self.res)
        mass = density * fill
        excess = np.zeros(self.res, dtype=np.float32)
        diagnostics = validate_state_fields(
            density,
            mass,
            fill,
            excess,
            np.zeros(self.res + (3,), dtype=np.float32),
            flags,
            periodic=self.model.periodic,
            require_interface_separation=validate_topology,
        )
        self.mass.assign(mass)
        self.fill_level.assign(fill)
        self.excess_mass.zero_()
        self.excess_momentum.zero_()
        self.flags.assign(flags)
        return diagnostics

    def copy_from(self, other: HomeFreeState) -> None:
        if self.res != other.res:
            raise ValueError("cannot copy HOME-FREE states with different resolutions")
        wp.copy(self.mass, other.mass)
        wp.copy(self.fill_level, other.fill_level)
        wp.copy(self.excess_mass, other.excess_mass)
        wp.copy(self.excess_momentum, other.excess_momentum)
        wp.copy(self.flags, other.flags)

    def validate(
        self,
        fluid_state: HomeLbmState,
        *,
        require_interface_separation: bool = True,
        allow_interface_endpoints: bool = False,
        require_mass_fill_consistency: bool = True,
    ) -> HomeFreeStateDiagnostics:
        if fluid_state.res != self.res:
            raise ValueError("fluid and HOME-FREE state resolutions must match")
        density = fluid_state.moments.numpy()[: self.cell_count].reshape(self.res)
        return validate_state_fields(
            density,
            self.mass.numpy(),
            self.fill_level.numpy(),
            self.excess_mass.numpy(),
            np.moveaxis(
                self.excess_momentum.numpy().reshape(3, *self.res), 0, -1
            ),
            self.flags.numpy(),
            periodic=self.model.periodic,
            require_interface_separation=require_interface_separation,
            allow_interface_endpoints=allow_interface_endpoints,
            require_mass_fill_consistency=require_mass_fill_consistency,
        )
