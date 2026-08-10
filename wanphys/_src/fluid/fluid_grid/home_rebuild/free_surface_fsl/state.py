# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent paper FSL state layered on a HomeCoreState."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..core.model import HomeCoreModel
from ..core.state import HomeCoreState
from .reference import FslStateDiagnostics, initialize_fsl_fields, validate_fsl_fields


class FslState:
    def __init__(self, model: HomeCoreModel) -> None:
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.cell_count = int(np.prod(self.res))
        self.device = model._device
        self.mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.fill_level = wp.zeros(self.res, dtype=float, device=self.device)
        self.excess_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.flags = wp.zeros(self.res, dtype=wp.int32, device=self.device)

    @property
    def persistent_scalar_count(self) -> int:
        return 4 * self.cell_count

    def initialize_from_fill_level(
        self,
        fluid_state: HomeCoreState,
        fill_level: np.ndarray,
        *,
        validate_topology: bool = True,
    ) -> FslStateDiagnostics:
        if fluid_state.res != self.res:
            raise ValueError("HOME core and FSL state resolutions must match")
        density = fluid_state.moments.numpy()[: self.cell_count].reshape(self.res)
        mass, excess, flags, diagnostics = initialize_fsl_fields(
            density,
            fill_level,
            periodic=self.model.periodic,
            validate_topology=validate_topology,
        )
        self.mass.assign(mass.astype(np.float32))
        self.fill_level.assign(np.asarray(fill_level, dtype=np.float32))
        self.excess_mass.assign(excess.astype(np.float32))
        self.flags.assign(flags)
        return diagnostics

    def validate(
        self,
        fluid_state: HomeCoreState,
        *,
        require_interface_separation: bool = True,
        require_mass_fill_consistency: bool = True,
        allow_interface_endpoints: bool = False,
    ) -> FslStateDiagnostics:
        if fluid_state.res != self.res:
            raise ValueError("HOME core and FSL state resolutions must match")
        density = fluid_state.moments.numpy()[: self.cell_count].reshape(self.res)
        return validate_fsl_fields(
            density,
            self.mass.numpy(),
            self.fill_level.numpy(),
            self.excess_mass.numpy(),
            self.flags.numpy(),
            periodic=self.model.periodic,
            require_interface_separation=require_interface_separation,
            require_mass_fill_consistency=require_mass_fill_consistency,
            allow_interface_endpoints=allow_interface_endpoints,
        )

    def copy_from(self, other: FslState) -> None:
        if self.res != other.res:
            raise ValueError("cannot copy FSL states with different resolutions")
        wp.copy(self.mass, other.mass)
        wp.copy(self.fill_level, other.fill_level)
        wp.copy(self.excess_mass, other.excess_mass)
        wp.copy(self.flags, other.flags)
