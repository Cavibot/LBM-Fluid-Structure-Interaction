# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent axis-aligned wall descriptor for the isolated HOME-Free composition."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..core.model import HomeCoreModel
from ..core.state import HomeCoreState
from .state import FslState
from .wall_reference import (
    FslHydrostaticFields,
    axis_aligned_wall_mask,
    closed_box_wall_mask,
    initialize_hydrostatic_fields,
    validate_wall_mask,
)


class FslWallMask:
    def __init__(
        self,
        model: HomeCoreModel,
        solid_mask: np.ndarray,
        *,
        closed_axes: tuple[bool, bool, bool] = (True, True, True),
    ) -> None:
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.closed_axes = tuple(bool(value) for value in closed_axes)
        solid = validate_wall_mask(solid_mask, closed_axes=self.closed_axes)
        if solid.shape != self.res:
            raise ValueError("solid_mask must match the HOME grid")
        self.host = solid.copy()
        self.device = wp.array(
            solid.astype(np.int32), dtype=wp.int32, device=model._device
        )

    @classmethod
    def closed_box(cls, model: HomeCoreModel) -> FslWallMask:
        return cls(
            model,
            closed_box_wall_mask((int(model.nx), int(model.ny), int(model.nz))),
        )

    @classmethod
    def periodic_depth_channel(cls, model: HomeCoreModel) -> FslWallMask:
        """Create x/y walls with a periodic z direction for a true 2-D benchmark."""

        closed_axes = (True, True, False)
        return cls(
            model,
            axis_aligned_wall_mask(
                (int(model.nx), int(model.ny), int(model.nz)),
                closed_axes=closed_axes,
            ),
            closed_axes=closed_axes,
        )

    @property
    def solid_cell_count(self) -> int:
        return int(np.count_nonzero(self.host))

    def initialize_hydrostatic(
        self,
        fluid_state: HomeCoreState,
        fsl_state: FslState,
        fill_level: np.ndarray,
        *,
        gas_density: float,
        gravity_axis: int,
        surface_coordinate: float,
    ) -> FslHydrostaticFields:
        if fluid_state.model is not self.model or fsl_state.model is not self.model:
            raise ValueError("wall, HOME, and FSL states must share one model")
        acceleration = float(self.model.lattice_acceleration[gravity_axis])
        fields = initialize_hydrostatic_fields(
            fill_level,
            self.host,
            gas_density=gas_density,
            gravity_axis=gravity_axis,
            lattice_acceleration=acceleration,
            surface_coordinate=surface_coordinate,
            closed_axes=self.closed_axes,
        )
        soa = np.ascontiguousarray(fields.moments.reshape(-1, 10).T.reshape(-1))
        fluid_state.moments.assign(soa.astype(np.float32))
        fsl_state.mass.assign(fields.mass.astype(np.float32))
        fsl_state.fill_level.assign(fields.fill_level.astype(np.float32))
        fsl_state.excess_mass.assign(fields.excess_mass.astype(np.float32))
        fsl_state.flags.assign(fields.flags)
        return fields
