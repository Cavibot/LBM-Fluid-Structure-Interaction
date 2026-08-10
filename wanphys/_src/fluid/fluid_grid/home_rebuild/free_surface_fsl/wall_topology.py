# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic dynamic topology transaction that excludes closed-wall cells."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..core.constants import D3Q27_DIRECTIONS
from ..core.model import HomeCoreModel
from ..core.state import HomeCoreState
from .state import FslState
from .topology import FslWarpTopologyDiagnostics
from .walls import FslWallMask
from . import wall_kernels


class FslWallTopologyUpdater:
    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        *,
        transition_tolerance: float = 0.0,
        relative_mass_drift_tolerance: float = 5.0e-6,
    ) -> None:
        if walls.model is not model:
            raise ValueError("wall topology updater and mask must share one model")
        if not math.isfinite(transition_tolerance) or transition_tolerance < 0.0:
            raise ValueError("transition_tolerance must be finite and nonnegative")
        if not math.isfinite(relative_mass_drift_tolerance) or relative_mass_drift_tolerance < 0.0:
            raise ValueError("relative_mass_drift_tolerance must be finite and nonnegative")
        self.model = model
        self.walls = walls
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.transition_tolerance = float(transition_tolerance)
        self.relative_mass_drift_tolerance = float(relative_mass_drift_tolerance)
        self.directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self.source_recipient_count = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.recipient_count = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.working_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.stranded = wp.zeros(self.res, dtype=float, device=self.device)
        self.candidates = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.transitions = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.final_flags = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self._invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._missing = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._direct = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._cancelled = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._counts = wp.zeros(4, dtype=wp.int32, device=self.device)
        self._initial_cells = wp.zeros(self.stride, dtype=wp.float64, device=self.device)
        self._final_cells = wp.zeros(self.stride, dtype=wp.float64, device=self.device)
        self._initial_total = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._final_total = wp.zeros(1, dtype=wp.float64, device=self.device)

    def resolve(
        self,
        fluid_in: HomeCoreState,
        source: FslState,
        advected_mass: wp.array,
        fluid_out: HomeCoreState,
        destination: FslState,
    ) -> FslWarpTopologyDiagnostics:
        states = (fluid_in, source, fluid_out, destination)
        if any(state.model is not self.model for state in states):
            raise ValueError("wall topology updater and states must share one model")
        if tuple(advected_mass.shape) != self.res or advected_mass.device != self.device:
            raise ValueError("advected_mass must match the wall topology grid")
        self._invalid.zero_()
        self._missing.zero_()
        self._direct.zero_()
        self._cancelled.zero_()
        self._counts.zero_()
        wp.copy(fluid_out.moments, fluid_in.moments)
        solid = self.walls.device
        wp.launch(
            wall_kernels.wall_count_active_neighbors_kernel,
            dim=self.res,
            inputs=[source.flags, solid, self.directions, self.source_recipient_count, *self.res],
            device=self.device,
        )
        wp.launch(
            wall_kernels.wall_receive_excess_kernel,
            dim=self.res,
            inputs=[
                advected_mass, source.excess_mass, source.flags, solid,
                self.source_recipient_count, self.directions, self.working_mass,
                self.stranded, self._initial_cells, self._invalid, *self.res,
            ],
            device=self.device,
        )
        wp.launch(
            wall_kernels.wall_mark_candidates_kernel,
            dim=self.res,
            inputs=[
                fluid_in.moments, self.working_mass, source.flags, solid,
                self.directions, self.candidates, self._invalid,
                self.transition_tolerance, *self.res, self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            wall_kernels.wall_resolve_growth_kernel,
            dim=self.res,
            inputs=[
                source.flags, solid, self.candidates, self.directions,
                self.transitions, self._cancelled, *self.res,
            ],
            device=self.device,
        )
        wp.launch(
            wall_kernels.wall_apply_transitions_kernel,
            dim=self.res,
            inputs=[
                source.flags, solid, self.transitions, self.directions,
                self.final_flags, *self.res,
            ],
            device=self.device,
        )
        wp.launch(
            wall_kernels.wall_initialize_fresh_kernel,
            dim=self.res,
            inputs=[
                fluid_in.moments, source.flags, solid, self.final_flags,
                self.transitions, self.directions, fluid_out.moments,
                self._missing, *self.res, self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            wall_kernels.wall_commit_topology_kernel,
            dim=self.res,
            inputs=[
                fluid_out.moments, self.working_mass, self.stranded,
                source.flags, solid, self.transitions, self.final_flags,
                destination.mass, destination.fill_level, destination.excess_mass,
                self._invalid, self._counts, self._final_cells,
                self.res[1], self.res[2], self.stride,
            ],
            device=self.device,
        )
        wp.copy(destination.flags, self.final_flags)
        wp.launch(
            wall_kernels.wall_count_active_neighbors_kernel,
            dim=self.res,
            inputs=[destination.flags, solid, self.directions, self.recipient_count, *self.res],
            device=self.device,
        )
        wp.launch(
            wall_kernels.wall_validate_separation_kernel,
            dim=self.res,
            inputs=[destination.flags, solid, self.directions, self._direct, *self.res],
            device=self.device,
        )
        wp.utils.array_sum(self._initial_cells, out=self._initial_total)
        wp.utils.array_sum(self._final_cells, out=self._final_total)
        wp.synchronize_device(self.device)
        invalid = int(self._invalid.numpy()[0])
        missing = int(self._missing.numpy()[0])
        direct = int(self._direct.numpy()[0])
        counts = self._counts.numpy()
        initial = float(self._initial_total.numpy()[0])
        final = float(self._final_total.numpy()[0])
        drift = (final - initial) / initial if initial != 0.0 else 0.0
        diagnostics = FslWarpTopologyDiagnostics(
            interface_to_liquid_count=int(counts[0]),
            interface_to_gas_count=int(counts[1]),
            gas_to_interface_count=int(counts[2]),
            liquid_to_interface_count=int(counts[3]),
            cancelled_interface_to_gas_count=int(self._cancelled.numpy()[0]),
            fresh_interface_without_donor_count=missing,
            direct_liquid_gas_link_count=direct,
            invalid_cell_count=invalid,
            initial_total_mass=initial,
            final_total_mass=final,
            relative_total_mass_drift=drift,
        )
        if invalid:
            raise FloatingPointError(f"wall topology contains {invalid} invalid cells")
        if missing:
            raise RuntimeError(f"{missing} fresh wall-interface cells have no donor")
        if direct:
            raise RuntimeError(f"wall topology produced {direct} liquid-gas links")
        if abs(drift) > self.relative_mass_drift_tolerance:
            raise FloatingPointError(
                f"wall topology mass drift {drift} exceeds "
                f"{self.relative_mass_drift_tolerance}"
            )
        return diagnostics
