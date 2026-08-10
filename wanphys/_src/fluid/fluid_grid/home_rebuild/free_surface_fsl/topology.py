# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Host transaction for deterministic CPU/CUDA HOME-Free topology changes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..core.constants import D3Q27_DIRECTIONS
from ..core.model import HomeCoreModel
from ..core.state import HomeCoreState
from .state import FslState
from . import topology_kernels


@dataclass(frozen=True)
class FslWarpTopologyDiagnostics:
    interface_to_liquid_count: int
    interface_to_gas_count: int
    gas_to_interface_count: int
    liquid_to_interface_count: int
    cancelled_interface_to_gas_count: int
    fresh_interface_without_donor_count: int
    direct_liquid_gas_link_count: int
    invalid_cell_count: int
    initial_total_mass: float
    final_total_mass: float
    relative_total_mass_drift: float


class FslTopologyUpdater:
    """Resolve dynamic topology through deterministic read-only stages."""

    def __init__(
        self,
        model: HomeCoreModel,
        *,
        transition_tolerance: float = 0.0,
        relative_mass_drift_tolerance: float = 5.0e-6,
    ) -> None:
        if not math.isfinite(transition_tolerance) or transition_tolerance < 0.0:
            raise ValueError("transition_tolerance must be finite and nonnegative")
        if (
            not math.isfinite(relative_mass_drift_tolerance)
            or relative_mass_drift_tolerance < 0.0
        ):
            raise ValueError("relative_mass_drift_tolerance must be finite and nonnegative")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.transition_tolerance = float(transition_tolerance)
        self.relative_mass_drift_tolerance = float(relative_mass_drift_tolerance)
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self.source_recipient_count = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self.recipient_count = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.working_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.stranded_excess = wp.zeros(self.res, dtype=float, device=self.device)
        self.candidates = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.transitions = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.final_flags = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self._invalid_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._missing_donor_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._direct_link_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._cancelled_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._transition_counts = wp.zeros(4, dtype=wp.int32, device=self.device)
        self._initial_mass_cells = wp.zeros(
            self.stride, dtype=wp.float64, device=self.device
        )
        self._final_mass_cells = wp.zeros(
            self.stride, dtype=wp.float64, device=self.device
        )
        self._initial_total = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._final_total = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.last_diagnostics: FslWarpTopologyDiagnostics | None = None

    def _validate(
        self,
        fluid_in: HomeCoreState,
        source: FslState,
        advected_mass: wp.array,
        fluid_out: HomeCoreState,
        destination: FslState,
    ) -> None:
        states = (fluid_in, source, fluid_out, destination)
        if any(state.model is not self.model for state in states):
            raise ValueError("topology updater and states must share one HomeCoreModel")
        if tuple(advected_mass.shape) != self.res or advected_mass.device != self.device:
            raise ValueError("advected_mass must match the topology grid and device")

    def resolve(
        self,
        fluid_in: HomeCoreState,
        source: FslState,
        advected_mass: wp.array,
        fluid_out: HomeCoreState,
        destination: FslState,
    ) -> FslWarpTopologyDiagnostics:
        self._validate(fluid_in, source, advected_mass, fluid_out, destination)
        self.last_diagnostics = None
        self._invalid_count.zero_()
        self._missing_donor_count.zero_()
        self._direct_link_count.zero_()
        self._cancelled_count.zero_()
        self._transition_counts.zero_()
        wp.copy(fluid_out.moments, fluid_in.moments)
        wp.launch(
            topology_kernels.count_active_neighbors_kernel,
            dim=self.res,
            inputs=[source.flags, self._directions, self.source_recipient_count, *self.res],
            device=self.device,
        )
        wp.launch(
            topology_kernels.receive_excess_kernel,
            dim=self.res,
            inputs=[
                advected_mass,
                source.excess_mass,
                source.flags,
                self.source_recipient_count,
                self._directions,
                self.working_mass,
                self.stranded_excess,
                self._initial_mass_cells,
                self._invalid_count,
                *self.res,
            ],
            device=self.device,
        )
        wp.launch(
            topology_kernels.mark_candidates_kernel,
            dim=self.res,
            inputs=[
                fluid_in.moments,
                self.working_mass,
                source.flags,
                self._directions,
                self.candidates,
                self._invalid_count,
                self.transition_tolerance,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            topology_kernels.resolve_growth_kernel,
            dim=self.res,
            inputs=[
                source.flags,
                self.candidates,
                self._directions,
                self.transitions,
                self._cancelled_count,
                *self.res,
            ],
            device=self.device,
        )
        wp.launch(
            topology_kernels.apply_transitions_kernel,
            dim=self.res,
            inputs=[
                source.flags,
                self.transitions,
                self._directions,
                self.final_flags,
                *self.res,
            ],
            device=self.device,
        )
        wp.launch(
            topology_kernels.initialize_fresh_moments_kernel,
            dim=self.res,
            inputs=[
                fluid_in.moments,
                source.flags,
                self.final_flags,
                self.transitions,
                self._directions,
                fluid_out.moments,
                self._missing_donor_count,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            topology_kernels.commit_topology_kernel,
            dim=self.res,
            inputs=[
                fluid_out.moments,
                self.working_mass,
                self.stranded_excess,
                source.flags,
                self.transitions,
                self.final_flags,
                destination.mass,
                destination.fill_level,
                destination.excess_mass,
                self._invalid_count,
                self._transition_counts,
                self._final_mass_cells,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.copy(destination.flags, self.final_flags)
        wp.launch(
            topology_kernels.count_active_neighbors_kernel,
            dim=self.res,
            inputs=[destination.flags, self._directions, self.recipient_count, *self.res],
            device=self.device,
        )
        wp.launch(
            topology_kernels.validate_separation_kernel,
            dim=self.res,
            inputs=[destination.flags, self._directions, self._direct_link_count, *self.res],
            device=self.device,
        )
        wp.utils.array_sum(self._initial_mass_cells, out=self._initial_total)
        wp.utils.array_sum(self._final_mass_cells, out=self._final_total)
        wp.synchronize_device(self.device)

        invalid = int(self._invalid_count.numpy()[0])
        missing = int(self._missing_donor_count.numpy()[0])
        direct = int(self._direct_link_count.numpy()[0])
        counts = self._transition_counts.numpy()
        initial_total = float(self._initial_total.numpy()[0])
        final_total = float(self._final_total.numpy()[0])
        relative_drift = (
            (final_total - initial_total) / initial_total if initial_total != 0.0 else 0.0
        )
        diagnostics = FslWarpTopologyDiagnostics(
            interface_to_liquid_count=int(counts[0]),
            interface_to_gas_count=int(counts[1]),
            gas_to_interface_count=int(counts[2]),
            liquid_to_interface_count=int(counts[3]),
            cancelled_interface_to_gas_count=int(self._cancelled_count.numpy()[0]),
            fresh_interface_without_donor_count=missing,
            direct_liquid_gas_link_count=direct,
            invalid_cell_count=invalid,
            initial_total_mass=initial_total,
            final_total_mass=final_total,
            relative_total_mass_drift=relative_drift,
        )
        if invalid:
            raise FloatingPointError(f"topology transaction contains {invalid} invalid cells")
        if missing:
            raise RuntimeError(f"{missing} fresh interface cells have no active donor")
        if direct:
            raise RuntimeError(f"topology transaction produced {direct} liquid-gas links")
        if abs(relative_drift) > self.relative_mass_drift_tolerance:
            raise FloatingPointError(
                f"topology relative mass drift {relative_drift} exceeds tolerance "
                f"{self.relative_mass_drift_tolerance}"
            )
        self.last_diagnostics = diagnostics
        return diagnostics
