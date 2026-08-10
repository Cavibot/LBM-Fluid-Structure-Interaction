# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Transactional Warp owner for strict geometric FSL topology commit."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..core import HomeCoreModel, HomeCoreState
from ..free_surface_fsl import FslState, FslWallMask
from .topology_reference import PlicTopologyDiagnostics
from . import topology_kernels


class PlicTopologyResolver:
    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        *,
        endpoint_tolerance: float = 4.0e-7,
    ) -> None:
        if walls.model is not model:
            raise ValueError("PLIC topology and walls must share one HOME model")
        if not 0.0 <= endpoint_tolerance < 0.5:
            raise ValueError("endpoint_tolerance must be in [0, 0.5)")
        self.model = model
        self.walls = walls
        self.res = walls.res
        self.stride = int(np.prod(self.res))
        self.endpoint_tolerance = float(endpoint_tolerance)
        self.classified_flags = wp.zeros(
            self.res, dtype=wp.int32, device=model._device
        )
        self.target_flags = wp.zeros(self.res, dtype=wp.int32, device=model._device)
        self.donor_count = wp.zeros(self.res, dtype=wp.int32, device=model._device)
        self._zero_momentum = wp.zeros(
            self.res, dtype=wp.vec3, device=model._device
        )
        self.closed_fill = wp.zeros(self.res, dtype=float, device=model._device)
        self.closed_mass = wp.zeros(self.res, dtype=float, device=model._device)
        self.closed_momentum = wp.zeros(
            self.res, dtype=wp.vec3, device=model._device
        )
        self._counts = wp.zeros(14, dtype=wp.int32, device=model._device)
        self._tail_volume_cells = wp.zeros(
            self.stride, dtype=wp.float64, device=model._device
        )
        self._tail_mass_cells = wp.zeros_like(self._tail_volume_cells)
        self._tail_momentum_x_cells = wp.zeros_like(self._tail_volume_cells)
        self._tail_momentum_y_cells = wp.zeros_like(self._tail_volume_cells)
        self._tail_momentum_z_cells = wp.zeros_like(self._tail_volume_cells)
        self._recipient_capacity_cells = wp.zeros_like(self._tail_volume_cells)
        self._tail_volume = wp.zeros(1, dtype=wp.float64, device=model._device)
        self._tail_mass = wp.zeros_like(self._tail_volume)
        self._tail_momentum_x = wp.zeros_like(self._tail_volume)
        self._tail_momentum_y = wp.zeros_like(self._tail_volume)
        self._tail_momentum_z = wp.zeros_like(self._tail_volume)
        self._recipient_capacity = wp.zeros_like(self._tail_volume)
        self.last_diagnostics: PlicTopologyDiagnostics | None = None

    def resolve_volume_mass(
        self,
        candidate_fluid: HomeCoreState,
        source_fsl: FslState,
        transported_fill: wp.array,
        transported_mass: wp.array,
        destination_fluid: HomeCoreState,
        destination_fsl: FslState,
    ) -> PlicTopologyDiagnostics:
        """Commit geometric volume and mass without a transported momentum ledger."""

        return self.resolve(
            candidate_fluid,
            source_fsl,
            transported_fill,
            transported_mass,
            self._zero_momentum,
            destination_fluid,
            destination_fsl,
        )

    def resolve(
        self,
        candidate_fluid: HomeCoreState,
        source_fsl: FslState,
        transported_fill: wp.array,
        transported_mass: wp.array,
        transported_momentum: wp.array,
        destination_fluid: HomeCoreState,
        destination_fsl: FslState,
    ) -> PlicTopologyDiagnostics:
        states = (candidate_fluid, destination_fluid, source_fsl, destination_fsl)
        if any(state.model is not self.model for state in states):
            raise ValueError("PLIC topology states must share one HOME model")
        if any(tuple(field.shape) != self.res for field in (transported_fill, transported_mass, transported_momentum)):
            raise ValueError("PLIC transported fields must match the cell grid")
        if transported_momentum.dtype != wp.vec3:
            raise TypeError("PLIC transported momentum must use wp.vec3")
        self._counts.zero_()
        periodic = tuple(int(not value) for value in self.walls.closed_axes)
        wp.launch(
            topology_kernels.prepare_endpoint_closure_kernel,
            dim=self.res,
            inputs=[
                transported_fill,
                transported_mass,
                transported_momentum,
                self.walls.device,
                self._tail_volume_cells,
                self._tail_mass_cells,
                self._tail_momentum_x_cells,
                self._tail_momentum_y_cells,
                self._tail_momentum_z_cells,
                self._recipient_capacity_cells,
                self._counts,
                self.endpoint_tolerance,
                self.res[1],
                self.res[2],
            ],
            device=self.model._device,
        )
        for cells, total in (
            (self._tail_volume_cells, self._tail_volume),
            (self._tail_mass_cells, self._tail_mass),
            (self._tail_momentum_x_cells, self._tail_momentum_x),
            (self._tail_momentum_y_cells, self._tail_momentum_y),
            (self._tail_momentum_z_cells, self._tail_momentum_z),
            (self._recipient_capacity_cells, self._recipient_capacity),
        ):
            wp.utils.array_sum(cells, out=total)
        wp.launch(
            topology_kernels.validate_endpoint_closure_totals_kernel,
            dim=1,
            inputs=[
                self._tail_volume,
                self._tail_mass,
                self._tail_momentum_x,
                self._tail_momentum_y,
                self._tail_momentum_z,
                self._recipient_capacity,
                self._counts,
            ],
            device=self.model._device,
        )
        wp.launch(
            topology_kernels.apply_endpoint_closure_kernel,
            dim=self.res,
            inputs=[
                transported_fill,
                transported_mass,
                transported_momentum,
                self.walls.device,
                self._tail_volume,
                self._tail_mass,
                self._tail_momentum_x,
                self._tail_momentum_y,
                self._tail_momentum_z,
                self._recipient_capacity,
                self.closed_fill,
                self.closed_mass,
                self.closed_momentum,
                self.endpoint_tolerance,
            ],
            device=self.model._device,
        )
        wp.launch(
            topology_kernels.validate_separation_kernel,
            dim=self.res,
            inputs=[
                source_fsl.flags,
                self.walls.device,
                self._counts,
                *periodic,
                *self.res,
            ],
            device=self.model._device,
        )
        wp.launch(
            topology_kernels.classify_topology_kernel,
            dim=self.res,
            inputs=[
                source_fsl.flags,
                self.closed_fill,
                self.closed_mass,
                self.closed_momentum,
                self.walls.device,
                self.classified_flags,
                self._counts,
                self.endpoint_tolerance,
            ],
            device=self.model._device,
        )
        wp.launch(
            topology_kernels.repair_separation_kernel,
            dim=self.res,
            inputs=[
                self.classified_flags,
                self.walls.device,
                self.target_flags,
                self._counts,
                *periodic,
                *self.res,
            ],
            device=self.model._device,
        )
        wp.launch(
            topology_kernels.count_topology_transitions_kernel,
            dim=self.res,
            inputs=[
                source_fsl.flags,
                self.target_flags,
                self.closed_fill,
                self.walls.device,
                self._counts,
                self.endpoint_tolerance,
            ],
            device=self.model._device,
        )
        wp.launch(
            topology_kernels.validate_separation_kernel,
            dim=self.res,
            inputs=[self.target_flags, self.walls.device, self._counts, *periodic, *self.res],
            device=self.model._device,
        )
        wp.launch(
            topology_kernels.count_donors_kernel,
            dim=self.res,
            inputs=[
                candidate_fluid.moments,
                source_fsl.flags,
                self.target_flags,
                self.walls.device,
                self.donor_count,
                self._counts,
                *periodic,
                *self.res,
                self.stride,
            ],
            device=self.model._device,
        )
        counts = self._counts.numpy()
        if counts[13]:
            raise RuntimeError("PLIC endpoint closure has invalid totals or capacity")
        if counts[5]:
            raise FloatingPointError(f"PLIC topology found {counts[5]} invalid cells")
        if counts[6]:
            raise RuntimeError("PLIC topology forbids a one-step gas-to-liquid jump")
        if counts[7]:
            raise RuntimeError("PLIC topology forbids a one-step liquid-to-gas jump")
        if counts[8]:
            raise RuntimeError(f"PLIC topology produced {counts[8]} direct liquid-gas links")
        if counts[9]:
            raise RuntimeError(f"{counts[9]} fresh PLIC interfaces have no persistent fluid donor")

        wp.copy(destination_fluid.moments, candidate_fluid.moments)
        wp.launch(
            topology_kernels.initialize_fresh_interface_kernel,
            dim=self.res,
            inputs=[
                candidate_fluid.moments,
                source_fsl.flags,
                self.target_flags,
                self.walls.device,
                self.donor_count,
                destination_fluid.moments,
                *periodic,
                *self.res,
                self.stride,
            ],
            device=self.model._device,
        )
        wp.copy(destination_fsl.fill_level, self.closed_fill)
        wp.copy(destination_fsl.mass, self.closed_mass)
        wp.copy(destination_fsl.flags, self.target_flags)
        destination_fsl.excess_mass.zero_()
        diagnostics = PlicTopologyDiagnostics(
            *(int(value) for value in counts[:5]),
            int(counts[0]),
            int(counts[8]),
            0.0,
            0.0,
            (0.0, 0.0, 0.0),
            endpoint_promoted_liquid_count=int(counts[10]),
            separation_repair_count=int(counts[11]),
            endpoint_removed_gas_count=int(counts[12]),
            endpoint_redistributed_volume=float(self._tail_volume.numpy()[0]),
            endpoint_redistributed_mass=float(self._tail_mass.numpy()[0]),
            endpoint_redistributed_momentum=(
                float(self._tail_momentum_x.numpy()[0]),
                float(self._tail_momentum_y.numpy()[0]),
                float(self._tail_momentum_z.numpy()[0]),
            ),
        )
        self.last_diagnostics = diagnostics
        return diagnostics
