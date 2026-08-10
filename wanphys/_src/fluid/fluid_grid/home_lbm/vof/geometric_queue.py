# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative bridge from HOME-Free excess queues to geometric PLIC."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..model import HomeLbmModel
from ..state import HomeLbmState
from . import geometric_queue_kernels
from . import kernels as mass_kernels
from .geometric_topology import (
    GeometricPlicTopologyDiagnostics,
    HomeFreeGeometricTopologyResolver,
)
from .state import HomeFreeState


@dataclass(frozen=True)
class HomeFreeGeometricQueueDiagnostics:
    queue_source_count: int
    materialized_cell_count: int
    represented_mass: float
    committed_mass: float
    relative_mass_error: float
    represented_momentum: tuple[float, float, float]
    committed_momentum: tuple[float, float, float]
    relative_momentum_error: float
    maximum_fill_delta: float
    topology: GeometricPlicTopologyDiagnostics | None


class HomeFreeGeometricQueueMaterializer:
    """Consume deferred per-recipient excess before geometric transport.

    This bridge gathers only queue fields; it never invokes the legacy HOME
    link-mass advection. Gathered mass becomes local liquid volume and queued
    momentum is committed with the same ten-moment Galilean correction used by
    the exact joint HOME-Free transaction.
    """

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        bound_tolerance: float = 2.0e-6,
        conservation_tolerance: float = 4.0e-6,
    ) -> None:
        if not math.isfinite(bound_tolerance) or bound_tolerance < 0.0:
            raise ValueError("queue bound tolerance must be finite and nonnegative")
        if not math.isfinite(conservation_tolerance) or conservation_tolerance <= 0.0:
            raise ValueError("queue conservation tolerance must be finite and positive")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.bound_tolerance = float(bound_tolerance)
        self.conservation_tolerance = float(conservation_tolerance)
        self.materialized_fill = wp.zeros(self.res, dtype=float, device=self.device)
        self.materialized_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.incoming_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self.incoming_momentum = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.represented_queue_mass = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self.represented_queue_momentum = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self.queue_capacity_sum = wp.zeros(
            self.res, dtype=float, device=self.device
        )
        self.queue_recipient_count = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._materialized_marker = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self.topology = HomeFreeGeometricTopologyResolver(model)
        self._queue_source_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._materialized_cell_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_cell_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._gather_invalid_counts = wp.zeros(
            4, dtype=wp.int32, device=self.device
        )
        self._source_ledger = wp.zeros(4, dtype=wp.float64, device=self.device)
        self._represented_queue_ledger = wp.zeros(
            4, dtype=wp.float64, device=self.device
        )
        self._gathered_ledger = wp.zeros(4, dtype=wp.float64, device=self.device)
        self._redistribution_ledger = wp.zeros(
            4, dtype=wp.float64, device=self.device
        )
        self._redistribution_capacity = wp.zeros(
            1, dtype=wp.float64, device=self.device
        )
        self._committed_ledger = wp.zeros(4, dtype=wp.float64, device=self.device)
        self._maximum_fill_delta = wp.zeros(1, dtype=float, device=self.device)

    def materialize(
        self,
        fluid: HomeLbmState,
        source: HomeFreeState,
        destination: HomeFreeState,
    ) -> HomeFreeGeometricQueueDiagnostics:
        self._validate_state(fluid, source, destination)
        self._queue_source_count.zero_()
        self._materialized_cell_count.zero_()
        self._invalid_cell_count.zero_()
        self._source_ledger.zero_()
        self._represented_queue_ledger.zero_()
        self._gathered_ledger.zero_()
        self._redistribution_ledger.zero_()
        self._redistribution_capacity.zero_()
        self._committed_ledger.zero_()
        self._maximum_fill_delta.zero_()
        periodic = tuple(int(value) for value in self.model.periodic)
        wp.launch(
            geometric_queue_kernels.prepare_geometric_excess_sources_kernel,
            dim=self.res,
            inputs=[
                fluid.moments,
                source.mass,
                source.fill_level,
                source.excess_mass,
                source.excess_momentum,
                source.flags,
                self.represented_queue_mass,
                self.represented_queue_momentum,
                self.queue_capacity_sum,
                self.queue_recipient_count,
                self._queue_source_count,
                self._invalid_cell_count,
                self._source_ledger,
                self._represented_queue_ledger,
                self.bound_tolerance,
                *periodic,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_cell_count.numpy()[0])
        if invalid:
            raise RuntimeError(
                f"geometric queue materialization found {invalid} invalid cells"
            )

        queue_sources = int(self._queue_source_count.numpy()[0])
        source_ledger = self._source_ledger.numpy().astype(np.float64)
        if queue_sources == 0:
            destination.copy_from(source)
            momentum = tuple(float(value) for value in source_ledger[1:])
            return HomeFreeGeometricQueueDiagnostics(
                queue_source_count=0,
                materialized_cell_count=0,
                represented_mass=float(source_ledger[0]),
                committed_mass=float(source_ledger[0]),
                relative_mass_error=0.0,
                represented_momentum=momentum,
                committed_momentum=momentum,
                relative_momentum_error=0.0,
                maximum_fill_delta=0.0,
                topology=None,
            )

        self._invalid_cell_count.zero_()
        self._gather_invalid_counts.zero_()
        wp.launch(
            geometric_queue_kernels.gather_prepared_geometric_excess_kernel,
            dim=self.res,
            inputs=[
                fluid.moments,
                source.mass,
                source.fill_level,
                source.flags,
                self.represented_queue_mass,
                self.represented_queue_momentum,
                self.queue_capacity_sum,
                self.queue_recipient_count,
                self.materialized_fill,
                self.materialized_mass,
                self.incoming_mass,
                self.incoming_momentum,
                self._materialized_marker,
                self._materialized_cell_count,
                self._invalid_cell_count,
                self._gather_invalid_counts,
                self._gathered_ledger,
                self._redistribution_ledger,
                self._maximum_fill_delta,
                self.bound_tolerance,
                *periodic,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            geometric_queue_kernels.sum_global_geometric_excess_capacity_kernel,
            dim=self.res,
            inputs=[
                source.mass,
                source.fill_level,
                source.flags,
                self.materialized_mass,
                self._redistribution_ledger,
                self._redistribution_capacity,
            ],
            device=self.device,
        )
        wp.launch(
            geometric_queue_kernels.apply_global_geometric_excess_kernel,
            dim=self.res,
            inputs=[
                source.mass,
                source.fill_level,
                source.flags,
                self.materialized_fill,
                self.materialized_mass,
                self.incoming_mass,
                self.incoming_momentum,
                self._materialized_marker,
                self._materialized_cell_count,
                self._redistribution_ledger,
                self._redistribution_capacity,
                self._invalid_cell_count,
                self._gather_invalid_counts,
                self._maximum_fill_delta,
                self.bound_tolerance,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_cell_count.numpy()[0])
        if invalid:
            categories = self._gather_invalid_counts.numpy()
            raise RuntimeError(
                "geometric queue gather found "
                f"{invalid} invalid cells/events "
                f"(source={int(categories[0])}, nonfinite={int(categories[1])}, "
                f"below_zero={int(categories[2])}, above_one={int(categories[3])})"
            )

        represented_queue = self._represented_queue_ledger.numpy().astype(np.float64)
        gathered_queue = self._gathered_ledger.numpy().astype(np.float64)
        queue_scale = max(abs(float(represented_queue[0])), 1.0)
        queue_mass_error = abs(
            float(gathered_queue[0] - represented_queue[0])
        ) / queue_scale
        queue_momentum_scale = max(
            float(np.linalg.norm(represented_queue[1:])),
            abs(float(represented_queue[0])),
            1.0,
        )
        queue_momentum_error = float(
            np.linalg.norm(gathered_queue[1:] - represented_queue[1:])
        ) / queue_momentum_scale
        if queue_mass_error > self.conservation_tolerance:
            raise RuntimeError(
                "geometric queue gather mass error "
                f"{queue_mass_error:.6e} exceeds tolerance "
                f"{self.conservation_tolerance:.6e}"
            )
        if queue_momentum_error > self.conservation_tolerance:
            raise RuntimeError(
                "geometric queue gather momentum error "
                f"{queue_momentum_error:.6e} exceeds tolerance "
                f"{self.conservation_tolerance:.6e}"
            )

        topology = self.topology.resolve(
            fluid,
            source,
            self.materialized_fill,
            destination,
            transported_mass=self.materialized_mass,
        )
        self._invalid_cell_count.zero_()
        wp.launch(
            mass_kernels.apply_queue_momentum_correction_kernel,
            dim=self.res,
            inputs=[
                fluid.moments,
                destination.mass,
                self.incoming_mass,
                self.incoming_momentum,
                destination.flags,
                self._invalid_cell_count,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            geometric_queue_kernels.accumulate_committed_geometric_ledger_kernel,
            dim=self.res,
            inputs=[
                fluid.moments,
                destination.mass,
                destination.flags,
                self._committed_ledger,
                self._invalid_cell_count,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_cell_count.numpy()[0])
        if invalid:
            raise RuntimeError(
                "geometric queue momentum commit produced "
                f"{invalid} invalid cells"
            )

        committed_ledger = self._committed_ledger.numpy().astype(np.float64)
        represented_mass = float(source_ledger[0])
        committed_mass = float(committed_ledger[0])
        mass_scale = max(abs(represented_mass), 1.0)
        relative_mass_error = abs(committed_mass - represented_mass) / mass_scale
        represented_momentum = source_ledger[1:]
        committed_momentum = committed_ledger[1:]
        momentum_scale = max(
            float(np.linalg.norm(represented_momentum)),
            abs(represented_mass),
            1.0,
        )
        relative_momentum_error = float(
            np.linalg.norm(committed_momentum - represented_momentum)
        ) / momentum_scale
        if relative_mass_error > self.conservation_tolerance:
            raise RuntimeError(
                "geometric queue committed mass error "
                f"{relative_mass_error:.6e} exceeds tolerance "
                f"{self.conservation_tolerance:.6e}"
            )
        if relative_momentum_error > self.conservation_tolerance:
            raise RuntimeError(
                "geometric queue committed momentum error "
                f"{relative_momentum_error:.6e} exceeds tolerance "
                f"{self.conservation_tolerance:.6e}"
            )
        destination.validate(
            fluid,
            allow_interface_endpoints=True,
            require_mass_fill_consistency=False,
        )
        return HomeFreeGeometricQueueDiagnostics(
            queue_source_count=queue_sources,
            materialized_cell_count=int(self._materialized_cell_count.numpy()[0]),
            represented_mass=represented_mass,
            committed_mass=committed_mass,
            relative_mass_error=relative_mass_error,
            represented_momentum=tuple(float(value) for value in represented_momentum),
            committed_momentum=tuple(float(value) for value in committed_momentum),
            relative_momentum_error=relative_momentum_error,
            maximum_fill_delta=float(self._maximum_fill_delta.numpy()[0]),
            topology=topology,
        )

    def _validate_state(
        self,
        fluid: HomeLbmState,
        source: HomeFreeState,
        destination: HomeFreeState,
    ) -> None:
        if fluid.model is not self.model:
            raise ValueError("queue materializer and HOME state must share a model")
        if source.model is not self.model or destination.model is not self.model:
            raise ValueError("queue materializer and HOME-Free states must share a model")
        if source is destination:
            raise ValueError("queue materialization requires distinct source and destination")
