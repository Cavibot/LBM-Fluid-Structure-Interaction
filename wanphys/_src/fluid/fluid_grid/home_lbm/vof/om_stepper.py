# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Transactional only-missing boundary comparator for geometric HOME-Free VOF."""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from ..diagnostics import HomeLbmDiagnostics
from ..model import HomeLbmModel
from ..solver import HomeLbmSolver
from ..state import HomeLbmState
from . import geometric_advection_kernels
from .geometric_advection import PlicAxisAdvectionDiagnostics
from .fsl_courant import FslFaceCourantDiagnostics
from .fsl_transition import (
    FslActiveTransitionDiagnostics,
    FslActiveTransitionRemapper,
)
from .geometric_topology import GeometricPlicTopologyDiagnostics
from .geometric_transport import HomeFreeGeometricTransport
from .geometry import HomeFreeInterfaceGeometry
from .geometric_queue import (
    HomeFreeGeometricQueueDiagnostics,
    HomeFreeGeometricQueueMaterializer,
)
from .courant_projection import HomeFreeCourantProjectionDiagnostics
from .state import HomeFreeState
from .surface_tension import (
    HomeFreeSurfaceTension,
    HomeFreeSurfaceTensionDiagnostics,
)


@dataclass(frozen=True)
class HomeFreeOmResearchStepDiagnostics:
    queue: HomeFreeGeometricQueueDiagnostics
    surface_tension: HomeFreeSurfaceTensionDiagnostics
    transition: FslActiveTransitionDiagnostics
    solver: HomeLbmDiagnostics
    face_courant: FslFaceCourantDiagnostics | None
    face_courant_by_axis: tuple[tuple[int, FslFaceCourantDiagnostics], ...]
    advection_by_axis: tuple[tuple[int, PlicAxisAdvectionDiagnostics], ...]
    topology_by_axis: tuple[tuple[int, GeometricPlicTopologyDiagnostics], ...]
    split_order: tuple[int, ...]
    invalid_active_count: int
    projection: HomeFreeCourantProjectionDiagnostics | None


class HomeFreeOmTransactionHistory:
    """Reusable checkpoint for scheduling and active-node history."""

    def __init__(self, model: HomeLbmModel) -> None:
        self.split_parity = 0
        self.remapper_exists = False
        self.previous_active = wp.zeros(
            (int(model.nx), int(model.ny), int(model.nz)),
            dtype=wp.int32,
            device=model._device,
        )
        self.primary_pressure_exists = False
        self.refinement_pressure_exists = False
        self.primary_pressure = wp.zeros(
            (int(model.nx), int(model.ny), int(model.nz)),
            dtype=float,
            device=model._device,
        )
        self.refinement_pressure = wp.zeros_like(self.primary_pressure)


class HomeFreeOmResearchStepper:
    """Advance geometric VOF with the official only-missing pressure boundary."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        advection_axis: int | None = None,
        advection_axes: tuple[int, ...] | None = None,
        gas_density: float = 1.0,
        surface_tension: float = 0.0,
        contact_angle_degrees: float | None = None,
        project_courant: bool = False,
        projection_max_iterations: int | None = None,
        solver: HomeLbmSolver | None = None,
    ) -> None:
        if (advection_axis is None) == (advection_axes is None):
            raise ValueError(
                "configure exactly one of advection_axis or advection_axes"
            )
        axes = (
            (int(advection_axis),)
            if advection_axis is not None
            else tuple(int(axis) for axis in advection_axes or ())
        )
        if not axes or any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError("OM research advection axes must use 0, 1, or 2")
        if len(set(axes)) != len(axes):
            raise ValueError("OM research advection axes must not repeat")
        if not math.isfinite(gas_density) or gas_density <= 0.0:
            raise ValueError("OM research gas density must be finite and positive")
        if solver is not None and solver.model is not model:
            raise ValueError("an injected HOME solver must own the same model")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.device = model._device
        self.advection_axes = axes
        self._split_parity = 0
        self.solver = solver or HomeLbmSolver(model)
        self.geometry = HomeFreeInterfaceGeometry(
            model, contact_angle_degrees=contact_angle_degrees
        )
        self.surface_tension = HomeFreeSurfaceTension(
            model,
            ambient_gas_density=gas_density,
            surface_tension=surface_tension,
            geometry=self.geometry,
        )
        self.gas_density = self.surface_tension.gas_density
        self.geometric_transport = HomeFreeGeometricTransport(
            model,
            axes=axes,
            contact_angle_degrees=contact_angle_degrees,
            project_courant=project_courant,
            projection_max_iterations=projection_max_iterations,
        )
        self.face_courant_builders = self.geometric_transport.face_courant_builders
        self.face_courant_builder = self.face_courant_builders[axes[0]]
        self.topology_resolver = self.geometric_transport.topology_resolver
        self.working_fluid = HomeLbmState(model)
        self.working_free_surface = HomeFreeState(model)
        self.queue_materializer = HomeFreeGeometricQueueMaterializer(model)
        self.candidate_fluid = HomeLbmState(model)
        self.candidate_free_surface = HomeFreeState(model)
        self.active = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self._invalid_active = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._body_force_density = wp.zeros(
            self.res, dtype=wp.vec3, device=self.device
        )
        self._invalid_body_force = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_momentum_transport = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._momentum_ledger = wp.zeros(
            10, dtype=wp.float64, device=self.device
        )
        self._velocity_correction = wp.zeros(
            1, dtype=wp.vec3, device=self.device
        )
        self._previous_active_backup = wp.zeros_like(self.active)
        self._remapper: FslActiveTransitionRemapper | None = None

    def reset_transaction_history(self) -> None:
        """Reset split scheduling and active history after domain reinitialization."""

        self._split_parity = 0
        self._remapper = None
        if self.geometric_transport.courant_projector is not None:
            self.geometric_transport.courant_projector.reset_pressure_guess()
        if self.geometric_transport.courant_refinement_projector is not None:
            refinement = self.geometric_transport.courant_refinement_projector
            refinement.reset_pressure_guess()

    def create_transaction_history(self) -> HomeFreeOmTransactionHistory:
        return HomeFreeOmTransactionHistory(self.model)

    def capture_transaction_history(
        self, history: HomeFreeOmTransactionHistory
    ) -> None:
        self._validate_history(history)
        history.split_parity = self._split_parity
        history.remapper_exists = self._remapper is not None
        if self._remapper is None:
            history.previous_active.zero_()
        else:
            wp.copy(history.previous_active, self._remapper.previous_active)
        primary = self.geometric_transport.courant_projector
        refinement = self.geometric_transport.courant_refinement_projector
        history.primary_pressure_exists = (
            primary.capture_pressure_guess(history.primary_pressure)
            if primary is not None
            else False
        )
        history.refinement_pressure_exists = (
            refinement.capture_pressure_guess(history.refinement_pressure)
            if refinement is not None
            else False
        )

    def restore_transaction_history(
        self, history: HomeFreeOmTransactionHistory
    ) -> None:
        self._validate_history(history)
        self._split_parity = history.split_parity
        if not history.remapper_exists:
            self._remapper = None
        else:
            if self._remapper is None:
                self._remapper = FslActiveTransitionRemapper(
                    self.working_fluid, history.previous_active
                )
            wp.copy(self._remapper.previous_active, history.previous_active)
        primary = self.geometric_transport.courant_projector
        refinement = self.geometric_transport.courant_refinement_projector
        if primary is not None:
            primary.restore_pressure_guess(
                history.primary_pressure, history.primary_pressure_exists
            )
        if refinement is not None:
            refinement.restore_pressure_guess(
                history.refinement_pressure,
                history.refinement_pressure_exists,
            )

    def _validate_history(self, history: HomeFreeOmTransactionHistory) -> None:
        if not isinstance(history, HomeFreeOmTransactionHistory):
            raise TypeError("OM transaction history has an incompatible type")
        if (
            tuple(history.previous_active.shape) != self.res
            or history.previous_active.device != self.device
        ):
            raise ValueError("OM transaction history does not match the stepper")
        if history.split_parity not in (0, 1):
            raise ValueError("OM split parity must be zero or one")

    def step(
        self,
        source_fluid: HomeLbmState,
        source_free_surface: HomeFreeState,
        destination_fluid: HomeLbmState,
        destination_free_surface: HomeFreeState,
        face_courant: wp.array | None = None,
    ) -> HomeFreeOmResearchStepDiagnostics:
        self._validate_state(source_fluid, source_free_surface)
        self._validate_state(destination_fluid, destination_free_surface)
        self.working_fluid.copy_from(source_fluid)
        queue_diagnostics = self.queue_materializer.materialize(
            self.working_fluid,
            source_free_surface,
            self.working_free_surface,
        )
        step_source = self.working_free_surface
        surface_diagnostics = self.surface_tension.update(
            step_source, solid_phi=self.working_fluid.solid_phi
        )
        self._invalid_active.zero_()
        wp.launch(
            geometric_advection_kernels.classify_only_missing_active_kernel,
            dim=self.res,
            inputs=[
                step_source.flags,
                self.active,
                self._invalid_active,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid_active = int(self._invalid_active.numpy()[0])
        if invalid_active:
            raise RuntimeError(
                f"OM active classification found {invalid_active} invalid cells"
            )

        created_remapper = self._remapper is None
        if created_remapper:
            self._remapper = FslActiveTransitionRemapper(
                self.working_fluid, self.active
            )
            transition = FslActiveTransitionDiagnostics(0, 0, 0)
        else:
            assert self._remapper is not None
            wp.copy(self._previous_active_backup, self._remapper.previous_active)
            transition = self._remapper.remap(self.working_fluid, self.active)

        try:
            split_order = (
                self.advection_axes
                if self._split_parity == 0
                else tuple(reversed(self.advection_axes))
            )
            transport = self.geometric_transport.transport(
                self.working_fluid,
                step_source,
                self.active,
                split_order=split_order,
                face_courant=face_courant,
            )
            self._invalid_body_force.zero_()
            wp.launch(
                geometric_advection_kernels.build_liquid_body_force_kernel,
                dim=self.res,
                inputs=[
                    step_source.mass,
                    step_source.flags,
                    wp.vec3(*self.model.lattice_acceleration),
                    self._body_force_density,
                    self._invalid_body_force,
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
            invalid_body_force = int(self._invalid_body_force.numpy()[0])
            if invalid_body_force:
                raise RuntimeError(
                    "OM liquid body-force construction found "
                    f"{invalid_body_force} invalid cells"
                )
            self.solver.step(
                self.working_fluid,
                self.candidate_fluid,
                self.model.time_step,
                free_surface_flags=step_source.flags,
                gas_density=self.surface_tension.ambient_gas_density,
                gas_density_field=self.gas_density,
                force_density=self._body_force_density,
            )
            self._momentum_ledger.zero_()
            wp.launch(
                geometric_advection_kernels.accumulate_expected_liquid_momentum_kernel,
                dim=self.res,
                inputs=[
                    self.working_fluid.moments,
                    step_source.mass,
                    step_source.flags,
                    self.candidate_fluid.moments,
                    self._momentum_ledger,
                    int(self.model.ny),
                    int(self.model.nz),
                    self.candidate_fluid.cell_count,
                ],
                device=self.device,
            )
            self._invalid_momentum_transport.zero_()
            wp.launch(
                geometric_advection_kernels.apply_geometric_momentum_transport_kernel,
                dim=self.res,
                inputs=[
                    self.working_fluid.moments,
                    step_source.mass,
                    self.candidate_fluid.moments,
                    transport.final_state.mass,
                    transport.final_state.fill_level,
                    transport.final_momentum,
                    transport.final_state.flags,
                    self._invalid_momentum_transport,
                    int(self.model.ny),
                    int(self.model.nz),
                    self.candidate_fluid.cell_count,
                ],
                device=self.device,
            )
            wp.launch(
                geometric_advection_kernels.accumulate_committed_liquid_momentum_kernel,
                dim=self.res,
                inputs=[
                    self.candidate_fluid.moments,
                    transport.final_state.mass,
                    transport.final_state.flags,
                    self._momentum_ledger,
                    int(self.model.ny),
                    int(self.model.nz),
                    self.candidate_fluid.cell_count,
                ],
                device=self.device,
            )
            wp.launch(
                geometric_advection_kernels.prepare_uniform_liquid_velocity_correction_kernel,
                dim=1,
                inputs=[
                    self._momentum_ledger,
                    self._velocity_correction,
                    self.model.max_lattice_speed,
                    self._invalid_momentum_transport,
                ],
                device=self.device,
            )
            wp.launch(
                geometric_advection_kernels.apply_device_liquid_velocity_correction_kernel,
                dim=self.res,
                inputs=[
                    self.candidate_fluid.moments,
                    transport.final_state.flags,
                    self._velocity_correction,
                    int(self.model.ny),
                    int(self.model.nz),
                    self.candidate_fluid.cell_count,
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
            invalid_momentum_transport = int(
                self._invalid_momentum_transport.numpy()[0]
            )
            if invalid_momentum_transport:
                raise RuntimeError(
                    "OM geometric momentum commit or conservative projection found "
                    f"{invalid_momentum_transport} invalid entries"
                )
            solver_diagnostics = self.solver.validate_state(
                self.candidate_fluid,
                free_surface_flags=transport.final_state.flags,
                gas_density_field=self.gas_density,
            )
            self.topology_resolver.resolve(
                self.candidate_fluid,
                transport.final_state,
                transport.final_state.fill_level,
                self.candidate_free_surface,
                transported_mass=transport.final_state.mass,
            )
            destination_fluid.copy_from(self.candidate_fluid)
            destination_free_surface.copy_from(self.candidate_free_surface)
            courant_diagnostics = (
                transport.face_courant_by_axis[0][1]
                if len(transport.face_courant_by_axis) == 1
                else None
            )
            diagnostics = HomeFreeOmResearchStepDiagnostics(
                queue=queue_diagnostics,
                surface_tension=surface_diagnostics,
                transition=transition,
                solver=solver_diagnostics,
                face_courant=courant_diagnostics,
                face_courant_by_axis=transport.face_courant_by_axis,
                advection_by_axis=transport.advection_by_axis,
                topology_by_axis=transport.topology_by_axis,
                split_order=split_order,
                invalid_active_count=invalid_active,
                projection=transport.projection,
            )
            self._split_parity = 1 - self._split_parity
            return diagnostics
        except Exception:
            if created_remapper:
                self._remapper = None
            else:
                assert self._remapper is not None
                wp.copy(
                    self._remapper.previous_active,
                    self._previous_active_backup,
                )
            raise

    def _validate_state(
        self, fluid: HomeLbmState, free_surface: HomeFreeState
    ) -> None:
        if fluid.model is not self.model or free_surface.model is not self.model:
            raise ValueError("OM research stepper and states must own the same model")
