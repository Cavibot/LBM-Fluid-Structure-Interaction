# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rollback-safe HOME-FSL step driven by projected geometric PLIC transport."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ..core import HomeCoreDiagnostics, HomeCoreModel, HomeCoreState
from ..free_surface_fsl import (
    FslState,
    FslWallActiveCollider,
    FslWallMask,
    FslWallOnlyMissingStreamer,
    FslWallStreamDiagnostics,
)
from .courant import (
    CourantProjectionDiagnostics,
    FaceCourantDiagnostics,
    HomeCourantProjector,
    HomeFaceCourantBuilder,
)
from .topology import PlicTopologyResolver
from .topology_reference import PlicTopologyDiagnostics
from .transport import PlicConservativeTransportDiagnostics, PlicGeometricTransport
from . import coupling_kernels


@dataclass(frozen=True)
class GeometricHomeFslDiagnostics:
    stream: FslWallStreamDiagnostics
    fluid: HomeCoreDiagnostics
    collided_fluid: HomeCoreDiagnostics
    face_courant: FaceCourantDiagnostics
    projection: CourantProjectionDiagnostics
    transport: PlicConservativeTransportDiagnostics
    topology: PlicTopologyDiagnostics


class GeometricHomeFslStepper:
    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        *,
        axes: tuple[int, ...] = (0, 1),
        gas_density: float = 1.0,
        projection_maximum_iterations: int = 320,
        projection_relative_tolerance: float = 5.0e-6,
        projection_absolute_divergence_tolerance: float = 5.0e-7,
        interface_roundoff_tolerance: float = 2.0e-6,
        contact_angle_degrees: float | None = None,
    ) -> None:
        if walls.model is not model:
            raise ValueError("geometric stepper and walls must share one HOME model")
        self.model = model
        self.walls = walls
        self.res = walls.res
        self.axes = tuple(axes)
        self.builder = HomeFaceCourantBuilder(model, walls)
        self.projector = HomeCourantProjector(
            model,
            walls,
            maximum_iterations=projection_maximum_iterations,
            relative_tolerance=projection_relative_tolerance,
            absolute_divergence_tolerance=projection_absolute_divergence_tolerance,
        )
        self.transport = PlicGeometricTransport(
            model,
            walls,
            endpoint_tolerance=interface_roundoff_tolerance,
            contact_angle_degrees=contact_angle_degrees,
        )
        self.streamer = FslWallOnlyMissingStreamer(
            model, walls, gas_density=gas_density
        )
        self.collider = FslWallActiveCollider(model, walls)
        self.topology = PlicTopologyResolver(
            model,
            walls,
            endpoint_tolerance=interface_roundoff_tolerance,
        )
        self.initial_momentum = wp.zeros(
            self.res, dtype=wp.vec3, device=model._device
        )
        self._collided = HomeCoreState(model)
        self._pre_momentum_fluid = HomeCoreState(model)
        self._candidate_fluid = HomeCoreState(model)
        self._candidate_fsl = FslState(model)
        self._invalid_momentum = wp.zeros(1, dtype=wp.int32, device=model._device)
        self._step_index = 0
        self.last_transported_fill = None
        self.last_transported_mass = None
        self.last_transported_momentum = None
        self.last_attempt_diagnostics: GeometricHomeFslDiagnostics | None = None
        self.last_diagnostics: GeometricHomeFslDiagnostics | None = None

    def _initialize_liquid_momentum(
        self,
        fluid: HomeCoreState,
        fsl: FslState,
        projected: tuple[wp.array, wp.array, wp.array],
    ) -> None:
        wp.launch(
            coupling_kernels.initialize_liquid_momentum_kernel,
            dim=self.res,
            inputs=[
                fluid.moments,
                fsl.mass,
                fsl.flags,
                self.walls.device,
                self.initial_momentum,
                self._invalid_momentum,
                self.res[1],
                self.res[2],
                fluid.cell_count,
            ],
            device=self.model._device,
        )

    def step(
        self,
        fluid_in: HomeCoreState,
        fsl_in: FslState,
        fluid_out: HomeCoreState,
        fsl_out: FslState,
        dt: float,
    ) -> GeometricHomeFslDiagnostics:
        states = (fluid_in, fsl_in, fluid_out, fsl_out)
        if any(state.model is not self.model for state in states):
            raise ValueError("geometric stepper and states must share one HOME model")
        self.model.validate_step(dt)
        self.last_diagnostics = None
        self.last_attempt_diagnostics = None
        self.last_transported_fill = None
        self.last_transported_mass = None
        self.last_transported_momentum = None
        split_order = self.axes if self._step_index % 2 == 0 else tuple(reversed(self.axes))
        faces = self.builder.build(fluid_in, fsl_in)
        projected = self.projector.project(faces, fsl_in)
        self._invalid_momentum.zero_()
        self._initialize_liquid_momentum(fluid_in, fsl_in, projected)
        transported_fill, transported_mass, transported_momentum, transport = (
            self.transport.transport_conservative(
                fsl_in.fill_level,
                fsl_in.mass,
                self.initial_momentum,
                projected,
                split_order=split_order,
            )
        )
        self.last_transported_fill = transported_fill
        self.last_transported_mass = transported_mass
        self.last_transported_momentum = transported_momentum
        streamed = self.streamer.stream(fluid_in, fsl_in)
        self.collider.collide(streamed, fsl_in, self._collided, dt)
        collided_fluid = self.collider.collect_diagnostics()
        topology = self.topology.resolve(
            self._collided,
            fsl_in,
            transported_fill,
            transported_mass,
            transported_momentum,
            self._candidate_fluid,
            self._candidate_fsl,
        )
        self.last_transported_fill = self.topology.closed_fill
        self.last_transported_mass = self.topology.closed_mass
        self.last_transported_momentum = self.topology.closed_momentum
        self._pre_momentum_fluid.copy_from(self._candidate_fluid)
        wp.launch(
            coupling_kernels.apply_transported_momentum_kernel,
            dim=self.res,
            inputs=[
                fluid_in.moments,
                fsl_in.mass,
                self._candidate_fluid.moments,
                self.topology.closed_mass,
                self.topology.closed_fill,
                self.topology.closed_momentum,
                self._candidate_fsl.flags,
                self.walls.device,
                self._invalid_momentum,
                self.res[1],
                self.res[2],
                fluid_in.cell_count,
            ],
            device=self.model._device,
        )
        stream = self.streamer.collect_diagnostics()
        fluid = self.collider.diagnose(self._candidate_fluid, self._candidate_fsl)
        invalid_momentum = int(self._invalid_momentum.numpy()[0])
        diagnostics = GeometricHomeFslDiagnostics(
            stream=stream,
            fluid=fluid,
            collided_fluid=collided_fluid,
            face_courant=self.builder.last_diagnostics,
            projection=self.projector.last_diagnostics,
            transport=transport,
            topology=topology,
        )
        self.last_attempt_diagnostics = diagnostics
        if (
            stream.invalid_cell_count
            or collided_fluid.invalid_cell_count
            or fluid.invalid_cell_count
            or invalid_momentum
        ):
            raise FloatingPointError(
                "geometric HOME-FSL transaction contains invalid cells"
            )
        if stream.direct_liquid_gas_link_count:
            raise RuntimeError("geometric HOME-FSL transaction contains liquid-gas links")
        if fluid.max_speed > self.model.max_lattice_speed:
            raise FloatingPointError(
                f"geometric HOME-FSL committed speed {fluid.max_speed} exceeds "
                f"{self.model.max_lattice_speed}; collided speed was "
                f"{collided_fluid.max_speed}"
            )
        fluid_out.copy_from(self._candidate_fluid)
        fsl_out.copy_from(self._candidate_fsl)
        self._step_index += 1
        self.last_diagnostics = diagnostics
        return diagnostics


class ProjectedVelocityGeometricHomeFslStepper(GeometricHomeFslStepper):
    """Experimental variant that initializes liquid momentum from projected faces."""

    def _initialize_liquid_momentum(
        self,
        fluid: HomeCoreState,
        fsl: FslState,
        projected: tuple[wp.array, wp.array, wp.array],
    ) -> None:
        wp.launch(
            coupling_kernels.initialize_projected_liquid_momentum_kernel,
            dim=self.res,
            inputs=[
                *projected,
                fsl.mass,
                fsl.flags,
                self.walls.device,
                self.initial_momentum,
                self._invalid_momentum,
            ],
            device=self.model._device,
        )
