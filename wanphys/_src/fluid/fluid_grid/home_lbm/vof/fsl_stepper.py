# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Transactional dynamic-topology research step for geometric VOF plus FSL."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from .. import kernels as home_kernels
from ..constants import D3Q27_DIRECTIONS, D3Q27_OPPOSITE, D3Q27_WEIGHTS
from ..model import HomeLbmModel
from ..state import HomeLbmState
from . import geometry_kernels, kernels as vof_kernels
from .geometric_advection import PlicAxisAdvectionDiagnostics
from .fsl_transition import (
    FslActiveTransitionDiagnostics,
    FslActiveTransitionRemapper,
)
from .fsl_courant import FslFaceCourantDiagnostics
from .fsl_stress import (
    FslStressClosureDiagnostics,
    HomeFreeFslStressClosure,
)
from .geometric_transport import HomeFreeGeometricTransport
from .courant_projection import HomeFreeCourantProjectionDiagnostics
from .geometric_topology import GeometricPlicTopologyDiagnostics
from .geometry import HomeFreeGeometryDiagnostics, HomeFreeInterfaceGeometry
from .state import HomeFreeState
from .surface_tension import (
    HomeFreeSurfaceTension,
    HomeFreeSurfaceTensionDiagnostics,
)


@dataclass(frozen=True)
class HomeFreeFslResearchStepDiagnostics:
    geometry: HomeFreeGeometryDiagnostics
    surface_tension: HomeFreeSurfaceTensionDiagnostics
    transition: FslActiveTransitionDiagnostics
    face_courant: FslFaceCourantDiagnostics | None
    face_courant_by_axis: tuple[tuple[int, FslFaceCourantDiagnostics], ...]
    advection_by_axis: tuple[tuple[int, PlicAxisAdvectionDiagnostics], ...]
    topology_by_axis: tuple[tuple[int, GeometricPlicTopologyDiagnostics], ...]
    stress: FslStressClosureDiagnostics
    split_order: tuple[int, ...]
    invalid_geometry_count: int
    invalid_velocity_count: int
    invalid_stream_count: int
    invalid_collision_count: int
    invalid_commit_count: int
    fallback_link_count: int
    projection: HomeFreeCourantProjectionDiagnostics | None


class HomeFreeFslResearchStepper:
    """Advance one force-free dynamic-topology FSL/PLIC research transaction.

    This class deliberately does not replace :class:`HomeFreeDomain`.  It
    closes geometric topology transitions and dynamic scheduling while surface
    tension, full Bogner stress closure, and FSI load ownership remain separate
    acceptance gates.
    """

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        advection_axis: int | None = None,
        advection_axes: tuple[int, ...] | None = None,
        gas_density: float = 1.0,
        surface_tension: float = 0.0,
        no_support_policy: str = "error",
        geometry_tolerance: float = 1.0e-6,
        project_courant: bool = False,
        projection_max_iterations: int = 160,
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
            raise ValueError("FSL research advection axes must use 0, 1, or 2")
        if len(set(axes)) != len(axes):
            raise ValueError("FSL research advection axes must not repeat")
        if not math.isfinite(gas_density) or gas_density <= 0.0:
            raise ValueError("FSL research gas density must be finite and positive")
        if not math.isfinite(geometry_tolerance) or geometry_tolerance <= 0.0:
            raise ValueError("FSL geometry tolerance must be finite and positive")
        if no_support_policy not in ("error", "only_missing"):
            raise ValueError(
                "no_support_policy must be 'error' or 'only_missing'"
            )
        if any(value != 0.0 for value in model.lattice_acceleration):
            raise ValueError("FSL research stepper currently requires zero body force")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.advection_axes = axes
        self._split_parity = 0
        self.geometry_tolerance = float(geometry_tolerance)
        self.no_support_policy = no_support_policy
        self._no_support_policy_code = (
            0 if no_support_policy == "error" else 1
        )
        self.geometry = HomeFreeInterfaceGeometry(model)
        self.surface_tension = HomeFreeSurfaceTension(
            model,
            ambient_gas_density=gas_density,
            surface_tension=surface_tension,
            geometry=self.geometry,
        )
        self.geometric_transport = HomeFreeGeometricTransport(
            model,
            axes=axes,
            project_courant=project_courant,
            projection_max_iterations=projection_max_iterations,
        )
        self.advection_geometry = self.geometric_transport.geometry
        self.geometric_advectors = self.geometric_transport.advectors
        self.face_courant_builders = self.geometric_transport.face_courant_builders
        self.geometric_advector = self.geometric_advectors[axes[0]]
        self.face_courant_builder = self.face_courant_builders[axes[0]]
        self.topology_resolver = self.geometric_transport.topology_resolver
        self.stress_closure = HomeFreeFslStressClosure(
            model, no_support_policy=no_support_policy
        )
        self._split_free_surface = self.geometric_transport.scratch_states
        self.working_fluid = HomeLbmState(model)
        self.candidate_fluid = HomeLbmState(model)
        self.candidate_free_surface = HomeFreeState(model)
        self.active = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.advection_active = self.active
        self.vof_compression = self.geometric_transport.compression
        self.fraction = wp.zeros(27 * self.stride, dtype=float, device=self.device)
        self.extrapolation = wp.zeros(
            27 * self.stride, dtype=float, device=self.device
        )
        self.link_status = wp.zeros(
            27 * self.stride, dtype=wp.int32, device=self.device
        )
        self.plane_owner = wp.zeros(
            27 * self.stride, dtype=wp.int32, device=self.device
        )
        self.boundary_velocity = wp.zeros(
            27 * self.stride, dtype=wp.vec3, device=self.device
        )
        self.boundary_strain = self.stress_closure.boundary_strain
        self.raw_moments = wp.zeros(
            10 * self.stride, dtype=float, device=self.device
        )
        self.gas_density = self.surface_tension.gas_density
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device
        )
        self._opposites = wp.array(
            D3Q27_OPPOSITE, dtype=wp.int32, device=self.device
        )
        self._invalid_geometry = wp.zeros(2, dtype=wp.int32, device=self.device)
        self._invalid_velocity = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_stream = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._invalid_collision = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._fallback_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._previous_active_backup = wp.zeros_like(self.active)
        self._remapper: FslActiveTransitionRemapper | None = None

    def step(
        self,
        source_fluid: HomeLbmState,
        source_free_surface: HomeFreeState,
        destination_fluid: HomeLbmState,
        destination_free_surface: HomeFreeState,
        face_courant: wp.array | None = None,
    ) -> HomeFreeFslResearchStepDiagnostics:
        self._validate_state(source_fluid, source_free_surface)
        self._validate_state(destination_fluid, destination_free_surface)
        self.working_fluid.copy_from(source_fluid)
        surface_tension_diagnostics = self.surface_tension.update(
            source_free_surface
        )
        geometry_diagnostics = surface_tension_diagnostics.geometry
        self._invalid_geometry.zero_()
        wp.launch(
            geometry_kernels.classify_fsl_hydrodynamic_nodes_kernel,
            dim=self.res,
            inputs=[
                source_free_surface.flags,
                self.geometry.normal,
                self.geometry.plane_offset,
                self.geometry.valid,
                self.active,
                self._invalid_geometry,
                self.geometry_tolerance,
            ],
            device=self.device,
        )
        wp.launch(
            geometry_kernels.construct_fsl_link_coverage_kernel,
            dim=27 * self.stride,
            inputs=[
                source_free_surface.flags,
                self.geometry.normal,
                self.geometry.plane_offset,
                self.geometry.valid,
                self.active,
                self._directions,
                self.fraction,
                self.extrapolation,
                self.link_status,
                self.plane_owner,
                self.geometry_tolerance,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                *self.res,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid_geometry = int(np.sum(self._invalid_geometry.numpy()))
        if invalid_geometry:
            raise RuntimeError(
                f"FSL active classification found {invalid_geometry} invalid cells"
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
            if face_courant is not None and len(self.advection_axes) != 1:
                raise ValueError(
                    "explicit face_courant is only valid for a single-axis step"
                )
            split_order = (
                self.advection_axes
                if self._split_parity == 0
                else tuple(reversed(self.advection_axes))
            )
            transport = self.geometric_transport.transport(
                self.working_fluid,
                source_free_surface,
                self.active,
                split_order=split_order,
                face_courant=face_courant,
            )
            courant_diagnostics_by_axis = list(transport.face_courant_by_axis)
            topology_diagnostics_by_axis = list(transport.topology_by_axis)
            current_free_surface = transport.final_state
            courant_diagnostics = (
                courant_diagnostics_by_axis[0][1]
                if len(courant_diagnostics_by_axis) == 1
                else None
            )
            self._invalid_velocity.zero_()
            self._invalid_stream.zero_()
            self._invalid_collision.zero_()
            self._fallback_count.zero_()
            wp.launch(
                vof_kernels.extrapolate_fsl_boundary_velocity_kernel,
                dim=27 * self.stride,
                inputs=[
                    self.working_fluid.moments,
                    self.active,
                    self.link_status,
                    self.fraction,
                    self._directions,
                    self.boundary_velocity,
                    self._invalid_velocity,
                    self._no_support_policy_code,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    *self.res,
                    self.stride,
                ],
                device=self.device,
            )
            self.stress_closure.build(
                self.working_fluid,
                self.active,
                self.link_status,
                self.plane_owner,
                self.fraction,
                self.geometry.normal,
                self.gas_density,
                self._directions,
            )
            assert self.stress_closure.last_diagnostics is not None
            wp.launch(
                vof_kernels.fsl_stream_moments_kernel,
                dim=self.res,
                inputs=[
                    self.working_fluid.moments,
                    self.active,
                    self.link_status,
                    self.plane_owner,
                    self.fraction,
                    self.gas_density,
                    self.boundary_velocity,
                    self.boundary_strain,
                    self._directions,
                    self._weights,
                    self._opposites,
                    self.raw_moments,
                    self._invalid_stream,
                    self._fallback_count,
                    self.model.shear_omega,
                    1,
                    self._no_support_policy_code,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    *self.res,
                    self.stride,
                ],
                device=self.device,
            )
            self.candidate_fluid.copy_from(self.working_fluid)
            wp.launch(
                home_kernels.collide_force_free_moments_kernel,
                dim=self.res,
                inputs=[
                    self.raw_moments,
                    self.active,
                    self.candidate_fluid.moments,
                    self._invalid_collision,
                    self.model.shear_omega,
                    self.res[1],
                    self.res[2],
                    self.stride,
                ],
                device=self.device,
            )
            wp.synchronize_device(self.device)
            invalid_velocity = int(self._invalid_velocity.numpy()[0])
            invalid_stream = int(self._invalid_stream.numpy()[0])
            invalid_collision = int(self._invalid_collision.numpy()[0])
            fallback_links = int(self._fallback_count.numpy()[0])
            invalid_total = invalid_velocity + invalid_stream + invalid_collision
            if invalid_total:
                raise RuntimeError(
                    f"FSL research transaction found {invalid_total} invalid operations"
                )
            if fallback_links and self._no_support_policy_code == 0:
                raise RuntimeError(
                    "FSL research transaction unexpectedly used "
                    f"{fallback_links} fallback links"
                )
            if (
                self._no_support_policy_code != 0
                and fallback_links
                != self.stress_closure.last_diagnostics.no_support_link_count
            ):
                raise RuntimeError(
                    "FSL hybrid stream/stress NO_SUPPORT ownership disagrees: "
                    f"{fallback_links} stream links versus "
                    f"{self.stress_closure.last_diagnostics.no_support_link_count} stress links"
                )
            self.topology_resolver.resolve(
                self.candidate_fluid,
                current_free_surface,
                current_free_surface.fill_level,
                self.candidate_free_surface,
                transported_mass=current_free_surface.mass,
            )
            destination_fluid.copy_from(self.candidate_fluid)
            destination_free_surface.copy_from(self.candidate_free_surface)
            diagnostics = HomeFreeFslResearchStepDiagnostics(
                geometry=geometry_diagnostics,
                surface_tension=surface_tension_diagnostics,
                transition=transition,
                face_courant=courant_diagnostics,
                face_courant_by_axis=tuple(courant_diagnostics_by_axis),
                advection_by_axis=transport.advection_by_axis,
                topology_by_axis=tuple(topology_diagnostics_by_axis),
                stress=self.stress_closure.last_diagnostics,
                split_order=split_order,
                invalid_geometry_count=invalid_geometry,
                invalid_velocity_count=invalid_velocity,
                invalid_stream_count=invalid_stream,
                invalid_collision_count=invalid_collision,
                invalid_commit_count=0,
                fallback_link_count=fallback_links,
                projection=transport.projection,
            )
            self._split_parity = 1 - self._split_parity
            return diagnostics
        except Exception:
            if created_remapper:
                self._remapper = None
            else:
                assert self._remapper is not None
                wp.copy(self._remapper.previous_active, self._previous_active_backup)
            raise

    def _validate_state(
        self, fluid: HomeLbmState, free_surface: HomeFreeState
    ) -> None:
        if fluid.model is not self.model or free_surface.model is not self.model:
            raise ValueError("FSL research stepper and states must own the same model")
