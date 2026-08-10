# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolated PLIC curvature to Laplace-pressure coupling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import warp as wp

from .curvature import (
    PlicCurvatureDiagnostics,
    PlicCurvatureEstimator3D,
    PlicCurvatureState,
)
from .geometry import (
    PlicGeometryDiagnostics,
    PlicGeometryReconstructor,
    PlicGeometryState,
)
from . import surface_tension_kernels

if TYPE_CHECKING:
    from ..core import HomeCoreModel, HomeCoreState
    from ..free_surface_fsl import FslState, FslWallMask
    from ..free_surface_fsl.wall_stepper import (
        FslWallDynamicDiagnostics,
        FslWallDynamicTopologyStepper,
    )


@dataclass(frozen=True)
class PlicSurfaceTensionDiagnostics:
    geometry: PlicGeometryDiagnostics
    curvature: PlicCurvatureDiagnostics
    invalid_gas_density_count: int
    minimum_interface_gas_density: float
    maximum_interface_gas_density: float


class PlicSurfaceTension:
    """Evaluate the HOME-Free Laplace boundary density without VOF feedback."""

    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        *,
        ambient_gas_density: float = 1.0,
        surface_tension: float,
        contact_angle_degrees: float | None = None,
    ) -> None:
        if walls.model is not model:
            raise ValueError("surface tension and walls must share one model")
        if not math.isfinite(ambient_gas_density) or ambient_gas_density <= 0.0:
            raise ValueError("ambient gas density must be finite and positive")
        if not math.isfinite(surface_tension) or surface_tension < 0.0:
            raise ValueError("surface tension must be finite and nonnegative")
        self.model = model
        self.walls = walls
        self.res = walls.res
        self.device = model._device
        self.ambient_gas_density = float(ambient_gas_density)
        self.surface_tension = float(surface_tension)
        self.contact_angle_degrees = contact_angle_degrees
        self.lattice_surface_tension = model.scaling.surface_tension_to_lattice(
            surface_tension
        )
        self.geometry = PlicGeometryState(model)
        self.reconstructor = PlicGeometryReconstructor(
            model,
            walls,
            contact_angle_degrees=contact_angle_degrees,
        )
        self.curvature = PlicCurvatureState(model)
        self.curvature_estimator = PlicCurvatureEstimator3D(
            model,
            walls,
            strict_bulk=False,
            fit_wall_contact=contact_angle_degrees is not None,
        )
        self.gas_density = wp.full(
            self.res,
            self.ambient_gas_density,
            dtype=float,
            device=self.device,
        )
        self._invalid_density_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._minimum_density = wp.zeros(1, dtype=float, device=self.device)
        self._maximum_density = wp.zeros(1, dtype=float, device=self.device)
        self.last_diagnostics: PlicSurfaceTensionDiagnostics | None = None

    def update(self, state: FslState) -> PlicSurfaceTensionDiagnostics:
        if state.model is not self.model:
            raise ValueError("surface tension and FSL state must share one model")
        geometry = self.reconstructor.reconstruct(state, self.geometry)
        curvature = self.curvature_estimator.reconstruct(
            state.fill_level,
            state.flags,
            self.geometry,
            self.curvature,
        )
        unresolved_bulk = (
            curvature.insufficient_neighbor_count
            + curvature.ill_conditioned_count
        )
        if unresolved_bulk:
            raise FloatingPointError(
                f"surface tension has {unresolved_bulk} unresolved bulk curvatures"
            )
        self._invalid_density_count.zero_()
        self._minimum_density.fill_(float("inf"))
        self._maximum_density.fill_(float("-inf"))
        wp.launch(
            surface_tension_kernels.compute_laplace_gas_density_kernel,
            dim=self.res,
            inputs=[
                state.flags,
                self.curvature.curvature,
                self.curvature.valid,
                self.curvature.required,
                self.ambient_gas_density,
                6.0 * self.lattice_surface_tension,
                self.gas_density,
                self._invalid_density_count,
                self._minimum_density,
                self._maximum_density,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_density_count.numpy()[0])
        diagnostics = PlicSurfaceTensionDiagnostics(
            geometry=geometry,
            curvature=curvature,
            invalid_gas_density_count=invalid,
            minimum_interface_gas_density=float(self._minimum_density.numpy()[0]),
            maximum_interface_gas_density=float(self._maximum_density.numpy()[0]),
        )
        self.last_diagnostics = diagnostics
        if invalid:
            raise FloatingPointError(
                f"surface tension produced {invalid} nonpositive gas densities"
            )
        return diagnostics


@dataclass(frozen=True)
class PlicCapillaryWallDiagnostics:
    surface_tension: PlicSurfaceTensionDiagnostics
    baseline: FslWallDynamicDiagnostics

    @property
    def stream(self):
        return self.baseline.stream

    @property
    def mass_exchange(self):
        return self.baseline.mass_exchange

    @property
    def fluid(self):
        return self.baseline.fluid

    @property
    def topology(self):
        return self.baseline.topology


class PlicCapillaryWallStepper:
    """Add only a curvature-driven pressure field to the frozen wall stepper."""

    def __init__(
        self,
        baseline: FslWallDynamicTopologyStepper,
        surface_tension: PlicSurfaceTension,
    ) -> None:
        if baseline.model is not surface_tension.model:
            raise ValueError("capillary and baseline steppers must share one model")
        if baseline.walls is not surface_tension.walls:
            raise ValueError("capillary and baseline steppers must share one wall mask")
        self.baseline = baseline
        self.surface_tension = surface_tension
        self.model = baseline.model
        self.walls = baseline.walls
        self.last_diagnostics: PlicCapillaryWallDiagnostics | None = None

    def step(
        self,
        fluid_in: HomeCoreState,
        fsl_in: FslState,
        fluid_out: HomeCoreState,
        fsl_out: FslState,
        dt: float,
    ) -> PlicCapillaryWallDiagnostics:
        surface = self.surface_tension.update(fsl_in)
        baseline = self.baseline.step(
            fluid_in,
            fsl_in,
            fluid_out,
            fsl_out,
            dt,
            gas_density_field=self.surface_tension.gas_density,
        )
        diagnostics = PlicCapillaryWallDiagnostics(surface, baseline)
        self.last_diagnostics = diagnostics
        return diagnostics
