# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Laplace-pressure boundary density derived from PLIC curvature."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..constants import D3Q27_DIRECTIONS, D3Q27_WEIGHTS
from ..model import HomeLbmModel
from ..state import HomeLbmState
from . import geometry_kernels, kernels
from .geometry import HomeFreeGeometryDiagnostics, HomeFreeInterfaceGeometry
from .state import HomeFreeState


@dataclass(frozen=True)
class HomeFreeSurfaceTensionDiagnostics:
    geometry: HomeFreeGeometryDiagnostics
    invalid_gas_density_count: int
    invalid_momentum_correction_count: int
    min_interface_gas_density: float
    max_interface_gas_density: float


class HomeFreeSurfaceTension:
    """Evaluate paper Eq. (12) in explicit physical/lattice units."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        ambient_gas_density: float = 1.0,
        surface_tension: float,
        geometry: HomeFreeInterfaceGeometry | None = None,
    ) -> None:
        if not math.isfinite(ambient_gas_density) or ambient_gas_density <= 0.0:
            raise ValueError("ambient_gas_density must be finite and positive")
        if not math.isfinite(surface_tension) or surface_tension < 0.0:
            raise ValueError("surface_tension must be finite and nonnegative")
        if geometry is not None and geometry.model is not model:
            raise ValueError("injected interface geometry must own the same model")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.ambient_gas_density = float(ambient_gas_density)
        self.surface_tension = float(surface_tension)
        self.lattice_surface_tension = model.scaling.surface_tension_to_lattice(
            surface_tension
        )
        self.geometry = geometry or HomeFreeInterfaceGeometry(model)
        self.gas_density = wp.full(
            self.res,
            self.ambient_gas_density,
            dtype=float,
            device=self.device,
        )
        self.capillary_momentum_correction: wp.array | None = None
        self._directions: wp.array | None = None
        self._weights: wp.array | None = None
        self._invalid_density_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._invalid_momentum_correction_count: wp.array | None = None
        self._invalid_applied_correction_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )
        self._min_density = wp.zeros(1, dtype=float, device=self.device)
        self._max_density = wp.zeros(1, dtype=float, device=self.device)
        self._updated_state: HomeFreeState | None = None

    def enable_plic_capillary_momentum(self) -> None:
        """Allocate the experimental PLIC-minus-staircase momentum ledger."""

        if self.capillary_momentum_correction is not None:
            return
        self.capillary_momentum_correction = wp.zeros(
            3 * self.stride, dtype=float, device=self.device
        )
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device
        )
        self._invalid_momentum_correction_count = wp.zeros(
            1, dtype=wp.int32, device=self.device
        )

    def update(
        self,
        state: HomeFreeState,
        *,
        solid_phi: wp.array | None = None,
    ) -> HomeFreeSurfaceTensionDiagnostics:
        if state.res != self.res or state.device != self.device:
            raise ValueError("HOME-Free state and surface tension must match")
        geometry_diagnostics = self.geometry.update(
            state,
            solid_phi=solid_phi,
            compute_curvature=self.lattice_surface_tension > 0.0,
        )
        self._invalid_density_count.zero_()
        if self._invalid_momentum_correction_count is not None:
            self._invalid_momentum_correction_count.zero_()
        self._min_density.fill_(float("inf"))
        self._max_density.fill_(float("-inf"))
        wp.launch(
            geometry_kernels.compute_laplace_gas_density_kernel,
            dim=self.res,
            inputs=[
                state.flags,
                self.geometry.curvature,
                self.geometry.curvature_valid,
                self.geometry.curvature_required,
                self.ambient_gas_density,
                6.0 * self.lattice_surface_tension,
                self.gas_density,
                self._invalid_density_count,
                self._min_density,
                self._max_density,
            ],
            device=self.device,
        )
        if self.capillary_momentum_correction is not None:
            assert self._directions is not None
            assert self._weights is not None
            assert self._invalid_momentum_correction_count is not None
            wp.launch(
                geometry_kernels.compute_plic_capillary_momentum_correction_kernel,
                dim=self.res,
                inputs=[
                    state.flags,
                    self.geometry.normal,
                    self.geometry.interface_area,
                    self.gas_density,
                    self.ambient_gas_density,
                    self._directions,
                    self._weights,
                    self.capillary_momentum_correction,
                    self._invalid_momentum_correction_count,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    *self.res,
                    self.stride,
                ],
                device=self.device,
            )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_density_count.numpy()[0])
        invalid_correction = (
            int(self._invalid_momentum_correction_count.numpy()[0])
            if self._invalid_momentum_correction_count is not None
            else 0
        )
        minimum = float(self._min_density.numpy()[0])
        maximum = float(self._max_density.numpy()[0])
        diagnostics = HomeFreeSurfaceTensionDiagnostics(
            geometry=geometry_diagnostics,
            invalid_gas_density_count=invalid,
            invalid_momentum_correction_count=invalid_correction,
            min_interface_gas_density=minimum,
            max_interface_gas_density=maximum,
        )
        if invalid:
            raise FloatingPointError(
                "HOME-Free Eq. (12) produced "
                f"{invalid} nonpositive, non-finite, or geometry-invalid gas densities"
            )
        if invalid_correction:
            raise FloatingPointError(
                "HOME-Free PLIC capillary correction produced "
                f"{invalid_correction} invalid interface momentum values"
            )
        self._updated_state = state
        return diagnostics

    def apply_momentum_correction(
        self,
        fluid_state: HomeLbmState,
        source_state: HomeFreeState,
    ) -> None:
        """Replace staircase capillary impulse on full-cell HOME moments."""

        if source_state is not self._updated_state:
            raise RuntimeError(
                "capillary momentum correction requires update() on the same source state"
            )
        if self.capillary_momentum_correction is None:
            raise RuntimeError(
                "capillary momentum correction requires "
                "enable_plic_capillary_momentum() before update()"
            )
        if fluid_state.res != self.res or fluid_state.device != self.device:
            raise ValueError("HOME state and surface tension must match")
        self._invalid_applied_correction_count.zero_()
        wp.launch(
            kernels.apply_full_cell_momentum_correction_kernel,
            dim=self.res,
            inputs=[
                fluid_state.moments,
                source_state.flags,
                self.capillary_momentum_correction,
                self._invalid_applied_correction_count,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        invalid = int(self._invalid_applied_correction_count.numpy()[0])
        if invalid:
            raise FloatingPointError(
                "HOME-Free PLIC capillary application produced "
                f"{invalid} invalid interface moments"
            )
