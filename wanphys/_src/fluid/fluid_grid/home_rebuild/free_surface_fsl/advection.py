# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Device orchestration for paper FSL link-wise mass exchange."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..core.constants import D3Q27_DIRECTIONS, D3Q27_OPPOSITE, D3Q27_WEIGHTS
from ..core.model import HomeCoreModel
from ..core.state import HomeCoreState
from . import kernels
from .state import FslState


@dataclass(frozen=True)
class FslMassExchangeDiagnostics:
    invalid_cell_count: int
    direct_liquid_gas_link_count: int
    initial_total_mass: float
    advected_total_mass: float
    relative_mass_drift: float
    max_abs_cell_delta: float


class FslMassAdvector:
    """Evaluate Eq. (9)-(10) without committing fill or topology."""

    def __init__(self, model: HomeCoreModel) -> None:
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.advected_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device
        )
        self._opposites = wp.array(D3Q27_OPPOSITE, dtype=wp.int32, device=self.device)
        self._invalid_cell_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._direct_liquid_gas_link_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._max_abs_mass_delta = wp.zeros(1, dtype=float, device=self.device)
        self._initial_total_mass = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._advected_total_mass = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._initial_mass_cells = wp.zeros(
            self.stride, dtype=wp.float64, device=self.device
        )
        self._advected_mass_cells = wp.zeros(
            self.stride, dtype=wp.float64, device=self.device
        )
        self._source_state: FslState | None = None

    def _validate_inputs(self, fluid_state: HomeCoreState, fsl_state: FslState) -> None:
        if fluid_state.model is not self.model or fsl_state.model is not self.model:
            raise ValueError("FSL advector and states must share one HomeCoreModel")
        if fluid_state.res != self.res or fsl_state.res != self.res:
            raise ValueError("HOME core, FSL state, and advector resolutions must match")

    def advect(self, fluid_state: HomeCoreState, fsl_state: FslState) -> wp.array:
        self._validate_inputs(fluid_state, fsl_state)
        self._invalid_cell_count.zero_()
        self._direct_liquid_gas_link_count.zero_()
        self._max_abs_mass_delta.zero_()
        wp.launch(
            kernels.link_mass_exchange_kernel,
            dim=self.res,
            inputs=[
                fluid_state.moments,
                fsl_state.mass,
                fsl_state.fill_level,
                fsl_state.flags,
                self._directions,
                self._weights,
                self._opposites,
                self.advected_mass,
                self._invalid_cell_count,
                self._direct_liquid_gas_link_count,
                self._max_abs_mass_delta,
                self._initial_mass_cells,
                self._advected_mass_cells,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.utils.array_sum(self._initial_mass_cells, out=self._initial_total_mass)
        wp.utils.array_sum(self._advected_mass_cells, out=self._advected_total_mass)
        self._source_state = fsl_state
        return self.advected_mass

    def collect_diagnostics(
        self, *, synchronize: bool = True
    ) -> FslMassExchangeDiagnostics:
        if self._source_state is None:
            raise RuntimeError("advect must run before diagnostics are collected")
        if synchronize:
            wp.synchronize_device(self.device)
        initial_mass = float(self._initial_total_mass.numpy()[0])
        advected_mass = float(self._advected_total_mass.numpy()[0])
        invalid = int(self._invalid_cell_count.numpy()[0])
        direct = int(self._direct_liquid_gas_link_count.numpy()[0])
        max_delta = float(self._max_abs_mass_delta.numpy()[0])
        relative_drift = (
            (advected_mass - initial_mass) / initial_mass if initial_mass != 0.0 else 0.0
        )
        return FslMassExchangeDiagnostics(
            invalid_cell_count=invalid,
            direct_liquid_gas_link_count=direct,
            initial_total_mass=initial_mass,
            advected_total_mass=advected_mass,
            relative_mass_drift=relative_drift,
            max_abs_cell_delta=max_delta,
        )

    def validate_result(self) -> FslMassExchangeDiagnostics:
        diagnostics = self.collect_diagnostics()
        if diagnostics.invalid_cell_count:
            raise FloatingPointError(
                f"FSL mass exchange encountered {diagnostics.invalid_cell_count} invalid cells"
            )
        if diagnostics.direct_liquid_gas_link_count:
            raise RuntimeError(
                "FSL mass exchange encountered "
                f"{diagnostics.direct_liquid_gas_link_count} direct liquid-gas links"
            )
        return diagnostics
