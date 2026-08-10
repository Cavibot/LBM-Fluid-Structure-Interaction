# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp orchestration for the HOME-Free only-missing pressure boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..core.constants import D3Q27_DIRECTIONS, D3Q27_OPPOSITE, D3Q27_WEIGHTS
from ..core.model import HomeCoreModel
from ..core.state import HomeCoreState
from . import kernels
from .state import FslState


@dataclass(frozen=True)
class FslOnlyMissingDiagnostics:
    invalid_cell_count: int
    direct_liquid_gas_link_count: int
    gas_link_count: int
    min_active_density: float
    max_active_density: float
    max_active_speed: float


class FslOnlyMissingStreamer:
    """Pull-stream populations and apply Eq. (11) only on gas-sourced links."""

    def __init__(self, model: HomeCoreModel, *, gas_density: float = 1.0) -> None:
        if not math.isfinite(gas_density) or gas_density <= 0.0:
            raise ValueError("gas_density must be finite and positive")
        if not all(model.periodic):
            raise ValueError("the isolated HOME core currently requires periodic lattice axes")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.gas_density = float(gas_density)
        self.streamed_state = HomeCoreState(model)
        self.gas_link_count = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device
        )
        self._opposites = wp.array(D3Q27_OPPOSITE, dtype=wp.int32, device=self.device)
        self._invalid_cell_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._direct_liquid_gas_link_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._total_gas_link_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._min_density = wp.zeros(1, dtype=float, device=self.device)
        self._max_density = wp.zeros(1, dtype=float, device=self.device)
        self._max_speed_squared = wp.zeros(1, dtype=float, device=self.device)
        self._has_result = False

    def _validate_inputs(self, fluid_state: HomeCoreState, fsl_state: FslState) -> None:
        if fluid_state.model is not self.model or fsl_state.model is not self.model:
            raise ValueError("FSL streamer and states must share one HomeCoreModel")
        if fluid_state.res != self.res or fsl_state.res != self.res:
            raise ValueError("HOME core, FSL state, and streamer resolutions must match")

    def stream(self, fluid_state: HomeCoreState, fsl_state: FslState) -> HomeCoreState:
        self._validate_inputs(fluid_state, fsl_state)
        self._invalid_cell_count.zero_()
        self._direct_liquid_gas_link_count.zero_()
        self._total_gas_link_count.zero_()
        self._min_density.fill_(float("inf"))
        self._max_density.fill_(float("-inf"))
        self._max_speed_squared.zero_()
        wp.launch(
            kernels.only_missing_stream_kernel,
            dim=self.res,
            inputs=[
                fluid_state.moments,
                fsl_state.flags,
                self._directions,
                self._weights,
                self._opposites,
                self.gas_density,
                self.streamed_state.moments,
                self.gas_link_count,
                self._invalid_cell_count,
                self._direct_liquid_gas_link_count,
                self._total_gas_link_count,
                self._min_density,
                self._max_density,
                self._max_speed_squared,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        self._has_result = True
        return self.streamed_state

    def collect_diagnostics(
        self, *, synchronize: bool = True
    ) -> FslOnlyMissingDiagnostics:
        if not self._has_result:
            raise RuntimeError("stream must run before diagnostics are collected")
        if synchronize:
            wp.synchronize_device(self.device)
        return FslOnlyMissingDiagnostics(
            invalid_cell_count=int(self._invalid_cell_count.numpy()[0]),
            direct_liquid_gas_link_count=int(self._direct_liquid_gas_link_count.numpy()[0]),
            gas_link_count=int(self._total_gas_link_count.numpy()[0]),
            min_active_density=float(self._min_density.numpy()[0]),
            max_active_density=float(self._max_density.numpy()[0]),
            max_active_speed=float(np.sqrt(self._max_speed_squared.numpy()[0])),
        )

    def validate_result(self) -> FslOnlyMissingDiagnostics:
        diagnostics = self.collect_diagnostics()
        if diagnostics.invalid_cell_count:
            raise FloatingPointError(
                f"FSL only-missing stream encountered {diagnostics.invalid_cell_count} invalid cells"
            )
        if diagnostics.direct_liquid_gas_link_count:
            raise RuntimeError(
                "FSL only-missing stream encountered "
                f"{diagnostics.direct_liquid_gas_link_count} direct liquid-gas links"
            )
        return diagnostics
