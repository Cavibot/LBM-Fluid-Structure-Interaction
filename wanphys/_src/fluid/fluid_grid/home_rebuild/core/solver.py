# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Host orchestration for the branch-minimal periodic HOME kernel."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from . import kernels
from .constants import D3Q27_DIRECTIONS, D3Q27_WEIGHTS
from .diagnostics import HomeCoreDiagnostics
from .model import HomeCoreModel
from .state import HomeCoreState


class HomeCoreSolver:
    def __init__(self, model: HomeCoreModel) -> None:
        self.model = model
        self.nx = int(model.nx)
        self.ny = int(model.ny)
        self.nz = int(model.nz)
        self.stride = self.nx * self.ny * self.nz
        self.device = model._device
        self._directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self._weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device
        )
        self._invalid_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._min_density = wp.zeros(1, dtype=float, device=self.device)
        self._max_density = wp.zeros(1, dtype=float, device=self.device)
        self._max_speed_squared = wp.zeros(1, dtype=float, device=self.device)
        self._max_stress_squared = wp.zeros(1, dtype=float, device=self.device)
        self._packed_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._packed_values = wp.zeros(4, dtype=float, device=self.device)
        self._diagnostics_state: HomeCoreState | None = None

    def _reset_diagnostics(self) -> None:
        self._invalid_count.zero_()
        self._min_density.fill_(float("inf"))
        self._max_density.fill_(float("-inf"))
        self._max_speed_squared.zero_()
        self._max_stress_squared.zero_()

    def initialize_uniform_lattice(
        self,
        state: HomeCoreState,
        rho: float = 1.0,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)))
        if not math.isfinite(rho) or rho <= 0.0:
            raise ValueError("rho must be finite and positive")
        if speed > self.model.max_lattice_speed:
            raise ValueError(
                f"initial lattice speed {speed} exceeds configured limit {self.model.max_lattice_speed}"
            )
        wp.launch(
            kernels.initialize_uniform_kernel,
            dim=self.stride,
            inputs=[state.moments, float(rho), wp.vec3(*velocity), self.stride],
            device=self.device,
        )
        self._diagnostics_state = None

    def step(self, state_in: HomeCoreState, state_out: HomeCoreState, dt: float) -> None:
        self.model.validate_step(dt)
        if state_in.model is not self.model or state_out.model is not self.model:
            raise ValueError("HOME core solver and states must share one model")
        self._reset_diagnostics()
        wp.launch(
            kernels.stream_collide_periodic_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_in.moments,
                self._directions,
                self._weights,
                state_out.moments,
                float(self.model.shear_omega),
                wp.vec3(*self.model.lattice_acceleration),
                self._invalid_count,
                self._min_density,
                self._max_density,
                self._max_speed_squared,
                self._max_stress_squared,
                self.nx,
                self.ny,
                self.nz,
                self.stride,
            ],
            device=self.device,
        )
        self._diagnostics_state = state_out

    def collide(self, streamed_state: HomeCoreState, state_out: HomeCoreState, dt: float) -> None:
        """Collide a pre-streamed ten-moment state without streaming it again."""

        self.model.validate_step(dt)
        if streamed_state.model is not self.model or state_out.model is not self.model:
            raise ValueError("HOME core solver and states must share one model")
        self._reset_diagnostics()
        wp.launch(
            kernels.collide_kernel,
            dim=self.stride,
            inputs=[
                streamed_state.moments,
                state_out.moments,
                float(self.model.shear_omega),
                wp.vec3(*self.model.lattice_acceleration),
                self._invalid_count,
                self._min_density,
                self._max_density,
                self._max_speed_squared,
                self._max_stress_squared,
                self.stride,
            ],
            device=self.device,
        )
        self._diagnostics_state = state_out

    def collect_diagnostics(
        self, state: HomeCoreState, *, force_recompute: bool = False
    ) -> HomeCoreDiagnostics:
        if force_recompute or state is not self._diagnostics_state:
            self._reset_diagnostics()
            wp.launch(
                kernels.diagnose_kernel,
                dim=self.stride,
                inputs=[
                    state.moments,
                    self._invalid_count,
                    self._min_density,
                    self._max_density,
                    self._max_speed_squared,
                    self._max_stress_squared,
                    self.stride,
                ],
                device=self.device,
            )
            self._diagnostics_state = state
        wp.launch(
            kernels.pack_diagnostics_kernel,
            dim=1,
            inputs=[
                self._invalid_count,
                self._min_density,
                self._max_density,
                self._max_speed_squared,
                self._max_stress_squared,
                self._packed_count,
                self._packed_values,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        count = self._packed_count.numpy()
        values = self._packed_values.numpy()
        return HomeCoreDiagnostics(
            invalid_cell_count=int(count[0]),
            min_density=float(values[0]),
            max_density=float(values[1]),
            max_speed=float(np.sqrt(values[2])),
            max_nonequilibrium_stress=float(np.sqrt(values[3])),
        )

    def validate_state(self, state: HomeCoreState) -> HomeCoreDiagnostics:
        diagnostics = self.collect_diagnostics(state)
        if diagnostics.invalid_cell_count:
            raise FloatingPointError(
                f"HOME core state contains {diagnostics.invalid_cell_count} invalid cells"
            )
        if diagnostics.max_speed > self.model.max_lattice_speed:
            raise FloatingPointError(
                f"HOME core maximum lattice speed {diagnostics.max_speed} exceeds "
                f"configured limit {self.model.max_lattice_speed}"
            )
        return diagnostics
