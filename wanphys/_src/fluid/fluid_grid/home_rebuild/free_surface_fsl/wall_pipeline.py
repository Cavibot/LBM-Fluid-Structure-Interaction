# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Host wrappers for closed-wall stream, mass exchange, and active collision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ..core.constants import D3Q27_DIRECTIONS, D3Q27_OPPOSITE, D3Q27_WEIGHTS
from ..core.diagnostics import HomeCoreDiagnostics
from ..core.model import HomeCoreModel
from ..core.state import HomeCoreState
from .state import FslState
from .walls import FslWallMask
from . import wall_kernels


@dataclass(frozen=True)
class FslWallStreamDiagnostics:
    invalid_cell_count: int
    direct_liquid_gas_link_count: int
    gas_link_count: int
    wall_link_count: int


@dataclass(frozen=True)
class FslWallMassDiagnostics:
    invalid_cell_count: int
    direct_liquid_gas_link_count: int
    initial_total_mass: float
    advected_total_mass: float
    relative_mass_drift: float


class _WallGridOperator:
    def __init__(self, model: HomeCoreModel, walls: FslWallMask) -> None:
        if walls.model is not model:
            raise ValueError("wall operator and mask must share one HomeCoreModel")
        self.model = model
        self.walls = walls
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.directions = wp.array(
            D3Q27_DIRECTIONS.astype(np.float32), dtype=wp.vec3, device=self.device
        )
        self.weights = wp.array(
            D3Q27_WEIGHTS.astype(np.float32), dtype=float, device=self.device
        )
        self.opposites = wp.array(
            D3Q27_OPPOSITE, dtype=wp.int32, device=self.device
        )

    def validate_states(self, fluid: HomeCoreState, fsl: FslState) -> None:
        if fluid.model is not self.model or fsl.model is not self.model:
            raise ValueError("wall operator and states must share one HomeCoreModel")


class FslWallOnlyMissingStreamer(_WallGridOperator):
    def __init__(
        self, model: HomeCoreModel, walls: FslWallMask, *, gas_density: float = 1.0
    ) -> None:
        super().__init__(model, walls)
        if not np.isfinite(gas_density) or gas_density <= 0.0:
            raise ValueError("gas_density must be finite and positive")
        self.gas_density = float(gas_density)
        self._uniform_gas_density_field = wp.full(
            self.res, self.gas_density, dtype=float, device=self.device
        )
        self.streamed_state = HomeCoreState(model)
        self.gas_link_count = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.wall_link_count = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self._invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._direct = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._total_gas = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._total_wall = wp.zeros(1, dtype=wp.int32, device=self.device)

    def stream(
        self,
        fluid: HomeCoreState,
        fsl: FslState,
        *,
        gas_density_field: wp.array | None = None,
    ) -> HomeCoreState:
        self.validate_states(fluid, fsl)
        if gas_density_field is not None:
            if tuple(gas_density_field.shape) != self.res:
                raise ValueError("gas density field must match the HOME grid")
            if gas_density_field.dtype != wp.float32:
                raise TypeError("gas density field must use Warp float")
            if gas_density_field.device != self.device:
                raise ValueError("gas density field must share the HOME device")
        boundary_density = (
            self._uniform_gas_density_field
            if gas_density_field is None
            else gas_density_field
        )
        self._invalid.zero_()
        self._direct.zero_()
        self._total_gas.zero_()
        self._total_wall.zero_()
        wp.launch(
            wall_kernels.wall_only_missing_stream_kernel,
            dim=self.res,
            inputs=[
                fluid.moments, fsl.flags, self.walls.device,
                self.directions, self.weights, self.opposites, self.gas_density,
                boundary_density, int(gas_density_field is not None),
                self.streamed_state.moments, self.gas_link_count, self.wall_link_count,
                self._invalid, self._direct, self._total_gas, self._total_wall,
                *self.res, self.stride,
            ],
            device=self.device,
        )
        return self.streamed_state

    def collect_diagnostics(self) -> FslWallStreamDiagnostics:
        wp.synchronize_device(self.device)
        return FslWallStreamDiagnostics(
            invalid_cell_count=int(self._invalid.numpy()[0]),
            direct_liquid_gas_link_count=int(self._direct.numpy()[0]),
            gas_link_count=int(self._total_gas.numpy()[0]),
            wall_link_count=int(self._total_wall.numpy()[0]),
        )


class FslWallMassAdvector(_WallGridOperator):
    def __init__(self, model: HomeCoreModel, walls: FslWallMask) -> None:
        super().__init__(model, walls)
        self.advected_mass = wp.zeros(self.res, dtype=float, device=self.device)
        self._invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._direct = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._initial_cells = wp.zeros(self.stride, dtype=wp.float64, device=self.device)
        self._final_cells = wp.zeros(self.stride, dtype=wp.float64, device=self.device)
        self._initial_total = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._final_total = wp.zeros(1, dtype=wp.float64, device=self.device)

    def advect(self, fluid: HomeCoreState, fsl: FslState) -> wp.array:
        self.validate_states(fluid, fsl)
        self._invalid.zero_()
        self._direct.zero_()
        wp.launch(
            wall_kernels.wall_mass_exchange_kernel,
            dim=self.res,
            inputs=[
                fluid.moments, fsl.mass, fsl.fill_level, fsl.flags, self.walls.device,
                self.directions, self.weights, self.opposites, self.advected_mass,
                self._invalid, self._direct, self._initial_cells, self._final_cells,
                *self.res, self.stride,
            ],
            device=self.device,
        )
        wp.utils.array_sum(self._initial_cells, out=self._initial_total)
        wp.utils.array_sum(self._final_cells, out=self._final_total)
        return self.advected_mass

    def collect_diagnostics(self) -> FslWallMassDiagnostics:
        wp.synchronize_device(self.device)
        initial = float(self._initial_total.numpy()[0])
        final = float(self._final_total.numpy()[0])
        return FslWallMassDiagnostics(
            invalid_cell_count=int(self._invalid.numpy()[0]),
            direct_liquid_gas_link_count=int(self._direct.numpy()[0]),
            initial_total_mass=initial,
            advected_total_mass=final,
            relative_mass_drift=(final - initial) / initial if initial != 0.0 else 0.0,
        )


class FslWallActiveCollider(_WallGridOperator):
    def __init__(self, model: HomeCoreModel, walls: FslWallMask) -> None:
        super().__init__(model, walls)
        self._invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._min_density = wp.zeros(1, dtype=float, device=self.device)
        self._max_density = wp.zeros(1, dtype=float, device=self.device)
        self._max_speed_squared = wp.zeros(1, dtype=float, device=self.device)
        self._max_stress_squared = wp.zeros(1, dtype=float, device=self.device)

    def _reset_diagnostics(self) -> None:
        self._invalid.zero_()
        self._min_density.fill_(float("inf"))
        self._max_density.fill_(float("-inf"))
        self._max_speed_squared.zero_()
        self._max_stress_squared.zero_()

    def collide(
        self,
        streamed: HomeCoreState,
        fsl: FslState,
        destination: HomeCoreState,
        dt: float,
    ) -> HomeCoreState:
        self.validate_states(streamed, fsl)
        if destination.model is not self.model:
            raise ValueError("wall collider and destination must share one model")
        self.model.validate_step(dt)
        self._reset_diagnostics()
        wp.launch(
            wall_kernels.wall_active_collide_kernel,
            dim=self.res,
            inputs=[
                streamed.moments, fsl.flags, self.walls.device, destination.moments,
                float(self.model.shear_omega), wp.vec3(*self.model.lattice_acceleration),
                self._invalid, self._min_density, self._max_density,
                self._max_speed_squared, self._max_stress_squared,
                self.res[1], self.res[2], self.stride,
            ],
            device=self.device,
        )
        return destination

    def diagnose(
        self, state: HomeCoreState, fsl: FslState
    ) -> HomeCoreDiagnostics:
        self.validate_states(state, fsl)
        self._reset_diagnostics()
        wp.launch(
            wall_kernels.diagnose_wall_active_state_kernel,
            dim=self.res,
            inputs=[
                state.moments,
                fsl.flags,
                self.walls.device,
                self._invalid,
                self._min_density,
                self._max_density,
                self._max_speed_squared,
                self._max_stress_squared,
                self.res[1],
                self.res[2],
                self.stride,
            ],
            device=self.device,
        )
        return self.collect_diagnostics()

    def collect_diagnostics(self) -> HomeCoreDiagnostics:
        wp.synchronize_device(self.device)
        return HomeCoreDiagnostics(
            invalid_cell_count=int(self._invalid.numpy()[0]),
            min_density=float(self._min_density.numpy()[0]),
            max_density=float(self._max_density.numpy()[0]),
            max_speed=float(np.sqrt(self._max_speed_squared.numpy()[0])),
            max_nonequilibrium_stress=float(np.sqrt(self._max_stress_squared.numpy()[0])),
        )
