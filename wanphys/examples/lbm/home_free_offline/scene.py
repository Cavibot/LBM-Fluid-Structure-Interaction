# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Common lifecycle and diagnostics for offline HOME-Free scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeCellFlag,
    HomeFreeGeometricDomain,
    HomeFreeRigidCoupling,
)
from wanphys.rigid import RigidDomain

from .config import HomeFreeOfflineConfig


@dataclass(frozen=True, slots=True)
class OfflineSceneMetrics:
    step: int
    physical_time: float
    represented_mass: float
    relative_mass_error: float
    liquid_volume: float
    maximum_fill: float
    liquid_centroid: tuple[float, float, float]
    occupied_minimum: tuple[int, int, int]
    occupied_maximum: tuple[int, int, int]
    interface_cells: int
    minimum_density: float
    maximum_density: float
    maximum_speed: float
    maximum_mach: float
    invalid_fluid_cells: int
    missing_cut_links: int
    direct_liquid_gas_links: int
    initial_max_divergence: float
    projected_max_divergence: float
    projection_iterations: int
    maximum_face_correction: float
    queue_relative_mass_error: float
    queue_relative_momentum_error: float
    queue_maximum_fill_delta: float
    gas_to_interface_cells: int
    liquid_to_interface_cells: int
    interface_to_gas_cells: int
    interface_to_liquid_cells: int
    snapped_cells: int
    snapped_volume_delta: float
    maximum_advection_bound_correction: float
    fresh_cells: int
    dead_cells: int
    wet_links: int
    dry_links: int
    invalid_rigid_flags: int
    dry_nonzero_impulses: int
    transition_relative_mass_error: float
    transition_relative_momentum_error: float
    maximum_lattice_displacement: float
    strong_coupling_iterations: int
    strong_coupling_residual: float
    rigid_geometry_capacity_ratio: float
    body_position: tuple[float, float, float] | None
    body_velocity: tuple[float, float, float] | None
    body_angular_velocity: tuple[float, float, float] | None


@dataclass
class HomeFreeOfflineScene:
    config: HomeFreeOfflineConfig
    fluid: HomeFreeGeometricDomain
    initial_fill: np.ndarray
    initial_mass: float
    rigid: RigidDomain | None = None
    coupling: HomeFreeRigidCoupling | None = None
    body_index: int | None = None
    step_index: int = 0
    last_step_diagnostics: Any | None = None

    @property
    def physical_time(self) -> float:
        return self.step_index * self.config.fluid.time_step

    def step(self) -> Any:
        if self.coupling is None:
            self.fluid.step(self.config.fluid.time_step)
            diagnostics = self.fluid.last_geometric_diagnostics
        else:
            diagnostics = self.coupling.step(self.config.fluid.time_step)
        self.step_index += 1
        self.last_step_diagnostics = diagnostics
        return diagnostics

    def run(
        self,
        steps: int | None = None,
        *,
        callback: Callable[[HomeFreeOfflineScene], None] | None = None,
    ) -> None:
        step_count = self.config.steps if steps is None else int(steps)
        if step_count < 0:
            raise ValueError("steps must be non-negative")
        for _ in range(step_count):
            self.step()
            if callback is not None:
                callback(self)

    def measure(self) -> OfflineSceneMetrics:
        free = self.fluid.free_surface_state
        mass = float(np.sum(free.mass.numpy(), dtype=np.float64))
        fill = free.fill_level.numpy()
        flags = free.flags.numpy()
        volume = float(np.sum(fill, dtype=np.float64))
        coordinates = np.indices(fill.shape, dtype=np.float64)
        centroid = tuple(
            float(np.sum(coordinates[axis] * fill, dtype=np.float64) / volume)
            for axis in range(3)
        )
        occupied = np.argwhere(fill > 0.0)
        occupied_minimum = tuple(int(value) for value in np.min(occupied, axis=0))
        occupied_maximum = tuple(int(value) for value in np.max(occupied, axis=0))
        geometric = self.fluid.last_geometric_diagnostics
        projection = None if geometric is None else geometric.projection
        queue = None if geometric is None else geometric.queue
        solver = None if geometric is None else geometric.solver

        topology = () if geometric is None else geometric.topology_by_axis
        topology_diagnostics = [diagnostics for _, diagnostics in topology]
        if queue is not None and queue.topology is not None:
            topology_diagnostics.append(queue.topology)
        advection = () if geometric is None else geometric.advection_by_axis

        fresh_cells = 0
        dead_cells = 0
        wet_links = 0
        dry_links = 0
        invalid_rigid_flags = 0
        dry_nonzero_impulses = 0
        transition_relative_mass_error = 0.0
        transition_relative_momentum_error = 0.0
        maximum_lattice_displacement = 0.0
        strong_coupling_iterations = 0
        strong_coupling_residual = 0.0
        rigid_geometry_capacity_ratio = 0.0
        if self.coupling is not None and self.last_step_diagnostics is not None:
            rigid_diagnostics = self.last_step_diagnostics
            interface = rigid_diagnostics.interface
            fresh_cells = int(interface.transitions.fresh_cell_count)
            dead_cells = int(interface.transitions.dead_cell_count)
            wet_links = int(interface.load.wet_link_count)
            dry_links = int(interface.load.dry_link_count)
            invalid_rigid_flags = int(interface.load.invalid_flag_count)
            dry_nonzero_impulses = int(interface.load.dry_nonzero_impulse_count)
            transition_relative_mass_error = float(
                interface.transitions.relative_mass_error
            )
            transition_relative_momentum_error = float(
                interface.transitions.relative_momentum_error
            )
            maximum_lattice_displacement = float(
                rigid_diagnostics.maximum_lattice_displacement
            )
            strong_coupling_iterations = int(
                rigid_diagnostics.strong_coupling_iterations
            )
            strong_coupling_residual = float(
                rigid_diagnostics.strong_coupling_residual
            )
            rigid_geometry_capacity_ratio = float(interface.geometry.capacity_ratio)

        body_position = None
        body_velocity = None
        body_angular_velocity = None
        if self.rigid is not None and self.body_index is not None:
            q = self.rigid.state.body_q
            qd = self.rigid.state.body_qd
            if q is None or qd is None:
                raise RuntimeError("rigid scene lost its body state arrays")
            body_position = tuple(
                float(value) for value in q.numpy()[self.body_index, :3]
            )
            body_velocity = tuple(
                float(value) for value in qd.numpy()[self.body_index, :3]
            )
            body_angular_velocity = tuple(
                float(value) for value in qd.numpy()[self.body_index, 3:]
            )

        return OfflineSceneMetrics(
            step=self.step_index,
            physical_time=self.physical_time,
            represented_mass=mass,
            relative_mass_error=abs(mass - self.initial_mass) / self.initial_mass,
            liquid_volume=volume,
            maximum_fill=float(np.max(fill)),
            liquid_centroid=centroid,
            occupied_minimum=occupied_minimum,
            occupied_maximum=occupied_maximum,
            interface_cells=int(
                np.count_nonzero(flags == int(HomeFreeCellFlag.INTERFACE))
            ),
            minimum_density=(0.0 if solver is None else float(solver.min_density)),
            maximum_density=(0.0 if solver is None else float(solver.max_density)),
            maximum_speed=(0.0 if solver is None else float(solver.max_speed)),
            maximum_mach=(
                0.0 if solver is None else float(solver.max_speed * np.sqrt(3.0))
            ),
            invalid_fluid_cells=(
                0 if solver is None else int(solver.invalid_cell_count)
            ),
            missing_cut_links=(
                0 if solver is None else int(solver.missing_cut_link_count)
            ),
            direct_liquid_gas_links=(
                0 if solver is None else int(solver.direct_liquid_gas_link_count)
            ),
            initial_max_divergence=(
                0.0 if projection is None else float(projection.initial_max_divergence)
            ),
            projected_max_divergence=(
                0.0 if projection is None else float(projection.projected_max_divergence)
            ),
            projection_iterations=(
                0 if projection is None else int(projection.iteration_count)
            ),
            maximum_face_correction=(
                0.0 if projection is None else float(projection.maximum_face_correction)
            ),
            queue_relative_mass_error=(
                0.0 if queue is None else float(queue.relative_mass_error)
            ),
            queue_relative_momentum_error=(
                0.0 if queue is None else float(queue.relative_momentum_error)
            ),
            queue_maximum_fill_delta=(
                0.0 if queue is None else float(queue.maximum_fill_delta)
            ),
            gas_to_interface_cells=sum(
                item.gas_to_interface_count for item in topology_diagnostics
            ),
            liquid_to_interface_cells=sum(
                item.liquid_to_interface_count for item in topology_diagnostics
            ),
            interface_to_gas_cells=sum(
                item.interface_to_gas_count for item in topology_diagnostics
            ),
            interface_to_liquid_cells=sum(
                item.interface_to_liquid_count for item in topology_diagnostics
            ),
            snapped_cells=sum(
                item.snapped_cell_count for item in topology_diagnostics
            ),
            snapped_volume_delta=sum(
                item.snapped_volume_delta for item in topology_diagnostics
            ),
            maximum_advection_bound_correction=max(
                (
                    float(item.maximum_bound_correction)
                    for _, item in advection
                ),
                default=0.0,
            ),
            fresh_cells=fresh_cells,
            dead_cells=dead_cells,
            wet_links=wet_links,
            dry_links=dry_links,
            invalid_rigid_flags=invalid_rigid_flags,
            dry_nonzero_impulses=dry_nonzero_impulses,
            transition_relative_mass_error=transition_relative_mass_error,
            transition_relative_momentum_error=transition_relative_momentum_error,
            maximum_lattice_displacement=maximum_lattice_displacement,
            strong_coupling_iterations=strong_coupling_iterations,
            strong_coupling_residual=strong_coupling_residual,
            rigid_geometry_capacity_ratio=rigid_geometry_capacity_ratio,
            body_position=body_position,
            body_velocity=body_velocity,
            body_angular_velocity=body_angular_velocity,
        )

    def synchronize(self) -> None:
        wp.synchronize_device(self.config.device)
