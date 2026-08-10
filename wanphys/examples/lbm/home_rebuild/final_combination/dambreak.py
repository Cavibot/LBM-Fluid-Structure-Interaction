# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Common-state fast/geometric HOME-Free dam-break acceptance scene."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeCellFlag,
    HomeFreeGeometricDomain,
    HomeFreeLegacyDomain,
    HomeLbmModel,
)


class CombinationBackend(str, Enum):
    """Frozen comparison profiles for the final combined scene."""

    FAST = "fast"
    BALANCED = "balanced"
    STRICT = "strict"


@dataclass(frozen=True, slots=True)
class CombinationDambreakConfig:
    backend: CombinationBackend
    resolution: tuple[int, int, int]
    cell_size: float
    time_step: float
    kinematic_viscosity: float
    gravity: float
    column_width_cells: float
    column_height_cells: float
    surface_tension: float
    contact_angle_degrees: float
    periodic: tuple[bool, bool, bool]
    advection_axes: tuple[int, ...]
    max_lattice_speed: float
    steps: int
    sample_steps: tuple[int, ...]
    device: str

    @property
    def lattice_viscosity(self) -> float:
        return self.kinematic_viscosity * self.time_step / self.cell_size**2

    @property
    def lattice_gravity(self) -> float:
        return self.gravity * self.time_step**2 / self.cell_size

    @property
    def lattice_surface_tension(self) -> float:
        return self.surface_tension * self.time_step**2 / self.cell_size**3

    @property
    def project_courant(self) -> bool:
        return self.backend is CombinationBackend.STRICT

    @property
    def relative_mass_tolerance(self) -> float:
        return 5.0e-3 if self.backend is CombinationBackend.FAST else 2.0e-5

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["backend"] = self.backend.value
        result["lattice_viscosity"] = self.lattice_viscosity
        result["lattice_gravity"] = self.lattice_gravity
        result["lattice_surface_tension"] = self.lattice_surface_tension
        result["project_courant"] = self.project_courant
        result["relative_mass_tolerance"] = self.relative_mass_tolerance
        return result


def make_combination_config(
    backend: CombinationBackend | str,
    *,
    scale: str = "gate",
    device: str = "cpu",
) -> CombinationDambreakConfig:
    """Build a same-parameter gate, preview, or final Viewer scene."""

    backend = CombinationBackend(backend)
    if scale == "gate":
        resolution = (24, 8, 16)
        width = 8.5
        height = 8.5
        steps = 100
        samples = (0, 1, 10, 25, 50, 100)
    elif scale == "preview":
        resolution = (64, 32, 56)
        width = 24.5
        height = 34.5
        steps = 3000
        samples = (0, 50, 100, 200, 400, 800, 1200, 2000, 3000)
    elif scale == "viewer-default":
        resolution = (128, 128, 128)
        width = 31.5
        height = 125.5
        steps = 20000
        samples = (0, 100, 250, 500, 1000, 2000, 4000, 8000, 12000, 20000)
    else:
        raise ValueError(f"unknown combination scale: {scale}")

    return CombinationDambreakConfig(
        backend=backend,
        resolution=resolution,
        cell_size=0.1,
        time_step=1.0,
        kinematic_viscosity=2.0e-4,
        gravity=-5.0e-6,
        column_width_cells=width,
        column_height_cells=height,
        surface_tension=0.0 if backend is CombinationBackend.FAST else 5.0e-6,
        contact_angle_degrees=120.0,
        periodic=(False, False, False),
        advection_axes=(0, 1, 2),
        max_lattice_speed=0.2,
        steps=steps,
        sample_steps=samples,
        device=device,
    )


def _fractional_interval(count: int, upper_cells: float) -> np.ndarray:
    lower = np.arange(count, dtype=np.float64)
    upper = lower + 1.0
    return np.clip(np.minimum(upper, upper_cells) - lower, 0.0, 1.0).astype(
        np.float32
    )


def build_combination_fill(config: CombinationDambreakConfig) -> np.ndarray:
    """Rasterize the same sharp rectangular column for every backend."""

    nx, ny, nz = config.resolution
    fill_x = _fractional_interval(nx, config.column_width_cells)
    fill_z = _fractional_interval(nz, config.column_height_cells)
    return np.broadcast_to(
        fill_x[:, None, None] * fill_z[None, None, :], (nx, ny, nz)
    ).copy()


@dataclass(frozen=True, slots=True)
class CombinationDambreakMetrics:
    step: int
    represented_mass: float
    relative_mass_error: float
    liquid_volume: float
    liquid_centroid: tuple[float, float, float]
    occupied_minimum: tuple[int, int, int]
    occupied_maximum: tuple[int, int, int]
    interface_cells: int
    maximum_speed: float
    minimum_density: float
    maximum_density: float
    invalid_fluid_cells: int
    direct_liquid_gas_links: int
    projected_max_divergence: float
    projection_iterations: int
    topology_change_count: int
    invalid_surface_tension_cells: int
    floor_gap_columns: int
    detached_bulk_columns: int
    floor_gap_maximum_height: int
    floor_gap_minimum_xy: tuple[int, int] | None
    floor_gap_maximum_xy: tuple[int, int] | None
    left_wall_liquid_volume: float
    ceiling_liquid_volume: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CombinationDambreakScene:
    """One comparison scene whose only variable is the free-surface backend."""

    def __init__(
        self,
        config: CombinationDambreakConfig,
        model: HomeLbmModel,
        fluid: HomeFreeLegacyDomain | HomeFreeGeometricDomain,
        initial_fill: np.ndarray,
    ) -> None:
        self.config = config
        self.model = model
        self.fluid = fluid
        self.initial_fill = initial_fill
        self.initial_mass = float(
            np.sum(fluid.free_surface_state.mass.numpy(), dtype=np.float64)
        )
        self.step_index = 0

    def step(self) -> None:
        self.fluid.step(self.config.time_step)
        self.step_index += 1

    def synchronize(self) -> None:
        wp.synchronize_device(self.model._device)

    def measure(self) -> CombinationDambreakMetrics:
        state = self.fluid.free_surface_state
        fill = state.fill_level.numpy()
        flags = state.flags.numpy()
        mass = float(np.sum(state.mass.numpy(), dtype=np.float64))
        volume = float(np.sum(fill, dtype=np.float64))
        if not math.isfinite(volume) or volume <= 0.0:
            raise FloatingPointError("combined dam-break lost its represented liquid")
        coordinates = np.indices(fill.shape, dtype=np.float64)
        centroid = tuple(
            float(np.sum(coordinates[axis] * fill, dtype=np.float64) / volume)
            for axis in range(3)
        )
        occupied = np.argwhere(fill > 0.0)
        thick_liquid_columns = np.sum(fill, axis=2) > 1.0
        floor_gap_mask = thick_liquid_columns & (fill[:, :, 0] <= 0.0)
        floor_gap_positions = np.argwhere(floor_gap_mask)
        floor_gap_columns = int(len(floor_gap_positions))
        detached_bulk_columns = int(
            np.count_nonzero(
                floor_gap_mask & (np.max(fill, axis=2) >= 0.5)
            )
        )
        floor_gap_maximum_height = 0
        floor_gap_minimum_xy = None
        floor_gap_maximum_xy = None
        if floor_gap_columns:
            floor_gap_minimum_xy = tuple(
                int(value) for value in np.min(floor_gap_positions, axis=0)
            )
            floor_gap_maximum_xy = tuple(
                int(value) for value in np.max(floor_gap_positions, axis=0)
            )
            for x, y in floor_gap_positions:
                occupied_z = np.flatnonzero(fill[x, y] > 0.0)
                floor_gap_maximum_height = max(
                    floor_gap_maximum_height, int(occupied_z[0])
                )

        solver = self.fluid.last_diagnostics
        geometric = (
            self.fluid.last_geometric_diagnostics
            if isinstance(self.fluid, HomeFreeGeometricDomain)
            else None
        )
        projection = None if geometric is None else geometric.projection
        topology = () if geometric is None else geometric.topology_by_axis
        queue = None if geometric is None else geometric.queue
        topology_count = sum(
            item.gas_to_interface_count
            + item.liquid_to_interface_count
            + item.interface_to_gas_count
            + item.interface_to_liquid_count
            for _, item in topology
        )
        if queue is not None and queue.topology is not None:
            item = queue.topology
            topology_count += (
                item.gas_to_interface_count
                + item.liquid_to_interface_count
                + item.interface_to_gas_count
                + item.interface_to_liquid_count
            )
        surface = self.fluid.last_surface_tension_diagnostics
        invalid_surface = 0
        if surface is not None:
            invalid_surface = int(surface.invalid_gas_density_count) + int(
                surface.invalid_momentum_correction_count
            )

        return CombinationDambreakMetrics(
            step=self.step_index,
            represented_mass=mass,
            relative_mass_error=abs(mass - self.initial_mass) / self.initial_mass,
            liquid_volume=volume,
            liquid_centroid=centroid,
            occupied_minimum=tuple(int(value) for value in np.min(occupied, axis=0)),
            occupied_maximum=tuple(int(value) for value in np.max(occupied, axis=0)),
            interface_cells=int(
                np.count_nonzero(flags == int(HomeFreeCellFlag.INTERFACE))
            ),
            maximum_speed=0.0 if solver is None else float(solver.max_speed),
            minimum_density=0.0 if solver is None else float(solver.min_density),
            maximum_density=0.0 if solver is None else float(solver.max_density),
            invalid_fluid_cells=(
                0 if solver is None else int(solver.invalid_cell_count)
            ),
            direct_liquid_gas_links=(
                0 if solver is None else int(solver.direct_liquid_gas_link_count)
            ),
            projected_max_divergence=(
                0.0
                if projection is None
                else float(projection.projected_max_divergence)
            ),
            projection_iterations=(
                0 if projection is None else int(projection.iteration_count)
            ),
            topology_change_count=topology_count,
            invalid_surface_tension_cells=invalid_surface,
            floor_gap_columns=floor_gap_columns,
            detached_bulk_columns=detached_bulk_columns,
            floor_gap_maximum_height=floor_gap_maximum_height,
            floor_gap_minimum_xy=floor_gap_minimum_xy,
            floor_gap_maximum_xy=floor_gap_maximum_xy,
            left_wall_liquid_volume=float(np.sum(fill[0], dtype=np.float64)),
            ceiling_liquid_volume=float(np.sum(fill[:, :, -1], dtype=np.float64)),
        )


def build_combination_dambreak(
    config: CombinationDambreakConfig,
) -> CombinationDambreakScene:
    model = HomeLbmModel(
        fluid_grid_res=config.resolution,
        fluid_grid_cell_size=config.cell_size,
        time_step=config.time_step,
        reference_density=1.0,
        kinematic_viscosity=config.kinematic_viscosity,
        body_acceleration=(0.0, 0.0, config.gravity),
        periodic=config.periodic,
        max_lattice_speed=config.max_lattice_speed,
        device=config.device,
    )
    if config.backend is CombinationBackend.FAST:
        fluid: HomeFreeLegacyDomain | HomeFreeGeometricDomain = (
            HomeFreeLegacyDomain(model)
        )
    else:
        fluid = HomeFreeGeometricDomain(
            model,
            surface_tension=config.surface_tension,
            contact_angle_degrees=config.contact_angle_degrees,
            advection_axes=config.advection_axes,
            project_courant=config.project_courant,
        )
    fill = build_combination_fill(config)
    fluid.initialize_anchored_hydrostatic_lattice(
        fill,
        reference_coordinate=(
            0.5 * config.resolution[0],
            0.5 * config.resolution[1],
            config.column_height_cells + 0.5,
        ),
    )
    return CombinationDambreakScene(config, model, fluid, fill)
