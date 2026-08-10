# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy oracle for closed walls and hydrostatic HOME-Free initialization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.constants import D3Q27_DIRECTIONS, D3Q27_OPPOSITE
from .flags import FslCellFlag
from .reference import (
    _neighbor_index,
    gas_pressure_boundary_populations,
    reconstruct_populations,
)


@dataclass(frozen=True)
class FslWallStreamResult:
    moments: np.ndarray
    gas_link_count: np.ndarray
    wall_link_count: np.ndarray


@dataclass(frozen=True)
class FslHydrostaticFields:
    moments: np.ndarray
    mass: np.ndarray
    fill_level: np.ndarray
    excess_mass: np.ndarray
    flags: np.ndarray


def closed_box_wall_mask(shape: tuple[int, int, int]) -> np.ndarray:
    if len(shape) != 3 or any(int(value) < 3 for value in shape):
        raise ValueError("closed wall boxes require three dimensions of at least 3 cells")
    solid = np.zeros(tuple(int(value) for value in shape), dtype=bool)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = 0
        upper[axis] = -1
        solid[tuple(lower)] = True
        solid[tuple(upper)] = True
    return solid


def axis_aligned_wall_mask(
    shape: tuple[int, int, int], *, closed_axes: tuple[bool, bool, bool]
) -> np.ndarray:
    """Build boundary planes on closed axes and leave periodic axes unmasked."""

    if len(shape) != 3 or len(closed_axes) != 3:
        raise ValueError("wall grids and closed_axes must be three-dimensional")
    normalized_shape = tuple(int(value) for value in shape)
    if any(size < (3 if closed else 1) for size, closed in zip(normalized_shape, closed_axes)):
        raise ValueError("closed axes require at least 3 cells; periodic axes require at least 1")
    if not any(closed_axes):
        raise ValueError("at least one axis must be closed")
    solid = np.zeros(normalized_shape, dtype=bool)
    for axis, closed in enumerate(closed_axes):
        if not closed:
            continue
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = 0
        upper[axis] = -1
        solid[tuple(lower)] = True
        solid[tuple(upper)] = True
    return solid


def validate_wall_mask(
    solid_mask: np.ndarray, *, closed_axes: tuple[bool, bool, bool]
) -> np.ndarray:
    """Validate solid planes while permitting unmasked periodic directions."""

    solid = np.asarray(solid_mask, dtype=bool)
    if solid.ndim != 3 or len(closed_axes) != 3:
        raise ValueError("solid_mask and closed_axes must be three-dimensional")
    if any(size < (3 if closed else 1) for size, closed in zip(solid.shape, closed_axes)):
        raise ValueError("closed axes require interior cells; periodic axes require one cell")
    if not any(closed_axes):
        raise ValueError("at least one axis must be closed")
    for axis, closed in enumerate(closed_axes):
        if not closed:
            continue
        if not np.all(np.take(solid, 0, axis=axis)):
            raise ValueError(f"solid_mask lower axis-{axis} plane must be closed")
        if not np.all(np.take(solid, -1, axis=axis)):
            raise ValueError(f"solid_mask upper axis-{axis} plane must be closed")
    if np.all(solid):
        raise ValueError("solid_mask must leave at least one non-solid cell")
    return solid


def validate_closed_wall_mask(solid_mask: np.ndarray) -> np.ndarray:
    return validate_wall_mask(solid_mask, closed_axes=(True, True, True))


def _validate_wall_fields(
    moments: np.ndarray, flags: np.ndarray, solid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    home = np.asarray(moments, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    solid = validate_closed_wall_mask(solid_mask)
    if home.ndim != 4 or home.shape[-1] != 10:
        raise ValueError("moments must have grid shape + (10,)")
    if cell_flags.shape != home.shape[:-1] or solid.shape != cell_flags.shape:
        raise ValueError("moments, flags, and solid_mask must share one grid")
    if not np.isfinite(home).all():
        raise ValueError("moments must be finite")
    if not np.isin(cell_flags, [int(flag) for flag in FslCellFlag]).all():
        raise ValueError("flags contain an unknown FSL category")
    if np.any((~solid) & (cell_flags != int(FslCellFlag.GAS)) & (home[..., 0] <= 0.0)):
        raise ValueError("active non-solid cells require positive density")
    return home, cell_flags, solid


def wall_only_missing_stream_moments(
    moments: np.ndarray,
    flags: np.ndarray,
    solid_mask: np.ndarray,
    *,
    gas_density: float | np.ndarray = 1.0,
) -> FslWallStreamResult:
    """Pull active populations with gas Eq. (11) and halfway wall bounce-back."""

    home, cell_flags, solid = _validate_wall_fields(moments, flags, solid_mask)
    shape = tuple(int(value) for value in cell_flags.shape)
    prescribed_rho = np.broadcast_to(np.asarray(gas_density, dtype=np.float64), shape)
    if not np.isfinite(prescribed_rho).all() or np.any(prescribed_rho <= 0.0):
        raise ValueError("gas_density must be finite, positive, and broadcast to the grid")
    populations = reconstruct_populations(home)
    gas_boundary = gas_pressure_boundary_populations(home, prescribed_rho)
    streamed = home.copy()
    gas_links = np.zeros(shape, dtype=np.int32)
    wall_links = np.zeros(shape, dtype=np.int32)
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    for destination in np.ndindex(shape):
        if solid[destination] or cell_flags[destination] == int(FslCellFlag.GAS):
            continue
        values = np.empty(27, dtype=np.float64)
        for q, direction in enumerate(directions):
            source = _neighbor_index(destination, -direction, shape, (True, True, True))
            assert source is not None
            if solid[source]:
                values[q] = populations[destination + (int(D3Q27_OPPOSITE[q]),)]
                wall_links[destination] += 1
            elif cell_flags[source] == int(FslCellFlag.GAS):
                if cell_flags[destination] == int(FslCellFlag.LIQUID):
                    raise ValueError("wall-aware stream found a direct liquid-gas link")
                values[q] = gas_boundary[destination + (q,)]
                gas_links[destination] += 1
            else:
                values[q] = populations[source + (q,)]
        c = directions.astype(np.float64)
        rho = float(np.sum(values))
        momentum = np.einsum("q,qa->a", values, c)
        second = np.einsum("q,qa,qb->ab", values, c, c)
        streamed[destination][0] = rho
        streamed[destination][1:4] = momentum
        streamed[destination][4] = second[0, 0] - rho / 3.0
        streamed[destination][5] = second[1, 1] - rho / 3.0
        streamed[destination][6] = second[2, 2] - rho / 3.0
        streamed[destination][7] = second[0, 1]
        streamed[destination][8] = second[0, 2]
        streamed[destination][9] = second[1, 2]
    return FslWallStreamResult(streamed, gas_links, wall_links)


def wall_link_mass_exchange(
    moments: np.ndarray,
    mass: np.ndarray,
    fill_level: np.ndarray,
    flags: np.ndarray,
    solid_mask: np.ndarray,
) -> np.ndarray:
    """Apply FSL link exchange while enforcing exactly zero wall-normal mass flux."""

    home, cell_flags, solid = _validate_wall_fields(moments, flags, solid_mask)
    liquid_mass = np.asarray(mass, dtype=np.float64)
    fill = np.asarray(fill_level, dtype=np.float64)
    if liquid_mass.shape != cell_flags.shape or fill.shape != cell_flags.shape:
        raise ValueError("mass and fill_level must match the wall grid")
    populations = reconstruct_populations(home)
    advected = liquid_mass.copy()
    advected[solid] = 0.0
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    shape = tuple(int(value) for value in cell_flags.shape)
    for destination in np.ndindex(shape):
        if solid[destination] or cell_flags[destination] == int(FslCellFlag.GAS):
            continue
        for q in range(1, 27):
            source = _neighbor_index(destination, -directions[q], shape, (True, True, True))
            assert source is not None
            if solid[source] or cell_flags[source] == int(FslCellFlag.GAS):
                continue
            alpha = 1.0
            if (
                cell_flags[destination] == int(FslCellFlag.INTERFACE)
                and cell_flags[source] == int(FslCellFlag.INTERFACE)
            ):
                alpha = 0.5 * (fill[destination] + fill[source])
            advected[destination] += alpha * (
                populations[source + (q,)]
                - populations[destination + (int(D3Q27_OPPOSITE[q]),)]
            )
    return advected


def initialize_hydrostatic_fields(
    fill_level: np.ndarray,
    solid_mask: np.ndarray,
    *,
    gas_density: float,
    gravity_axis: int,
    lattice_acceleration: float,
    surface_coordinate: float,
    closed_axes: tuple[bool, bool, bool] = (True, True, True),
) -> FslHydrostaticFields:
    """Initialize isothermal hydrostatic density from ``c_s^2=1/3``."""

    fill = np.asarray(fill_level, dtype=np.float64)
    solid = validate_wall_mask(solid_mask, closed_axes=closed_axes)
    if fill.shape != solid.shape:
        raise ValueError("fill_level and solid_mask must share one grid")
    if gravity_axis not in (0, 1, 2):
        raise ValueError("gravity_axis must be 0, 1, or 2")
    if not np.isfinite(gas_density) or gas_density <= 0.0:
        raise ValueError("gas_density must be finite and positive")
    if not np.isfinite(lattice_acceleration) or lattice_acceleration == 0.0:
        raise ValueError("lattice_acceleration must be finite and nonzero")
    if not np.isfinite(surface_coordinate):
        raise ValueError("surface_coordinate must be finite")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must be finite and remain in [0, 1]")
    if np.any(fill[solid] != 0.0):
        raise ValueError("solid cells must have zero fill")
    coordinates = np.arange(fill.shape[gravity_axis], dtype=np.float64)
    reshape = [1, 1, 1]
    reshape[gravity_axis] = fill.shape[gravity_axis]
    coordinate_grid = coordinates.reshape(reshape)
    density = gas_density * np.exp(
        3.0 * lattice_acceleration * (coordinate_grid - surface_coordinate)
    )
    density = np.broadcast_to(density, fill.shape).copy()
    density[fill == 0.0] = gas_density
    density[solid] = gas_density
    flags = np.full(fill.shape, int(FslCellFlag.INTERFACE), dtype=np.int32)
    flags[fill == 0.0] = int(FslCellFlag.GAS)
    flags[fill == 1.0] = int(FslCellFlag.LIQUID)
    flags[solid] = int(FslCellFlag.GAS)
    moments = np.zeros(fill.shape + (10,), dtype=np.float64)
    moments[..., 0] = density
    mass = density * fill
    mass[solid] = 0.0
    return FslHydrostaticFields(
        moments=moments,
        mass=mass,
        fill_level=fill.copy(),
        excess_mass=np.zeros(fill.shape, dtype=np.float64),
        flags=flags,
    )
