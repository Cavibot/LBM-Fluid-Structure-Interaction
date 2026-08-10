# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy oracle for the paper HOME-Free state and link-wise mass exchange."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.constants import D3Q27_DIRECTIONS, D3Q27_WEIGHTS
from .flags import FslCellFlag


@dataclass(frozen=True)
class FslStateDiagnostics:
    total_mass: float
    liquid_cell_count: int
    interface_cell_count: int
    gas_cell_count: int
    direct_liquid_gas_link_count: int
    invalid_cell_count: int
    max_mass_fill_error: float


@dataclass(frozen=True)
class FslOnlyMissingStreamResult:
    moments: np.ndarray
    gas_link_count: np.ndarray


def classify_fill(fill_level: np.ndarray) -> np.ndarray:
    fill = np.asarray(fill_level, dtype=np.float64)
    if not np.isfinite(fill).all():
        raise ValueError("fill_level must be finite")
    if np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must remain in [0, 1]")
    flags = np.full(fill.shape, int(FslCellFlag.INTERFACE), dtype=np.int32)
    flags[fill == 0.0] = int(FslCellFlag.GAS)
    flags[fill == 1.0] = int(FslCellFlag.LIQUID)
    return flags


def _neighbor_index(
    index: tuple[int, int, int],
    offset: np.ndarray,
    shape: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[int, int, int] | None:
    neighbor = [index[axis] + int(offset[axis]) for axis in range(3)]
    for axis in range(3):
        if neighbor[axis] < 0 or neighbor[axis] >= shape[axis]:
            if periodic[axis]:
                neighbor[axis] %= shape[axis]
            else:
                return None
    return tuple(neighbor)


def validate_fsl_fields(
    density: np.ndarray,
    mass: np.ndarray,
    fill_level: np.ndarray,
    excess_mass: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    require_interface_separation: bool = True,
    require_mass_fill_consistency: bool = True,
    allow_interface_endpoints: bool = False,
) -> FslStateDiagnostics:
    rho = np.asarray(density, dtype=np.float64)
    liquid_mass = np.asarray(mass, dtype=np.float64)
    fill = np.asarray(fill_level, dtype=np.float64)
    excess = np.asarray(excess_mass, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if rho.ndim != 3:
        raise ValueError("FSL fields must be three-dimensional lattice arrays")
    if any(field.shape != rho.shape for field in (liquid_mass, fill, excess, cell_flags)):
        raise ValueError("all FSL fields must share one grid shape")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")

    legal = np.isin(cell_flags, [int(flag) for flag in FslCellFlag])
    finite = np.isfinite(rho) & np.isfinite(liquid_mass) & np.isfinite(fill) & np.isfinite(excess)
    in_range = (fill >= 0.0) & (fill <= 1.0)
    active = cell_flags != int(FslCellFlag.GAS)
    positive_active_density = (~active) | (rho > 0.0)
    interface_fill = (
        (fill >= 0.0) & (fill <= 1.0)
        if allow_interface_endpoints
        else (fill > 0.0) & (fill < 1.0)
    )
    flag_fill = (
        ((cell_flags == int(FslCellFlag.GAS)) & (fill == 0.0))
        | ((cell_flags == int(FslCellFlag.LIQUID)) & (fill == 1.0))
        | ((cell_flags == int(FslCellFlag.INTERFACE)) & interface_fill)
    )
    expected_mass = rho * fill
    mass_error = np.abs(liquid_mass - expected_mass)
    mass_consistent = np.ones(rho.shape, dtype=bool)
    if require_mass_fill_consistency:
        tolerance = 2.0e-6 * np.maximum(1.0, np.abs(expected_mass))
        mass_consistent = mass_error <= tolerance
    invalid = ~(legal & finite & in_range & positive_active_density & flag_fill & mass_consistent)

    direct_links = 0
    if require_interface_separation:
        directions = D3Q27_DIRECTIONS.astype(np.int32)
        shape = tuple(int(value) for value in rho.shape)
        for index in np.ndindex(shape):
            if cell_flags[index] != int(FslCellFlag.LIQUID):
                continue
            for offset in directions[1:]:
                neighbor = _neighbor_index(index, offset, shape, periodic)
                if neighbor is not None and cell_flags[neighbor] == int(FslCellFlag.GAS):
                    direct_links += 1

    diagnostics = FslStateDiagnostics(
        total_mass=float(np.sum(liquid_mass, dtype=np.float64) + np.sum(excess, dtype=np.float64)),
        liquid_cell_count=int(np.count_nonzero(cell_flags == int(FslCellFlag.LIQUID))),
        interface_cell_count=int(np.count_nonzero(cell_flags == int(FslCellFlag.INTERFACE))),
        gas_cell_count=int(np.count_nonzero(cell_flags == int(FslCellFlag.GAS))),
        direct_liquid_gas_link_count=direct_links,
        invalid_cell_count=int(np.count_nonzero(invalid)),
        max_mass_fill_error=float(np.max(mass_error, initial=0.0)),
    )
    if diagnostics.invalid_cell_count:
        raise ValueError(f"FSL state contains {diagnostics.invalid_cell_count} invalid cells")
    if diagnostics.direct_liquid_gas_link_count:
        raise ValueError(
            "FSL topology contains "
            f"{diagnostics.direct_liquid_gas_link_count} direct liquid-gas links"
        )
    return diagnostics


def initialize_fsl_fields(
    density: np.ndarray,
    fill_level: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    validate_topology: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FslStateDiagnostics]:
    rho = np.asarray(density, dtype=np.float64)
    fill = np.asarray(fill_level, dtype=np.float64)
    if rho.shape != fill.shape or rho.ndim != 3:
        raise ValueError("density and fill_level must share one three-dimensional shape")
    flags = classify_fill(fill)
    mass = rho * fill
    excess = np.zeros(fill.shape, dtype=np.float64)
    diagnostics = validate_fsl_fields(
        rho,
        mass,
        fill,
        excess,
        flags,
        periodic=periodic,
        require_interface_separation=validate_topology,
    )
    return mass, excess, flags, diagnostics


def reconstruct_populations(moments: np.ndarray) -> np.ndarray:
    home = np.asarray(moments, dtype=np.float64)
    if home.ndim < 1 or home.shape[-1] != 10:
        raise ValueError("moments must have a trailing dimension of 10")
    if not np.isfinite(home).all():
        raise ValueError("moments must be finite")
    rho = home[..., 0]
    if np.any(rho <= 0.0):
        raise ValueError("population reconstruction requires positive density")
    velocity = home[..., 1:4] / rho[..., None]
    stress = np.empty(home.shape[:-1] + (3, 3), dtype=np.float64)
    stress[..., 0, 0] = home[..., 4]
    stress[..., 1, 1] = home[..., 5]
    stress[..., 2, 2] = home[..., 6]
    stress[..., 0, 1] = stress[..., 1, 0] = home[..., 7]
    stress[..., 0, 2] = stress[..., 2, 0] = home[..., 8]
    stress[..., 1, 2] = stress[..., 2, 1] = home[..., 9]
    trace_stress = np.trace(stress, axis1=-2, axis2=-1)
    u2 = np.sum(velocity * velocity, axis=-1)
    populations = np.empty(home.shape[:-1] + (27,), dtype=np.float64)
    for q, (direction, weight) in enumerate(zip(D3Q27_DIRECTIONS, D3Q27_WEIGHTS)):
        c = direction.astype(np.float64)
        cu = np.einsum("...i,i->...", velocity, c)
        cj = np.einsum("...i,i->...", home[..., 1:4], c)
        csc = np.einsum("i,...ij,j->...", c, stress, c)
        su = np.einsum("...ij,...j->...i", stress, velocity)
        trace_a3 = 2.0 * su + trace_stress[..., None] * velocity
        trace_a3 -= 2.0 * rho[..., None] * u2[..., None] * velocity
        a3_ccc = 3.0 * csc * cu - 2.0 * rho * cu**3
        h3a3 = a3_ccc - np.einsum("...i,i->...", trace_a3, c)
        populations[..., q] = weight * (
            rho + 3.0 * cj + 4.5 * (csc - trace_stress / 3.0) + 4.5 * h3a3
        )
    return populations


def gas_pressure_boundary_populations(
    interface_moments: np.ndarray, gas_density: float | np.ndarray
) -> np.ndarray:
    """Evaluate Eq. (11); callers select only pull links sourced from gas."""

    moments = np.asarray(interface_moments, dtype=np.float64)
    if moments.ndim < 1 or moments.shape[-1] != 10:
        raise ValueError("interface_moments must have a trailing dimension of 10")
    rho = moments[..., 0]
    if not np.isfinite(moments).all() or np.any(rho <= 0.0):
        raise ValueError("interface moments must be finite with positive density")
    prescribed_rho = np.asarray(gas_density, dtype=np.float64)
    try:
        prescribed_rho = np.broadcast_to(prescribed_rho, rho.shape)
    except ValueError as error:
        raise ValueError("gas_density must broadcast to the interface grid") from error
    if not np.isfinite(prescribed_rho).all() or np.any(prescribed_rho <= 0.0):
        raise ValueError("gas_density must be finite and positive")

    velocity = moments[..., 1:4] / rho[..., None]
    cu = np.einsum("...a,qa->...q", velocity, D3Q27_DIRECTIONS)
    speed_squared = np.sum(velocity * velocity, axis=-1)
    equilibrium = D3Q27_WEIGHTS * prescribed_rho[..., None] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared[..., None]
    )
    filtered = reconstruct_populations(moments)
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    lookup = {tuple(direction): q for q, direction in enumerate(directions)}
    opposites = np.asarray([lookup[tuple(-direction)] for direction in directions])
    return equilibrium + equilibrium[..., opposites] - filtered[..., opposites]


def only_missing_stream_moments(
    moments: np.ndarray,
    flags: np.ndarray,
    *,
    gas_density: float | np.ndarray = 1.0,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> FslOnlyMissingStreamResult:
    """Pull-stream active cells and reconstruct only populations missing from gas."""

    home = np.asarray(moments, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if home.ndim != 4 or home.shape[-1] != 10:
        raise ValueError("moments must have grid shape + (10,)")
    shape = home.shape[:-1]
    if cell_flags.shape != shape:
        raise ValueError("flags must match the HOME grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not np.isin(cell_flags, [int(flag) for flag in FslCellFlag]).all():
        raise ValueError("flags contain an unknown FSL category")
    prescribed_rho = np.asarray(gas_density, dtype=np.float64)
    try:
        prescribed_rho = np.broadcast_to(prescribed_rho, shape)
    except ValueError as error:
        raise ValueError("gas_density must broadcast to the HOME grid") from error
    if not np.isfinite(prescribed_rho).all() or np.any(prescribed_rho <= 0.0):
        raise ValueError("gas_density must be finite and positive")

    populations = reconstruct_populations(home)
    boundary = gas_pressure_boundary_populations(home, prescribed_rho)
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    streamed = home.copy()
    gas_link_count = np.zeros(shape, dtype=np.int32)
    grid_shape = tuple(int(value) for value in shape)
    for destination in np.ndindex(grid_shape):
        destination_flag = FslCellFlag(int(cell_flags[destination]))
        if destination_flag == FslCellFlag.GAS:
            continue
        rho = 0.0
        momentum = np.zeros(3, dtype=np.float64)
        raw_second = np.zeros((3, 3), dtype=np.float64)
        for q, direction in enumerate(directions):
            source = _neighbor_index(destination, -direction, grid_shape, periodic)
            if source is None:
                raise ValueError("only-missing oracle requires a boundary owner on every link")
            source_flag = FslCellFlag(int(cell_flags[source]))
            if source_flag == FslCellFlag.GAS:
                if destination_flag == FslCellFlag.LIQUID:
                    raise ValueError("only-missing boundary encountered a direct liquid-gas link")
                value = boundary[destination + (q,)]
                gas_link_count[destination] += 1
            else:
                value = populations[source + (q,)]
            c = direction.astype(np.float64)
            rho += value
            momentum += value * c
            raw_second += value * np.outer(c, c)
        streamed[destination + (0,)] = rho
        streamed[destination + (slice(1, 4),)] = momentum
        streamed[destination + (4,)] = raw_second[0, 0] - rho / 3.0
        streamed[destination + (5,)] = raw_second[1, 1] - rho / 3.0
        streamed[destination + (6,)] = raw_second[2, 2] - rho / 3.0
        streamed[destination + (7,)] = raw_second[0, 1]
        streamed[destination + (8,)] = raw_second[0, 2]
        streamed[destination + (9,)] = raw_second[1, 2]
    return FslOnlyMissingStreamResult(moments=streamed, gas_link_count=gas_link_count)


def collide_streamed_moments(
    moments: np.ndarray,
    *,
    shear_omega: float,
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Apply the HOME NOCM collision to an already streamed ten-moment field."""

    streamed = np.asarray(moments, dtype=np.float64)
    if streamed.ndim < 1 or streamed.shape[-1] != 10:
        raise ValueError("moments must have a trailing dimension of 10")
    if not np.isfinite(streamed).all() or np.any(streamed[..., 0] <= 0.0):
        raise ValueError("streamed moments must be finite with positive density")
    if not np.isfinite(shear_omega) or not 0.0 < shear_omega < 2.0:
        raise ValueError("shear_omega must be finite and lie in (0, 2)")
    lattice_acceleration = np.asarray(acceleration, dtype=np.float64)
    if lattice_acceleration.shape != (3,) or not np.isfinite(lattice_acceleration).all():
        raise ValueError("acceleration must contain three finite components")

    rho = streamed[..., 0]
    momentum = streamed[..., 1:4]
    force = rho[..., None] * lattice_acceleration
    velocity = (momentum + 0.5 * force) / rho[..., None]
    stress = np.empty(streamed.shape[:-1] + (3, 3), dtype=np.float64)
    stress[..., 0, 0] = streamed[..., 4] / rho
    stress[..., 1, 1] = streamed[..., 5] / rho
    stress[..., 2, 2] = streamed[..., 6] / rho
    stress[..., 0, 1] = stress[..., 1, 0] = streamed[..., 7] / rho
    stress[..., 0, 2] = stress[..., 2, 0] = streamed[..., 8] / rho
    stress[..., 1, 2] = stress[..., 2, 1] = streamed[..., 9] / rho
    trace_stress = np.trace(stress, axis1=-2, axis2=-1)
    speed_squared = np.sum(velocity * velocity, axis=-1)
    equilibrium = velocity[..., :, None] * velocity[..., None, :]
    deviatoric = stress - equilibrium
    isotropic_correction = (speed_squared - trace_stress) / 3.0
    indices = np.arange(3)
    deviatoric[..., indices, indices] += isotropic_correction[..., None]
    post_stress = equilibrium + (1.0 - shear_omega) * deviatoric

    force_power = np.sum(force * velocity, axis=-1)
    diagonal_force = force * velocity
    diagonal_force += (1.0 - shear_omega) * (
        3.0 * force * velocity - force_power[..., None]
    ) / 3.0
    post_stress[..., indices, indices] += diagonal_force / rho[..., None]
    force_shear = (1.0 - 0.5 * shear_omega) / rho
    post_stress[..., 0, 1] += force_shear * (
        force[..., 0] * velocity[..., 1] + force[..., 1] * velocity[..., 0]
    )
    post_stress[..., 0, 2] += force_shear * (
        force[..., 0] * velocity[..., 2] + force[..., 2] * velocity[..., 0]
    )
    post_stress[..., 1, 2] += force_shear * (
        force[..., 1] * velocity[..., 2] + force[..., 2] * velocity[..., 1]
    )
    post_stress[..., 1, 0] = post_stress[..., 0, 1]
    post_stress[..., 2, 0] = post_stress[..., 0, 2]
    post_stress[..., 2, 1] = post_stress[..., 1, 2]

    collided = streamed.copy()
    collided[..., 1:4] = momentum + force
    collided[..., 4] = rho * post_stress[..., 0, 0]
    collided[..., 5] = rho * post_stress[..., 1, 1]
    collided[..., 6] = rho * post_stress[..., 2, 2]
    collided[..., 7] = rho * post_stress[..., 0, 1]
    collided[..., 8] = rho * post_stress[..., 0, 2]
    collided[..., 9] = rho * post_stress[..., 1, 2]
    return collided


def link_mass_exchange(
    moments: np.ndarray,
    mass: np.ndarray,
    fill_level: np.ndarray,
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (True, True, True),
) -> np.ndarray:
    """Apply Eq. (9)-(10) on active-active D3Q27 links only."""

    home = np.asarray(moments, dtype=np.float64)
    liquid_mass = np.asarray(mass, dtype=np.float64)
    fill = np.asarray(fill_level, dtype=np.float64)
    cell_flags = np.asarray(flags, dtype=np.int32)
    if home.ndim != 4 or home.shape[-1] != 10:
        raise ValueError("moments must have grid shape + (10,)")
    shape = home.shape[:-1]
    if any(field.shape != shape for field in (liquid_mass, fill, cell_flags)):
        raise ValueError("mass, fill_level, and flags must match the HOME grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not np.isfinite(liquid_mass).all() or not np.isfinite(fill).all():
        raise ValueError("mass exchange fields must be finite")
    if np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("fill_level must remain in [0, 1]")
    if not np.isin(cell_flags, [int(flag) for flag in FslCellFlag]).all():
        raise ValueError("flags contain an unknown FSL category")

    populations = reconstruct_populations(home)
    directions = D3Q27_DIRECTIONS.astype(np.int32)
    opposites = np.empty(27, dtype=np.int32)
    direction_lookup = {tuple(direction): q for q, direction in enumerate(directions)}
    for q, direction in enumerate(directions):
        opposites[q] = direction_lookup[tuple(-direction)]

    advected = liquid_mass.copy()
    grid_shape = tuple(int(value) for value in shape)
    for destination in np.ndindex(grid_shape):
        destination_flag = FslCellFlag(int(cell_flags[destination]))
        if destination_flag == FslCellFlag.GAS:
            continue
        for q in range(1, 27):
            source = _neighbor_index(destination, -directions[q], grid_shape, periodic)
            if source is None:
                continue
            source_flag = FslCellFlag(int(cell_flags[source]))
            if source_flag == FslCellFlag.GAS:
                continue
            alpha = 1.0
            if destination_flag == FslCellFlag.INTERFACE and source_flag == FslCellFlag.INTERFACE:
                alpha = 0.5 * (fill[destination] + fill[source])
            incoming = populations[source + (q,)]
            outgoing = populations[destination + (int(opposites[q]),)]
            advected[destination] += alpha * (incoming - outgoing)
    return advected
