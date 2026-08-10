# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Strict NumPy oracle for committing geometrically transported FSL state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.constants import D3Q27_DIRECTIONS
from ..free_surface_fsl import FslCellFlag


@dataclass(frozen=True)
class PlicTopologyDiagnostics:
    gas_to_interface_count: int
    liquid_to_interface_count: int
    interface_to_gas_count: int
    interface_to_liquid_count: int
    sub_tolerance_interface_count: int
    fresh_interface_count: int
    direct_liquid_gas_link_count: int
    volume_drift: float
    mass_drift: float
    momentum_drift: tuple[float, float, float]
    endpoint_promoted_liquid_count: int = 0
    separation_repair_count: int = 0
    endpoint_removed_gas_count: int = 0
    endpoint_redistributed_volume: float = 0.0
    endpoint_redistributed_mass: float = 0.0
    endpoint_redistributed_momentum: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PlicTopologyResult:
    moments: np.ndarray
    fill_level: np.ndarray
    mass: np.ndarray
    momentum: np.ndarray
    flags: np.ndarray
    donor_count: np.ndarray
    diagnostics: PlicTopologyDiagnostics


def _neighbor(
    index: tuple[int, int, int],
    offset: np.ndarray,
    shape: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[int, int, int] | None:
    result = [index[axis] + int(offset[axis]) for axis in range(3)]
    for axis in range(3):
        if result[axis] < 0 or result[axis] >= shape[axis]:
            if not periodic[axis]:
                return None
            result[axis] %= shape[axis]
    return tuple(result)


def _equilibrium_moments(rho: float, velocity: np.ndarray) -> np.ndarray:
    ux, uy, uz = velocity
    return np.asarray(
        (
            rho,
            rho * ux,
            rho * uy,
            rho * uz,
            rho * ux * ux,
            rho * uy * uy,
            rho * uz * uz,
            rho * ux * uy,
            rho * ux * uz,
            rho * uy * uz,
        ),
        dtype=np.float64,
    )


def resolve_plic_topology_reference(
    moments: np.ndarray,
    source_flags: np.ndarray,
    transported_fill: np.ndarray,
    transported_mass: np.ndarray,
    transported_momentum: np.ndarray,
    solid: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    endpoint_tolerance: float = 4.0e-7,
) -> PlicTopologyResult:
    """Classify and commit one PLIC step without modifying any conserved ledger."""

    home = np.asarray(moments, dtype=np.float64)
    source = np.asarray(source_flags, dtype=np.int32)
    fill = np.asarray(transported_fill, dtype=np.float64)
    mass = np.asarray(transported_mass, dtype=np.float64)
    momentum = np.asarray(transported_momentum, dtype=np.float64)
    wall = np.asarray(solid, dtype=bool)
    shape = tuple(int(value) for value in source.shape)
    if home.shape != shape + (10,) or momentum.shape != shape + (3,):
        raise ValueError("PLIC topology moments or momentum shape is invalid")
    if any(field.shape != shape for field in (fill, mass, wall)):
        raise ValueError("PLIC topology scalar fields must share one grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not 0.0 <= endpoint_tolerance < 0.5:
        raise ValueError("endpoint_tolerance must be in [0, 0.5)")
    if not all(np.isfinite(field).all() for field in (home, fill, mass, momentum)):
        raise ValueError("PLIC topology inputs must be finite")
    if np.any(fill < 0.0) or np.any(fill > 1.0) or np.any(mass < 0.0):
        raise ValueError("PLIC topology fill and mass must be nonnegative and bounded")
    if np.any(fill[wall] != 0.0) or np.any(mass[wall] != 0.0) or np.any(momentum[wall] != 0.0):
        raise ValueError("solid cells cannot contain transported ledgers")

    gas = int(FslCellFlag.GAS)
    interface = int(FslCellFlag.INTERFACE)
    liquid = int(FslCellFlag.LIQUID)
    legal = np.isin(source, (gas, interface, liquid))
    if not legal.all():
        raise ValueError("source flags contain an unknown FSL category")
    source_direct_links = 0
    for index in np.ndindex(shape):
        if wall[index] or source[index] != liquid:
            continue
        for offset in D3Q27_DIRECTIONS[1:]:
            neighbor = _neighbor(index, offset, shape, periodic)
            if neighbor is not None and not wall[neighbor] and source[neighbor] == gas:
                source_direct_links += 1
    if source_direct_links:
        raise RuntimeError(
            f"PLIC topology source contains {source_direct_links} direct liquid-gas links"
        )

    endpoint_tail = ~wall & (fill > 0.0) & (fill <= endpoint_tolerance)
    recipient = (
        ~wall
        & (fill > endpoint_tolerance)
        & (fill < 1.0 - endpoint_tolerance)
    )
    closed_fill = fill.copy()
    closed_mass = mass.copy()
    closed_momentum = momentum.copy()
    redistributed_volume = float(np.sum(fill[endpoint_tail], dtype=np.float64))
    redistributed_mass = float(np.sum(mass[endpoint_tail], dtype=np.float64))
    redistributed_momentum = np.sum(
        momentum[endpoint_tail], axis=0, dtype=np.float64
    )
    if redistributed_volume > 0.0:
        capacity = np.where(recipient, 1.0 - fill, 0.0)
        total_capacity = float(np.sum(capacity, dtype=np.float64))
        if not np.isfinite(total_capacity) or total_capacity < redistributed_volume:
            raise RuntimeError("PLIC endpoint closure has insufficient interface capacity")
        weights = capacity / total_capacity
        closed_fill[endpoint_tail] = 0.0
        closed_mass[endpoint_tail] = 0.0
        closed_momentum[endpoint_tail] = 0.0
        closed_fill += redistributed_volume * weights
        closed_mass += redistributed_mass * weights
        closed_momentum += weights[..., None] * redistributed_momentum
    if np.any(closed_fill < 0.0) or np.any(closed_fill > 1.0):
        raise RuntimeError("PLIC endpoint closure produced an invalid fill fraction")

    classified = np.full(shape, interface, dtype=np.int32)
    classified[closed_fill == 0.0] = gas
    classified[closed_fill >= 1.0 - endpoint_tolerance] = liquid
    classified[wall] = gas
    if np.any((source == gas) & (classified == liquid)):
        raise RuntimeError("PLIC topology forbids a one-step gas-to-liquid jump")
    if np.any((source == liquid) & (classified == gas)):
        raise RuntimeError("PLIC topology forbids a one-step liquid-to-gas jump")
    target = classified.copy()
    separation_repair = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        if wall[index] or classified[index] != liquid:
            continue
        for offset in D3Q27_DIRECTIONS[1:]:
            neighbor = _neighbor(index, offset, shape, periodic)
            if (
                neighbor is not None
                and not wall[neighbor]
                and classified[neighbor] == gas
            ):
                target[index] = interface
                separation_repair[index] = True
                break
    direct_links = 0
    for index in np.ndindex(shape):
        if wall[index] or target[index] != liquid:
            continue
        for offset in D3Q27_DIRECTIONS[1:]:
            neighbor = _neighbor(index, offset, shape, periodic)
            if neighbor is not None and not wall[neighbor] and target[neighbor] == gas:
                direct_links += 1
    if direct_links:
        raise RuntimeError(
            f"PLIC topology produced {direct_links} direct liquid-gas links"
        )

    fresh = (source == gas) & (target == interface)
    donor_count = np.zeros(shape, dtype=np.int32)
    committed_moments = home.copy()
    for index in np.ndindex(shape):
        if not fresh[index]:
            continue
        donors: list[tuple[int, int, int]] = []
        for offset in D3Q27_DIRECTIONS[1:]:
            neighbor = _neighbor(index, offset, shape, periodic)
            if neighbor is None or wall[neighbor]:
                continue
            if source[neighbor] != gas and target[neighbor] != gas:
                donors.append(neighbor)
        donor_count[index] = len(donors)
        if not donors:
            raise RuntimeError("fresh PLIC interface has no persistent fluid donor")
        density = np.asarray([home[cell][0] for cell in donors])
        if np.any(density <= 0.0):
            raise RuntimeError("fresh PLIC interface donor has invalid density")
        velocity = np.asarray([home[cell][1:4] / home[cell][0] for cell in donors])
        committed_moments[index] = _equilibrium_moments(
            float(np.mean(density, dtype=np.float64)),
            np.mean(velocity, axis=0, dtype=np.float64),
        )

    active = (target != gas) & ~wall
    if np.any(committed_moments[..., 0][active] <= 0.0):
        raise RuntimeError("committed PLIC active cells require positive HOME density")
    sliver = active & (
        (closed_fill <= endpoint_tolerance)
        | (closed_fill >= 1.0 - endpoint_tolerance)
    )
    diagnostics = PlicTopologyDiagnostics(
        int(np.count_nonzero((source == gas) & (target == interface))),
        int(np.count_nonzero((source == liquid) & (target == interface))),
        int(np.count_nonzero((source == interface) & (target == gas))),
        int(np.count_nonzero((source == interface) & (target == liquid))),
        int(np.count_nonzero(sliver)),
        int(np.count_nonzero(fresh)),
        direct_links,
        float(
            np.sum(closed_fill, dtype=np.float64)
            - np.sum(transported_fill, dtype=np.float64)
        ),
        float(
            np.sum(closed_mass, dtype=np.float64)
            - np.sum(transported_mass, dtype=np.float64)
        ),
        tuple(
            float(value)
            for value in (
                np.sum(closed_momentum, axis=(0, 1, 2), dtype=np.float64)
                - np.sum(transported_momentum, axis=(0, 1, 2), dtype=np.float64)
            )
        ),
        int(
            np.count_nonzero(
                (closed_fill < 1.0) & (classified == liquid) & ~wall
            )
        ),
        int(np.count_nonzero(separation_repair)),
        int(np.count_nonzero(endpoint_tail)),
        redistributed_volume,
        redistributed_mass,
        tuple(float(value) for value in redistributed_momentum),
    )
    return PlicTopologyResult(
        committed_moments,
        closed_fill,
        closed_mass,
        closed_momentum,
        target,
        donor_count,
        diagnostics,
    )
