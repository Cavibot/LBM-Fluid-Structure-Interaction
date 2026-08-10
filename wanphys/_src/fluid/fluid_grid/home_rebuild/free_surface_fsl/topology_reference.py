# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic NumPy oracle for the HOME-Free topology transaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.constants import D3Q27_DIRECTIONS
from .flags import FslCellFlag
from .reference import _neighbor_index, validate_fsl_fields


@dataclass(frozen=True)
class FslTopologyDiagnostics:
    interface_to_liquid_count: int
    interface_to_gas_count: int
    gas_to_interface_count: int
    liquid_to_interface_count: int
    cancelled_interface_to_gas_count: int
    fresh_interface_without_donor_count: int
    direct_liquid_gas_link_count: int
    initial_total_mass: float
    final_total_mass: float
    relative_total_mass_drift: float


@dataclass(frozen=True)
class FslTopologyResult:
    moments: np.ndarray
    mass: np.ndarray
    fill_level: np.ndarray
    excess_mass: np.ndarray
    flags: np.ndarray
    recipient_count: np.ndarray
    diagnostics: FslTopologyDiagnostics


def _active_neighbor_count(
    flags: np.ndarray, periodic: tuple[bool, bool, bool]
) -> np.ndarray:
    shape = tuple(int(value) for value in flags.shape)
    count = np.zeros(shape, dtype=np.int32)
    for index in np.ndindex(shape):
        for offset in D3Q27_DIRECTIONS[1:]:
            neighbor = _neighbor_index(index, offset, shape, periodic)
            if neighbor is not None and flags[neighbor] != int(FslCellFlag.GAS):
                count[index] += 1
    return count


def _receive_queued_excess(
    advected_mass: np.ndarray,
    queued_excess: np.ndarray,
    source_flags: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Deliver donor-total queues equally to active source-topology neighbors."""

    shape = tuple(int(value) for value in source_flags.shape)
    recipient_count = _active_neighbor_count(source_flags, periodic)
    mass = advected_mass.copy()
    stranded = np.zeros(shape, dtype=np.float64)
    for donor in np.ndindex(shape):
        queued = float(queued_excess[donor])
        if queued == 0.0:
            continue
        count = int(recipient_count[donor])
        if count == 0:
            stranded[donor] = queued
            continue
        share = queued / count
        for offset in D3Q27_DIRECTIONS[1:]:
            recipient = _neighbor_index(donor, offset, shape, periodic)
            if recipient is not None and source_flags[recipient] != int(FslCellFlag.GAS):
                mass[recipient] += share
    return mass, stranded


def _has_neighbor_flag(
    index: tuple[int, int, int],
    flags: np.ndarray,
    flag: FslCellFlag,
    periodic: tuple[bool, bool, bool],
) -> bool:
    shape = tuple(int(value) for value in flags.shape)
    for offset in D3Q27_DIRECTIONS[1:]:
        neighbor = _neighbor_index(index, offset, shape, periodic)
        if neighbor is not None and flags[neighbor] == int(flag):
            return True
    return False


def _initialize_fresh_interface_moments(
    moments: np.ndarray,
    source_flags: np.ndarray,
    final_flags: np.ndarray,
    fresh: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, int]:
    result = moments.copy()
    shape = tuple(int(value) for value in source_flags.shape)
    missing = 0
    for index in np.ndindex(shape):
        if not fresh[index]:
            continue
        densities: list[float] = []
        velocities: list[np.ndarray] = []
        for offset in D3Q27_DIRECTIONS[1:]:
            neighbor = _neighbor_index(index, offset, shape, periodic)
            if neighbor is None:
                continue
            if source_flags[neighbor] == int(FslCellFlag.GAS):
                continue
            if final_flags[neighbor] == int(FslCellFlag.GAS):
                continue
            rho = float(moments[neighbor][0])
            if not np.isfinite(rho) or rho <= 0.0:
                continue
            densities.append(rho)
            velocities.append(moments[neighbor][1:4] / rho)
        if not densities:
            missing += 1
            continue
        rho = float(np.mean(densities, dtype=np.float64))
        velocity = np.mean(np.asarray(velocities), axis=0, dtype=np.float64)
        result[index][0] = rho
        result[index][1:4] = rho * velocity
        result[index][4] = rho * velocity[0] * velocity[0]
        result[index][5] = rho * velocity[1] * velocity[1]
        result[index][6] = rho * velocity[2] * velocity[2]
        result[index][7] = rho * velocity[0] * velocity[1]
        result[index][8] = rho * velocity[0] * velocity[2]
        result[index][9] = rho * velocity[1] * velocity[2]
    return result, missing


def resolve_fsl_topology(
    moments: np.ndarray,
    source_flags: np.ndarray,
    advected_mass: np.ndarray,
    queued_excess_mass: np.ndarray | None = None,
    *,
    periodic: tuple[bool, bool, bool] = (True, True, True),
    transition_tolerance: float = 0.0,
) -> FslTopologyResult:
    """Resolve one paper-style topology update with deterministic staging.

    ``queued_excess_mass`` stores a donor total, unlike the official CUDA
    implementation's per-recipient share. Delivery is mathematically identical,
    while the represented total remains ``sum(mass) + sum(excess_mass)``.
    """

    home = np.asarray(moments, dtype=np.float64)
    flags = np.asarray(source_flags, dtype=np.int32)
    transported = np.asarray(advected_mass, dtype=np.float64)
    excess = (
        np.zeros(flags.shape, dtype=np.float64)
        if queued_excess_mass is None
        else np.asarray(queued_excess_mass, dtype=np.float64)
    )
    if home.ndim != 4 or home.shape[-1] != 10:
        raise ValueError("moments must have grid shape + (10,)")
    if flags.shape != home.shape[:-1] or transported.shape != flags.shape or excess.shape != flags.shape:
        raise ValueError("topology fields must share the HOME grid shape")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not np.isfinite(home).all() or not np.isfinite(transported).all() or not np.isfinite(excess).all():
        raise ValueError("topology inputs must be finite")
    if not np.isin(flags, [int(flag) for flag in FslCellFlag]).all():
        raise ValueError("source_flags contain an unknown FSL category")
    if not np.isfinite(transition_tolerance) or transition_tolerance < 0.0:
        raise ValueError("transition_tolerance must be finite and nonnegative")

    initial_total = float(np.sum(transported, dtype=np.float64) + np.sum(excess, dtype=np.float64))
    working_mass, stranded = _receive_queued_excess(
        transported, excess, flags, periodic
    )
    rho = home[..., 0]
    if np.any((flags != int(FslCellFlag.GAS)) & (rho <= 0.0)):
        raise ValueError("active source cells require positive density")

    to_liquid = np.zeros(flags.shape, dtype=bool)
    to_gas = np.zeros(flags.shape, dtype=bool)
    for index in np.ndindex(flags.shape):
        if flags[index] != int(FslCellFlag.INTERFACE):
            continue
        no_gas = not _has_neighbor_flag(index, flags, FslCellFlag.GAS, periodic)
        no_liquid = not _has_neighbor_flag(index, flags, FslCellFlag.LIQUID, periodic)
        if working_mass[index] > rho[index] + transition_tolerance or no_gas:
            to_liquid[index] = True
        elif working_mass[index] < -transition_tolerance or no_liquid:
            to_gas[index] = True

    cancelled_to_gas = np.zeros(flags.shape, dtype=bool)
    fresh = np.zeros(flags.shape, dtype=bool)
    for index in np.ndindex(flags.shape):
        if not to_liquid[index]:
            continue
        for offset in D3Q27_DIRECTIONS[1:]:
            neighbor = _neighbor_index(index, offset, flags.shape, periodic)
            if neighbor is None:
                continue
            if to_gas[neighbor]:
                cancelled_to_gas[neighbor] = True
            if flags[neighbor] == int(FslCellFlag.GAS):
                fresh[neighbor] = True
    to_gas &= ~cancelled_to_gas

    liquid_to_interface = np.zeros(flags.shape, dtype=bool)
    for index in np.ndindex(flags.shape):
        if not to_gas[index]:
            continue
        for offset in D3Q27_DIRECTIONS[1:]:
            neighbor = _neighbor_index(index, offset, flags.shape, periodic)
            if neighbor is not None and (
                flags[neighbor] == int(FslCellFlag.LIQUID) or to_liquid[neighbor]
            ):
                liquid_to_interface[neighbor] = True

    final_flags = flags.copy()
    final_flags[to_gas] = int(FslCellFlag.GAS)
    final_flags[to_liquid] = int(FslCellFlag.LIQUID)
    final_flags[liquid_to_interface | fresh] = int(FslCellFlag.INTERFACE)
    final_moments, missing_donors = _initialize_fresh_interface_moments(
        home, flags, final_flags, fresh, periodic
    )
    if missing_donors:
        raise RuntimeError(
            f"{missing_donors} fresh interface cells have no persistent active donor"
        )

    final_rho = final_moments[..., 0]
    committed_mass = np.zeros(flags.shape, dtype=np.float64)
    fill = np.zeros(flags.shape, dtype=np.float64)
    new_excess = stranded.copy()
    liquid = final_flags == int(FslCellFlag.LIQUID)
    interface = final_flags == int(FslCellFlag.INTERFACE)
    gas = final_flags == int(FslCellFlag.GAS)
    committed_mass[liquid] = final_rho[liquid]
    fill[liquid] = 1.0
    new_excess[liquid] += working_mass[liquid] - final_rho[liquid]
    committed_mass[interface] = np.clip(working_mass[interface], 0.0, final_rho[interface])
    fill[interface] = committed_mass[interface] / final_rho[interface]
    new_excess[interface] += working_mass[interface] - committed_mass[interface]
    new_excess[gas] += working_mass[gas]

    recipient_count = _active_neighbor_count(final_flags, periodic)
    validate_fsl_fields(
        final_rho,
        committed_mass,
        fill,
        new_excess,
        final_flags,
        periodic=periodic,
        allow_interface_endpoints=True,
    )
    final_total = float(
        np.sum(committed_mass, dtype=np.float64) + np.sum(new_excess, dtype=np.float64)
    )
    relative_drift = (
        (final_total - initial_total) / initial_total if initial_total != 0.0 else 0.0
    )
    diagnostics = FslTopologyDiagnostics(
        interface_to_liquid_count=int(np.count_nonzero(to_liquid & ~liquid_to_interface)),
        interface_to_gas_count=int(np.count_nonzero(to_gas)),
        gas_to_interface_count=int(np.count_nonzero(fresh)),
        liquid_to_interface_count=int(np.count_nonzero(liquid_to_interface)),
        cancelled_interface_to_gas_count=int(np.count_nonzero(cancelled_to_gas)),
        fresh_interface_without_donor_count=missing_donors,
        direct_liquid_gas_link_count=0,
        initial_total_mass=initial_total,
        final_total_mass=final_total,
        relative_total_mass_drift=relative_drift,
    )
    return FslTopologyResult(
        moments=final_moments,
        mass=committed_mass,
        fill_level=fill,
        excess_mass=new_excess,
        flags=final_flags,
        recipient_count=recipient_count,
        diagnostics=diagnostics,
    )
