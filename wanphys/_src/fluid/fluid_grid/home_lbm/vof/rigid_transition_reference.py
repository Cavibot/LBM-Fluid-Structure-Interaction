# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""NumPy phase-classification contract for moving rigid HOME-Free cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flags import HomeFreeCellFlag
from .reference import validate_state_fields


@dataclass(frozen=True)
class HomeFreeRigidPhaseRemap:
    """Deterministic phase proposal before coupled mass/moment correction."""

    flags: np.ndarray
    fill_level: np.ndarray
    fresh_mask: np.ndarray
    dead_mask: np.ndarray
    fresh_liquid_cells: int
    fresh_interface_cells: int
    fresh_gas_cells: int


def classify_moving_solid_phases(
    previous_solid: np.ndarray,
    current_solid: np.ndarray,
    previous_flags: np.ndarray,
    previous_fill: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> HomeFreeRigidPhaseRemap:
    """Classify cells exposed or covered by a sub-cell rigid-body motion.

    Fresh-cell donors must remain fluid in both masks. This prevents a chain
    of newly exposed cells from depending on thread order in the GPU version.
    The function proposes phase/fill only; conservation of moments and liquid
    mass is the responsibility of the coupled runtime remapper.
    """

    old_solid = np.asarray(previous_solid, dtype=bool)
    new_solid = np.asarray(current_solid, dtype=bool)
    flags = np.asarray(previous_flags, dtype=np.int32)
    fill = np.asarray(previous_fill, dtype=np.float64)
    if not (old_solid.shape == new_solid.shape == flags.shape == fill.shape):
        raise ValueError("moving-solid phase fields must have identical shapes")
    if old_solid.ndim != 3:
        raise ValueError("moving-solid phase fields must be three-dimensional")
    if len(periodic) != 3:
        raise ValueError("periodic must contain one flag per axis")
    if not np.isfinite(fill).all() or np.any(fill < 0.0) or np.any(fill > 1.0):
        raise ValueError("previous_fill must contain finite values in [0, 1]")
    expected_old_solid = flags == int(HomeFreeCellFlag.SOLID)
    if not np.array_equal(expected_old_solid, old_solid):
        raise ValueError("previous solid mask and HOME-Free SOLID flags disagree")

    persistent = ~old_solid & ~new_solid
    fresh = old_solid & ~new_solid
    dead = ~old_solid & new_solid
    output_flags = flags.copy()
    output_fill = fill.copy()
    output_flags[new_solid] = int(HomeFreeCellFlag.SOLID)
    output_fill[new_solid] = 0.0

    fresh_liquid = 0
    fresh_interface = 0
    fresh_gas = 0
    for index in zip(*np.nonzero(fresh), strict=True):
        donors = list(_persistent_neighbors(index, persistent, periodic))
        if not donors:
            raise RuntimeError(f"fresh cell {index} has no persistent fluid donor")
        donor_flags = np.asarray([flags[donor] for donor in donors], dtype=np.int32)
        donor_fill = np.asarray([fill[donor] for donor in donors], dtype=np.float64)
        if np.any(donor_flags == int(HomeFreeCellFlag.SOLID)):
            raise RuntimeError("persistent donor cannot be SOLID")
        has_liquid = np.any(donor_flags == int(HomeFreeCellFlag.LIQUID))
        has_interface = np.any(donor_flags == int(HomeFreeCellFlag.INTERFACE))
        has_gas = np.any(donor_flags == int(HomeFreeCellFlag.GAS))
        if has_interface or (has_liquid and has_gas):
            output_flags[index] = int(HomeFreeCellFlag.INTERFACE)
            output_fill[index] = float(np.mean(donor_fill))
            fresh_interface += 1
        elif has_liquid:
            output_flags[index] = int(HomeFreeCellFlag.LIQUID)
            output_fill[index] = 1.0
            fresh_liquid += 1
        elif has_gas:
            output_flags[index] = int(HomeFreeCellFlag.GAS)
            output_fill[index] = 0.0
            fresh_gas += 1
        else:
            raise RuntimeError(f"fresh cell {index} has no valid phase donor")

    # Reuse the central topology validator with synthetic positive density and
    # phase-consistent mass. Endpoint interfaces are allowed for one remap.
    density = np.ones(flags.shape, dtype=np.float64)
    mass = output_fill.copy()
    excess = np.zeros(flags.shape, dtype=np.float64)
    validate_state_fields(
        density,
        mass,
        output_fill,
        excess,
        np.zeros(flags.shape + (3,), dtype=np.float64),
        output_flags,
        periodic=periodic,
        require_interface_separation=True,
        allow_interface_endpoints=True,
    )
    return HomeFreeRigidPhaseRemap(
        flags=output_flags,
        fill_level=output_fill,
        fresh_mask=fresh,
        dead_mask=dead,
        fresh_liquid_cells=fresh_liquid,
        fresh_interface_cells=fresh_interface,
        fresh_gas_cells=fresh_gas,
    )


def _persistent_neighbors(
    index: tuple[int, int, int],
    persistent: np.ndarray,
    periodic: tuple[bool, bool, bool],
):
    shape = persistent.shape
    for offset in np.ndindex(3, 3, 3):
        delta = tuple(value - 1 for value in offset)
        if delta == (0, 0, 0):
            continue
        neighbor = []
        valid = True
        for axis in range(3):
            value = index[axis] + delta[axis]
            if periodic[axis]:
                value %= shape[axis]
            elif value < 0 or value >= shape[axis]:
                valid = False
                break
            neighbor.append(value)
        if valid:
            candidate = tuple(neighbor)
            if persistent[candidate]:
                yield candidate
