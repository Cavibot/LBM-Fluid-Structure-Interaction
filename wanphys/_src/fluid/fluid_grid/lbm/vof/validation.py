# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only invariant checks for authoritative VOF initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .contracts import VofCellType
from .initialization import (
    validate_initial_topology,
    validate_initialized_density,
    validate_no_solid_cells,
)

if TYPE_CHECKING:
    from ..state import LbmStateBase
    from .state import VofGridState


def validate_empty_vof_state(vof: VofGridState) -> None:
    """Validate the canonical all-GAS state created by construction or clear."""

    mass = np.asarray(vof.mass.numpy())
    phi = np.asarray(vof.phi.numpy())
    cell_type = np.asarray(vof.cell_type.numpy())
    if not np.all(np.isfinite(mass)) or not np.all(mass == 0.0):
        raise ValueError("empty VOF mass must be finite and exactly zero")
    if not np.all(np.isfinite(phi)) or not np.all(phi == 0.0):
        raise ValueError("empty VOF phi must be finite and exactly zero")
    if not np.all(cell_type == int(VofCellType.GAS)):
        raise ValueError("empty VOF cell_type must be GAS at every cell")


def validate_initialized_vof_state(
    state: LbmStateBase,
    *,
    periodic: tuple[bool, bool, bool],
    atol: float = 1.0e-6,
    rtol: float = 1.0e-5,
) -> None:
    """Validate density, mass, fill, type, and initial topology invariants."""

    if state.vof is None:
        raise ValueError("initialized VOF validation requires state.vof storage")
    validate_no_solid_cells(state.solid_phi)
    validate_initialized_density(state.density)

    density = np.asarray(state.density.numpy())
    mass = np.asarray(state.vof.mass.numpy())
    phi = np.asarray(state.vof.phi.numpy())
    cell_type = np.asarray(state.vof.cell_type.numpy())
    expected_shape = tuple(int(value) for value in state.res)
    for name, values in (
        ("density", density),
        ("mass", mass),
        ("phi", phi),
        ("cell_type", cell_type),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} shape {values.shape} does not match grid {expected_shape}"
            )

    if not np.all(np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError("initialized VOF mass must be finite and non-negative")
    if not np.all(np.isfinite(phi)) or np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("initialized VOF phi must be finite and lie in [0, 1]")

    legal_types = np.array(
        [
            int(VofCellType.GAS),
            int(VofCellType.INTERFACE),
            int(VofCellType.LIQUID),
        ],
        dtype=np.uint8,
    )
    if not np.all(np.isin(cell_type, legal_types)):
        raise ValueError("initialized VOF cell_type contains an unknown value")

    gas = cell_type == int(VofCellType.GAS)
    interface = cell_type == int(VofCellType.INTERFACE)
    liquid = cell_type == int(VofCellType.LIQUID)
    if np.any(mass[gas] != 0.0) or np.any(phi[gas] != 0.0):
        raise ValueError("GAS cells must have mass=0 and phi=0")
    if np.any(phi[interface] <= 0.0) or np.any(phi[interface] >= 1.0):
        raise ValueError("INTERFACE cells must satisfy 0 < phi < 1")
    if np.any(phi[liquid] != 1.0):
        raise ValueError("LIQUID cells must have phi=1")
    if not np.allclose(mass, density * phi, atol=atol, rtol=rtol):
        raise ValueError("initialized VOF must satisfy mass ~= density * phi")

    validate_initial_topology(cell_type, periodic=periodic)


def validate_vof_buffer_pair(
    state_in: LbmStateBase,
    state_out: LbmStateBase,
) -> None:
    """Require equal authoritative values backed by independent arrays."""

    if state_in.vof is None or state_out.vof is None:
        raise ValueError("both state buffers must contain authoritative VOF storage")
    for name in ("mass", "phi", "cell_type"):
        source = getattr(state_in.vof, name)
        target = getattr(state_out.vof, name)
        if source is target or int(source.ptr) == int(target.ptr):
            raise ValueError(f"VOF buffer {name} shares storage")
        if not np.array_equal(source.numpy(), target.numpy()):
            raise ValueError(f"VOF buffer {name} values differ")


__all__ = [
    "validate_empty_vof_state",
    "validate_initialized_vof_state",
    "validate_vof_buffer_pair",
]
