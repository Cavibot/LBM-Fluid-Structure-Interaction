# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Host validation and one-shot initialization for authoritative VOF state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..constants import CX, CY, CZ, OPPOSITE
from .contracts import VofCellType

if TYPE_CHECKING:
    from .state import VofGridState


# One direction from each D3Q19 opposite pair.  The fixed order makes
# topology diagnostics deterministic and avoids counting ordinary links twice.
_UNIQUE_D3Q19_DIRECTIONS: tuple[int, ...] = tuple(
    direction
    for direction in range(1, len(OPPOSITE))
    if direction < OPPOSITE[direction]
)


def classify_initial_phi(phi0: np.ndarray) -> np.ndarray:
    """Classify a canonical float32 fill field using exact endpoint semantics."""

    if phi0.dtype != np.float32 or not phi0.flags.c_contiguous:
        raise TypeError("canonical phi0 must be a C-contiguous float32 array")
    cell_type = np.full(
        phi0.shape,
        int(VofCellType.INTERFACE),
        dtype=np.uint8,
    )
    cell_type[phi0 == np.float32(0.0)] = int(VofCellType.GAS)
    cell_type[phi0 == np.float32(1.0)] = int(VofCellType.LIQUID)
    return cell_type


def _resolve_neighbor(
    cell: tuple[int, int, int],
    direction: tuple[int, int, int],
    shape: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[int, int, int] | None:
    neighbor = [cell[axis] + direction[axis] for axis in range(3)]
    for axis in range(3):
        if 0 <= neighbor[axis] < shape[axis]:
            continue
        if not periodic[axis]:
            return None
        neighbor[axis] %= shape[axis]
    return neighbor[0], neighbor[1], neighbor[2]


def validate_initial_topology(
    cell_type: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool],
) -> None:
    """Reject unique D3Q19 links that connect LIQUID directly to GAS."""

    if cell_type.ndim != 3:
        raise ValueError(
            f"cell_type must be three-dimensional, got shape {cell_type.shape}"
        )
    if len(periodic) != 3:
        raise ValueError(f"periodic must contain three flags, got {periodic!r}")

    shape = tuple(int(value) for value in cell_type.shape)
    seen_links: set[tuple[int, int]] = set()
    invalid_count = 0
    first_invalid: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ] | None = None

    for cell in np.ndindex(shape):
        source_linear = int(np.ravel_multi_index(cell, shape))
        source_type = int(cell_type[cell])
        for direction_index in _UNIQUE_D3Q19_DIRECTIONS:
            direction = (
                CX[direction_index],
                CY[direction_index],
                CZ[direction_index],
            )
            neighbor = _resolve_neighbor(cell, direction, shape, periodic)
            if neighbor is None or neighbor == cell:
                continue
            neighbor_linear = int(np.ravel_multi_index(neighbor, shape))
            link_key = (
                min(source_linear, neighbor_linear),
                max(source_linear, neighbor_linear),
            )
            if link_key in seen_links:
                continue
            seen_links.add(link_key)

            pair = {source_type, int(cell_type[neighbor])}
            if pair != {int(VofCellType.GAS), int(VofCellType.LIQUID)}:
                continue
            invalid_count += 1
            if first_invalid is None:
                first_invalid = (cell, direction, neighbor)

    if first_invalid is not None:
        cell, direction, neighbor = first_invalid
        raise ValueError(
            "Initial VOF topology contains "
            f"{invalid_count} direct D3Q19 LIQUID-GAS link(s); "
            f"first cell={cell}, direction={direction}, neighbor={neighbor}"
        )


def prepare_initial_vof(
    phi0: np.ndarray,
    *,
    shape: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, canonicalize, classify, and topology-check user fill data."""

    phi = np.asarray(phi0)
    if not np.issubdtype(phi.dtype, np.floating):
        raise TypeError(f"phi0 must have a floating dtype, got {phi.dtype}")
    if phi.shape != shape:
        raise ValueError(f"phi0 shape {phi.shape} does not match grid shape {shape}")
    if not np.all(np.isfinite(phi)):
        raise ValueError("phi0 must contain only finite values")
    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("phi0 values must lie in the closed interval [0, 1]")

    canonical_phi = np.ascontiguousarray(phi, dtype=np.float32)
    cell_type = classify_initial_phi(canonical_phi)
    validate_initial_topology(cell_type, periodic=periodic)
    return canonical_phi, cell_type


def validate_no_solid_cells(solid_phi: wp.array3d) -> None:
    """Reject embedded solid cells in a P1 authoritative VOF domain."""

    values = np.asarray(solid_phi.numpy())
    if not np.all(np.isfinite(values)):
        raise ValueError("solid_phi must contain only finite values")
    if np.any(values < 0.0):
        raise NotImplementedError(
            "P1 authoritative VOF does not support embedded solid cells"
        )


def validate_initialized_density(density: wp.array3d) -> None:
    """Require an initialized positive LBM density at every grid cell."""

    values = np.asarray(density.numpy())
    if not np.all(np.isfinite(values)):
        raise ValueError("initialized LBM density must be finite at every cell")
    if np.any(values <= 0.0):
        raise ValueError("initialized LBM density must be positive at every cell")


@wp.kernel
def initialize_vof_mass_kernel(
    density: wp.array3d(dtype=float),
    phi_input: wp.array3d(dtype=float),
    cell_type_input: wp.array3d(dtype=wp.uint8),
    mass_out: wp.array3d(dtype=float),
    phi_out: wp.array3d(dtype=float),
    cell_type_out: wp.array3d(dtype=wp.uint8),
) -> None:
    """Build canonical mass, phi, and type from initialized LBM density."""

    i, j, k = wp.tid()
    rho = density[i, j, k]
    kind = cell_type_input[i, j, k]

    mass = 0.0
    phi = 0.0
    if kind == wp.uint8(2):
        mass = rho
        phi = 1.0
    elif kind == wp.uint8(1):
        mass = rho * phi_input[i, j, k]
        phi = mass / rho

    mass_out[i, j, k] = mass
    phi_out[i, j, k] = phi
    cell_type_out[i, j, k] = kind


def initialize_vof_fields(
    density: wp.array3d,
    target: VofGridState,
    phi0: np.ndarray,
    cell_type: np.ndarray,
) -> None:
    """Initialize one authoritative VOF state from prepared host data."""

    validate_initialized_density(density)
    if phi0.shape != target.shape or cell_type.shape != target.shape:
        raise ValueError(
            "prepared VOF arrays must match target shape "
            f"{target.shape}, got phi={phi0.shape}, type={cell_type.shape}"
        )
    if phi0.dtype != np.float32 or not phi0.flags.c_contiguous:
        raise TypeError("prepared phi0 must be a C-contiguous float32 array")
    if cell_type.dtype != np.uint8 or not cell_type.flags.c_contiguous:
        raise TypeError("prepared cell_type must be a C-contiguous uint8 array")

    phi_device = wp.array(phi0, dtype=float, device=target.device)
    cell_type_device = wp.array(cell_type, dtype=wp.uint8, device=target.device)
    wp.launch(
        initialize_vof_mass_kernel,
        dim=target.shape,
        inputs=[
            density,
            phi_device,
            cell_type_device,
            target.mass,
            target.phi,
            target.cell_type,
        ],
        device=target.device,
    )


__all__ = [
    "classify_initial_phi",
    "initialize_vof_fields",
    "initialize_vof_mass_kernel",
    "prepare_initial_vof",
    "validate_initial_topology",
    "validate_initialized_density",
    "validate_no_solid_cells",
]
