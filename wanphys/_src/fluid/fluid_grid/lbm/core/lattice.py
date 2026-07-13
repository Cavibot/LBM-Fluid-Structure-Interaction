# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Lattice velocity sets for distribution-based LBM backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import warp as wp

# ---------------------------------------------------------------------------
# D3Q19 discrete velocity set (lattice units)
# ---------------------------------------------------------------------------
# Direction ordering:
#   0:   rest       ( 0,  0,  0)   w = 1/3
#   1-2: x-axis     (±1,  0,  0)   w = 1/18
#   3-4: y-axis     ( 0, ±1,  0)   w = 1/18
#   5-6: z-axis     ( 0,  0, ±1)   w = 1/18
#   7-10: xy-plane  (±1, ±1,  0)   w = 1/36
#  11-14: xz-plane  (±1,  0, ±1)   w = 1/36
#  15-18: yz-plane  ( 0, ±1, ±1)   w = 1/36

_D3Q19_CX: tuple[int, ...] = (
    0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0,
)
_D3Q19_CY: tuple[int, ...] = (
    0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1,
)
_D3Q19_CZ: tuple[int, ...] = (
    0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1,
)

_W_REST: Final[float] = 1.0 / 3.0
_W_FACE: Final[float] = 1.0 / 18.0
_W_EDGE: Final[float] = 1.0 / 36.0

_D3Q19_WEIGHTS: tuple[float, ...] = (
    _W_REST,
    _W_FACE, _W_FACE,
    _W_FACE, _W_FACE,
    _W_FACE, _W_FACE,
    _W_EDGE, _W_EDGE, _W_EDGE, _W_EDGE,
    _W_EDGE, _W_EDGE, _W_EDGE, _W_EDGE,
    _W_EDGE, _W_EDGE, _W_EDGE, _W_EDGE,
)

_D3Q19_OPPOSITE: tuple[int, ...] = (
    0,
    2, 1,
    4, 3,
    6, 5,
    10, 9, 8, 7,
    14, 13, 12, 11,
    18, 17, 16, 15,
)


@dataclass(frozen=True)
class LatticeSpec:
    """Static discrete-velocity lattice configuration."""

    name: str
    num_dirs: int
    cx: tuple[int, ...]
    cy: tuple[int, ...]
    cz: tuple[int, ...]
    weights: tuple[float, ...]
    opposite: tuple[int, ...]
    cs2: float

    @property
    def distribution_count(self) -> int:
        """Number of distribution functions per grid node."""
        return self.num_dirs

    def flat_array_length(self, num_cells: int) -> int:
        """Length of the flattened distribution buffer for *num_cells* nodes."""
        return self.num_dirs * int(num_cells)


D3Q19: Final[LatticeSpec] = LatticeSpec(
    name="D3Q19",
    num_dirs=19,
    cx=_D3Q19_CX,
    cy=_D3Q19_CY,
    cz=_D3Q19_CZ,
    weights=_D3Q19_WEIGHTS,
    opposite=_D3Q19_OPPOSITE,
    cs2=1.0 / 3.0,
)

_LATTICE_REGISTRY: dict[str, LatticeSpec] = {
    D3Q19.name: D3Q19,
}


def get_lattice_spec(name: str) -> LatticeSpec:
    """Return a registered :class:`LatticeSpec` by name."""
    key = str(name).upper()
    try:
        return _LATTICE_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_LATTICE_REGISTRY))
        raise ValueError(
            f"Unknown LBM lattice {name!r}.  Supported: {known}"
        ) from exc


# Host-side aliases used by kernels and legacy constants module.
NUM_DIRS: Final[int] = D3Q19.num_dirs
CX: Final[list[int]] = list(D3Q19.cx)
CY: Final[list[int]] = list(D3Q19.cy)
CZ: Final[list[int]] = list(D3Q19.cz)
W: Final[list[float]] = list(D3Q19.weights)
OPPOSITE: Final[list[int]] = list(D3Q19.opposite)
W_REST: Final[float] = _W_REST
W_FACE: Final[float] = _W_FACE
W_EDGE: Final[float] = _W_EDGE

# Warp compile-time constant (lattice speed of sound squared).
CS2: wp.constant = wp.constant(D3Q19.cs2)
