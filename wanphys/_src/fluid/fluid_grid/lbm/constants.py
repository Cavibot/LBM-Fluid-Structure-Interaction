# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 lattice constants for the Lattice Boltzmann Method.

The D3Q19 model uses 19 discrete velocities in 3D:
  - 1 rest particle
  - 6 face-centred neighbours (speed 1)
  - 12 edge-centred neighbours (speed sqrt(2))
"""

from __future__ import annotations

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

NUM_DIRS: int = 19

# Direction components (Python lists for host-side use;
# inside Warp kernels the directions are hard-coded for unrolling).
CX: list[int] = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
CY: list[int] = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1]
CZ: list[int] = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1]

# Lattice weights
W_REST: float = 1.0 / 3.0   # rest particle
W_FACE: float = 1.0 / 18.0  # face-centred (6 directions)
W_EDGE: float = 1.0 / 36.0  # edge-centred (12 directions)

# Weight by direction index (for host-side reference)
W: list[float] = [
    W_REST,                                    #  0: ( 0, 0, 0)
    W_FACE, W_FACE,                            #  1-2: x-axis
    W_FACE, W_FACE,                            #  3-4: y-axis
    W_FACE, W_FACE,                            #  5-6: z-axis
    W_EDGE, W_EDGE, W_EDGE, W_EDGE,            #  7-10: xy-plane
    W_EDGE, W_EDGE, W_EDGE, W_EDGE,            # 11-14: xz-plane
    W_EDGE, W_EDGE, W_EDGE, W_EDGE,            # 15-18: yz-plane
]

# Opposite (reversed) direction index for each d = 0..18.
# opposite[d] is the index of -c_d.
# Pairs: 1↔2, 3↔4, 5↔6, 7↔10, 8↔9, 11↔14, 12↔13, 15↔18, 16↔17.
OPPOSITE: list[int] = [
    0,   #  0: ( 0, 0, 0) ↔ ( 0, 0, 0)
    2,   #  1: ( 1, 0, 0) ↔ (-1, 0, 0)
    1,   #  2: (-1, 0, 0) ↔ ( 1, 0, 0)
    4,   #  3: ( 0, 1, 0) ↔ ( 0,-1, 0)
    3,   #  4: ( 0,-1, 0) ↔ ( 0, 1, 0)
    6,   #  5: ( 0, 0, 1) ↔ ( 0, 0,-1)
    5,   #  6: ( 0, 0,-1) ↔ ( 0, 0, 1)
    10,  #  7: ( 1, 1, 0) ↔ (-1,-1, 0)
    9,   #  8: (-1, 1, 0) ↔ ( 1,-1, 0)
    8,   #  9: ( 1,-1, 0) ↔ (-1, 1, 0)
    7,   # 10: (-1,-1, 0) ↔ ( 1, 1, 0)
    14,  # 11: ( 1, 0, 1) ↔ (-1, 0,-1)
    13,  # 12: (-1, 0, 1) ↔ ( 1, 0,-1)
    12,  # 13: ( 1, 0,-1) ↔ (-1, 0, 1)
    11,  # 14: (-1, 0,-1) ↔ ( 1, 0, 1)
    18,  # 15: ( 0, 1, 1) ↔ ( 0,-1,-1)
    17,  # 16: ( 0,-1, 1) ↔ ( 0, 1,-1)
    16,  # 17: ( 0, 1,-1) ↔ ( 0,-1, 1)
    15,  # 18: ( 0,-1,-1) ↔ ( 0, 1, 1)
]

# Lattice speed of sound squared (Warp compile-time constant)
CS2: wp.constant = wp.constant(1.0 / 3.0)

# ---------------------------------------------------------------------------
# Boundary condition types
# ---------------------------------------------------------------------------
BC_BOUNCE_BACK: int = 0     # default: halfway bounce-back (handled in collide-stream)
BC_VELOCITY_INLET: int = 1  # Zou-He velocity boundary
BC_OUTFLOW: int = 2         # convective outflow (zero-gradient copy)
BC_PERIODIC: int = 3        # periodic boundary (neighbour wraps to opposite face)

# ---------------------------------------------------------------------------
# Shan-Chen pseudopotential types
# ---------------------------------------------------------------------------
PSI_RHO: int = 0  # ψ(ρ) = ρ (simplest, good for moderate density ratios)
PSI_EXP: int = 1  # ψ(ρ) = 1 − exp(−ρ / ρ_ref) (Shan-Chen original, better stability)

# Boundary face indices
FACE_XMIN: int = 0
FACE_XMAX: int = 1
FACE_YMIN: int = 2
FACE_YMAX: int = 3
FACE_ZMIN: int = 4
FACE_ZMAX: int = 5
