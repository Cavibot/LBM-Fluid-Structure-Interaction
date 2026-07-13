# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 lattice constants for the Lattice Boltzmann Method.

Geometry and weights are defined in :mod:`lbm.core.lattice` and re-exported
here for backward compatibility.
"""

from __future__ import annotations

from .core.lattice import (
    CS2,
    CX,
    CY,
    CZ,
    D3Q19,
    NUM_DIRS,
    OPPOSITE,
    W,
    W_EDGE,
    W_FACE,
    W_REST,
)

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
