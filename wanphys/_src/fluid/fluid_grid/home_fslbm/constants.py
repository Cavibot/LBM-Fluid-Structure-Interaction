# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q27 / D3Q7 lattice constants and cell flag enumerations for HOME-FSLBM.

All constants are derived from the reference implementation:
- ``mrConstantParamsGpu3D.h`` — lattice velocities, weights
- ``mlLbmCommon.h`` — cell flag bitfield definitions
- ``mrLbmSolverGpu3D.cu:1822-1840`` — CMR-MRT relaxation rates
"""

from __future__ import annotations

import warp as wp

# ============================================================================
# D3Q27 discrete velocity set
# ============================================================================
# Direction ordering (0-based, 27 directions):
#   0:   ( 0,  0,  0)  rest particle
#   1-2: (±1,  0,  0)  face x-axis
#   3-4: ( 0, ±1,  0)  face y-axis
#   5-6: ( 0,  0, ±1)  face z-axis
#   7-10: (±1, ±1,  0) edge xy-plane
#  11-14: (±1,  0, ±1) edge xz-plane
#  15-18: ( 0, ±1, ±1) edge yz-plane
#  19-26: (±1, ±1, ±1) corners
#
# Ref: ``mrConstantParamsGpu3D.h`` constants ``ex3d_gpu``, ``ey3d_gpu``, ``ez3d_gpu``

NUM_DIRS: int = 27

# Velocity components (Python lists — inside Warp kernels these are hard-coded
# through the explicit per-direction formula in `reconstruct_distribution`).
CX: list[int] = [
    0,   #  0
    1, -1,  0,  0,  0,  0,                               #  1-6   faces
    1, -1,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0,        #  7-18  edges
    1, -1,  1, -1,  1, -1, -1,  1,                         # 19-26  corners
]
CY: list[int] = [
    0,   #  0
    0,  0,  1, -1,  0,  0,                                 #  1-6
    1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  1, -1,          #  7-18
    1, -1,  1, -1, -1,  1,  1, -1,                          # 19-26
]
CZ: list[int] = [
    0,   #  0
    0,  0,  0,  0,  1, -1,                                  #  1-6
    0,  0,  1, -1,  1, -1,  0,  0, -1,  1, -1,  1,           #  7-18
    1, -1, -1,  1,  1, -1,  1, -1,                           # 19-26
]

# ============================================================================
# Lattice weights (D3Q27)
# ============================================================================
# Ref: ``mrConstantParamsGpu3D.h`` — ``def_w0``, ``def_ws``, ``def_we``, ``def_wc``
# Equivalent rational forms: W0=8/27, W1=2/27, W2=1/54, W3=1/216

W0: float = 1.0 / 3.375    # rest particle      (= 8/27)
WS: float = 1.0 / 13.5     # face neighbours     (= 2/27, 6 directions)
WE: float = 1.0 / 54.0     # edge neighbours     (= 1/54, 12 directions)
WC: float = 1.0 / 216.0    # corner neighbours   (= 1/216, 8 directions)

# Per-direction weights list (host-side reference)
W: list[float] = [
    W0,                                         #  0: ( 0, 0, 0)
    WS, WS,   WS, WS,   WS, WS,                 #  1-6: faces
    WE, WE,   WE, WE,   WE, WE,   WE, WE,       #  7-18: edges
    WE, WE,   WE, WE,
    WC, WC,   WC, WC,   WC, WC,   WC, WC,       # 19-26: corners
]

# ============================================================================
# Opposite direction indices
# ============================================================================
# ``OPPOSITE[i]`` gives the index of ``-c_i``.
# Ref: ``mrConstantParamsGpu3D.h`` — ``index3dInv_gpu``

OPPOSITE: list[int] = [
    0,   #  0: ( 0, 0, 0) ↔ ( 0, 0, 0)
    2,   #  1: ( 1, 0, 0) ↔ (-1, 0, 0)
    1,   #  2: (-1, 0, 0) ↔ ( 1, 0, 0)
    4,   #  3: ( 0, 1, 0) ↔ ( 0,-1, 0)
    3,   #  4: ( 0,-1, 0) ↔ ( 0, 1, 0)
    6,   #  5: ( 0, 0, 1) ↔ ( 0, 0,-1)
    5,   #  6: ( 0, 0,-1) ↔ ( 0, 0, 1)
    8,   #  7: ( 1, 1, 0) ↔ (-1,-1, 0)
    7,   #  8: (-1, 1, 0) ↔ ( 1,-1, 0)
    10,  #  9: ( 1,-1, 0) ↔ (-1, 1, 0)
    9,   # 10: (-1,-1, 0) ↔ ( 1, 1, 0)
    12,  # 11: ( 1, 0, 1) ↔ (-1, 0,-1)
    11,  # 12: (-1, 0, 1) ↔ ( 1, 0,-1)
    14,  # 13: ( 1, 0,-1) ↔ (-1, 0, 1)
    13,  # 14: (-1, 0,-1) ↔ ( 1, 0, 1)
    16,  # 15: ( 0, 1, 1) ↔ ( 0,-1,-1)
    15,  # 16: ( 0,-1, 1) ↔ ( 0, 1,-1)
    18,  # 17: ( 0, 1,-1) ↔ ( 0,-1, 1)
    17,  # 18: ( 0,-1,-1) ↔ ( 0, 1, 1)
    20,  # 19: ( 1, 1, 1) ↔ (-1,-1,-1)
    19,  # 20: (-1, 1, 1) ↔ ( 1,-1,-1)
    22,  # 21: ( 1,-1, 1) ↔ (-1, 1,-1)
    21,  # 22: (-1,-1, 1) ↔ ( 1, 1,-1)
    24,  # 23: ( 1, 1,-1) ↔ (-1,-1, 1)
    23,  # 24: (-1, 1,-1) ↔ ( 1,-1, 1)
    26,  # 25: (-1,-1,-1) ↔ ( 1, 1, 1)  — wait, let me recheck
    25,  # 26: ( 1,-1,-1) ↔ (-1, 1, 1)
]
# Re-verifying index3dInv_gpu from reference: [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17,20,19,22,21,24,23,26,25]
# So OPPOSITE[i] = index3dInv_gpu[i].

# ============================================================================
# Lattice speed of sound
# ============================================================================
CS2: float = 1.0 / 3.0       # c_s^2
CS: float = 0.57735027       # c_s = 1/sqrt(3)
INV_CS2: float = 3.0         # 1 / c_s^2

# ============================================================================
# HOME-LBM: Number of stored moments per node
# ============================================================================
# HOME-LBM stores first 3 velocity moments (10 scalars total):
#   ρ, ρu_x, ρu_y, ρu_z, ρΠ_xx, ρΠ_xy, ρΠ_xz, ρΠ_yy, ρΠ_yz, ρΠ_zz
# where Π_αβ are the non-equilibrium stress components (minus c_s^2 for diagonal).
NUM_MOMENTS: int = 10

# Moment index aliases for readability
M_RHO: int = 0
M_UX: int = 1
M_UY: int = 2
M_UZ: int = 3
M_SXX: int = 4     # Π_xx - c_s^2  (stored traceless part)
M_SXY: int = 5     # Π_xy
M_SXZ: int = 6     # Π_xz
M_SYY: int = 7     # Π_yy - c_s^2
M_SYZ: int = 8     # Π_yz
M_SZZ: int = 9     # Π_zz - c_s^2

# ============================================================================
# Cell flag bitfield (MLLATTICENODE_SURFACE_FLAG)
# ============================================================================
# Ref: ``mlLbmCommon.h``


class CellFlag:
    """Bitfield constants for per-cell type classification.

    The flag byte uses the following bit layout:
        bit 0 (0x01): TYPE_S — solid boundary
        bit 1 (0x02): TYPE_E — equilibrium boundary
        bit 2 (0x04): TYPE_T — temperature boundary
        bit 3 (0x08): TYPE_F — fluid (filled)
        bit 4 (0x10): TYPE_I — interface (partially filled)
        bit 5 (0x20): TYPE_G — gas (empty)
        bit 6 (0x40): TYPE_X — reserved
        bit 7 (0x80): TYPE_Y — reserved
    """

    # ---- primitive flags --------------------------------------------------
    TYPE_S: int = 0x01   # solid (stationary or moving)
    TYPE_E: int = 0x02   # equilibrium boundary (inflow/outflow)
    TYPE_T: int = 0x04   # temperature boundary
    TYPE_F: int = 0x08   # fluid (filled cell)
    TYPE_I: int = 0x10   # interface (partially filled)
    TYPE_G: int = 0x20   # gas (empty)
    TYPE_X: int = 0x40   # reserved type X
    TYPE_Y: int = 0x80   # reserved type Y

    # ---- composite / transition flags ------------------------------------
    TYPE_MS: int = 0x03  # cell next to moving solid (S | E)
    TYPE_BO: int = 0x03  # boundary mask (S | E)
    TYPE_IF: int = 0x18  # interface → fluid transition (I | F)
    TYPE_IG: int = 0x30  # interface → gas transition (I | G)
    TYPE_GI: int = 0x38  # gas → interface transition (G | I | F)
    TYPE_SU: int = 0x38  # surface mask (G | I | F)

    # The SU (surface) mask covers all non-solid fluid types.
    # In the reference code this is used as:
    #   flagsn_su = flag & TYPE_SU_MASK  → one of {F, I, G}
    #   flagsn_bo = flag & TYPE_BO_MASK  → one of {S, E, MS}

    @staticmethod
    def is_solid(flag: int) -> int:
        """Return 1 if the cell is solid (flag & TYPE_BO == TYPE_S)."""
        # In reference: ``flagsn_bo == TYPE_S``
        return 1 if (flag & 0x03) == CellFlag.TYPE_S else 0

    @staticmethod
    def is_gas(flag: int) -> int:
        """Return 1 if the cell is pure gas (flag & TYPE_SU == TYPE_G)."""
        return 1 if (flag & CellFlag.TYPE_SU) == CellFlag.TYPE_G else 0

    @staticmethod
    def is_fluid(flag: int) -> int:
        """Return 1 if the cell is pure fluid."""
        return 1 if (flag & CellFlag.TYPE_SU) == CellFlag.TYPE_F else 0

    @staticmethod
    def is_interface(flag: int) -> int:
        """Return 1 if the cell is interface."""
        return 1 if (flag & CellFlag.TYPE_SU) == CellFlag.TYPE_I else 0


# Convenience aliases (module-level)
TYPE_S: int = CellFlag.TYPE_S
TYPE_E: int = CellFlag.TYPE_E
TYPE_G: int = CellFlag.TYPE_G
TYPE_F: int = CellFlag.TYPE_F
TYPE_I: int = CellFlag.TYPE_I
TYPE_IF: int = CellFlag.TYPE_IF
TYPE_IG: int = CellFlag.TYPE_IG
TYPE_GI: int = CellFlag.TYPE_GI
TYPE_BO_MASK: int = CellFlag.TYPE_BO
TYPE_SU_MASK: int = CellFlag.TYPE_SU

# ============================================================================
# D3Q7 dissolved gas model (CMR-MRT)
# ============================================================================
# Ref: ``mrConstantParamsGpu3D.h`` — ``d3q7_w``
#      ``mrLbmSolverGpu3D.cu:1822-1840`` — CMR-MRT relaxation rates

NUM_DIRS_GAS: int = 7

# D3Q7 discrete velocities (first 7 directions of D3Q27):
#   0: (0,0,0), 1:(1,0,0), 2:(-1,0,0), 3:(0,1,0), 4:(0,-1,0), 5:(0,0,1), 6:(0,0,-1)

# D3Q7 weights
W_GAS: list[float] = [1.0 / 4.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0]

# CMR-MRT relaxation rate vector (7 independent rates)
# s[0] = 1.0             → density moment → instantaneous equilibrium (mass conservation)
# s[1-3] = 1/0.9 ≈ 1.111 → momentum moments → τ_m = (1/s - 0.5) = 0.4
# s[4-6] = 1.5            → higher-order moments → fast relaxation, suppress numerical oscillations
#
# Ref: ``mrLbmSolverGpu3D.cu:1837-1840``
GAS_CMR_S: list[float] = [1.0, 1.0 / 0.9, 1.0 / 0.9, 1.0 / 0.9, 1.5, 1.5, 1.5]

# ============================================================================
# Physical constants
# ============================================================================
SURFACE_TENSION: float = 6.0 * 4e-3    # ``def_6_sigma`` — surface tension ×6 for D3Q27
HENRY_CONSTANT: float = 1e-3           # ``K_h`` — Henry's law constant
DISJOINT_FACTOR: float = 0.032         # disjoining pressure strength multiplier

# ============================================================================
# Boundary condition types (compatible with LBM convention)
# ============================================================================
BC_BOUNCE_BACK: int = 0
BC_VELOCITY_INLET: int = 1
BC_OUTFLOW: int = 2
BC_PERIODIC: int = 3

# Boundary face indices
FACE_XMIN: int = 0
FACE_XMAX: int = 1
FACE_YMIN: int = 2
FACE_YMAX: int = 3
FACE_ZMIN: int = 4
FACE_ZMAX: int = 5
