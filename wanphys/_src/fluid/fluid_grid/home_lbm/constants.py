# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q27 lattice and persistent HOME moment layout."""

from __future__ import annotations

import numpy as np

CS2 = 1.0 / 3.0
CS4 = CS2 * CS2
CS6 = CS4 * CS2

# Rest, axis, face-diagonal, then body-diagonal directions.  This ordering is
# fixed because boundary-link IDs and their opposite mapping become persistent
# data once moving solids and VOF are added.
D3Q27_DIRECTIONS = np.asarray(
    [
        (0, 0, 0),
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
        (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
        (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
    ],
    dtype=np.float64,
)

D3Q27_WEIGHTS = np.asarray(
    [8.0 / 27.0]
    + [2.0 / 27.0] * 6
    + [1.0 / 54.0] * 12
    + [1.0 / 216.0] * 8,
    dtype=np.float64,
)

D3Q27_OPPOSITE = np.asarray(
    [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15, 26, 25, 24, 23, 22, 21, 20, 19],
    dtype=np.int32,
)

# Persistent component-major SoA layout: moment[component * cell_count + cell].
M_RHO = 0
M_JX = 1
M_JY = 2
M_JZ = 3
M_RHO_SXX = 4
M_RHO_SYY = 5
M_RHO_SZZ = 6
M_RHO_SXY = 7
M_RHO_SXZ = 8
M_RHO_SYZ = 9
MOMENT_COUNT = 10

MOMENT_NAMES = (
    "rho",
    "rho_u_x",
    "rho_u_y",
    "rho_u_z",
    "rho_S_xx",
    "rho_S_yy",
    "rho_S_zz",
    "rho_S_xy",
    "rho_S_xz",
    "rho_S_yz",
)
