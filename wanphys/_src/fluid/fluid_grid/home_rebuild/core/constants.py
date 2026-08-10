# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q27 quadrature and the persistent ten-moment SoA layout."""

from __future__ import annotations

import numpy as np

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
    [8.0 / 27.0] + [2.0 / 27.0] * 6 + [1.0 / 54.0] * 12 + [1.0 / 216.0] * 8,
    dtype=np.float64,
)

D3Q27_OPPOSITE = np.asarray(
    [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15, 26, 25, 24, 23, 22, 21, 20, 19],
    dtype=np.int32,
)

MOMENT_COUNT = 10
