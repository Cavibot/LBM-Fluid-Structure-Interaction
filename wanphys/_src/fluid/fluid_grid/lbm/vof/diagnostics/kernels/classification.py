# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Reusable kernels for initializing VOF cell types from bounded fill data."""

import warp as wp


@wp.kernel
def classify_bounded_fill_fraction_kernel(
    phi: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),
    cell_type: wp.array3d(dtype=wp.uint8),
    gas_max: float,
    liquid_min: float,
) -> None:
    """Classify a bounded fill field without performing VOF transitions.

    This operation is suitable for initialization and observation.  It is not
    the conservative I/L/G reclassification used after future mass transport.
    """

    i, j, k = wp.tid()
    value = wp.clamp(phi[i, j, k], 0.0, 1.0)
    phi[i, j, k] = value

    result = wp.uint8(1)
    if solid_phi[i, j, k] < 0.0 or value <= gas_max:
        result = wp.uint8(0)
    elif value >= liquid_min:
        result = wp.uint8(2)
    cell_type[i, j, k] = result


__all__ = ["classify_bounded_fill_fraction_kernel"]
