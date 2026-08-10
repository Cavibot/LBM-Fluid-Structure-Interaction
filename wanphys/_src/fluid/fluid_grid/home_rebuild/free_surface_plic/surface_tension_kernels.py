# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for PLIC curvature-driven Laplace pressure."""

import warp as wp


@wp.kernel
def compute_laplace_gas_density_kernel(
    flags: wp.array3d(dtype=wp.int32),
    curvature: wp.array3d(dtype=float),
    curvature_valid: wp.array3d(dtype=wp.int32),
    curvature_required: wp.array3d(dtype=wp.int32),
    ambient_gas_density: float,
    six_surface_tension: float,
    gas_density: wp.array3d(dtype=float),
    invalid_density_count: wp.array(dtype=wp.int32),
    min_interface_density: wp.array(dtype=float),
    max_interface_density: wp.array(dtype=float),
) -> None:
    i, j, k = wp.tid()
    value = ambient_gas_density
    if (
        flags[i, j, k] == 1
        and curvature_required[i, j, k] != 0
        and curvature_valid[i, j, k] != 0
    ):
        value -= six_surface_tension * curvature[i, j, k]
    if not wp.isfinite(value) or value <= 0.0:
        wp.atomic_add(invalid_density_count, 0, 1)
        gas_density[i, j, k] = 0.0
        return
    gas_density[i, j, k] = value
    if flags[i, j, k] == 1:
        wp.atomic_min(min_interface_density, 0, value)
        wp.atomic_max(max_interface_density, 0, value)
