"""VOF authoritative state initialization kernels."""

from __future__ import annotations

import warp as wp


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


__all__ = ["initialize_vof_mass_kernel"]
