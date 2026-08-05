"""Shan-Chen debug observation kernels."""

from __future__ import annotations

import warp as wp


@wp.kernel
def density_to_debug_fill_fraction_kernel(
    density: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    rho_gas: float,
    inverse_density_span: float,
) -> None:
    i, j, k = wp.tid()
    value = (density[i, j, k] - rho_gas) * inverse_density_span
    phi[i, j, k] = wp.clamp(value, 0.0, 1.0)


__all__ = ["density_to_debug_fill_fraction_kernel"]
