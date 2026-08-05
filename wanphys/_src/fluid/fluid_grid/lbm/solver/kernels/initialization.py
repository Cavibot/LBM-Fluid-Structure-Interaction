"""Equilibrium initialization kernels."""

from __future__ import annotations

import warp as wp

from .common import equilibrium_f



@wp.kernel
def initialize_equilibrium_kernel(
    f: wp.array(dtype=float),
    rho0: float,
    u0x: float,
    u0y: float,
    u0z: float,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Set *f* to the Maxwell-Boltzmann equilibrium for uniform (rho0, u0)."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    f[0 * stride + idx] = equilibrium_f(1.0 / 3.0, rho0, u0x, u0y, u0z, 0, 0, 0)
    f[1 * stride + idx] = equilibrium_f(1.0 / 18.0, rho0, u0x, u0y, u0z, 1, 0, 0)
    f[2 * stride + idx] = equilibrium_f(1.0 / 18.0, rho0, u0x, u0y, u0z, -1, 0, 0)
    f[3 * stride + idx] = equilibrium_f(1.0 / 18.0, rho0, u0x, u0y, u0z, 0, 1, 0)
    f[4 * stride + idx] = equilibrium_f(1.0 / 18.0, rho0, u0x, u0y, u0z, 0, -1, 0)
    f[5 * stride + idx] = equilibrium_f(1.0 / 18.0, rho0, u0x, u0y, u0z, 0, 0, 1)
    f[6 * stride + idx] = equilibrium_f(1.0 / 18.0, rho0, u0x, u0y, u0z, 0, 0, -1)
    f[7 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, 1, 1, 0)
    f[8 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, -1, 1, 0)
    f[9 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, 1, -1, 0)
    f[10 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, -1, -1, 0)
    f[11 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, 1, 0, 1)
    f[12 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, -1, 0, 1)
    f[13 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, 1, 0, -1)
    f[14 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, -1, 0, -1)
    f[15 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, 0, 1, 1)
    f[16 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, 0, -1, 1)
    f[17 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, 0, 1, -1)
    f[18 * stride + idx] = equilibrium_f(1.0 / 36.0, rho0, u0x, u0y, u0z, 0, -1, -1)
