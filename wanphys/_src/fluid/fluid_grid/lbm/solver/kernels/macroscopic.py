"""Macroscopic observable publication kernels."""

from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------------------


@wp.kernel
def moments_to_mac_u_kernel(
    ux: wp.array3d(dtype=float),
    vel_u: wp.array3d(dtype=float),
    nx: int,
) -> None:
    """Interpolate cell-centred *ux* -> MAC x-face velocity ``vel_u``.

    Launched over ``(nx+1, ny, nz)``.
    """
    i, j, k = wp.tid()
    if i == 0:
        vel_u[i, j, k] = ux[0, j, k]
    elif i == nx:
        vel_u[i, j, k] = ux[nx - 1, j, k]
    else:
        vel_u[i, j, k] = 0.5 * (ux[i - 1, j, k] + ux[i, j, k])


@wp.kernel
def moments_to_mac_v_kernel(
    uy: wp.array3d(dtype=float),
    vel_v: wp.array3d(dtype=float),
    ny: int,
) -> None:
    """Interpolate cell-centred *uy* -> MAC y-face velocity ``vel_v``.

    Launched over ``(nx, ny+1, nz)``.
    """
    i, j, k = wp.tid()
    if j == 0:
        vel_v[i, j, k] = uy[i, 0, k]
    elif j == ny:
        vel_v[i, j, k] = uy[i, ny - 1, k]
    else:
        vel_v[i, j, k] = 0.5 * (uy[i, j - 1, k] + uy[i, j, k])


@wp.kernel
def moments_to_mac_w_kernel(
    uz: wp.array3d(dtype=float),
    vel_w: wp.array3d(dtype=float),
    nz: int,
) -> None:
    """Interpolate cell-centred *uz* -> MAC z-face velocity ``vel_w``.

    Launched over ``(nx, ny, nz+1)``.
    """
    i, j, k = wp.tid()
    if k == 0:
        vel_w[i, j, k] = uz[i, j, 0]
    elif k == nz:
        vel_w[i, j, k] = uz[i, j, nz - 1]
    else:
        vel_w[i, j, k] = 0.5 * (uz[i, j, k - 1] + uz[i, j, k])
