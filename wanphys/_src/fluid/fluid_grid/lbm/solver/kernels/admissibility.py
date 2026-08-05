"""Population admissibility kernels."""

import warp as wp

from .common import direction_x, direction_y, direction_z, equilibrium_population


@wp.kernel
def enforce_population_admissibility_kernel(
    f: wp.array(dtype=float),
    population_floor: float,
    max_lattice_speed: float,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Move inadmissible populations minimally toward a local equilibrium."""

    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    rho = float(0.0)
    jx = float(0.0)
    jy = float(0.0)
    jz = float(0.0)
    needs_filter = int(0)
    for q in range(19):
        value = f[q * stride + idx]
        rho += value
        jx += float(direction_x(q)) * value
        jy += float(direction_y(q)) * value
        jz += float(direction_z(q)) * value
        if value < population_floor:
            needs_filter = 1

    if not needs_filter or rho <= 0.0:
        return

    inverse_rho = 1.0 / rho
    ux = jx * inverse_rho
    uy = jy * inverse_rho
    uz = jz * inverse_rho
    if max_lattice_speed > 0.0:
        speed_sq = ux * ux + uy * uy + uz * uz
        limit_sq = max_lattice_speed * max_lattice_speed
        if speed_sq > limit_sq:
            scale = wp.sqrt(limit_sq / speed_sq)
            ux *= scale
            uy *= scale
            uz *= scale

    alpha = float(1.0)
    for q in range(19):
        value = f[q * stride + idx]
        equilibrium = equilibrium_population(q, rho, ux, uy, uz)
        if value < population_floor and equilibrium > population_floor:
            candidate = (equilibrium - population_floor) / (equilibrium - value)
            alpha = wp.min(alpha, wp.max(candidate, 0.0))

    if alpha < 1.0:
        for q in range(19):
            value = f[q * stride + idx]
            equilibrium = equilibrium_population(q, rho, ux, uy, uz)
            f[q * stride + idx] = equilibrium + alpha * (value - equilibrium)

__all__ = ["enforce_population_admissibility_kernel"]
