"""Compact device-side diagnostics kernels."""

from __future__ import annotations

import warp as wp

from ....solver.kernels.common import (
    direction_x,
    direction_y,
    direction_z,
    opposite_direction,
)

@wp.func
def _wrap_once(value: int, size: int) -> int:
    mapped = value
    if mapped < 0:
        mapped += size
    elif mapped >= size:
        mapped -= size
    return mapped


@wp.kernel
def initialize_vof_device_metrics_kernel(
    metrics: wp.array(dtype=wp.float64),
    epoch: int,
    geometry_epoch: int,
    cell_count: int,
) -> None:
    """Initialize the single compact diagnostics structure on device."""

    if wp.tid() == 0:
        for index in range(33):
            metrics[index] = wp.float64(0.0)
        metrics[1] = wp.float64(1.0e30)
        metrics[2] = wp.float64(-1.0e30)
        metrics[14] = wp.float64(epoch)
        metrics[15] = wp.float64(geometry_epoch)
        metrics[18] = wp.float64(cell_count)
        metrics[19] = wp.float64(cell_count)


@wp.kernel
def reduce_vof_device_diagnostics_kernel(
    mass: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    pending_excess: wp.array3d(dtype=float),
    pending_receiver_count: wp.array3d(dtype=wp.uint8),
    cell_type: wp.array3d(dtype=wp.uint8),
    density: wp.array3d(dtype=float),
    velocity_x: wp.array3d(dtype=float),
    velocity_y: wp.array3d(dtype=float),
    velocity_z: wp.array3d(dtype=float),
    curvature: wp.array3d(dtype=float),
    normal: wp.array3d(dtype=wp.vec3),
    plic_offset: wp.array3d(dtype=float),
    metrics: wp.array(dtype=wp.float64),
    atmosphere_pressure: float,
    surface_tension: float,
    mass_tolerance: float,
    phi_tolerance: float,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Reduce material invariants without copying full fields to the host."""

    i, j, k = wp.tid()
    local_mass = mass[i, j, k]
    local_phi = phi[i, j, k]
    local_pending = pending_excess[i, j, k]
    local_count = pending_receiver_count[i, j, k]
    local_type = cell_type[i, j, k]
    local_density = density[i, j, k]
    ux = velocity_x[i, j, k]
    uy = velocity_y[i, j, k]
    uz = velocity_z[i, j, k]
    local_curvature = curvature[i, j, k]
    local_normal = normal[i, j, k]
    local_offset = plic_offset[i, j, k]

    wp.atomic_add(
        metrics,
        0,
        wp.float64(local_mass) + wp.float64(local_pending),
    )
    wp.atomic_min(metrics, 1, wp.float64(local_phi))
    wp.atomic_max(metrics, 2, wp.float64(local_phi))
    speed_squared = ux * ux + uy * uy + uz * uz
    wp.atomic_max(metrics, 3, wp.float64(speed_squared))
    wp.atomic_max(metrics, 4, wp.float64(wp.abs(local_pending)))
    wp.atomic_add(metrics, 13, wp.float64(local_pending))

    invalid_finite = int(0)
    if not wp.isfinite(local_mass):
        invalid_finite += 1
    if not wp.isfinite(local_phi):
        invalid_finite += 1
    if not wp.isfinite(local_pending):
        invalid_finite += 1
    if not wp.isfinite(local_density):
        invalid_finite += 1
    if not wp.isfinite(ux):
        invalid_finite += 1
    if not wp.isfinite(uy):
        invalid_finite += 1
    if not wp.isfinite(uz):
        invalid_finite += 1
    if not wp.isfinite(local_curvature):
        invalid_finite += 1
    if not wp.isfinite(local_normal[0]):
        invalid_finite += 1
    if not wp.isfinite(local_normal[1]):
        invalid_finite += 1
    if not wp.isfinite(local_normal[2]):
        invalid_finite += 1
    if not wp.isfinite(local_offset):
        invalid_finite += 1
    if invalid_finite > 0:
        wp.atomic_add(metrics, 6, wp.float64(invalid_finite))

    legal_type = (
        local_type == wp.uint8(0)
        or local_type == wp.uint8(1)
        or local_type == wp.uint8(2)
    )
    if not legal_type:
        wp.atomic_add(metrics, 10, wp.float64(1.0))
    if local_type == wp.uint8(1):
        wp.atomic_add(metrics, 8, wp.float64(1.0))
    if local_type != wp.uint8(0) and (
        not wp.isfinite(local_density) or local_density <= 0.0
    ):
        wp.atomic_add(metrics, 7, wp.float64(1.0))
    if (
        local_phi < 0.0 - phi_tolerance
        or local_phi > 1.0 + phi_tolerance
    ):
        wp.atomic_add(metrics, 11, wp.float64(1.0))

    if local_type == wp.uint8(1):
        rho_g = 3.0 * (
            atmosphere_pressure
            - 2.0 * surface_tension * local_curvature
        )
        if not wp.isfinite(rho_g) or rho_g <= 0.0:
            wp.atomic_add(metrics, 12, wp.float64(1.0))

    actual_receiver_count = int(0)
    for q in range(1, 19):
        ni = i - direction_x(q)
        nj = j - direction_y(q)
        nk = k - direction_z(q)
        outside = bool(False)
        if ni < 0 or ni >= nx:
            if periodic_x != 0:
                ni = _wrap_once(ni, nx)
            else:
                outside = True
        if nj < 0 or nj >= ny:
            if periodic_y != 0:
                nj = _wrap_once(nj, ny)
            else:
                outside = True
        if nk < 0 or nk >= nz:
            if periodic_z != 0:
                nk = _wrap_once(nk, nz)
            else:
                outside = True
        if not outside:
            neighbor_type = cell_type[ni, nj, nk]
            if neighbor_type == wp.uint8(1):
                actual_receiver_count += 1
            if q < opposite_direction(q):
                invalid_pair = (
                    local_type == wp.uint8(0)
                    and neighbor_type == wp.uint8(2)
                ) or (
                    local_type == wp.uint8(2)
                    and neighbor_type == wp.uint8(0)
                )
                if invalid_pair:
                    wp.atomic_add(metrics, 5, wp.float64(1.0))

    material_pending = wp.abs(local_pending) > mass_tolerance
    linear_index = i * ny * nz + j * nz + k
    if material_pending and int(local_count) == 0 and actual_receiver_count == 0:
        wp.atomic_add(metrics, 16, wp.float64(1.0))
        wp.atomic_min(metrics, 18, wp.float64(linear_index))
    elif material_pending and int(local_count) != actual_receiver_count:
        wp.atomic_add(metrics, 9, wp.float64(1.0))
        wp.atomic_add(metrics, 17, wp.float64(1.0))
        wp.atomic_min(metrics, 19, wp.float64(linear_index))


@wp.kernel
def describe_first_invalid_pending_kernel(
    pending_excess: wp.array3d(dtype=float),
    pending_receiver_count: wp.array3d(dtype=wp.uint8),
    previous_cell_type: wp.array3d(dtype=wp.uint8),
    proposed_cell_type: wp.array3d(dtype=wp.uint8),
    final_cell_type: wp.array3d(dtype=wp.uint8),
    metrics: wp.array(dtype=wp.float64),
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    nx: int,
    ny: int,
    nz: int,
) -> None:
    """Describe the lowest-index invalid pending route after reduction."""

    if wp.tid() != 0:
        return
    linear_index = int(metrics[19])
    if metrics[16] > wp.float64(0.0):
        linear_index = int(metrics[18])
    if linear_index >= nx * ny * nz:
        return

    i = linear_index // (ny * nz)
    remainder = linear_index - i * ny * nz
    j = remainder // nz
    k = remainder - j * nz
    actual_receiver_count = int(0)
    valid_mask = int(0)
    previous_interface_mask = int(0)
    previous_liquid_mask = int(0)
    proposed_interface_mask = int(0)
    proposed_liquid_mask = int(0)
    final_interface_mask = int(0)
    final_liquid_mask = int(0)
    for q in range(1, 19):
        ni = i - direction_x(q)
        nj = j - direction_y(q)
        nk = k - direction_z(q)
        outside = bool(False)
        if ni < 0 or ni >= nx:
            if periodic_x != 0:
                ni = _wrap_once(ni, nx)
            else:
                outside = True
        if nj < 0 or nj >= ny:
            if periodic_y != 0:
                nj = _wrap_once(nj, ny)
            else:
                outside = True
        if nk < 0 or nk >= nz:
            if periodic_z != 0:
                nk = _wrap_once(nk, nz)
            else:
                outside = True
        if not outside:
            bit = int(1) << (q - 1)
            valid_mask = valid_mask | bit
            previous_neighbor = previous_cell_type[ni, nj, nk]
            proposed_neighbor = proposed_cell_type[ni, nj, nk]
            final_neighbor = final_cell_type[ni, nj, nk]
            if previous_neighbor == wp.uint8(1):
                previous_interface_mask = previous_interface_mask | bit
            elif previous_neighbor == wp.uint8(2):
                previous_liquid_mask = previous_liquid_mask | bit
            if proposed_neighbor == wp.uint8(1):
                proposed_interface_mask = proposed_interface_mask | bit
            elif proposed_neighbor == wp.uint8(2):
                proposed_liquid_mask = proposed_liquid_mask | bit
            if final_neighbor == wp.uint8(1):
                actual_receiver_count += 1
                final_interface_mask = final_interface_mask | bit
            elif final_neighbor == wp.uint8(2):
                final_liquid_mask = final_liquid_mask | bit

    metrics[20] = wp.float64(pending_excess[i, j, k])
    metrics[21] = wp.float64(pending_receiver_count[i, j, k])
    metrics[22] = wp.float64(actual_receiver_count)
    metrics[23] = wp.float64(final_cell_type[i, j, k])
    metrics[24] = wp.float64(previous_cell_type[i, j, k])
    metrics[25] = wp.float64(proposed_cell_type[i, j, k])
    metrics[26] = wp.float64(valid_mask)
    metrics[27] = wp.float64(previous_interface_mask)
    metrics[28] = wp.float64(previous_liquid_mask)
    metrics[29] = wp.float64(proposed_interface_mask)
    metrics[30] = wp.float64(proposed_liquid_mask)
    metrics[31] = wp.float64(final_interface_mask)
    metrics[32] = wp.float64(final_liquid_mask)




__all__ = [
    "describe_first_invalid_pending_kernel",
    "initialize_vof_device_metrics_kernel",
    "reduce_vof_device_diagnostics_kernel",
]
