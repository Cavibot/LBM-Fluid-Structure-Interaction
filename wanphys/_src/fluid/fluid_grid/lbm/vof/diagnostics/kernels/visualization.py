"""Read-only interface visualization kernels."""

from __future__ import annotations

import warp as wp


@wp.kernel
def compact_interface_visuals_kernel(
    cell_type: wp.array3d(dtype=wp.uint8),
    normal: wp.array3d(dtype=wp.vec3),
    solid_phi: wp.array3d(dtype=float),
    count: wp.array(dtype=wp.int32),
    points: wp.array(dtype=wp.vec3),
    normal_ends: wp.array(dtype=wp.vec3),
    origin_x: float,
    origin_y: float,
    origin_z: float,
    cell_size: float,
    normal_length: float,
) -> None:
    i, j, k = wp.tid()
    if cell_type[i, j, k] != wp.uint8(1) or solid_phi[i, j, k] < 0.0:
        return
    output_index = wp.atomic_add(count, 0, 1)
    point = wp.vec3(
        origin_x + (float(i) + 0.5) * cell_size,
        origin_y + (float(j) + 0.5) * cell_size,
        origin_z + (float(k) + 0.5) * cell_size,
    )
    points[output_index] = point
    normal_ends[output_index] = point + normal[i, j, k] * normal_length


__all__ = ["compact_interface_visuals_kernel"]
