# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Native NxN and explicit broad phase collision detection for rigid shapes."""

from __future__ import annotations

import numpy as np
import warp as wp

from .broad_phase_common import (
    check_aabb_overlap,
    is_pair_excluded,
    precompute_world_map,
    test_world_and_group_pair,
    write_pair,
)


@wp.kernel
def _nxn_broadphase_precomputed_pairs(
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    nxn_shape_pair: wp.array(dtype=wp.vec2i, ndim=1),
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    candidate_pair_count: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):
    elementid = wp.tid()
    pair = nxn_shape_pair[elementid]
    shape1 = pair[0]
    shape2 = pair[1]

    gap1 = 0.0
    gap2 = 0.0
    if shape_gap.shape[0] > 0:
        gap1 = shape_gap[shape1]
        gap2 = shape_gap[shape2]

    if check_aabb_overlap(
        shape_bounding_box_lower[shape1],
        shape_bounding_box_upper[shape1],
        gap1,
        shape_bounding_box_lower[shape2],
        shape_bounding_box_upper[shape2],
        gap2,
    ):
        write_pair(pair, candidate_pair, candidate_pair_count, max_candidate_pair)


@wp.func
def _get_lower_triangular_indices(index: int, matrix_size: int) -> tuple[int, int]:
    total = (matrix_size * (matrix_size - 1)) >> 1
    if index >= total:
        return -1, -1

    low = int(0)
    high = matrix_size - 1
    while low < high:
        mid = (low + high) >> 1
        count = (mid * (2 * matrix_size - mid - 1)) >> 1
        if count <= index:
            low = mid + 1
        else:
            high = mid
    row = low - 1
    first = (row * (2 * matrix_size - row - 1)) >> 1
    col = (index - first) + row + 1
    return row, col


@wp.func
def _find_world_and_local_id(tid: int, world_cumsum_lower_tri: wp.array(dtype=int, ndim=1)):
    world_count = world_cumsum_lower_tri.shape[0]
    low = int(0)
    high = int(world_count - 1)
    world_id = int(0)

    while low <= high:
        mid = (low + high) >> 1
        if tid < world_cumsum_lower_tri[mid]:
            high = mid - 1
            world_id = mid
        else:
            low = mid + 1

    local_id = tid
    if world_id > 0:
        local_id = tid - world_cumsum_lower_tri[world_id - 1]

    return world_id, local_id


@wp.kernel
def _nxn_broadphase_kernel(
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    shape_world: wp.array(dtype=int, ndim=1),
    world_cumsum_lower_tri: wp.array(dtype=int, ndim=1),
    world_slice_ends: wp.array(dtype=int, ndim=1),
    world_index_map: wp.array(dtype=int, ndim=1),
    num_regular_worlds: int,
    filter_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_filter_pairs: int,
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    candidate_pair_count: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):
    tid = wp.tid()
    world_id, local_id = _find_world_and_local_id(tid, world_cumsum_lower_tri)

    world_slice_start = 0
    if world_id > 0:
        world_slice_start = world_slice_ends[world_id - 1]
    world_slice_end = world_slice_ends[world_id]
    num_shapes_in_world = world_slice_end - world_slice_start

    local_shape1, local_shape2 = _get_lower_triangular_indices(local_id, num_shapes_in_world)

    shape1_tmp = world_index_map[world_slice_start + local_shape1]
    shape2_tmp = world_index_map[world_slice_start + local_shape2]
    shape1 = wp.min(shape1_tmp, shape2_tmp)
    shape2 = wp.max(shape1_tmp, shape2_tmp)

    world1 = shape_world[shape1]
    world2 = shape_world[shape2]
    collision_group1 = collision_group[shape1]
    collision_group2 = collision_group[shape2]

    is_dedicated_minus_one_segment = world_id >= num_regular_worlds
    if world1 == -1 and world2 == -1 and not is_dedicated_minus_one_segment:
        return

    if not test_world_and_group_pair(world1, world2, collision_group1, collision_group2):
        return

    gap1 = 0.0
    gap2 = 0.0
    if shape_gap.shape[0] > 0:
        gap1 = shape_gap[shape1]
        gap2 = shape_gap[shape2]

    if check_aabb_overlap(
        shape_bounding_box_lower[shape1],
        shape_bounding_box_upper[shape1],
        gap1,
        shape_bounding_box_lower[shape2],
        shape_bounding_box_upper[shape2],
        gap2,
    ):
        if num_filter_pairs > 0 and is_pair_excluded(wp.vec2i(shape1, shape2), filter_pairs, num_filter_pairs):
            return
        write_pair(wp.vec2i(shape1, shape2), candidate_pair, candidate_pair_count, max_candidate_pair)


class BroadPhaseAllPairs:
    def __init__(
        self,
        shape_world: wp.array(dtype=wp.int32, ndim=1) | np.ndarray,
        shape_flags: wp.array(dtype=wp.int32, ndim=1) | np.ndarray | None = None,
        device=None,
    ) -> None:
        if isinstance(shape_world, wp.array):
            shape_world_np = shape_world.numpy()
            if device is None:
                device = shape_world.device
        else:
            shape_world_np = shape_world
            if device is None:
                device = "cpu"

        shape_flags_np = None
        if shape_flags is not None:
            shape_flags_np = shape_flags.numpy() if isinstance(shape_flags, wp.array) else shape_flags

        index_map_np, slice_ends_np = precompute_world_map(shape_world_np, shape_flags_np)
        num_regular_worlds = max(0, len(slice_ends_np) - 1)

        world_count = len(slice_ends_np)
        world_cumsum_lower_tri_np = np.zeros(world_count, dtype=np.int32)

        start_idx = 0
        cumsum = 0
        for world_idx in range(world_count):
            end_idx = slice_ends_np[world_idx]
            num_geoms_in_world = end_idx - start_idx
            num_lower_tri = (num_geoms_in_world * (num_geoms_in_world - 1)) // 2
            cumsum += num_lower_tri
            world_cumsum_lower_tri_np[world_idx] = cumsum
            start_idx = end_idx

        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)
        self.world_slice_ends = wp.array(slice_ends_np, dtype=wp.int32, device=device)
        self.world_cumsum_lower_tri = wp.array(world_cumsum_lower_tri_np, dtype=wp.int32, device=device)
        self.num_kernel_threads = int(world_cumsum_lower_tri_np[-1]) if world_count > 0 else 0
        self.num_regular_worlds = int(num_regular_worlds)

    def launch(
        self,
        shape_lower: wp.array(dtype=wp.vec3, ndim=1),
        shape_upper: wp.array(dtype=wp.vec3, ndim=1),
        shape_gap: wp.array(dtype=float, ndim=1) | None,
        shape_collision_group: wp.array(dtype=int, ndim=1),
        shape_world: wp.array(dtype=int, ndim=1),
        shape_count: int,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
        candidate_pair_count: wp.array(dtype=int, ndim=1),
        device=None,
        filter_pairs: wp.array(dtype=wp.vec2i, ndim=1) | None = None,
        num_filter_pairs: int | None = None,
    ) -> None:
        del shape_count
        max_candidate_pair = candidate_pair.shape[0]
        candidate_pair_count.zero_()

        if device is None:
            device = shape_lower.device
        if shape_gap is None:
            shape_gap = wp.empty(0, dtype=wp.float32, device=device)

        if filter_pairs is None or filter_pairs.shape[0] == 0:
            filter_pairs_arr = wp.empty(0, dtype=wp.vec2i, device=device)
            n_filter = 0
        else:
            filter_pairs_arr = filter_pairs
            n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]

        wp.launch(
            _nxn_broadphase_kernel,
            dim=self.num_kernel_threads,
            inputs=[
                shape_lower,
                shape_upper,
                shape_gap,
                shape_collision_group,
                shape_world,
                self.world_cumsum_lower_tri,
                self.world_slice_ends,
                self.world_index_map,
                self.num_regular_worlds,
                filter_pairs_arr,
                n_filter,
            ],
            outputs=[candidate_pair, candidate_pair_count, max_candidate_pair],
            device=device,
        )


class BroadPhaseExplicit:
    def __init__(self) -> None:
        pass

    def launch(
        self,
        shape_lower: wp.array(dtype=wp.vec3, ndim=1),
        shape_upper: wp.array(dtype=wp.vec3, ndim=1),
        shape_gap: wp.array(dtype=float, ndim=1) | None,
        shape_pairs: wp.array(dtype=wp.vec2i, ndim=1),
        shape_pair_count: int,
        candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
        candidate_pair_count: wp.array(dtype=int, ndim=1),
        device=None,
    ) -> None:
        max_candidate_pair = candidate_pair.shape[0]
        candidate_pair_count.zero_()

        if device is None:
            device = shape_lower.device
        if shape_gap is None:
            shape_gap = wp.empty(0, dtype=wp.float32, device=device)

        wp.launch(
            kernel=_nxn_broadphase_precomputed_pairs,
            dim=shape_pair_count,
            inputs=[
                shape_lower,
                shape_upper,
                shape_gap,
                shape_pairs,
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
            ],
            device=device,
        )
