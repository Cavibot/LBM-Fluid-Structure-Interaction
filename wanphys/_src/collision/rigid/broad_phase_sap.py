# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Native sweep-and-prune broad phase collision detection for rigid shapes."""

from __future__ import annotations

from typing import Literal

import numpy as np
import warp as wp

from .broad_phase_common import (
    binary_search,
    check_aabb_overlap,
    is_pair_excluded,
    precompute_world_map,
    test_world_and_group_pair,
    write_pair,
)

wp.set_module_options({"enable_backward": False})

SAPSortMode = Literal["segmented", "tile"]


def _normalize_sort_mode(mode: str) -> SAPSortMode:
    normalized = mode.strip().lower()
    if normalized not in ("segmented", "tile"):
        raise ValueError(f"Unsupported SAP sort mode: {mode!r}. Expected 'segmented' or 'tile'.")
    return normalized


@wp.func
def _sap_project_aabb(
    elementid: int,
    direction: wp.vec3,
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
) -> wp.vec2:
    lower = shape_bounding_box_lower[elementid]
    upper = shape_bounding_box_upper[elementid]

    gap = 0.0
    if shape_gap.shape[0] > 0:
        gap = shape_gap[elementid]

    half_size = 0.5 * (upper - lower)
    half_size = wp.vec3(half_size[0] + gap, half_size[1] + gap, half_size[2] + gap)
    radius = wp.dot(direction, half_size)
    center = wp.dot(direction, 0.5 * (lower + upper))
    return wp.vec2(center - radius, center + radius)


@wp.func
def binary_search_segment(
    arr: wp.array(dtype=float, ndim=1),
    base_idx: int,
    value: float,
    start: int,
    end: int,
) -> int:
    low = int(start)
    high = int(end)
    while low < high:
        mid = (low + high) // 2
        if arr[base_idx + mid] < value:
            low = mid + 1
        else:
            high = mid
    return low


def _create_tile_sort_kernel(tile_size: int):
    @wp.kernel
    def tile_sort_kernel(
        sap_projection_lower: wp.array(dtype=float, ndim=1),
        sap_sort_index: wp.array(dtype=int, ndim=1),
        max_geoms_per_world: int,
    ):
        world_id = wp.tid()
        base_idx = world_id * max_geoms_per_world
        keys = wp.tile_load(sap_projection_lower, shape=(tile_size,), offset=(base_idx,), storage="shared")
        values = wp.tile_load(sap_sort_index, shape=(tile_size,), offset=(base_idx,), storage="shared")
        wp.tile_sort(keys, values)
        wp.tile_store(sap_projection_lower, keys, offset=(base_idx,))
        wp.tile_store(sap_sort_index, values, offset=(base_idx,))

    return tile_sort_kernel


@wp.kernel
def _sap_project_kernel(
    direction: wp.vec3,
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    world_index_map: wp.array(dtype=int, ndim=1),
    world_slice_ends: wp.array(dtype=int, ndim=1),
    max_shapes_per_world: int,
    sap_projection_lower_out: wp.array(dtype=float, ndim=1),
    sap_projection_upper_out: wp.array(dtype=float, ndim=1),
    sap_sort_index_out: wp.array(dtype=int, ndim=1),
):
    world_id, local_shape_id = wp.tid()
    idx = world_id * max_shapes_per_world + local_shape_id

    world_slice_start = 0
    if world_id > 0:
        world_slice_start = world_slice_ends[world_id - 1]
    world_slice_end = world_slice_ends[world_id]
    num_shapes_in_world = world_slice_end - world_slice_start

    if local_shape_id >= num_shapes_in_world:
        sap_projection_lower_out[idx] = 1e30
        sap_projection_upper_out[idx] = 1e30
        sap_sort_index_out[idx] = -1
        return

    shape_id = world_index_map[world_slice_start + local_shape_id]
    projection = _sap_project_aabb(shape_id, direction, shape_bounding_box_lower, shape_bounding_box_upper, shape_gap)
    sap_projection_lower_out[idx] = projection[0]
    sap_projection_upper_out[idx] = projection[1]
    sap_sort_index_out[idx] = local_shape_id


@wp.kernel
def _sap_range_kernel(
    world_slice_ends: wp.array(dtype=int, ndim=1),
    max_shapes_per_world: int,
    sap_projection_lower_in: wp.array(dtype=float, ndim=1),
    sap_projection_upper_in: wp.array(dtype=float, ndim=1),
    sap_sort_index_in: wp.array(dtype=int, ndim=1),
    sap_range_out: wp.array(dtype=int, ndim=1),
):
    world_id, local_shape_id = wp.tid()
    idx = world_id * max_shapes_per_world + local_shape_id

    world_slice_start = 0
    if world_id > 0:
        world_slice_start = world_slice_ends[world_id - 1]
    world_slice_end = world_slice_ends[world_id]
    num_shapes_in_world = world_slice_end - world_slice_start

    if local_shape_id >= num_shapes_in_world:
        sap_range_out[idx] = 0
        return

    sort_idx = sap_sort_index_in[idx]
    if sort_idx < 0:
        sap_range_out[idx] = 0
        return

    upper_idx = world_id * max_shapes_per_world + sort_idx
    upper = sap_projection_upper_in[upper_idx]
    world_base_idx = world_id * max_shapes_per_world
    limit = binary_search_segment(
        sap_projection_lower_in, world_base_idx, upper, local_shape_id + 1, num_shapes_in_world
    )
    limit = wp.min(num_shapes_in_world, limit)
    sap_range_out[idx] = limit - local_shape_id - 1


@wp.func
def _process_single_sap_pair(
    pair: wp.vec2i,
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    candidate_pair_count: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
    filter_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_filter_pairs: int,
):
    shape1 = pair[0]
    shape2 = pair[1]

    if num_filter_pairs > 0 and is_pair_excluded(pair, filter_pairs, num_filter_pairs):
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
        write_pair(pair, candidate_pair, candidate_pair_count, max_candidate_pair)


@wp.kernel
def _sap_broadphase_kernel(
    shape_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
    shape_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
    shape_gap: wp.array(dtype=float, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    shape_world: wp.array(dtype=int, ndim=1),
    world_index_map: wp.array(dtype=int, ndim=1),
    world_slice_ends: wp.array(dtype=int, ndim=1),
    sap_sort_index_in: wp.array(dtype=int, ndim=1),
    sap_cumulative_sum_in: wp.array(dtype=int, ndim=1),
    world_count: int,
    max_shapes_per_world: int,
    nsweep_in: int,
    num_regular_worlds: int,
    filter_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_filter_pairs: int,
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    candidate_pair_count: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):
    tid = wp.tid()
    total_work_packages = sap_cumulative_sum_in[world_count * max_shapes_per_world - 1]

    workid = tid
    while workid < total_work_packages:
        flat_id = binary_search(sap_cumulative_sum_in, workid, 0, world_count * max_shapes_per_world)
        j = flat_id + workid + 1
        if flat_id > 0:
            j -= sap_cumulative_sum_in[flat_id - 1]

        world_id = flat_id // max_shapes_per_world
        i = flat_id % max_shapes_per_world
        j = j % max_shapes_per_world

        world_slice_start = 0
        if world_id > 0:
            world_slice_start = world_slice_ends[world_id - 1]
        world_slice_end = world_slice_ends[world_id]
        num_shapes_in_world = world_slice_end - world_slice_start

        if i >= num_shapes_in_world or j >= num_shapes_in_world:
            workid += nsweep_in
            continue
        if i >= j:
            workid += nsweep_in
            continue

        idx_i = world_id * max_shapes_per_world + i
        idx_j = world_id * max_shapes_per_world + j
        local_shape1 = sap_sort_index_in[idx_i]
        local_shape2 = sap_sort_index_in[idx_j]

        if local_shape1 < 0 or local_shape2 < 0:
            workid += nsweep_in
            continue

        shape1_tmp = world_index_map[world_slice_start + local_shape1]
        shape2_tmp = world_index_map[world_slice_start + local_shape2]
        if shape1_tmp == shape2_tmp:
            workid += nsweep_in
            continue

        shape1 = wp.min(shape1_tmp, shape2_tmp)
        shape2 = wp.max(shape1_tmp, shape2_tmp)
        col_group1 = collision_group[shape1]
        col_group2 = collision_group[shape2]
        world1 = shape_world[shape1]
        world2 = shape_world[shape2]

        is_dedicated_minus_one_segment = world_id >= num_regular_worlds
        if world1 == -1 and world2 == -1 and not is_dedicated_minus_one_segment:
            workid += nsweep_in
            continue

        if test_world_and_group_pair(world1, world2, col_group1, col_group2):
            _process_single_sap_pair(
                wp.vec2i(shape1, shape2),
                shape_bounding_box_lower,
                shape_bounding_box_upper,
                shape_gap,
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
                filter_pairs,
                num_filter_pairs,
            )

        workid += nsweep_in


class BroadPhaseSAP:
    def __init__(
        self,
        shape_world: wp.array(dtype=wp.int32, ndim=1) | np.ndarray,
        shape_flags: wp.array(dtype=wp.int32, ndim=1) | np.ndarray | None = None,
        sweep_thread_count_multiplier: int = 5,
        sort_type: Literal["segmented", "tile"] = "segmented",
        tile_block_dim: int | None = None,
        device=None,
    ) -> None:
        self.sweep_thread_count_multiplier = sweep_thread_count_multiplier
        self.sort_type = _normalize_sort_mode(sort_type)
        self.tile_block_dim_override = tile_block_dim

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

        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)
        self.world_slice_ends = wp.array(slice_ends_np, dtype=wp.int32, device=device)
        self.world_count = len(slice_ends_np)
        self.num_regular_worlds = int(num_regular_worlds)
        self.max_shapes_per_world = 0

        start_idx = 0
        for end_idx in slice_ends_np:
            num_shapes = end_idx - start_idx
            self.max_shapes_per_world = max(self.max_shapes_per_world, num_shapes)
            start_idx = end_idx

        self.tile_sort_kernel = None
        if self.sort_type == "tile":
            if self.tile_block_dim_override is not None:
                self.tile_block_dim = max(32, min(self.tile_block_dim_override, 1024))
            else:
                block_dim = 1
                while block_dim < self.max_shapes_per_world:
                    block_dim *= 2
                self.tile_block_dim = max(32, min(block_dim, 512))

            self.tile_size = int(self.max_shapes_per_world)
            self.tile_sort_kernel = _create_tile_sort_kernel(self.tile_size)

        total_elements = int(self.world_count * self.max_shapes_per_world)
        self.sap_projection_lower = wp.zeros(2 * total_elements, dtype=wp.float32, device=device)
        self.sap_projection_upper = wp.zeros(total_elements, dtype=wp.float32, device=device)
        self.sap_sort_index = wp.zeros(2 * total_elements, dtype=wp.int32, device=device)
        self.sap_range = wp.zeros(total_elements, dtype=wp.int32, device=device)
        self.sap_cumulative_sum = wp.zeros(total_elements, dtype=wp.int32, device=device)

        segment_indices_np = np.array(
            [i * self.max_shapes_per_world for i in range(self.world_count + 1)],
            dtype=np.int32,
        )
        self.segment_indices = wp.array(segment_indices_np, dtype=wp.int32, device=device)

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

        direction = wp.vec3(0.5935, 0.7790, 0.1235)
        direction = wp.normalize(direction)
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
            kernel=_sap_project_kernel,
            dim=(self.world_count, self.max_shapes_per_world),
            inputs=[
                direction,
                shape_lower,
                shape_upper,
                shape_gap,
                self.world_index_map,
                self.world_slice_ends,
                self.max_shapes_per_world,
                self.sap_projection_lower,
                self.sap_projection_upper,
                self.sap_sort_index,
            ],
            device=device,
        )

        if self.sort_type == "tile" and self.tile_sort_kernel is not None:
            wp.launch_tiled(
                kernel=self.tile_sort_kernel,
                dim=self.world_count,
                inputs=[
                    self.sap_projection_lower,
                    self.sap_sort_index,
                    self.max_shapes_per_world,
                ],
                block_dim=self.tile_block_dim,
                device=device,
            )
        else:
            wp.utils.segmented_sort_pairs(
                keys=self.sap_projection_lower,
                values=self.sap_sort_index,
                count=self.world_count * self.max_shapes_per_world,
                segment_start_indices=self.segment_indices,
            )

        wp.launch(
            kernel=_sap_range_kernel,
            dim=(self.world_count, self.max_shapes_per_world),
            inputs=[
                self.world_slice_ends,
                self.max_shapes_per_world,
                self.sap_projection_lower,
                self.sap_projection_upper,
                self.sap_sort_index,
                self.sap_range,
            ],
            device=device,
        )

        wp.utils.array_scan(self.sap_range, self.sap_cumulative_sum, True)

        total_elements = self.world_count * self.max_shapes_per_world
        nsweep_in = int(self.sweep_thread_count_multiplier * total_elements)

        wp.launch(
            kernel=_sap_broadphase_kernel,
            dim=nsweep_in,
            inputs=[
                shape_lower,
                shape_upper,
                shape_gap,
                shape_collision_group,
                shape_world,
                self.world_index_map,
                self.world_slice_ends,
                self.sap_sort_index,
                self.sap_cumulative_sum,
                self.world_count,
                self.max_shapes_per_world,
                nsweep_in,
                self.num_regular_worlds,
                filter_pairs_arr,
                n_filter,
            ],
            outputs=[
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
            ],
            device=device,
        )
