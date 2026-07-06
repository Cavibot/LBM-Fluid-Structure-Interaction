# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Common utilities for native rigid broad phase collision detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp


_COLLIDE_SHAPES = 1 << 1


@wp.func
def check_aabb_overlap(
    box1_lower: wp.vec3,
    box1_upper: wp.vec3,
    box1_cutoff: float,
    box2_lower: wp.vec3,
    box2_upper: wp.vec3,
    box2_cutoff: float,
) -> bool:
    cutoff_combined = box1_cutoff + box2_cutoff
    return (
        box1_lower[0] <= box2_upper[0] + cutoff_combined
        and box1_upper[0] >= box2_lower[0] - cutoff_combined
        and box1_lower[1] <= box2_upper[1] + cutoff_combined
        and box1_upper[1] >= box2_lower[1] - cutoff_combined
        and box1_lower[2] <= box2_upper[2] + cutoff_combined
        and box1_upper[2] >= box2_lower[2] - cutoff_combined
    )


@wp.func
def binary_search(values: wp.array(dtype=Any), value: Any, lower: int, upper: int) -> int:
    while lower < upper:
        mid = (lower + upper) >> 1
        if values[mid] > value:
            upper = mid
        else:
            lower = mid + 1
    return upper


@wp.func
def _vec2i_less(p: wp.vec2i, q: wp.vec2i) -> bool:
    if p[0] < q[0]:
        return True
    if p[0] > q[0]:
        return False
    return p[1] < q[1]


@wp.func
def _vec2i_equal(p: wp.vec2i, q: wp.vec2i) -> bool:
    return p[0] == q[0] and p[1] == q[1]


@wp.func
def is_pair_excluded(
    pair: wp.vec2i,
    filter_pairs: wp.array(dtype=wp.vec2i, ndim=1),
    num_filter_pairs: int,
) -> bool:
    if num_filter_pairs <= 0:
        return False
    low = int(0)
    high = num_filter_pairs - 1
    while low <= high:
        mid = (low + high) >> 1
        value = filter_pairs[mid]
        if _vec2i_equal(pair, value):
            return True
        if _vec2i_less(pair, value):
            high = mid - 1
        else:
            low = mid + 1
    return False


@wp.func
def write_pair(
    pair: wp.vec2i,
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    candidate_pair_count: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):
    pairid = wp.atomic_add(candidate_pair_count, 0, 1)
    if pairid >= max_candidate_pair:
        return
    candidate_pair[pairid] = pair


@wp.func
def test_group_pair(group_a: int, group_b: int) -> bool:
    if group_a == 0 or group_b == 0:
        return False
    if group_a > 0:
        return group_a == group_b or group_b < 0
    if group_a < 0:
        return group_a != group_b
    return False


@wp.func
def test_world_and_group_pair(world_a: int, world_b: int, collision_group_a: int, collision_group_b: int) -> bool:
    if world_a != -1 and world_b != -1 and world_a != world_b:
        return False
    return test_group_pair(collision_group_a, collision_group_b)


def precompute_world_map(shape_world: np.ndarray | list[int], shape_flags: np.ndarray | list[int] | None = None):
    if not isinstance(shape_world, np.ndarray):
        shape_world = np.array(shape_world)

    if shape_flags is not None:
        if not isinstance(shape_flags, np.ndarray):
            shape_flags = np.array(shape_flags)
        if shape_flags.shape[0] != shape_world.shape[0]:
            raise ValueError("shape_flags and shape_world must have the same length")
        colliding_mask = (shape_flags & _COLLIDE_SHAPES) != 0
    else:
        colliding_mask = np.ones(len(shape_world), dtype=bool)

    valid_indices = np.where(colliding_mask)[0]
    filtered_world_ids = shape_world[valid_indices]

    invalid_worlds = shape_world[(shape_world < -1)]
    if len(invalid_worlds) > 0:
        unique_invalid = np.unique(invalid_worlds)
        raise ValueError(
            f"Invalid world IDs detected: {unique_invalid.tolist()}. "
            f"Only world ID -1 and non-negative IDs are supported."
        )

    negative_mask = filtered_world_ids == -1
    num_shared = np.sum(negative_mask)
    shared_local_indices = np.where(negative_mask)[0]
    shared_indices = valid_indices[shared_local_indices]

    positive_mask = filtered_world_ids >= 0
    positive_world_ids = filtered_world_ids[positive_mask]
    unique_worlds = np.unique(positive_world_ids)
    world_count = len(unique_worlds)

    num_positive = np.sum(positive_mask)
    total_size = num_positive + (num_shared * world_count) + num_shared

    index_map = np.empty(total_size, dtype=np.int32)
    slice_ends = np.empty(world_count + 1, dtype=np.int32)

    current_pos = 0
    for world_idx, world_id in enumerate(unique_worlds):
        world_local_indices = np.where(filtered_world_ids == world_id)[0]
        world_indices = valid_indices[world_local_indices]
        world_shape_count = len(world_indices)

        index_map[current_pos : current_pos + world_shape_count] = world_indices
        current_pos += world_shape_count

        index_map[current_pos : current_pos + num_shared] = shared_indices
        current_pos += num_shared

        slice_ends[world_idx] = current_pos

    index_map[current_pos : current_pos + num_shared] = shared_indices
    current_pos += num_shared
    slice_ends[world_count] = current_pos

    return index_map, slice_ends
