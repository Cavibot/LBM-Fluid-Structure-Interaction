# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Point-to-collision-shape distance queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton._src.core.types import Axis
from newton._src.geometry.kernels import (
    sdf_box,
    sdf_capsule,
    sdf_cone,
    sdf_cylinder,
    sdf_ellipsoid,
    sdf_plane,
    sdf_sphere,
)
from newton._src.geometry.sdf_utils import SDF, SDFData
from newton._src.geometry.types import GeoType
from newton._src.utils.heightfield import HeightfieldData, sample_sdf_grad_heightfield

if TYPE_CHECKING:
    from wanphys._src.rigid.domain import RigidDomain
    from wanphys._src.rigid.model import RigidModel


_INF = 1.0e20
_MESH_QUERY_MAX_DIST = 1.0e6


@wp.func
def _sample_sdf_for_grid_rasterization(sdf_data: SDFData, sdf_pos: wp.vec3) -> float:
    """Sample a mesh SDF for dense fluid-grid rasterization.

    Fluid coupling queries cell centers across the whole grid, including points
    far from the mesh surface and deep inside the mesh. The sparse SDF is only
    reliable near the narrow band around the surface, so unusually large sparse
    values are treated as invalid and fall back to the coarse SDF.
    """
    lower = sdf_data.center - sdf_data.half_extents
    upper = sdf_data.center + sdf_data.half_extents

    inside_extent = (
        sdf_pos[0] >= lower[0]
        and sdf_pos[0] <= upper[0]
        and sdf_pos[1] >= lower[1]
        and sdf_pos[1] <= upper[1]
        and sdf_pos[2] >= lower[2]
        and sdf_pos[2] <= upper[2]
    )

    if inside_extent:
        sparse_idx = wp.volume_world_to_index(sdf_data.sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_f(sdf_data.sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR)
        # Sparse voxels are narrow-band data; large interpolated values here
        # are not useful for dense grid rasterization and should use coarse SDF.
        sparse_valid_limit = sdf_data.sparse_voxel_radius * 8.0

        if wp.abs(sparse_dist) > sparse_valid_limit or sparse_dist >= sdf_data.background_value * 0.99 or wp.isnan(sparse_dist):
            coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, sdf_pos)
            return wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)

        return sparse_dist

    eps = 1e-2 * sdf_data.sparse_voxel_size
    clamped_pos = wp.min(wp.max(sdf_pos, lower + eps), upper - eps)
    dist_to_boundary = wp.length(sdf_pos - clamped_pos)
    coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, clamped_pos)
    boundary_dist = wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)
    return boundary_dist + dist_to_boundary


@wp.struct
class RigidShapeQueryData:
    """Warp-side static data needed by point-to-shape queries."""

    # Static model arrays passed through directly.
    shape_body: wp.array(dtype=int)
    shape_transform: wp.array(dtype=wp.transform)
    shape_type: wp.array(dtype=int)
    shape_source_ptr: wp.array(dtype=wp.uint64)
    shape_scale: wp.array(dtype=wp.vec3)
    shape_collision_radius: wp.array(dtype=float)
    shape_collision_aabb_lower: wp.array(dtype=wp.vec3)
    shape_collision_aabb_upper: wp.array(dtype=wp.vec3)
    shape_heightfield_data: wp.array(dtype=HeightfieldData)
    heightfield_elevation_data: wp.array(dtype=wp.float32)
    sdf_data: wp.array(dtype=SDFData)
    shape_sdf_index: wp.array(dtype=wp.int32)

    # Baked lookup/acceleration data owned by RigidShapeQuery.
    body_shape_offsets: wp.array(dtype=int)
    body_shape_indices: wp.array(dtype=int)

    body_count: int
    shape_count: int


@wp.func
def _mesh_sdf_distance(point_local: wp.vec3, shape_idx: int, data: RigidShapeQueryData) -> float:
    scale = data.shape_scale[shape_idx]

    min_scale = wp.min(scale)
    if min_scale <= 0.0:
        return _INF

    sdf_idx = data.shape_sdf_index[shape_idx]
    if sdf_idx >= 0:
        sdf = data.sdf_data[sdf_idx]
        if sdf.sparse_sdf_ptr != wp.uint64(0):
            return _sample_sdf_for_grid_rasterization(sdf, point_local)

    mesh_id = data.shape_source_ptr[shape_idx]
    if mesh_id != wp.uint64(0):
        query_point = wp.cw_div(point_local, scale)
        query_max_dist = _MESH_QUERY_MAX_DIST / min_scale

        sign = float(0.0)
        face = int(0)
        u = float(0.0)
        v = float(0.0)
        if wp.mesh_query_point(mesh_id, query_point, query_max_dist, sign, face, u, v):
            closest = wp.mesh_eval_position(mesh_id, face, u, v)
            closest_scaled = wp.cw_mul(closest, scale)
            return wp.length(point_local - closest_scaled) * sign

        return _MESH_QUERY_MAX_DIST

    return _INF


@wp.func
def _shape_distance_local(point_local: wp.vec3, shape_idx: int, data: RigidShapeQueryData) -> float:
    geo_type = data.shape_type[shape_idx]
    scale = data.shape_scale[shape_idx]

    if geo_type == GeoType.SPHERE:
        return sdf_sphere(point_local, scale[0])

    if geo_type == GeoType.BOX:
        return sdf_box(point_local, scale[0], scale[1], scale[2])

    if geo_type == GeoType.CAPSULE:
        return sdf_capsule(point_local, scale[0], scale[1], int(Axis.Z))

    if geo_type == GeoType.CYLINDER:
        return sdf_cylinder(point_local, scale[0], scale[1], int(Axis.Z))

    if geo_type == GeoType.CONE:
        return sdf_cone(point_local, scale[0], scale[1], int(Axis.Z))

    if geo_type == GeoType.ELLIPSOID:
        return sdf_ellipsoid(point_local, scale)

    if geo_type == GeoType.PLANE:
        return sdf_plane(point_local, scale[0] * 0.5, scale[1] * 0.5)

    if geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
        return _mesh_sdf_distance(point_local, shape_idx, data)

    if geo_type == GeoType.HFIELD:
        hfd = data.shape_heightfield_data[shape_idx]
        if hfd.nrow > 1 and hfd.ncol > 1:
            d, _n = sample_sdf_grad_heightfield(hfd, data.heightfield_elevation_data, point_local)
            return d

    return _INF


@wp.func
def _shape_point_local(
    point_world: wp.vec3,
    shape_idx: int,
    data: RigidShapeQueryData,
    body_q: wp.array(dtype=wp.transform),
) -> wp.vec3:
    body_idx = data.shape_body[shape_idx]
    X_ws = data.shape_transform[shape_idx]
    if body_idx >= 0:
        X_ws = wp.transform_multiply(body_q[body_idx], X_ws)
    return wp.transform_point(wp.transform_inverse(X_ws), point_world)


@wp.func
def _shape_bounding_sphere_distance_lower_bound(
    point_local: wp.vec3, shape_idx: int, data: RigidShapeQueryData
) -> float:
    return wp.length(point_local) - data.shape_collision_radius[shape_idx]


@wp.func
def _point_aabb_distance(point: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> float:
    delta_lower = lower - point
    delta_upper = point - upper
    outside = wp.max(wp.max(delta_lower, delta_upper), wp.vec3(0.0))
    return wp.length(outside)


@wp.func
def _shape_distance_prune_lower_bound(point_local: wp.vec3, shape_idx: int, data: RigidShapeQueryData) -> float:
    geo_type = data.shape_type[shape_idx]
    sphere_bound = _shape_bounding_sphere_distance_lower_bound(point_local, shape_idx, data)

    if geo_type == GeoType.SPHERE or geo_type == GeoType.PLANE or geo_type == GeoType.HFIELD:
        return sphere_bound

    aabb_bound = _point_aabb_distance(
        point_local,
        data.shape_collision_aabb_lower[shape_idx],
        data.shape_collision_aabb_upper[shape_idx],
    )
    return wp.max(sphere_bound, aabb_bound)


@wp.func
def point_shape_distance(
    point_world: wp.vec3,
    shape_idx: int,
    data: RigidShapeQueryData,
    body_q: wp.array(dtype=wp.transform),
) -> float:
    """Return signed distance from a world-space point to one collision shape.
    """

    if shape_idx < 0 or shape_idx >= data.shape_count:
        return _INF

    body_idx = data.shape_body[shape_idx]
    if body_idx >= data.body_count:
        return _INF

    point_local = _shape_point_local(point_world, shape_idx, data, body_q)
    return _shape_distance_local(point_local, shape_idx, data)


@wp.func
def point_body_distance(
    point_world: wp.vec3,
    body_idx: int,
    data: RigidShapeQueryData,
    body_q: wp.array(dtype=wp.transform),
) -> float:
    """Return signed distance from a point to all shapes attached to a body.

    The body distance is the minimum signed distance over all collision shapes
    attached to ``body_idx``. Positive values are outside, negative values are
    inside, and zero is on the collision surface.
    """

    if body_idx < 0 or body_idx >= data.body_count:
        return _INF

    start = data.body_shape_offsets[body_idx]
    end = data.body_shape_offsets[body_idx + 1]

    best = _INF
    i = start
    while i < end:
        shape_idx = data.body_shape_indices[i]
        point_local = _shape_point_local(point_world, shape_idx, data, body_q)
        lower_bound = _shape_distance_prune_lower_bound(point_local, shape_idx, data)
        if best < 0.0 or lower_bound <= best:
            d = _shape_distance_local(point_local, shape_idx, data)
            best = wp.min(best, d)
        i += 1

    return best


@dataclass
class RigidShapeQuery:
    """Host-side wrapper that owns precomputed body-to-shape lookup arrays."""

    data: RigidShapeQueryData
    device: wp.Device
    body_count: int
    shape_count: int
    model_id: int
    sdf_volumes: list[object] | None = None
    sdf_coarse_volumes: list[object] | None = None

    @classmethod
    def from_domain(
        cls,
        domain: RigidDomain,
    ) -> "RigidShapeQuery":
        """Build static query data from a rigid domain."""

        model = domain.model
        body_count = int(model.body_count)
        shape_body = model.shape_body.numpy()

        body_shapes: list[list[int]] = [[] for _ in range(body_count)]
        for shape_idx, owner in enumerate(shape_body):
            body_idx = int(owner)
            if 0 <= body_idx < body_count:
                body_shapes[body_idx].append(int(shape_idx))

        offsets = [0]
        indices: list[int] = []
        for shapes in body_shapes:
            indices.extend(shapes)
            offsets.append(len(indices))

        device = model.device
        body_shape_offsets = wp.array(np.asarray(offsets, dtype=np.int32), dtype=int, device=device)
        body_shape_indices = wp.array(np.asarray(indices, dtype=np.int32), dtype=int, device=device)
        sdf_data, shape_sdf_index, sdf_volumes, sdf_coarse_volumes = cls._build_mesh_sdf_table(model, device)

        data = RigidShapeQueryData()
        data.shape_body = model.shape_body
        data.shape_transform = model.shape_transform
        data.shape_type = model.shape_type
        data.shape_source_ptr = model.shape_source_ptr
        data.shape_scale = model.shape_scale
        data.shape_collision_radius = model.shape_collision_radius
        data.shape_collision_aabb_lower = model.shape_collision_aabb_lower
        data.shape_collision_aabb_upper = model.shape_collision_aabb_upper
        data.shape_heightfield_data = model.shape_heightfield_data
        data.heightfield_elevation_data = model.heightfield_elevation_data
        data.sdf_data = sdf_data
        data.shape_sdf_index = shape_sdf_index
        data.body_shape_offsets = body_shape_offsets
        data.body_shape_indices = body_shape_indices
        data.body_count = body_count
        data.shape_count = int(model.shape_count)
        return cls(
            data=data,
            device=device,
            body_count=body_count,
            shape_count=int(model.shape_count),
            model_id=id(model),
            sdf_volumes=sdf_volumes,
            sdf_coarse_volumes=sdf_coarse_volumes,
        )

    @staticmethod
    def _build_mesh_sdf_table(
        model: RigidModel,
        device: str | wp.Device,
    ) -> tuple[wp.array, wp.array, list[object], list[object]]:
        shape_count = int(model.shape_count)
        shape_types = model.shape_type.numpy()
        shape_scale = model.shape_scale.numpy()
        shape_sdf_index = [-1] * shape_count
        sdf_payloads: list[SDFData] = []
        sdf_volumes: list[object] = []
        sdf_coarse_volumes: list[object] = []
        sdf_cache: dict[tuple[int, tuple[float, float, float]], int] = {}

        current_device = wp.get_device(device)
        for shape_idx, geo_type_value in enumerate(shape_types):
            geo_type = int(geo_type_value)
            if geo_type != int(GeoType.MESH) and geo_type != int(GeoType.CONVEX_MESH):
                continue

            source = model.shape_source[shape_idx] if shape_idx < len(model.shape_source) else None
            if source is None:
                continue

            if not current_device.is_cuda:
                continue

            scale_tuple = tuple(float(v) for v in shape_scale[shape_idx])
            if min(scale_tuple) <= 0.0:
                continue

            cache_key = (id(source), scale_tuple)
            sdf_idx = sdf_cache.get(cache_key)
            if sdf_idx is None:
                sdf = SDF.create_from_mesh(source, max_resolution=64, scale=scale_tuple, device=device)
                if sdf.is_empty():
                    continue

                sdf_idx = len(sdf_payloads)
                sdf_cache[cache_key] = sdf_idx
                sdf_payloads.append(sdf.to_kernel_data())
                sdf_volumes.append(sdf.sparse_volume)
                sdf_coarse_volumes.append(sdf.coarse_volume)
            shape_sdf_index[shape_idx] = sdf_idx

        return (
            wp.array(sdf_payloads, dtype=SDFData, device=device) if sdf_payloads else wp.array([], dtype=SDFData, device=device),
            wp.array(np.asarray(shape_sdf_index, dtype=np.int32), dtype=wp.int32, device=device),
            sdf_volumes,
            sdf_coarse_volumes,
        )
