# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for rigid-fluid particle-shape contact generation."""

from __future__ import annotations

import warp as wp

from newton._src.geometry.flags import ParticleFlags, ShapeFlags
from newton._src.geometry.kernels import counter_increment, counter_increment_replay
from newton._src.geometry.types import GeoType
from newton._src.utils.heightfield import HeightfieldData

from .collision_particles import (
    collide_box_particle,
    collide_capsule_particle,
    collide_cone_particle,
    collide_cylinder_particle,
    collide_ellipsoid_particle,
    collide_heightfield_particle,
    collide_mesh_particle,
    collide_plane_particle,
    collide_sphere_particle,
)


@wp.kernel
def create_soft_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_world: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_world: wp.array(dtype=int),
    margin: float,
    soft_contact_max: int,
    shape_count: int,
    shape_flags: wp.array(dtype=wp.int32),
    shape_heightfield_data: wp.array(dtype=HeightfieldData),
    heightfield_elevation_data: wp.array(dtype=wp.float32),
    # outputs
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    soft_contact_tids: wp.array(dtype=int),
):
    tid = wp.tid()
    particle_index, shape_index = tid // shape_count, tid % shape_count
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    particle_world_id = particle_world[particle_index]
    shape_world_id = shape_world[shape_index]
    if particle_world_id != -1 and shape_world_id != -1 and particle_world_id != shape_world_id:
        return

    rigid_index = shape_body[shape_index]
    px = particle_q[particle_index]
    radius = particle_radius[particle_index]

    X_wb = wp.transform_identity()
    if rigid_index >= 0:
        X_wb = body_q[rigid_index]

    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)
    x_local = wp.transform_point(X_sw, px)

    geo_type = shape_type[shape_index]
    geo_scale = shape_scale[shape_index]

    contact_distance = 1.0e6
    contact_pos_local = wp.vec3()
    normal_local = wp.vec3()
    v = wp.vec3()

    if geo_type == GeoType.SPHERE:
        contact_distance, contact_pos_local, normal_local = collide_sphere_particle(x_local, geo_scale[0], radius)
    if geo_type == GeoType.BOX:
        contact_distance, contact_pos_local, normal_local = collide_box_particle(x_local, geo_scale, radius)
    if geo_type == GeoType.CAPSULE:
        contact_distance, contact_pos_local, normal_local = collide_capsule_particle(
            x_local, geo_scale[0], geo_scale[1], radius
        )
    if geo_type == GeoType.CYLINDER:
        contact_distance, contact_pos_local, normal_local = collide_cylinder_particle(
            x_local, geo_scale[0], geo_scale[1], radius
        )
    if geo_type == GeoType.CONE:
        contact_distance, contact_pos_local, normal_local = collide_cone_particle(
            x_local, geo_scale[0], geo_scale[1], radius
        )
    if geo_type == GeoType.ELLIPSOID:
        contact_distance, contact_pos_local, normal_local = collide_ellipsoid_particle(x_local, geo_scale, radius)

    if geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
        min_scale = wp.min(geo_scale)
        contact_distance, contact_pos_local, normal_local, v = collide_mesh_particle(
            shape_source_ptr[shape_index], x_local, geo_scale, radius, margin + radius / min_scale
        )

    if geo_type == GeoType.PLANE:
        contact_distance, contact_pos_local, normal_local = collide_plane_particle(
            x_local, geo_scale[0] * 0.5, geo_scale[1] * 0.5, radius
        )

    if geo_type == GeoType.HFIELD:
        hfd = shape_heightfield_data[shape_index]
        contact_distance, contact_pos_local, normal_local = collide_heightfield_particle(
            x_local, hfd, heightfield_elevation_data, radius
        )

    if contact_distance < margin:
        index = counter_increment(soft_contact_count, 0, soft_contact_tids, tid)
        if index < soft_contact_max:
            body_pos = wp.transform_point(X_bs, contact_pos_local)
            body_vel = wp.transform_vector(X_bs, v)
            world_normal = wp.transform_vector(X_ws, normal_local)
            soft_contact_shape[index] = shape_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal
