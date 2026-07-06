# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle-shape collision functions.

Conventions match :mod:`collision_primitive`: ``collide_A_B`` returns a normal
from A into B. SDF-style particle contacts return the shape surface point rather
than a midpoint.

The shape-particle SDF functions take the particle center in shape-local space
and return the contact point and normal in shape-local space.
"""

import warp as wp

from newton._src.core.types import MAXVAL
from newton._src.core.types import Axis
from newton._src.geometry.kernels import (
    sdf_box,
    sdf_box_grad,
    sdf_capsule,
    sdf_capsule_grad,
    sdf_cone,
    sdf_cone_grad,
    sdf_cylinder,
    sdf_cylinder_grad,
    sdf_ellipsoid,
    sdf_ellipsoid_grad,
    sdf_plane,
    sdf_plane_grad,
    sdf_sphere,
    sdf_sphere_grad,
    triangle_closest_point,
)
from newton._src.math import normalize_with_norm
from newton._src.utils.heightfield import HeightfieldData, sample_sdf_grad_heightfield


@wp.func
def _shape_particle_contact_from_local(
    particle_pos_local: wp.vec3,
    particle_radius: float,
    shape_dist: float,
    shape_normal_local: wp.vec3,
) -> tuple[float, wp.vec3, wp.vec3]:
    contact_distance = shape_dist - particle_radius
    contact_position_local = particle_pos_local - shape_normal_local * shape_dist
    return contact_distance, contact_position_local, shape_normal_local


@wp.func
def _triangle_fallback_normal(tri_a: wp.vec3, tri_b: wp.vec3, tri_c: wp.vec3, toward: wp.vec3) -> wp.vec3:
    normal, normal_len = normalize_with_norm(wp.cross(tri_b - tri_a, tri_c - tri_a))
    if normal_len <= 1.0e-8:
        normal = wp.vec3(1.0, 0.0, 0.0)

    tri_center = (tri_a + tri_b + tri_c) / 3.0
    if wp.dot(normal, toward - tri_center) < 0.0:
        normal = -normal

    return normal


@wp.func
def collide_triangle_particle(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    particle_pos: wp.vec3,
    particle_radius: float,
) -> tuple[float, wp.vec3, wp.vec3]:
    """Whole-triangle vs particle closest-point collision calculation.

    The particle is queried by its center, and particle_radius offsets the
    reported distance. The normal points from the triangle into the particle.
    """

    closest, _bary, _feature_type = triangle_closest_point(tri_a, tri_b, tri_c, particle_pos)
    delta = particle_pos - closest
    normal, shape_dist = normalize_with_norm(delta)

    if shape_dist <= 1.0e-8:
        normal = _triangle_fallback_normal(tri_a, tri_b, tri_c, particle_pos)

    contact_distance = shape_dist - particle_radius
    contact_position = closest

    return contact_distance, contact_position, normal


@wp.func
def collide_sphere_particle(
    particle_pos_local: wp.vec3,
    sphere_radius: float,
    particle_radius: float,
) -> tuple[float, wp.vec3, wp.vec3]:
    """Sphere-particle SDF-style collision calculation."""

    shape_dist = sdf_sphere(particle_pos_local, sphere_radius)
    normal_local = sdf_sphere_grad(particle_pos_local, sphere_radius)
    return _shape_particle_contact_from_local(particle_pos_local, particle_radius, shape_dist, normal_local)


@wp.func
def collide_box_particle(
    particle_pos_local: wp.vec3,
    box_size: wp.vec3,
    particle_radius: float,
) -> tuple[float, wp.vec3, wp.vec3]:
    """Box-particle SDF-style collision calculation."""

    shape_dist = sdf_box(particle_pos_local, box_size[0], box_size[1], box_size[2])
    normal_local = sdf_box_grad(particle_pos_local, box_size[0], box_size[1], box_size[2])
    return _shape_particle_contact_from_local(particle_pos_local, particle_radius, shape_dist, normal_local)


@wp.func
def collide_capsule_particle(
    particle_pos_local: wp.vec3,
    capsule_radius: float,
    capsule_half_length: float,
    particle_radius: float,
) -> tuple[float, wp.vec3, wp.vec3]:
    """Capsule-particle SDF-style collision calculation."""

    shape_dist = sdf_capsule(particle_pos_local, capsule_radius, capsule_half_length, int(Axis.Z))
    normal_local = sdf_capsule_grad(particle_pos_local, capsule_radius, capsule_half_length, int(Axis.Z))
    return _shape_particle_contact_from_local(particle_pos_local, particle_radius, shape_dist, normal_local)


@wp.func
def collide_cylinder_particle(
    particle_pos_local: wp.vec3,
    cylinder_radius: float,
    cylinder_half_height: float,
    particle_radius: float,
) -> tuple[float, wp.vec3, wp.vec3]:
    """Cylinder-particle SDF-style collision calculation."""

    shape_dist = sdf_cylinder(particle_pos_local, cylinder_radius, cylinder_half_height, int(Axis.Z))
    normal_local = sdf_cylinder_grad(particle_pos_local, cylinder_radius, cylinder_half_height, int(Axis.Z))
    return _shape_particle_contact_from_local(particle_pos_local, particle_radius, shape_dist, normal_local)


@wp.func
def collide_cone_particle(
    particle_pos_local: wp.vec3,
    cone_radius: float,
    cone_half_height: float,
    particle_radius: float,
) -> tuple[float, wp.vec3, wp.vec3]:
    """Cone-particle SDF-style collision calculation."""

    shape_dist = sdf_cone(particle_pos_local, cone_radius, cone_half_height, int(Axis.Z))
    normal_local = sdf_cone_grad(particle_pos_local, cone_radius, cone_half_height, int(Axis.Z))
    return _shape_particle_contact_from_local(particle_pos_local, particle_radius, shape_dist, normal_local)


@wp.func
def collide_ellipsoid_particle(
    particle_pos_local: wp.vec3,
    ellipsoid_size: wp.vec3,
    particle_radius: float,
) -> tuple[float, wp.vec3, wp.vec3]:
    """Ellipsoid-particle SDF-style collision calculation."""

    shape_dist = sdf_ellipsoid(particle_pos_local, ellipsoid_size)
    normal_local = sdf_ellipsoid_grad(particle_pos_local, ellipsoid_size)
    return _shape_particle_contact_from_local(particle_pos_local, particle_radius, shape_dist, normal_local)


@wp.func
def collide_plane_particle(
    particle_pos_local: wp.vec3,
    plane_width: float,
    plane_length: float,
    particle_radius: float,
) -> tuple[float, wp.vec3, wp.vec3]:
    """Plane-particle SDF-style collision calculation."""

    shape_dist = sdf_plane(particle_pos_local, plane_width, plane_length)
    normal_local = sdf_plane_grad(particle_pos_local, plane_width, plane_length)
    return _shape_particle_contact_from_local(particle_pos_local, particle_radius, shape_dist, normal_local)


@wp.func
def collide_mesh_particle(
    mesh: wp.uint64,
    particle_pos_local: wp.vec3,
    mesh_scale: wp.vec3,
    particle_radius: float,
    max_dist: float,
) -> tuple[float, wp.vec3, wp.vec3, wp.vec3]:
    """Mesh-particle SDF-style collision calculation."""

    face_index = int(0)
    face_u = float(0.0)
    face_v = float(0.0)
    sign = float(0.0)

    if wp.mesh_query_point_sign_normal(
        mesh, wp.cw_div(particle_pos_local, mesh_scale), max_dist, sign, face_index, face_u, face_v
    ):
        shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
        shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)
        shape_p = wp.cw_mul(shape_p, mesh_scale)
        shape_v = wp.cw_mul(shape_v, mesh_scale)
        delta = particle_pos_local - shape_p
        shape_dist = wp.length(delta) * sign
        normal_local = wp.normalize(delta) * sign
        contact_distance, contact_position_local, normal_local = _shape_particle_contact_from_local(
            particle_pos_local, particle_radius, shape_dist, normal_local
        )
        return contact_distance, contact_position_local, normal_local, shape_v

    return MAXVAL, wp.vec3(), wp.vec3(), wp.vec3()


@wp.func
def collide_heightfield_particle(
    particle_pos_local: wp.vec3,
    heightfield_data: HeightfieldData,
    heightfield_elevation_data: wp.array(dtype=wp.float32),
    particle_radius: float,
) -> tuple[float, wp.vec3, wp.vec3]:
    """Heightfield-particle SDF-style collision calculation."""

    if heightfield_data.nrow <= 1 or heightfield_data.ncol <= 1:
        return MAXVAL, wp.vec3(), wp.vec3()

    shape_dist, normal_local = sample_sdf_grad_heightfield(
        heightfield_data, heightfield_elevation_data, particle_pos_local
    )
    return _shape_particle_contact_from_local(particle_pos_local, particle_radius, shape_dist, normal_local)
