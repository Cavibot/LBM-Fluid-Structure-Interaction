# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Triangle collision functions.

Triangle-particle is implemented in :mod:`wanphys._src.collision.particles.collision_particles`. Rigid
triangle-shape contacts use the same support-map GJK/MPR manifold path as mesh
triangles.
"""

import warp as wp

from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.support_function import (
    GenericShapeData,
    GeoTypeEx,
    SupportMapDataProvider,
    pack_mesh_ptr,
    support_map,
)
from newton._src.geometry.types import GeoType
from newton._src.math import normalize_with_norm

from .collision_convex import create_solve_convex_multi_contact_values
from .collision_core import post_process_axial_on_discrete_contact


# Rigid Triangle Simulation

_SMALL_ROUNDED_RADIUS = 0.0001
_vec5 = wp.types.vector(5, wp.float32)
_mat53f = wp.types.matrix((5, 3), wp.float32)

@wp.func
def _make_triangle_shape(tri_a: wp.vec3, tri_b: wp.vec3, tri_c: wp.vec3) -> GenericShapeData:
    shape = GenericShapeData()
    shape.shape_type = int(GeoTypeEx.TRIANGLE)
    shape.scale = tri_b - tri_a
    shape.auxiliary = tri_c - tri_a
    return shape


@wp.func
def _make_axis_quat(axis: wp.vec3) -> wp.quat:
    axis_n, axis_len = normalize_with_norm(axis)
    if axis_len <= 1.0e-8:
        return wp.quat_identity()

    local_z = wp.vec3(0.0, 0.0, 1.0)
    cross = wp.cross(local_z, axis_n)
    cross_len = wp.length(cross)
    dot = wp.clamp(wp.dot(local_z, axis_n), -1.0, 1.0)

    if cross_len <= 1.0e-8:
        if dot < 0.0:
            return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi)
        return wp.quat_identity()

    return wp.quat_from_axis_angle(cross / cross_len, wp.atan2(cross_len, dot))


@wp.func
def _collide_triangle_convex_values(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    shape_b_data: GenericShapeData,
    pos_b: wp.vec3,
    quat_b: wp.quat,
    radius_eff_b: float,
    gap: float,
    skip_manifold: bool,
    shape_a: int,
    shape_b: int,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    shape_a_data = _make_triangle_shape(tri_a, tri_b, tri_c)
    quat_a = wp.quat_identity()
    pos_a = tri_a

    contact_template = ContactData()
    contact_template.radius_eff_a = 0.0
    contact_template.radius_eff_b = radius_eff_b
    contact_template.margin_a = 0.0
    contact_template.margin_b = 0.0
    contact_template.shape_a = shape_a
    contact_template.shape_b = shape_b
    contact_template.gap_sum = gap

    data_provider = SupportMapDataProvider()

    return wp.static(
        create_solve_convex_multi_contact_values(support_map, post_process_axial_on_discrete_contact)
    )(
        shape_a_data,
        shape_b_data,
        quat_a,
        quat_b,
        pos_a,
        pos_b,
        0.0,
        data_provider,
        gap + radius_eff_b,
        skip_manifold,
        contact_template,
    )


@wp.func
def collide_triangle_sphere(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    sphere_pos: wp.vec3,
    sphere_radius: float,
    gap: float = 0.0,
    shape_a: int = -1,
    shape_b: int = -1,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    sphere_data = GenericShapeData()
    sphere_data.shape_type = int(GeoType.SPHERE)
    sphere_data.scale = wp.vec3(_SMALL_ROUNDED_RADIUS, 0.0, 0.0)
    sphere_data.auxiliary = wp.vec3()

    return _collide_triangle_convex_values(
        tri_a,
        tri_b,
        tri_c,
        sphere_data,
        sphere_pos,
        wp.quat_identity(),
        sphere_radius,
        gap,
        True,
        shape_a,
        shape_b,
    )


@wp.func
def collide_triangle_triangle(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    other_tri_a: wp.vec3,
    other_tri_b: wp.vec3,
    other_tri_c: wp.vec3,
    gap: float = 0.0,
    shape_a: int = -1,
    shape_b: int = -1,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    other_triangle_data = _make_triangle_shape(other_tri_a, other_tri_b, other_tri_c)

    return _collide_triangle_convex_values(
        tri_a,
        tri_b,
        tri_c,
        other_triangle_data,
        other_tri_a,
        wp.quat_identity(),
        0.0,
        gap,
        True,
        shape_a,
        shape_b,
    )


@wp.func
def collide_triangle_capsule(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    capsule_pos: wp.vec3,
    capsule_axis: wp.vec3,
    capsule_radius: float,
    capsule_half_length: float,
    gap: float = 0.0,
    shape_a: int = -1,
    shape_b: int = -1,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    capsule_data = GenericShapeData()
    capsule_data.shape_type = int(GeoType.CAPSULE)
    capsule_data.scale = wp.vec3(_SMALL_ROUNDED_RADIUS, capsule_half_length, 0.0)
    capsule_data.auxiliary = wp.vec3()

    return _collide_triangle_convex_values(
        tri_a,
        tri_b,
        tri_c,
        capsule_data,
        capsule_pos,
        _make_axis_quat(capsule_axis),
        capsule_radius,
        gap,
        False,
        shape_a,
        shape_b,
    )


@wp.func
def collide_triangle_box(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    box_pos: wp.vec3,
    box_quat: wp.quat,
    box_size: wp.vec3,
    gap: float = 0.0,
    shape_a: int = -1,
    shape_b: int = -1,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    box_data = GenericShapeData()
    box_data.shape_type = int(GeoType.BOX)
    box_data.scale = box_size
    box_data.auxiliary = wp.vec3()

    return _collide_triangle_convex_values(
        tri_a,
        tri_b,
        tri_c,
        box_data,
        box_pos,
        box_quat,
        0.0,
        gap,
        False,
        shape_a,
        shape_b,
    )


@wp.func
def collide_triangle_ellipsoid(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    ellipsoid_pos: wp.vec3,
    ellipsoid_quat: wp.quat,
    ellipsoid_size: wp.vec3,
    gap: float = 0.0,
    shape_a: int = -1,
    shape_b: int = -1,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    ellipsoid_data = GenericShapeData()
    ellipsoid_data.shape_type = int(GeoType.ELLIPSOID)
    ellipsoid_data.scale = ellipsoid_size
    ellipsoid_data.auxiliary = wp.vec3()

    return _collide_triangle_convex_values(
        tri_a,
        tri_b,
        tri_c,
        ellipsoid_data,
        ellipsoid_pos,
        ellipsoid_quat,
        0.0,
        gap,
        True,
        shape_a,
        shape_b,
    )


@wp.func
def collide_triangle_cylinder(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    cylinder_pos: wp.vec3,
    cylinder_axis: wp.vec3,
    cylinder_radius: float,
    cylinder_half_height: float,
    gap: float = 0.0,
    shape_a: int = -1,
    shape_b: int = -1,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    cylinder_data = GenericShapeData()
    cylinder_data.shape_type = int(GeoType.CYLINDER)
    cylinder_data.scale = wp.vec3(cylinder_radius, cylinder_half_height, 0.0)
    cylinder_data.auxiliary = wp.vec3()

    return _collide_triangle_convex_values(
        tri_a,
        tri_b,
        tri_c,
        cylinder_data,
        cylinder_pos,
        _make_axis_quat(cylinder_axis),
        0.0,
        gap,
        False,
        shape_a,
        shape_b,
    )


@wp.func
def collide_triangle_cone(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    cone_pos: wp.vec3,
    cone_axis: wp.vec3,
    cone_radius: float,
    cone_half_height: float,
    gap: float = 0.0,
    shape_a: int = -1,
    shape_b: int = -1,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    cone_data = GenericShapeData()
    cone_data.shape_type = int(GeoType.CONE)
    cone_data.scale = wp.vec3(cone_radius, cone_half_height, 0.0)
    cone_data.auxiliary = wp.vec3()

    return _collide_triangle_convex_values(
        tri_a,
        tri_b,
        tri_c,
        cone_data,
        cone_pos,
        _make_axis_quat(cone_axis),
        0.0,
        gap,
        False,
        shape_a,
        shape_b,
    )


@wp.func
def collide_triangle_plane(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    plane_pos: wp.vec3,
    plane_quat: wp.quat,
    plane_size: wp.vec2,
    gap: float = 0.0,
    shape_a: int = -1,
    shape_b: int = -1,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    plane_data = GenericShapeData()
    plane_data.shape_type = int(GeoType.PLANE)
    plane_data.scale = wp.vec3(plane_size[0], plane_size[1], 0.0)
    plane_data.auxiliary = wp.vec3()

    return _collide_triangle_convex_values(
        tri_a,
        tri_b,
        tri_c,
        plane_data,
        plane_pos,
        plane_quat,
        0.0,
        gap,
        False,
        shape_a,
        shape_b,
    )


@wp.func
def collide_triangle_convex_mesh(
    tri_a: wp.vec3,
    tri_b: wp.vec3,
    tri_c: wp.vec3,
    mesh_pos: wp.vec3,
    mesh_quat: wp.quat,
    mesh_scale: wp.vec3,
    mesh_id: wp.uint64,
    gap: float = 0.0,
    shape_a: int = -1,
    shape_b: int = -1,
) -> tuple[int, _vec5, _vec5, _mat53f, _mat53f, _mat53f]:
    mesh_data = GenericShapeData()
    mesh_data.shape_type = int(GeoType.CONVEX_MESH)
    mesh_data.scale = mesh_scale
    mesh_data.auxiliary = pack_mesh_ptr(mesh_id)

    return _collide_triangle_convex_values(
        tri_a,
        tri_b,
        tri_c,
        mesh_data,
        mesh_pos,
        mesh_quat,
        0.0,
        gap,
        False,
        shape_a,
        shape_b,
    )
