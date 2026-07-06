# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from .utils.geometry_utils import (
    intersection_gradient_vector,
    triangle_normal,
    triangle_barycentric
)

########################################################################################################################
#################################################    Style3D Kernel   ##################################################
########################################################################################################################

@wp.kernel
def handle_edge_face_contacts_geometry_kernel(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=int, ndim=2),
    edge_indices: wp.array(dtype=int, ndim=2),
    broad_phase_ef: wp.array(dtype=int, ndim=2),
    max_contacts: int,
    # outputs
    contact_count: wp.array(dtype=int),
    contact_fid: wp.array(dtype=int, ndim=2),
    contact_dir: wp.array(dtype=wp.vec3, ndim=2),
    contact_bary: wp.array(dtype=wp.vec3, ndim=2),
    contact_edge_bary: wp.array(dtype=wp.vec2, ndim=2),
    contact_point: wp.array(dtype=wp.vec3, ndim=2),
):
    eid = wp.tid()
    edge = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])
    v0 = pos[edge[0]]
    v1 = pos[edge[1]]

    out_count = wp.int32(0)

    # Skip invalid edge
    len0 = wp.length(v0 - v1)
    if len0 < 5e-4:
        contact_count[eid] = out_count
        return

    # Edge direction
    E = wp.normalize(v0 - v1)
    N2 = wp.vec3(0.0) if edge[2] < 0 else triangle_normal(v0, v1, pos[edge[2]])
    N3 = wp.vec3(0.0) if edge[3] < 0 else triangle_normal(v0, v1, pos[edge[3]])

    count = wp.min(broad_phase_ef[0, eid], max_contacts)
    for i in range(count):
        fid = broad_phase_ef[i + 1, eid]
        face = wp.vec3i(tri_indices[fid, 0], tri_indices[fid, 1], tri_indices[fid, 2])

        if face[0] == edge[0] or face[0] == edge[1]:
            continue
        if face[1] == edge[0] or face[1] == edge[1]:
            continue
        if face[2] == edge[0] or face[2] == edge[1]:
            continue

        x0 = pos[face[0]]
        x1 = pos[face[1]]
        x2 = pos[face[2]]
        face_normal = wp.cross(x1 - x0, x2 - x1)
        normal_len = wp.length(face_normal)
        if normal_len < 1e-8:
            continue  # invalid triangle

        face_normal = wp.normalize(face_normal)
        d1 = wp.dot(face_normal, v0 - x0)
        d2 = wp.dot(face_normal, v1 - x0)
        if d1 * d2 >= 0.0:
            continue  # on same side

        d1, d2 = wp.abs(d1), wp.abs(d2)
        hit_point = (v0 * d2 + v1 * d1) / (d2 + d1)
        bary_coord = triangle_barycentric(x0, x1, x2, hit_point)

        if (bary_coord[0] < 1e-2) or (bary_coord[1] < 1e-2) or (bary_coord[2] < 1e-2):
            continue  # hit outside

        G = wp.vec3(0.0)

        if edge[2] >= 0:
            R = wp.cross(face_normal, N2)
            R = wp.vec3(0.0) if wp.length(R) < 1e-6 else wp.normalize(R)
            if wp.dot(wp.cross(E, R), wp.cross(E, pos[edge[2]] - hit_point)) < 0.0:
                R *= -1.0
            G += intersection_gradient_vector(R, E, face_normal)

        if edge[3] >= 0:
            R = wp.cross(face_normal, N3)
            R = wp.vec3(0.0) if wp.length(R) < 1e-6 else wp.normalize(R)
            if wp.dot(wp.cross(E, R), wp.cross(E, pos[edge[3]] - hit_point)) < 0.0:
                R *= -1.0
            G += intersection_gradient_vector(R, E, face_normal)

        if wp.length(G) < 1.0e-12:
            continue
        G = wp.normalize(G)

        contact_fid[out_count, eid] = fid
        contact_dir[out_count, eid] = G
        contact_bary[out_count, eid] = bary_coord
        contact_edge_bary[out_count, eid] = wp.vec2(d2, d1) / (d1 + d2)
        contact_point[out_count, eid] = hit_point
        out_count += 1

    contact_count[eid] = out_count


@wp.kernel
def handle_edge_face_contacts_force_kernel(
    thickness: float,
    stiff_factor: float,
    tri_indices: wp.array(dtype=int, ndim=2),
    edge_indices: wp.array(dtype=int, ndim=2),
    static_diags: wp.array(dtype=float),
    max_contacts: int,
    contact_count: wp.array(dtype=int),
    contact_fid: wp.array(dtype=int, ndim=2),
    contact_dir: wp.array(dtype=wp.vec3, ndim=2),
    contact_bary: wp.array(dtype=wp.vec3, ndim=2),
    contact_edge_bary: wp.array(dtype=wp.vec2, ndim=2),
    # outputs
    forces: wp.array(dtype=wp.vec3),
    hessian_diags: wp.array(dtype=wp.mat33),
):
    eid = wp.tid()
    edge = wp.vec4i(edge_indices[eid, 2], edge_indices[eid, 3], edge_indices[eid, 0], edge_indices[eid, 1])

    force0 = wp.vec3(0.0)
    force1 = wp.vec3(0.0)
    hess0 = wp.identity(n=3, dtype=float) * 0.0
    hess1 = wp.identity(n=3, dtype=float) * 0.0
    stiff_0 = (static_diags[edge[0]] + static_diags[edge[1]]) / 2.0
    is_collided = wp.int32(0)

    count = wp.min(contact_count[eid], max_contacts)
    for i in range(count):
        fid = contact_fid[i, eid]
        face = wp.vec3i(tri_indices[fid, 0], tri_indices[fid, 1], tri_indices[fid, 2])
        bary_coord = contact_bary[i, eid]
        edge_bary = contact_edge_bary[i, eid]
        G = contact_dir[i, eid]

        stiff_1 = (static_diags[face[0]] + static_diags[face[1]] + static_diags[face[2]]) / 3.0
        stiff = stiff_factor * (stiff_0 * stiff_1) / (stiff_0 + stiff_1)
        disp = 2.0 * thickness

        force = stiff * G * disp
        hess = stiff * wp.outer(G, G)

        force0 += force * edge_bary[0]
        force1 += force * edge_bary[1]
        hess0 += hess * edge_bary[0] * edge_bary[0]
        hess1 += hess * edge_bary[1] * edge_bary[1]

        wp.atomic_add(forces, face[0], -force * bary_coord[0])
        wp.atomic_add(forces, face[1], -force * bary_coord[1])
        wp.atomic_add(forces, face[2], -force * bary_coord[2])

        wp.atomic_add(hessian_diags, face[0], hess * bary_coord[0] * bary_coord[0])
        wp.atomic_add(hessian_diags, face[1], hess * bary_coord[1] * bary_coord[1])
        wp.atomic_add(hessian_diags, face[2], hess * bary_coord[2] * bary_coord[2])
        is_collided = 1

    if is_collided != 0:
        wp.atomic_add(forces, edge[0], force0)
        wp.atomic_add(forces, edge[1], force1)
        wp.atomic_add(hessian_diags, edge[0], hess0)
        wp.atomic_add(hessian_diags, edge[1], hess1)