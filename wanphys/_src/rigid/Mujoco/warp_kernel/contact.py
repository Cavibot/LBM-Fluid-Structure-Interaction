# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact conversion kernels for the WanPhys MuJoCo adapter.

There are two dataflow directions in this module:
- external contacts prepared by WanPhys -> MuJoCo warp contact buffers
- MuJoCo warp solver contacts -> WanPhys-facing exported contact arrays
"""

from __future__ import annotations

import warp as wp

from .common import contact_params, make_frame, vec5, write_contact


@wp.kernel
def convert_newton_contacts_to_mjwarp_kernel(
    body_q: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    geom_condim: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    rigid_contact_count: wp.array(dtype=wp.int32),
    rigid_contact_shape0: wp.array(dtype=wp.int32),
    rigid_contact_shape1: wp.array(dtype=wp.int32),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    rigid_contact_margin0: wp.array(dtype=wp.float32),
    rigid_contact_margin1: wp.array(dtype=wp.float32),
    rigid_contact_stiffness: wp.array(dtype=wp.float32),
    rigid_contact_damping: wp.array(dtype=wp.float32),
    rigid_contact_friction_scale: wp.array(dtype=wp.float32),
    shape_margin: wp.array(dtype=float),
    bodies_per_world: int,
    newton_shape_to_mjc_geom: wp.array(dtype=wp.int32),
    naconmax: int,
    nacon_out: wp.array(dtype=int),
    contact_dist_out: wp.array(dtype=float),
    contact_pos_out: wp.array(dtype=wp.vec3),
    contact_frame_out: wp.array(dtype=wp.mat33),
    contact_includemargin_out: wp.array(dtype=float),
    contact_friction_out: wp.array(dtype=vec5),
    contact_solref_out: wp.array(dtype=wp.vec2),
    contact_solreffriction_out: wp.array(dtype=wp.vec2),
    contact_solimp_out: wp.array(dtype=vec5),
    contact_dim_out: wp.array(dtype=int),
    contact_geom_out: wp.array(dtype=wp.vec2i),
    contact_worldid_out: wp.array(dtype=int),
    nworld_in: int,
    ncollision_out: wp.array(dtype=int),
):
    """Translate externally prepared contact arrays into MuJoCo contact storage.

    Each thread handles one source contact. The kernel computes the world-space
    contact point/frame, looks up MuJoCo geom ids, merges per-geom contact
    parameters, then writes a MuJoCo-style contact record.
    """
    tid = wp.tid()
    count = rigid_contact_count[0]
    if tid == 0:
        if count > naconmax:
            wp.printf("Number of Newton contacts (%d) exceeded MJWarp limit (%d). Increase nconmax.\n", count, naconmax)
            count = naconmax
        nacon_out[0] = count
        ncollision_out[0] = 0
    if count > naconmax:
        count = naconmax
    if tid >= count:
        return
    shape_a = rigid_contact_shape0[tid]
    shape_b = rigid_contact_shape1[tid]
    if shape_a < 0 or shape_b < 0:
        return

    # Contacts are authored in the local frames of the attached bodies, so
    # recover world-space points from the current body transforms first.
    body_a = shape_body[shape_a]
    body_b = shape_body[shape_b]
    X_wb_a = wp.transform_identity()
    X_wb_b = wp.transform_identity()
    if body_a >= 0:
        X_wb_a = body_q[body_a]
    if body_b >= 0:
        X_wb_b = body_q[body_b]
    bx_a = wp.transform_point(X_wb_a, rigid_contact_point0[tid])
    bx_b = wp.transform_point(X_wb_b, rigid_contact_point1[tid])
    n = -rigid_contact_normal[tid]
    # rigid_contact_margin0/1 = radius_eff + shape_margin per shape.
    # Subtract only radius_eff so dist remains a surface-to-surface distance.
    # The per-shape margin is still handled by MuJoCo's includemargin path.
    radius_eff = (rigid_contact_margin0[tid] - shape_margin[shape_a]) + (
        rigid_contact_margin1[tid] - shape_margin[shape_b]
    )
    dist = wp.dot(n, bx_b - bx_a) - radius_eff
    pos = 0.5 * (bx_a + bx_b)
    frame = make_frame(n)
    geoms = wp.vec2i(newton_shape_to_mjc_geom[shape_a], newton_shape_to_mjc_geom[shape_b])

    # Static shapes are shared across worlds, so world assignment is recovered
    # from whichever shape is attached to a dynamic body.
    worldid = body_a // bodies_per_world
    if body_a < 0:
        worldid = body_b // bodies_per_world
    margin, gap, condim, friction, solref, solreffriction, solimp = contact_params(
        geom_condim, geom_priority, geom_solmix, geom_solref, geom_solimp, geom_friction, geom_margin, geom_gap, geoms, worldid
    )
    if rigid_contact_stiffness:
        # Optional per-contact overrides let the upstream contact pipeline tune
        # stiffness/damping/friction without rewriting MuJoCo geom materials.
        contact_ke = rigid_contact_stiffness[tid]
        if contact_ke > 0.0:
            imp = solimp[1]
            solimp = vec5(imp, imp, 0.001, 1.0, 0.5)
            contact_ke = contact_ke * (1.0 - imp)
            kd = rigid_contact_damping[tid]
            if kd > 0.0:
                timeconst = 2.0 / kd
                dampratio = wp.sqrt(1.0 / (timeconst * timeconst * contact_ke))
            else:
                timeconst = wp.sqrt(1.0 / contact_ke)
                dampratio = 1.0
            solref = wp.vec2(timeconst, dampratio)
        friction_scale = rigid_contact_friction_scale[tid]
        if friction_scale > 0.0:
            friction = vec5(friction[0] * friction_scale, friction[1] * friction_scale, friction[2], friction[3], friction[4])
    write_contact(
        dist_in=dist,
        pos_in=pos,
        frame_in=frame,
        margin_in=margin,
        gap_in=gap,
        condim_in=condim,
        friction_in=friction,
        solref_in=solref,
        solreffriction_in=solreffriction,
        solimp_in=solimp,
        geoms_in=geoms,
        worldid_in=worldid,
        contact_id_in=tid,
        contact_dist_out=contact_dist_out,
        contact_pos_out=contact_pos_out,
        contact_frame_out=contact_frame_out,
        contact_includemargin_out=contact_includemargin_out,
        contact_friction_out=contact_friction_out,
        contact_solref_out=contact_solref_out,
        contact_solreffriction_out=contact_solreffriction_out,
        contact_solimp_out=contact_solimp_out,
        contact_dim_out=contact_dim_out,
        contact_geom_out=contact_geom_out,
        contact_worldid_out=contact_worldid_out,
    )


@wp.kernel
def convert_mjw_contact_to_warp_kernel(
    mjc_geom_to_newton_shape: wp.array2d(dtype=wp.int32),
    pyramidal_cone: bool,
    mj_nacon: wp.array(dtype=wp.int32),
    mj_contact_frame: wp.array(dtype=wp.mat33f),
    mj_contact_dim: wp.array(dtype=int),
    mj_contact_geom: wp.array(dtype=wp.vec2i),
    mj_contact_efc_address: wp.array2d(dtype=int),
    mj_contact_worldid: wp.array(dtype=wp.int32),
    mj_efc_force: wp.array2d(dtype=float),
    rigid_contact_count: wp.array(dtype=wp.int32),
    rigid_contact_shape0: wp.array(dtype=wp.int32),
    rigid_contact_shape1: wp.array(dtype=wp.int32),
    rigid_contact_point0: wp.array(dtype=wp.vec3),
    rigid_contact_point1: wp.array(dtype=wp.vec3),
    rigid_contact_normal: wp.array(dtype=wp.vec3),
    contact_force: wp.array(dtype=wp.spatial_vector),
):
    """Export MuJoCo solver contacts into WanPhys-facing arrays.

    The output matches Newton's ``Contacts`` contract so WanPhys sensors and
    viewers can consume the data without MuJoCo-specific side channels.
    """
    contact_idx = wp.tid()

    n_contacts = mj_nacon[0]
    if contact_idx == 0:
        rigid_contact_count[0] = n_contacts

    if contact_idx >= n_contacts:
        return

    world = mj_contact_worldid[contact_idx]
    geoms_mjw = mj_contact_geom[contact_idx]

    normal = mj_contact_frame[contact_idx][0]
    rigid_contact_shape0[contact_idx] = mjc_geom_to_newton_shape[world, geoms_mjw[0]]
    rigid_contact_shape1[contact_idx] = mjc_geom_to_newton_shape[world, geoms_mjw[1]]
    rigid_contact_point0[contact_idx] = wp.vec3(0.0)
    rigid_contact_point1[contact_idx] = wp.vec3(0.0)
    rigid_contact_normal[contact_idx] = -normal

    if not contact_force:
        return

    normalforce = wp.float(-1.0)
    efc_address0 = mj_contact_efc_address[contact_idx, 0]
    if efc_address0 >= 0:
        normalforce = mj_efc_force[world, efc_address0]
        if pyramidal_cone:
            dim = mj_contact_dim[contact_idx]
            for i in range(1, 2 * (dim - 1)):
                normalforce += mj_efc_force[world, mj_contact_efc_address[contact_idx, i]]
    force = wp.where(normalforce > 0.0, -normalforce * normal, wp.vec3(0.0))
    contact_force[contact_idx] = wp.spatial_vector(force, wp.vec3(0.0))


@wp.kernel(enable_backward=False)
def _create_inverse_shape_mapping_kernel(
    mjc_geom_to_newton_shape: wp.array2d(dtype=wp.int32),
    newton_shape_to_mjc_geom: wp.array(dtype=wp.int32),
):
    """Build the shape -> geom lookup used when injecting external contacts."""
    world, geom_idx = wp.tid()
    source_shape_idx = mjc_geom_to_newton_shape[world, geom_idx]
    if source_shape_idx >= 0:
        newton_shape_to_mjc_geom[source_shape_idx] = geom_idx


__all__ = [
    "convert_newton_contacts_to_mjwarp_kernel",
    "convert_mjw_contact_to_warp_kernel",
    "_create_inverse_shape_mapping_kernel",
]
