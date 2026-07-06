# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared types, helpers, and small utility kernels for MuJoCo warp kernels.

This module holds the tiny pieces that several kernel families depend on:
- quaternion storage-order conversion between Warp and MuJoCo
- contact-frame construction and contact-parameter merging
- small tiling utilities used after compile-time world expansion
"""

from __future__ import annotations

from typing import Any

import warp as wp

from newton._src.core.types import vec5

vec10 = wp.types.vector(length=10, dtype=wp.float32)
MJ_MINVAL = 2.220446049250313e-16


@wp.func
def orthogonals(a: wp.vec3):
    """Build two unit vectors orthogonal to ``a``.

    MuJoCo contact frames store a normal plus two tangent directions.  This
    helper chooses a stable fallback axis and constructs an orthonormal basis.
    """
    y = wp.vec3(0.0, 1.0, 0.0)
    z = wp.vec3(0.0, 0.0, 1.0)
    b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
    b = b - a * wp.dot(a, b)
    b = wp.normalize(b)
    if wp.length(a) == 0.0:
        b = wp.vec3(0.0, 0.0, 0.0)
    c = wp.cross(a, b)
    return b, c


@wp.func
def make_frame(a: wp.vec3):
    """Build a contact frame whose first axis is aligned with ``a``."""
    a = wp.normalize(a)
    b, c = orthogonals(a)
    # fmt: off
    return wp.mat33(
        a.x, a.y, a.z,
        b.x, b.y, b.z,
        c.x, c.y, c.z
    )
    # fmt: on


@wp.func
def mj_quat_to_warp(q: wp.quat):
    """Convert MuJoCo quaternion storage order ``wxyz`` into Warp ``xyzw``."""
    return wp.quat(q[1], q[2], q[3], q[0])


@wp.func
def warp_quat_to_mj(q: wp.quat):
    """Convert Warp quaternion storage order ``xyzw`` into MuJoCo ``wxyz``."""
    return wp.quat(q[3], q[0], q[1], q[2])


@wp.func
def decode_actuator_axis(raw_value: int):
    """Decode the signed actuator mapping produced by the compiler.

    Encoding:
    - ``-1``: no source axis is mapped to this actuator
    - ``>= 0``: position actuator for that source axis
    - ``< -1``: velocity actuator, decoded by ``-(raw + 2)``
    """
    if raw_value == -1:
        return -1, False, False
    if raw_value >= 0:
        return raw_value, False, True
    return -raw_value - 2, True, True


@wp.func
def write_contact(
    dist_in: float,
    pos_in: wp.vec3,
    frame_in: wp.mat33,
    margin_in: float,
    gap_in: float,
    condim_in: int,
    friction_in: vec5,
    solref_in: wp.vec2f,
    solreffriction_in: wp.vec2f,
    solimp_in: vec5,
    geoms_in: wp.vec2i,
    worldid_in: int,
    contact_id_in: int,
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
):
    """Write one fully prepared contact record into MuJoCo warp buffers."""
    cid = contact_id_in
    contact_dist_out[cid] = dist_in
    contact_pos_out[cid] = pos_in
    contact_frame_out[cid] = frame_in
    contact_geom_out[cid] = geoms_in
    contact_worldid_out[cid] = worldid_in
    contact_includemargin_out[cid] = margin_in - gap_in
    contact_dim_out[cid] = condim_in
    contact_friction_out[cid] = friction_in
    contact_solref_out[cid] = solref_in
    contact_solreffriction_out[cid] = solreffriction_in
    contact_solimp_out[cid] = solimp_in


@wp.func
def contact_params(
    geom_condim: wp.array(dtype=int),
    geom_priority: wp.array(dtype=int),
    geom_solmix: wp.array2d(dtype=float),
    geom_solref: wp.array2d(dtype=wp.vec2),
    geom_solimp: wp.array2d(dtype=vec5),
    geom_friction: wp.array2d(dtype=wp.vec3),
    geom_margin: wp.array2d(dtype=float),
    geom_gap: wp.array2d(dtype=float),
    geoms: wp.vec2i,
    worldid: int,
):
    """Merge two geom material/contact records into one effective contact.

    MuJoCo combines per-geom parameters such as friction, condim, margin, and
    solver settings into per-contact values.  This helper mirrors that logic
    so externally supplied contacts can be written in MuJoCo's native layout.
    """
    g1 = geoms[0]
    g2 = geoms[1]
    p1 = geom_priority[g1]
    p2 = geom_priority[g2]
    solmix1 = geom_solmix[worldid, g1]
    solmix2 = geom_solmix[worldid, g2]
    mix = solmix1 / (solmix1 + solmix2)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 < MJ_MINVAL), 0.5, mix)
    mix = wp.where((solmix1 < MJ_MINVAL) and (solmix2 >= MJ_MINVAL), 0.0, mix)
    mix = wp.where((solmix1 >= MJ_MINVAL) and (solmix2 < MJ_MINVAL), 1.0, mix)
    mix = wp.where(p1 == p2, mix, wp.where(p1 > p2, 1.0, 0.0))
    margin = wp.max(geom_margin[worldid, g1], geom_margin[worldid, g2])
    gap = wp.max(geom_gap[worldid, g1], geom_gap[worldid, g2])
    condim1 = geom_condim[g1]
    condim2 = geom_condim[g2]
    condim = wp.where(p1 == p2, wp.max(condim1, condim2), wp.where(p1 > p2, condim1, condim2))
    max_geom_friction = wp.max(geom_friction[worldid, g1], geom_friction[worldid, g2])
    friction = vec5(max_geom_friction[0], max_geom_friction[0], max_geom_friction[1], max_geom_friction[2], max_geom_friction[2])
    if geom_solref[worldid, g1].x > 0.0 and geom_solref[worldid, g2].x > 0.0:
        solref = mix * geom_solref[worldid, g1] + (1.0 - mix) * geom_solref[worldid, g2]
    else:
        solref = wp.min(geom_solref[worldid, g1], geom_solref[worldid, g2])
    solreffriction = wp.vec2(0.0, 0.0)
    solimp = mix * geom_solimp[worldid, g1] + (1.0 - mix) * geom_solimp[worldid, g2]
    return margin, gap, condim, friction, solref, solreffriction, solimp


@wp.func
def convert_solref(ke: float, kd: float, d_width: float, d_r: float) -> wp.vec2:
    """Convert stiffness/damping into MuJoCo ``solref`` parameters."""
    if ke > 0.0 and kd > 0.0:
        timeconst = 2.0 / (kd * d_width)
        dampratio = kd / 2.0 * wp.sqrt(d_r / ke)
    else:
        timeconst = 0.02
        dampratio = 1.0
    return wp.vec2(timeconst, dampratio)


@wp.kernel(module="unique", enable_backward=False)
def repeat_array_kernel(
    src: wp.array(dtype=Any),
    nelems_per_world: int,
    dst: wp.array(dtype=Any),
):
    """Repeat a template-world array across worlds after compile-time expansion."""
    tid = wp.tid()
    src_idx = tid % nelems_per_world
    dst[tid] = src[src_idx]


__all__ = [
    "MJ_MINVAL",
    "vec5",
    "vec10",
    "orthogonals",
    "make_frame",
    "mj_quat_to_warp",
    "warp_quat_to_mj",
    "decode_actuator_axis",
    "write_contact",
    "contact_params",
    "convert_solref",
    "repeat_array_kernel",
]
