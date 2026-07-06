# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys-owned type definitions for the SemiImplicit solver.

Replaces direct newton imports so the SemiImplicit package has no
dependency on newton's internal type hierarchy.
"""

from __future__ import annotations

from enum import IntEnum

import warp as wp
from warp import DeviceLike as Devicelike

try:
    from typing import override
except ImportError:
    try:
        from typing_extensions import override
    except ImportError:
        # Fallback no-op decorator if override is not available
        def override(func):
            return func
# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


class ParticleFlags(IntEnum):
    """Flags for particle properties."""

    ACTIVE = 1 << 0
    """Indicates that the particle is active."""


# ---------------------------------------------------------------------------
# Joint types
# ---------------------------------------------------------------------------


class JointType(IntEnum):
    """Enumeration of joint types."""

    PRISMATIC = 0
    REVOLUTE = 1
    BALL = 2
    FIXED = 3
    FREE = 4
    DISTANCE = 5
    D6 = 6
    CABLE = 7


# ---------------------------------------------------------------------------
# Quaternion utilities (inlined from newton._src.core.spatial)
# ---------------------------------------------------------------------------


@wp.func
def quat_twist(axis: wp.vec3, q: wp.quat):
    """Return the twist component of a quaternion around an axis."""
    a = wp.vec3(q[0], q[1], q[2])
    proj = wp.dot(a, axis)
    a = proj * axis
    return wp.normalize(wp.quat(a[0], a[1], a[2], q[3]))


@wp.func
def quat_decompose(q: wp.quat):
    """Decompose a quaternion into extrinsic XYZ Euler angles."""
    R = wp.matrix_from_cols(
        wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)),
    )
    phi = wp.atan2(R[1, 2], R[2, 2])
    sinp = -R[0, 2]
    if wp.abs(sinp) >= 1.0:
        theta = wp.HALF_PI * wp.sign(sinp)
    else:
        theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])
    return -wp.vec3(phi, theta, psi)


# ---------------------------------------------------------------------------
# Geometry utilities (inlined from newton._src.geometry.kernels)
# ---------------------------------------------------------------------------


@wp.func
def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    """Return barycentric coordinates of the closest point on triangle (a,b,c) to p."""
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return wp.vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return wp.vec3(0.0, 1.0 - w, w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.vec3(1.0 - v - w, v, w)
