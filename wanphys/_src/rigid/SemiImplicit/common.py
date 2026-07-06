# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared type definitions and geometry utilities for the symplectic solver.

This module centralises flag constants, articulation (joint) type enumerations,
quaternion helpers, and triangle geometry routines so that every sub-module can
import from a single place without circular dependencies.
"""

from __future__ import annotations

from enum import IntEnum

import warp as wp


# ---------------------------------------------------------------------------
# Particle activity flags
# ---------------------------------------------------------------------------


class ParticleFlag(IntEnum):
    """Bit-flags that annotate per-particle properties."""

    ACTIVE = 1 << 0
    """The particle participates in the simulation."""


# ---------------------------------------------------------------------------
# Articulation (joint / constraint) types
# ---------------------------------------------------------------------------


class ArticulationType(IntEnum):
    """Enumeration of supported articulation constraint types."""

    PRISMATIC = 0
    REVOLUTE = 1
    BALL = 2
    FIXED = 3
    FREE = 4
    DISTANCE = 5
    D6 = 6
    CABLE = 7


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


@wp.func
def quat_twist_component(axis: wp.vec3, q: wp.quat):
    """Extract the twist part of *q* about *axis* (swing-twist decomposition).

    Given a unit quaternion *q* and a reference *axis*, returns a quaternion
    that represents only the rotation component around *axis*.
    """
    imag = wp.vec3(q[0], q[1], q[2])
    proj = wp.dot(imag, axis)
    twist_imag = proj * axis
    return wp.normalize(wp.quat(twist_imag[0], twist_imag[1], twist_imag[2], q[3]))


@wp.func
def quat_to_euler_xyz(q: wp.quat):
    """Decompose quaternion *q* into extrinsic XYZ Euler angles (radians).

    Returns a ``wp.vec3(roll, pitch, yaw)`` following the extrinsic
    rotation convention.
    """
    R = wp.matrix_from_cols(
        wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)),
        wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)),
    )
    phi = wp.atan2(R[1, 2], R[2, 2])
    sin_pitch = -R[0, 2]
    if wp.abs(sin_pitch) >= 1.0:
        theta = wp.HALF_PI * wp.sign(sin_pitch)
    else:
        theta = wp.asin(-R[0, 2])
    psi = wp.atan2(R[0, 1], R[0, 0])
    return -wp.vec3(phi, theta, psi)


# ---------------------------------------------------------------------------
# Triangle geometry
# ---------------------------------------------------------------------------


@wp.func
def triangle_closest_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    """Barycentric coordinates of the closest point on triangle *abc* to *p*.

    Returns ``wp.vec3(u, v, w)`` such that ``closest = u*a + v*b + w*c``.
    Handles all vertex / edge / face Voronoi regions.
    """
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
