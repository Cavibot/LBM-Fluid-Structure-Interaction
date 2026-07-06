# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Muscle / tendon actuator force kernels.

Physics
-------
A muscle is modelled as a series of straight-line segments connecting
ordered *waypoints* that are rigidly attached to body frames.  Each
segment (a, b) spanning two *different* bodies contributes a tensile
force along the unit vector from a to b:

    d̂  = (p_b - p_a) / ‖p_b - p_a‖
    F   = activation · d̂

Newton's third law: equal-and-opposite wrenches are applied to the two
bodies.  The moment arm for each wrench is the world-space waypoint
position (origin convention matches the rigid-body wrench accumulator).

Waypoints are stored in body-local coordinates relative to the body
origin (not the CoM).  The world-space position is therefore:

    p = transform_point(body_tf, waypoint - body_com)
"""

from __future__ import annotations

import warp as wp


# =====================================================================
# Geometry helpers
# =====================================================================


@wp.func
def waypoint_world_pos(
    body_tf: wp.transform,
    body_com: wp.vec3,
    local_pt: wp.vec3,
) -> wp.vec3:
    """Map a body-local waypoint to world space.

    The waypoint is stored relative to the body origin; subtracting the
    CoM offset converts it to the CoM-relative frame expected by
    ``wp.transform_point``.
    """
    return wp.transform_point(body_tf, local_pt - body_com)


@wp.func
def tensile_wrench(
    anchor: wp.vec3,
    direction: wp.vec3,
    magnitude: float,
) -> wp.spatial_vector:
    """Compute the spatial wrench (force + moment) at an anchor point.

    wrench = (f, r × f)   where f = magnitude · direction
    """
    f = direction * magnitude
    return wp.spatial_vector(f, wp.cross(anchor, f))


# =====================================================================
# Per-segment evaluation
# =====================================================================


@wp.func
def eval_muscle_segment(
    seg: int,
    body_tf: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    waypoint_body: wp.array(dtype=int),
    waypoint_local: wp.array(dtype=wp.vec3),
    activation: float,
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """Apply tensile forces for one waypoint segment of a muscle.

    Segments that share the same body at both ends produce no inter-body
    force and are skipped.
    """
    body_a = waypoint_body[seg]
    body_b = waypoint_body[seg + 1]

    if body_a == body_b:
        return 0   # intra-body segment — no net wrench

    p_a = waypoint_world_pos(body_tf[body_a], body_com[body_a], waypoint_local[seg])
    p_b = waypoint_world_pos(body_tf[body_b], body_com[body_b], waypoint_local[seg + 1])

    direction = wp.normalize(p_b - p_a)

    wp.atomic_sub(body_wrench, body_a, tensile_wrench(p_a,  direction, activation))
    wp.atomic_add(body_wrench, body_b, tensile_wrench(p_b,  direction, activation))


# =====================================================================
# Muscle kernel
# =====================================================================


@wp.kernel
def muscle_actuator_kernel(
    body_tf: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    muscle_start: wp.array(dtype=int),
    muscle_params: wp.array(dtype=float),
    waypoint_body: wp.array(dtype=int),
    waypoint_local: wp.array(dtype=wp.vec3),
    activations: wp.array(dtype=float),
    # output
    body_wrench: wp.array(dtype=wp.spatial_vector),
):
    """Evaluate all waypoint segments for one muscle.

    One thread per muscle.  Segments are iterated sequentially because
    a muscle's waypoints form a chain — no parallelism within a single
    muscle is possible without race conditions.
    """
    m = wp.tid()

    seg_begin = muscle_start[m]
    seg_end   = muscle_start[m + 1] - 1   # last valid segment index
    act       = activations[m]

    for seg in range(seg_begin, seg_end):
        eval_muscle_segment(
            seg,
            body_tf, body_com,
            waypoint_body, waypoint_local,
            act,
            body_wrench,
        )


# =====================================================================
# Python-side launcher
# =====================================================================


def apply_muscle_actuators(model, state, control, body_wrench: wp.array):
    """Launch the muscle actuator kernel if the model contains muscles."""
    if model.muscle_count:
        wp.launch(
            kernel=muscle_actuator_kernel,
            dim=model.muscle_count,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.muscle_start,
                model.muscle_params,
                model.muscle_bodies,
                model.muscle_points,
                control.muscle_activations,
            ],
            outputs=[body_wrench],
            device=model.device,
        )
