# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Symplectic Euler time-stepping for particles and rigid bodies.

Physics background
------------------
Symplectic (semi-implicit) Euler is a first-order variational integrator.
Unlike explicit Euler it updates velocity *before* position, which makes it
energy-preserving on average and unconditionally stable for conservative
systems (given a small enough time-step).

    v_{n+1} = v_n + a(x_n) * dt      # velocity first
    x_{n+1} = x_n + v_{n+1} * dt     # position uses new velocity

Rigid body angular integration uses Euler's rotation equation in the body
frame to correctly account for gyroscopic (omega x L) effects:

    I * alpha = tau - omega x (I * omega)

All kernels write to separate output arrays so that state_in is never
mutated during a step.
"""

from __future__ import annotations

import warp as wp

from .common import ParticleFlag


# =====================================================================
# Particle helpers
# =====================================================================


@wp.func
def particle_acceleration(
    force: wp.vec3,
    inv_mass: float,
    gravity: wp.vec3,
) -> wp.vec3:
    """Compute particle acceleration: a = F/m + g.

    Static particles (inv_mass == 0) receive zero acceleration.
    """
    if inv_mass > 0.0:
        return force * inv_mass + gravity
    return wp.vec3(0.0)


@wp.func
def clamp_speed(v: wp.vec3, v_max: float) -> wp.vec3:
    """Clamp velocity magnitude to v_max for numerical stability."""
    speed = wp.length(v)
    if speed > v_max and speed > 0.0:
        return v * (v_max / speed)
    return v


# =====================================================================
# Particle integration kernel
# =====================================================================


@wp.kernel
def advance_particles(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    flags: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    # outputs
    pos_out: wp.array(dtype=wp.vec3),
    vel_out: wp.array(dtype=wp.vec3),
):
    """Symplectic Euler integration for a single particle.

    Inactive particles (ACTIVE flag unset) are frozen in place.
    """
    i = wp.tid()

    if (flags[i] & ParticleFlag.ACTIVE) == 0:
        pos_out[i] = pos[i]
        return

    a = particle_acceleration(force[i], inv_mass[i], gravity[0])
    v1 = clamp_speed(vel[i] + a * dt, v_max)
    pos_out[i] = pos[i] + v1 * dt
    vel_out[i] = v1


# =====================================================================
# Rigid body helpers
# =====================================================================


@wp.func
def linear_step(
    x_com: wp.vec3,
    v: wp.vec3,
    force: wp.vec3,
    inv_mass: float,
    gravity: wp.vec3,
    dt: float,
) -> tuple[wp.vec3, wp.vec3]:
    """Symplectic Euler for the translational DOFs.

    Returns (x_com_new, v_new).
    Static bodies (inv_mass == 0) are not moved.
    """
    if inv_mass > 0.0:
        a = force * inv_mass + gravity
    else:
        a = wp.vec3(0.0)
    v_new = v + a * dt
    x_new = x_com + v_new * dt      # symplectic: position uses updated velocity
    return x_new, v_new


@wp.func
def angular_step(
    rot: wp.quat,
    omega: wp.vec3,
    torque: wp.vec3,
    inertia_body: wp.mat33,
    inv_inertia_body: wp.mat33,
    dt: float,
) -> tuple[wp.quat, wp.vec3]:
    """Symplectic Euler for the rotational DOFs via Euler's equation.

    Euler's rotation equation in the body frame::

        I * alpha = tau - omega x (I * omega)

    where the ``omega x (I * omega)`` term captures gyroscopic effects.
    Returns (rot_new, omega_new) both in world frame.
    """
    # --- rotate quantities to body frame ---
    omega_b  = wp.quat_rotate_inv(rot, omega)
    torque_b = wp.quat_rotate_inv(rot, torque)

    # angular momentum in body frame
    L = inertia_body * omega_b

    # Euler's equation: alpha = I^{-1} * (tau - omega x L)
    alpha_b = inv_inertia_body * (torque_b - wp.cross(omega_b, L))

    omega_b_new = omega_b + alpha_b * dt
    omega_new   = wp.quat_rotate(rot, omega_b_new)

    # quaternion kinematics: dq/dt = 0.5 * [omega, 0] * q
    dq      = wp.quat(omega_new, 0.0) * rot * 0.5
    rot_new = wp.normalize(rot + dq * dt)

    return rot_new, omega_new


# =====================================================================
# Rigid body integration kernel
# =====================================================================


@wp.kernel
def advance_bodies(
    body_tf: wp.array(dtype=wp.transform),
    body_vel: wp.array(dtype=wp.spatial_vector),
    body_force: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    gravity: wp.array(dtype=wp.vec3),
    angular_damping: float,
    dt: float,
    # outputs
    tf_out: wp.array(dtype=wp.transform),
    vel_out: wp.array(dtype=wp.spatial_vector),
):
    """Symplectic Euler integration for a single rigid body.

    Spatial vectors follow the Newton convention:
      - ``spatial_top``    → linear  (force / linear velocity)
      - ``spatial_bottom`` → angular (torque / angular velocity)
    """
    i = wp.tid()

    tf  = body_tf[i]
    rot = wp.transform_get_rotation(tf)
    x0  = wp.transform_get_translation(tf)
    com = body_com[i]

    # unpack velocity twist
    v0 = wp.spatial_top(body_vel[i])      # linear velocity
    w0 = wp.spatial_bottom(body_vel[i])   # angular velocity

    # unpack force wrench
    force  = wp.spatial_top(body_force[i])    # net force
    torque = wp.spatial_bottom(body_force[i]) # net torque

    # centre-of-mass position in world frame
    x_com = x0 + wp.quat_rotate(rot, com)

    # --- translational integration ---
    x_new, v_new = linear_step(x_com, v0, force, body_inv_mass[i], gravity[0], dt)

    # --- rotational integration ---
    rot_new, w_new = angular_step(
        rot, w0, torque,
        body_inertia[i], body_inv_inertia[i],
        dt,
    )

    # apply angular damping after integration
    w_new = w_new * (1.0 - angular_damping * dt)

    # reconstruct transform: origin = x_com - R_new * com
    tf_out[i]  = wp.transform(x_new - wp.quat_rotate(rot_new, com), rot_new)
    vel_out[i] = wp.spatial_vector(v_new, w_new)
