# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys Sequential Impulse solver for rigid body simulation.

Velocity-level constraint resolution via iterative impulse application.
Integration sequence each step:
  1. Apply joint actuation forces, integrate velocities.
  2. Iteratively solve joint and contact constraints (Jacobi impulse).
  3. Apply restitution impulses (optional).
  4. Update positions from corrected velocities.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ..model import RigidModel
from ..solver import RigidSolver
from ..state import RigidState
from .kernels import (
    apply_joint_actuation,
    si_apply_body_contact_deltas,
    si_apply_body_joint_deltas,
    si_apply_restitution_deltas,
    si_apply_rigid_restitution,
    si_integrate_body_velocities,
    si_update_body_positions,
    solve_body_contact_velocities_si,
    solve_body_joints_si,
)


class WanPhysSequentialImpulseSolver(RigidSolver):
    """Sequential Impulse (SI) solver for the WanPhys rigid body framework.

    Implements velocity-level constraint resolution via iterative impulse
    application.  The integration sequence is:

    1. Apply joint forces and integrate velocities semi-implicitly.
    2. Iteratively solve joint and contact velocity constraints (Jacobi).
    3. (Optional) Apply restitution impulses for bouncy collisions.
    4. Update positions from the corrected velocities.

    Example::

        from wanphys._src.rigid import create_si_solver
        solver = create_si_solver(model, iterations=10)
        for _ in range(n_steps):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    Args:
        model: WanPhys RigidModel instance.
        iterations: Number of constraint solver iterations per step.
        joint_linear_relaxation: Baumgarte stabilisation factor for
            linear (position) joint constraint errors.
        joint_angular_relaxation: Baumgarte stabilisation factor for
            angular (orientation) joint constraint errors.
        joint_impulse_relaxation: Per-constraint impulse scaling factor.
        rigid_contact_relaxation: Baumgarte coefficient for contact
            penetration depth correction.
        angular_damping: Angular velocity damping applied during velocity
            integration.
        enable_restitution: When True (default), bouncing behaviour from
            material ``restitution`` coefficients is active.
    """

    def __init__(
        self,
        model: RigidModel,
        iterations: int = 2,
        joint_linear_relaxation: float = 0.2,
        joint_angular_relaxation: float = 0.2,
        joint_impulse_relaxation: float = 0.2,
        rigid_contact_relaxation: float = 1.0,
        angular_damping: float = 0.0,
        enable_restitution: bool = True,
    ) -> None:
        self.model: RigidModel = model

        self.iterations: int = iterations
        self.joint_linear_relaxation: float = joint_linear_relaxation
        self.joint_angular_relaxation: float = joint_angular_relaxation
        self.joint_impulse_relaxation: float = joint_impulse_relaxation
        self.rigid_contact_relaxation: float = rigid_contact_relaxation
        self.angular_damping: float = angular_damping
        self.enable_restitution: bool = enable_restitution

        self.body_contact_deltas: wp.array | None = None
        self.body_contact_count: wp.array | None = None
        self.body_joint_deltas: wp.array | None = None
        self.body_joint_count: wp.array | None = None
        self.body_restitution_deltas: wp.array | None = None

        if model.body_count:
            self.body_contact_deltas = wp.zeros(
                model.body_count, dtype=wp.spatial_vector, device=model.device
            )
            self.body_contact_count = wp.zeros(model.body_count, dtype=float, device=model.device)
            self.body_joint_deltas = wp.zeros(
                model.body_count, dtype=wp.spatial_vector, device=model.device
            )
            self.body_joint_count = wp.zeros(model.body_count, dtype=float, device=model.device)
            if enable_restitution:
                self.body_restitution_deltas = wp.zeros(
                    model.body_count, dtype=wp.spatial_vector, device=model.device
                )

    def step(
        self,
        state_in: RigidState,
        state_out: RigidState,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> None:
        """Advance simulation by one timestep.

        Args:
            state_in: Current state (input, read-only).
            state_out: Next state (output, written to).
            control: Control inputs.  If ``None``, model defaults are used.
            contacts: Collision contact data.  If ``None``, no contacts are
                applied.
            dt: Timestep in seconds.
        """
        m: RigidModel = self.model

        if control is None:
            control = m.control()

        # ==================== Step 1: velocity integration ====================

        body_q_prev: wp.array | None = None
        body_qd_prev: wp.array | None = None

        if m.body_count:
            body_q_prev = wp.clone(state_in.body_q)

            if m.joint_count:
                wp.launch(
                    kernel=apply_joint_actuation,
                    dim=m.joint_count,
                    inputs=[
                        state_in.body_q,
                        m.body_com,
                        m.joint_type,
                        m.joint_parent,
                        m.joint_child,
                        m.joint_X_p,
                        m.joint_qd_start,
                        m.joint_dof_dim,
                        m.joint_axis,
                        control.joint_f,
                    ],
                    outputs=[state_in.body_f],
                    device=m.device,
                )

            wp.launch(
                kernel=si_integrate_body_velocities,
                dim=m.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    m.body_com,
                    m.body_mass,
                    m.body_inertia,
                    m.body_inv_mass,
                    m.body_inv_inertia,
                    m.gravity,
                    self.angular_damping,
                    dt,
                ],
                outputs=[state_out.body_qd],
                device=m.device,
            )

            wp.copy(state_out.body_q, state_in.body_q)

            if self.enable_restitution:
                body_qd_prev = wp.clone(state_out.body_qd)

        # ==================== Step 2: constraint iterations ====================

        for _ in range(self.iterations):

            if (
                m.joint_count
                and self.body_joint_deltas is not None
                and self.body_joint_count is not None
            ):
                self.body_joint_deltas.zero_()
                self.body_joint_count.zero_()

                wp.launch(
                    kernel=solve_body_joints_si,
                    dim=m.joint_count,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        m.body_com,
                        m.body_inv_mass,
                        m.body_inv_inertia,
                        m.joint_type,
                        m.joint_enabled,
                        m.joint_parent,
                        m.joint_child,
                        m.joint_X_p,
                        m.joint_X_c,
                        m.joint_axis,
                        m.joint_qd_start,
                        self.joint_linear_relaxation,
                        self.joint_angular_relaxation,
                        self.joint_impulse_relaxation,
                        dt,
                    ],
                    outputs=[
                        self.body_joint_deltas,
                        self.body_joint_count,
                    ],
                    device=m.device,
                )

                wp.launch(
                    kernel=si_apply_body_joint_deltas,
                    dim=m.body_count,
                    inputs=[
                        self.body_joint_deltas,
                        self.body_joint_count,
                        m.body_inv_mass,
                    ],
                    outputs=[state_out.body_qd],
                    device=m.device,
                )

            if (
                m.body_count
                and contacts is not None
                and self.body_contact_deltas is not None
                and self.body_contact_count is not None
            ):
                self.body_contact_deltas.zero_()
                self.body_contact_count.zero_()

                wp.launch(
                    kernel=solve_body_contact_velocities_si,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        m.body_com,
                        m.body_inv_mass,
                        m.body_inv_inertia,
                        m.shape_body,
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_offset0,
                        contacts.rigid_contact_offset1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_margin0,
                        contacts.rigid_contact_margin1,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        m.shape_material_mu,
                        m.shape_material_mu_torsional,
                        m.shape_material_mu_rolling,
                        self.rigid_contact_relaxation,
                        dt,
                    ],
                    outputs=[
                        self.body_contact_deltas,
                        self.body_contact_count,
                    ],
                    device=m.device,
                )

                wp.launch(
                    kernel=si_apply_body_contact_deltas,
                    dim=m.body_count,
                    inputs=[
                        self.body_contact_deltas,
                        self.body_contact_count,
                        m.body_inv_mass,
                    ],
                    outputs=[state_out.body_qd],
                    device=m.device,
                )

        # ==================== Step 3: restitution ====================

        if (
            m.body_count
            and self.enable_restitution
            and body_qd_prev is not None
            and contacts is not None
            and self.body_restitution_deltas is not None
        ):
            self.body_restitution_deltas.zero_()

            wp.launch(
                    kernel=si_apply_rigid_restitution,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        state_out.body_q,
                        state_out.body_qd,
                        body_qd_prev,
                        m.body_com,
                    m.body_inv_mass,
                    m.body_inv_inertia,
                    m.shape_body,
                    m.shape_material_restitution,
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_margin0,
                    contacts.rigid_contact_margin1,
                    m.gravity,
                    dt,
                ],
                outputs=[self.body_restitution_deltas],
                device=m.device,
            )

            wp.launch(
                kernel=si_apply_restitution_deltas,
                dim=m.body_count,
                inputs=[
                    self.body_restitution_deltas,
                    m.body_inv_mass,
                ],
                outputs=[state_out.body_qd],
                device=m.device,
            )

        # ==================== Step 4: position update ====================

        if m.body_count and body_q_prev is not None:
            wp.launch(
                kernel=si_update_body_positions,
                dim=m.body_count,
                inputs=[
                    body_q_prev,
                    state_out.body_qd,
                    m.body_com,
                    m.body_inv_mass,
                    dt,
                ],
                outputs=[state_out.body_q],
                device=m.device,
            )
