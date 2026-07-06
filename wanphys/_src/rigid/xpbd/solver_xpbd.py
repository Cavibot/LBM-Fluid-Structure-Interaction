# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import warp as wp

from ._types import override
from ..solver import RigidSolver
from ..model import RigidModel
from ..state import RigidState
from .constraint_base import ConstraintGroup
from .constraints import (
    BendingConstraint,
    BodyJointConstraint,
    ParticleParticleContact,
    ParticleShapeContact,
    RigidContactConstraint,
    SpringConstraint,
    TetrahedraConstraint,
)
from .context import ConstraintPhase, XPBDContext
from .integrator import apply_body_deltas, apply_particle_deltas
from ._types import ParticleFlags


def _replace_state_array(state: RigidState, private_name: str, view_name: str, value: wp.array) -> None:
    """Replace a RigidState-owned array and keep its cached Newton view coherent."""
    setattr(state, private_name, value)
    if state._newton_view is not None:
        setattr(state._newton_view, view_name, value)


@wp.func
def _integrate_rigid_body(
    q: wp.transform,
    qd: wp.spatial_vector,
    f: wp.spatial_vector,
    com: wp.vec3,
    inertia: wp.mat33,
    inv_mass: float,
    inv_inertia: wp.mat33,
    gravity: wp.array(dtype=wp.vec3),
    angular_damping: float,
    dt: float,
):
    # unpack transform
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # unpack spatial twist
    w0 = wp.spatial_bottom(qd)
    v0 = wp.spatial_top(qd)

    # unpack spatial wrench
    t0 = wp.spatial_bottom(f)
    f0 = wp.spatial_top(f)

    x_com = x0 + wp.quat_rotate(r0, com)

    # linear part
    v1 = v0 + (f0 * inv_mass + gravity[0] * wp.nonzero(inv_mass)) * dt
    x1 = x_com + v1 * dt

    # angular part (compute in body frame)
    wb = wp.quat_rotate_inv(r0, w0)
    tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces

    w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)
    r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)

    # angular damping
    w1 *= 1.0 - angular_damping * dt

    q_new = wp.transform(x1 - wp.quat_rotate(r1, com), r1)
    qd_new = wp.spatial_vector(v1, w1)

    return q_new, qd_new

@wp.kernel
def _integrate_particles_kernel(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    w: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    dt: float,
    v_max: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    x0 = x[tid]

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        x_new[tid] = x0
        return

    v0 = v[tid]
    f0 = f[tid]

    inv_mass = w[tid]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + gravity[0] * wp.step(-inv_mass)) * dt
    # enforce velocity limit to prevent instability
    v1_mag = wp.length(v1)
    if v1_mag > v_max:
        v1 *= v_max / v1_mag
    x1 = x0 + v1 * dt

    x_new[tid] = x1
    v_new[tid] = v1

@wp.kernel
def _integrate_bodies_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    I: wp.array(dtype=wp.mat33),
    inv_m: wp.array(dtype=float),
    inv_I: wp.array(dtype=wp.mat33),
    gravity: wp.array(dtype=wp.vec3),
    angular_damping: float,
    dt: float,
    # outputs
    body_q_new: wp.array(dtype=wp.transform),
    body_qd_new: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    # positions
    q = body_q[tid]
    qd = body_qd[tid]
    f = body_f[tid]

    # masses
    inv_mass = inv_m[tid]  # 1 / mass

    inertia = I[tid]
    inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix

    com = body_com[tid]

    q_new, qd_new = _integrate_rigid_body(
        q,
        qd,
        f,
        com,
        inertia,
        inv_mass,
        inv_inertia,
        gravity,
        angular_damping,
        dt,
    )

    body_q_new[tid] = q_new
    body_qd_new[tid] = qd_new



class SolverXPBD(RigidSolver):
    """An implicit integrator using eXtended Position-Based Dynamics (XPBD) for rigid and soft body simulation.

    References:
        - Miles Macklin, Matthias Müller, and Nuttapong Chentanez. 2016. XPBD: position-based simulation of compliant constrained dynamics. In Proceedings of the 9th International Conference on Motion in Games (MIG '16). Association for Computing Machinery, New York, NY, USA, 49-54. https://doi.org/10.1145/2994258.2994272
        - Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. 2020. Detailed rigid body simulation with extended position based dynamics. In Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation (SCA '20). Eurographics Association, Goslar, DEU, Article 10, 1-12. https://doi.org/10.1111/cgf.14105

    After constructing :class:`Model`, :class:`State`, and :class:`Control` (optional) objects, this time-integrator
    may be used to advance the simulation state forward in time.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverXPBD(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    def __init__(
        self,
        model: RigidModel,
        iterations: int = 2,
        soft_body_relaxation: float = 0.9,
        soft_contact_relaxation: float = 0.9,
        joint_linear_relaxation: float = 0.7,
        joint_angular_relaxation: float = 0.4,
        joint_linear_compliance: float = 0.0,
        joint_angular_compliance: float = 0.0,
        rigid_contact_relaxation: float = 0.8,
        rigid_contact_con_weighting: bool = True,
        angular_damping: float = 0.0,
        enable_restitution: bool = False,
    ) -> None:

        self.model: RigidModel = model
        self.iterations: int = iterations
        self.angular_damping: float = angular_damping
        self.enable_restitution: bool = enable_restitution

        # Kept for backward-compatible attribute access
        self.soft_body_relaxation: float = soft_body_relaxation
        self.soft_contact_relaxation: float = soft_contact_relaxation
        self.joint_linear_relaxation: float = joint_linear_relaxation
        self.joint_angular_relaxation: float = joint_angular_relaxation
        self.joint_linear_compliance: float = joint_linear_compliance
        self.joint_angular_compliance: float = joint_angular_compliance
        self.rigid_contact_relaxation: float = rigid_contact_relaxation
        self.rigid_contact_con_weighting: bool = rigid_contact_con_weighting

        self.compute_body_velocity_from_position_delta: bool = False

        # Double-buffering counters (managed by _apply_*_deltas helpers)
        self._particle_delta_counter: int = 0
        self._body_delta_counter: int = 0

        # ── Build constraint groups (ordered by phase, then priority) ──
        self._constraints: list[ConstraintGroup] = [
            # PARTICLE phase
            ParticleShapeContact(relaxation=soft_contact_relaxation, enable_restitution=enable_restitution),
            ParticleParticleContact(relaxation=soft_contact_relaxation),
            SpringConstraint(),
            BendingConstraint(),
            TetrahedraConstraint(relaxation=soft_body_relaxation),
            # BODY_JOINT phase
            BodyJointConstraint(
                linear_relaxation=joint_linear_relaxation,
                angular_relaxation=joint_angular_relaxation,
                linear_compliance=joint_linear_compliance,
                angular_compliance=joint_angular_compliance,
            ),
            # RIGID_CONTACT phase
            RigidContactConstraint(
                relaxation=rigid_contact_relaxation,
                con_weighting=rigid_contact_con_weighting,
                enable_restitution=enable_restitution,
                compute_velocity_from_position_delta=False,
            ),
        ]

        # ── CUDA streams for parallel particle / body execution ──
        self._particle_stream: wp.Stream | None = None
        self._body_stream: wp.Stream | None = None
        device: wp.Device = wp.get_device(model.device)
        if device.is_cuda:
            self._particle_stream = wp.Stream(device)
            self._body_stream = wp.Stream(device)

        # Populated per step by _prepare()
        self._ctx: XPBDContext | None = None
        self._active: list[ConstraintGroup] = []
        self._phase_groups: dict[ConstraintPhase, list[ConstraintGroup]] = {}

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    @override
    def step(
        self,
        state_in: RigidState,
        state_out: RigidState,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> RigidState:
        # print("*****************")

        with wp.ScopedTimer("simulate", False):
            self._prepare(state_in, state_out, control, contacts, dt)
            self._apply_external_forces()
            self._predict_positions()

            for i in range(self.iterations):
                self._solve_constraints(i)

            self._stash_lambdas()
            self._finalize_state()
            self._apply_restitution()

        return state_out

    # ────────────────────────────────────────────────────────────
    # Step sub-routines
    # ────────────────────────────────────────────────────────────

    def _prepare(
        self,
        state_in: RigidState,
        state_out: RigidState,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> None:
        """Build context, determine active constraints, allocate working buffers."""
        model: RigidModel = self.model
        self._particle_delta_counter = 0
        self._body_delta_counter = 0

        if control is None:
            control = model.control()

        # Sync dynamic flags to constraint instances
        for c in self._constraints:
            if isinstance(c, RigidContactConstraint):
                c.compute_velocity_from_position_delta = self.compute_body_velocity_from_position_delta

        # Build shared context
        ctx = XPBDContext(
            model=model,
            state_in=state_in,
            state_out=state_out,
            control=control,
            contacts=contacts,
            dt=dt,
        )

        # ── Particle buffers ──
        if model.particle_count:
            ctx.particle_q = state_out.particle_q
            ctx.particle_qd = state_out.particle_qd

            self.particle_q_init = wp.clone(state_in.particle_q)
            ctx.particle_q_init = self.particle_q_init


            if self.enable_restitution:
                self.particle_qd_init = wp.clone(state_in.particle_qd)
                ctx.particle_qd_init = self.particle_qd_init
            ctx.particle_deltas = wp.empty_like(state_out.particle_qd)

        # ── Body buffers ──
        if model.body_count:
            ctx.body_q = state_out.body_q
            ctx.body_qd = state_out.body_qd

            if self.compute_body_velocity_from_position_delta or self.enable_restitution:
                ctx.body_q_init = wp.clone(state_in.body_q)
                ctx.body_qd_init = wp.clone(state_in.body_qd)

            ctx.body_deltas = wp.empty_like(state_out.body_qd)

            if contacts is not None and self.rigid_contact_con_weighting:
                ctx.rigid_contact_inv_weight = wp.zeros_like(contacts.rigid_contact_margin0)

        # ── Determine active constraints & group by phase ──
        active: list[ConstraintGroup] = []
        phase_groups: dict[ConstraintPhase, list[ConstraintGroup]] = {p: [] for p in ConstraintPhase}
        for c in self._constraints:
            if c.is_active(model, contacts):
                active.append(c)
                phase_groups[c.phase].append(c)

        self._ctx = ctx
        self._active = active
        self._phase_groups = phase_groups

    def _apply_external_forces(self) -> None:
        """Initialize constraints — apply joint forces, allocate lambda buffers, etc."""
        for c in self._active:
            c.initialize(self._ctx)

    def _predict_positions(self) -> None:
        """Semi-implicit Euler integration for particles and rigid bodies."""
        ctx = self._ctx
        model = ctx.model
        has_particles = model.particle_count > 0
        has_bodies = model.body_count > 0

        if has_particles and has_bodies and self._particle_stream is not None:
            main = wp.get_stream(model.device)
            fork_evt = main.record_event()

            self._particle_stream.wait_event(fork_evt)
            with wp.ScopedStream(self._particle_stream):
                self._integrate_particles(model, ctx.state_in, ctx.state_out, ctx.dt)
            p_done = self._particle_stream.record_event()

            self._body_stream.wait_event(fork_evt)
            with wp.ScopedStream(self._body_stream):
                self._integrate_bodies(model, ctx.state_in, ctx.state_out, ctx.dt, self.angular_damping)
            b_done = self._body_stream.record_event()

            main.wait_event(p_done)
            main.wait_event(b_done)
        else:
            if has_particles:
                self._integrate_particles(model, ctx.state_in, ctx.state_out, ctx.dt)
            if has_bodies:
                self._integrate_bodies(model, ctx.state_in, ctx.state_out, ctx.dt, self.angular_damping)

    def _solve_constraints(self, iteration: int) -> None:
        """One full iteration: project all constraint phases and apply deltas."""
        ctx = self._ctx
        model = ctx.model

        with wp.ScopedTimer(f"iteration_{iteration}", False):
            self._zero_deltas(iteration)

            particle_groups = self._phase_groups[ConstraintPhase.PARTICLE]
            body_joint_groups = self._phase_groups[ConstraintPhase.BODY_JOINT]
            rigid_contact_groups = self._phase_groups[ConstraintPhase.RIGID_CONTACT]

            # Try to run PARTICLE and BODY_JOINT phases in parallel on separate streams
            use_streams = (
                particle_groups
                and body_joint_groups
                and self._particle_stream is not None
            )

            if use_streams:
                main = wp.get_stream(model.device)
                fork_evt = main.record_event()

                # ── PARTICLE phase on particle stream ──
                self._particle_stream.wait_event(fork_evt)
                with wp.ScopedStream(self._particle_stream):
                    for c in particle_groups:
                        c.reset_iteration(ctx, iteration)
                        c.project(ctx, iteration)
                    if model.particle_count:
                        ctx.particle_q, ctx.particle_qd = self._apply_particle_deltas()
                p_done = self._particle_stream.record_event()

                # ── BODY_JOINT phase on body stream ──
                self._body_stream.wait_event(fork_evt)
                with wp.ScopedStream(self._body_stream):
                    for c in body_joint_groups:
                        c.reset_iteration(ctx, iteration)
                        c.project(ctx, iteration)
                    ctx.body_q, ctx.body_qd = self._apply_body_deltas()
                b_done = self._body_stream.record_event()

                # Sync back to main before RIGID_CONTACT
                main.wait_event(p_done)
                main.wait_event(b_done)
            else:
                # ── Sequential fallback ──
                for c in particle_groups:
                    c.reset_iteration(ctx, iteration)
                    c.project(ctx, iteration)
                if model.particle_count:
                    ctx.particle_q, ctx.particle_qd = self._apply_particle_deltas()

                for c in body_joint_groups:
                    c.reset_iteration(ctx, iteration)
                    c.project(ctx, iteration)
                if body_joint_groups:
                    ctx.body_q, ctx.body_qd = self._apply_body_deltas()

            # ── RIGID_CONTACT phase (always sequential, re-zeros body_deltas) ──
            if rigid_contact_groups:
                ctx.body_deltas.zero_()

                for c in rigid_contact_groups:
                    c.reset_iteration(ctx, iteration)
                    c.project(ctx, iteration)

                ctx.body_q, ctx.body_qd = self._apply_body_deltas(
                    ctx.rigid_contact_inv_weight,
                )

    def _finalize_state(self) -> None:
        """Copy final double-buffered arrays back to *state_out*."""
        ctx = self._ctx
        model = ctx.model

        need_particle_copy = (
            model.particle_count > 0
            and ctx.particle_q.ptr != ctx.state_out.particle_q.ptr
        )
        need_body_copy = (
            model.body_count > 0
            and ctx.body_q.ptr != ctx.state_out.body_q.ptr
        )

        if need_particle_copy and need_body_copy and self._particle_stream is not None:
            main = wp.get_stream(model.device)
            fork_evt = main.record_event()

            self._particle_stream.wait_event(fork_evt)
            with wp.ScopedStream(self._particle_stream):
                ctx.state_out.particle_q.assign(ctx.particle_q)
                ctx.state_out.particle_qd.assign(ctx.particle_qd)
            p_done = self._particle_stream.record_event()

            self._body_stream.wait_event(fork_evt)
            with wp.ScopedStream(self._body_stream):
                ctx.state_out.body_q.assign(ctx.body_q)
                ctx.state_out.body_qd.assign(ctx.body_qd)
            b_done = self._body_stream.record_event()

            main.wait_event(p_done)
            main.wait_event(b_done)
        else:
            if need_particle_copy:
                ctx.state_out.particle_q.assign(ctx.particle_q)
                ctx.state_out.particle_qd.assign(ctx.particle_qd)
            if need_body_copy:
                ctx.state_out.body_q.assign(ctx.body_q)
                ctx.state_out.body_qd.assign(ctx.body_qd)

    def _stash_lambdas(self) -> None:
        """Save accumulated lambdas for warm-starting the next step."""
        for c in self._active:
            c.stash_lambdas(self._ctx)

    def _apply_restitution(self) -> None:
        """Post-solve velocity correction for contacts."""
        for c in self._active:
            c.apply_restitution(self._ctx)

    # ────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────

    def _zero_deltas(self, iteration: int) -> None:
        """Zero delta buffers; allocate fresh arrays when gradients are tracked."""
        ctx = self._ctx
        model = ctx.model
        requires_grad = ctx.requires_grad

        if model.body_count:
            if requires_grad and iteration > 0:
                ctx.body_deltas = wp.zeros_like(ctx.body_deltas)
            else:
                ctx.body_deltas.zero_()

        if model.particle_count:
            if requires_grad and iteration > 0:
                ctx.particle_deltas = wp.zeros_like(ctx.particle_deltas)
            else:
                ctx.particle_deltas.zero_()

    def _apply_particle_deltas(self):
        """Apply accumulated particle constraint deltas (double-buffered for gradients)."""
        ctx = self._ctx
        model = ctx.model
        state_in = ctx.state_in
        state_out = ctx.state_out

        if state_in.requires_grad:
            particle_q = state_out.particle_q
            new_particle_q = wp.empty_like(state_out.particle_q)
            new_particle_qd = wp.empty_like(state_out.particle_qd)
            self._particle_delta_counter += 1
        else:
            if self._particle_delta_counter == 0:
                particle_q = state_out.particle_q
                new_particle_q = state_in.particle_q
                new_particle_qd = state_in.particle_qd
            else:
                particle_q = state_in.particle_q
                new_particle_q = state_out.particle_q
                new_particle_qd = state_out.particle_qd
            self._particle_delta_counter = 1 - self._particle_delta_counter

        wp.launch(
            kernel=apply_particle_deltas,
            dim=model.particle_count,
            inputs=[
                self.particle_q_init,
                particle_q,
                model.particle_flags,
                ctx.particle_deltas,
                ctx.dt,
                model.particle_max_velocity,
            ],
            outputs=[new_particle_q, new_particle_qd],
            device=model.device,
        )

        if state_in.requires_grad:
            _replace_state_array(state_out, "_particle_q", "particle_q", new_particle_q)
            _replace_state_array(state_out, "_particle_qd", "particle_qd", new_particle_qd)

        return new_particle_q, new_particle_qd

    def _apply_body_deltas(self, rigid_contact_inv_weight: wp.array = None):
        """Apply accumulated body constraint deltas (double-buffered for gradients)."""
        ctx = self._ctx
        model = ctx.model
        state_in = ctx.state_in
        state_out = ctx.state_out

        with wp.ScopedTimer("apply_body_deltas", False):
            if state_in.requires_grad:
                body_q = state_out.body_q
                body_qd = state_out.body_qd
                new_body_q = wp.clone(body_q)
                new_body_qd = wp.clone(body_qd)
                self._body_delta_counter += 1
            else:
                if self._body_delta_counter == 0:
                    body_q = state_out.body_q
                    body_qd = state_out.body_qd
                    new_body_q = state_in.body_q
                    new_body_qd = state_in.body_qd
                else:
                    body_q = state_in.body_q
                    body_qd = state_in.body_qd
                    new_body_q = state_out.body_q
                    new_body_qd = state_out.body_qd
                self._body_delta_counter = 1 - self._body_delta_counter

            wp.launch(
                kernel=apply_body_deltas,
                dim=model.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    ctx.body_deltas,
                    rigid_contact_inv_weight,
                    ctx.dt,
                ],
                outputs=[
                    new_body_q,
                    new_body_qd,
                ],
                device=model.device,
            )

            if state_in.requires_grad:
                _replace_state_array(state_out, "_body_q", "body_q", new_body_q)
                _replace_state_array(state_out, "_body_qd", "body_qd", new_body_qd)

        return new_body_q, new_body_qd

    def _integrate_bodies(
            self,
            model,
            state_in,
            state_out,
            dt: float,
            angular_damping: float = 0.0,
    ) -> None:
        if model.body_count:
            wp.launch(
                kernel=_integrate_bodies_kernel,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_mass,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.gravity,
                    angular_damping,
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )

    def _integrate_particles(
            self,
            model,
            state_in,
            state_out,
            dt: float,
    ) -> None:
        if model.particle_count:
            wp.launch(
                kernel=_integrate_particles_kernel,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_in.particle_f,
                    model.particle_inv_mass,
                    model.particle_flags,
                    model.gravity,
                    dt,
                    model.particle_max_velocity,
                ],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=model.device,
            )
