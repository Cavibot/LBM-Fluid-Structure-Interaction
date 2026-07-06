# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Symplectic Euler integrator for WanPhys rigid-body simulation.

This module provides :class:`SymplecticEulerSolver`, a variational
integrator that composes force modules via a :class:`ForcePipeline` and
advances state using symplectic (semi-implicit) Euler time-stepping.

Integration sequence per step:

1. **Force accumulation** – each registered force module evaluates its
   contribution and atomically accumulates into particle / body force
   buffers.
2. **Time advance** – symplectic Euler updates velocities then positions.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ..model import RigidModel
from ..solver import RigidSolver
from ..state import RigidState
from .forces import (
    ArticulationDispatcher,
    MaterialLaw,
    apply_articulation_forces,
    apply_articulation_forces_dispatched,
    apply_hinge_bending,
    apply_membrane_stress,
    apply_muscle_actuators,
    apply_particle_proximity,
    apply_mesh_particle_contact,
    apply_particle_shape_contact,
    apply_rigid_contacts,
    apply_solid_stress,
    apply_spring_dashpot,
)
from .pipeline import ForcePipeline
from .time_stepping import advance_bodies, advance_particles


class SymplecticEulerSolver(RigidSolver):
    """Symplectic (semi-implicit) Euler integrator for WanPhys.

    This is a variational integrator that preserves energy but is
    conditionally stable — the time-step must be small enough to support
    the required stiffness and damping forces.

    Contact response uses the penalty method: bodies settle at a small
    equilibrium penetration depth proportional to ``1/ke``.

    The solver organises force evaluation via a :class:`ForcePipeline`.
    Force modules can be enabled / disabled at runtime via
    ``solver.pipeline.enable("name")`` / ``solver.pipeline.disable("name")``.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    Example::

        from wanphys.rigid import RigidModelBuilder
        builder = RigidModelBuilder()
        # ... build scene ...
        model = builder.finalize()
        solver = SymplecticEulerSolver(model)
        s0 = model.state()
        s1 = model.state()
        solver.step(s0, s1, control=None, contacts=None, dt=1/60)
    """

    def __init__(
        self,
        model: RigidModel,
        *,
        angular_damping: float = 0.05,
        friction_smoothing: float = 1.0,
        joint_attach_ke: float = 1.0e4,
        joint_attach_kd: float = 1.0e2,
        enable_mesh_particle_contact: bool = True,
        material_law: int | None = None,
        enable_articulation_dispatch: bool = False,
    ) -> None:
        """Create a new symplectic Euler solver.

        Args:
            model: WanPhys RigidModel describing the scene.
            angular_damping: Damping factor applied to rigid-body angular
                velocities each step.
            friction_smoothing: Huber-norm delta for smoothing tangential
                friction velocity in rigid contacts.
            joint_attach_ke: Spring stiffness for joint attachment
                (positional + orientation).
            joint_attach_kd: Damping coefficient for joint attachment.
            enable_mesh_particle_contact: Whether to evaluate triangle-mesh
                vs. particle contact forces.
            material_law: Selects the hyperelastic constitutive law for FEM
                elements.  ``None`` (default) uses the built-in Neo-Hookean
                kernel; an integer selects from :class:`MaterialLaw`.
            enable_articulation_dispatch: Pre-sort joints by type and
                dispatch specialised kernels to avoid branch divergence.
                D6 joints fall back to the unified kernel automatically.
        """
        self.model: RigidModel = model
        self.device: Any = model.device
        self._step_count: int = 0

        # Solver parameters
        self.angular_damping: float = angular_damping
        self.friction_smoothing: float = friction_smoothing
        self.joint_attach_ke: float = joint_attach_ke
        self.joint_attach_kd: float = joint_attach_kd
        self.enable_mesh_particle_contact: bool = enable_mesh_particle_contact
        self.material_law: int | None = material_law
        self.enable_articulation_dispatch: bool = enable_articulation_dispatch

        # Pre-build articulation dispatcher if requested
        self._art_dispatcher: ArticulationDispatcher | None = None
        if enable_articulation_dispatch and model.joint_count > 0:
            self._art_dispatcher = ArticulationDispatcher(model)

        # Build the force pipeline
        self.pipeline: ForcePipeline = self._build_default_pipeline()

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_default_pipeline(self) -> ForcePipeline:
        pipe: ForcePipeline = ForcePipeline()

        # Elastic forces
        pipe.register("spring_dashpot", self._eval_springs)
        pipe.register("membrane_fem", self._eval_membrane)
        pipe.register("hinge_bending", self._eval_bending)
        pipe.register("solid_fem", self._eval_solid)

        # Articulation constraints
        pipe.register("articulation", self._eval_articulation)

        # Muscle actuators (disabled by default — experimental)
        pipe.register("muscle_actuator", self._eval_muscles, enabled=False)

        # Contact interactions
        pipe.register("particle_proximity", self._eval_particle_proximity)
        pipe.register("mesh_particle_contact", self._eval_mesh_particle_contact,
                       enabled=self.enable_mesh_particle_contact)
        pipe.register("rigid_contact", self._eval_rigid_contact)
        pipe.register("particle_shape_contact", self._eval_particle_shape_contact)

        return pipe

    # ------------------------------------------------------------------
    # Force module callbacks  (each receives the shared context dict)
    # ------------------------------------------------------------------

    def _eval_springs(self, ctx: dict[str, Any]) -> None:
        apply_spring_dashpot(ctx["model"], ctx["state"], ctx["pforce"])

    def _eval_membrane(self, ctx: dict[str, Any]) -> None:
        apply_membrane_stress(ctx["model"], ctx["state"], ctx["control"],
                              ctx["pforce"], self.material_law)

    def _eval_bending(self, ctx: dict[str, Any]) -> None:
        apply_hinge_bending(ctx["model"], ctx["state"], ctx["pforce"])

    def _eval_solid(self, ctx: dict[str, Any]) -> None:
        apply_solid_stress(ctx["model"], ctx["state"], ctx["control"],
                           ctx["pforce"], self.material_law)

    def _eval_articulation(self, ctx: dict[str, Any]) -> None:
        model: RigidModel = ctx["model"]
        state: RigidState = ctx["state"]
        control: Any = ctx["control"]
        bw: wp.array | None = ctx["body_wrench"]

        if self.enable_articulation_dispatch and self._art_dispatcher is not None:
            if self._art_dispatcher.get_type_count("d6") > 0:
                apply_articulation_forces(
                    model, state, control, bw,
                    self.joint_attach_ke, self.joint_attach_kd,
                )
            else:
                apply_articulation_forces_dispatched(
                    model, state, control, bw,
                    self.joint_attach_ke, self.joint_attach_kd,
                    self._art_dispatcher,
                )
        else:
            apply_articulation_forces(
                model, state, control, bw,
                self.joint_attach_ke, self.joint_attach_kd,
            )

    def _eval_muscles(self, ctx: dict[str, Any]) -> None:
        apply_muscle_actuators(ctx["model"], ctx["state"], ctx["control"],
                               ctx["body_wrench"])

    def _eval_particle_proximity(self, ctx: dict[str, Any]) -> None:
        apply_particle_proximity(ctx["model"], ctx["state"], ctx["pforce"])

    def _eval_mesh_particle_contact(self, ctx: dict[str, Any]) -> None:
        apply_mesh_particle_contact(ctx["model"], ctx["state"], ctx["pforce"])

    def _eval_rigid_contact(self, ctx: dict[str, Any]) -> None:
        apply_rigid_contacts(
            ctx["model"], ctx["state"], ctx["contacts"],
            friction_smoothing=self.friction_smoothing,
        )

    def _eval_particle_shape_contact(self, ctx: dict[str, Any]) -> None:
        apply_particle_shape_contact(
            ctx["model"], ctx["state"], ctx["contacts"],
            ctx["pforce"], ctx["body_wrench"],
            wrench_in_world=False,
        )

    # ------------------------------------------------------------------
    # Time-stepping helpers
    # ------------------------------------------------------------------

    def _advance_particles(
        self,
        model: RigidModel,
        state_in: RigidState,
        state_out: RigidState,
        dt: float,
    ) -> None:
        if model.particle_count:
            wp.launch(
                kernel=advance_particles,
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

    def _advance_bodies(
        self,
        model: RigidModel,
        state_in: RigidState,
        state_out: RigidState,
        dt: float,
        ang_damp: float,
    ) -> None:
        if model.body_count:
            wp.launch(
                kernel=advance_bodies,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_inertia,
                    model.body_inv_mass,
                    model.body_inv_inertia,
                    model.gravity,
                    ang_damp,
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=model.device,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        state_in: RigidState,
        state_out: RigidState,
        control: Any,
        contacts: Any,
        dt: float,
    ) -> None:
        """Advance the simulation by one time-step.

        Args:
            state_in: Current state (read-only input).
            state_out: Next state (written as output).
            control: Joint targets and actuator commands.
                ``None`` uses model defaults.
            contacts: Contact data from the collision pipeline.
                ``None`` disables contact forces.
            dt: Time-step size in seconds.
        """
        model: RigidModel = self.model

        with wp.ScopedTimer("simulate", False):
            pforce: wp.array | None = None
            body_wrench: wp.array | None = None

            if model.particle_count:
                pforce = state_in.particle_f

            if model.body_count:
                body_wrench = state_in.body_f

            if control is None:
                control = model.control()

            # Build evaluation context
            ctx: dict[str, Any] = {
                "model": model,
                "state": state_in,
                "control": control,
                "contacts": contacts,
                "pforce": pforce,
                "body_wrench": body_wrench,
            }

            # Run force pipeline
            self.pipeline.evaluate(ctx)

            # Time advance
            self._advance_particles(model, state_in, state_out, dt)
            self._advance_bodies(model, state_in, state_out, dt, self.angular_damping)

        self._step_count += 1
