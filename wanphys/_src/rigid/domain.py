# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid body domain - isolates Newton dependency via RigidModel/Solver/State."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from wanphys._src.core.domain import Domain
from .model import RigidModel
from .solver import RigidSolver, create_xpbd_solver
from .state import RigidState

if TYPE_CHECKING:
    import numpy as np


class RigidDomain(Domain):
    """Domain for rigid body and articulation simulation.

    RigidDomain provides rigid body physics including:
    - Rigid bodies with mass, inertia, collision shapes
    - Joints and articulations (revolute, prismatic, ball, fixed, free, etc.)
    - Collision detection and contact resolution
    - Joint motors and limits

    This domain uses an isolation layer (RigidModel/Solver/State) to decouple
    from Newton, allowing future replacement with custom WanPhys implementations.

    Example:
        >>> from wanphys.rigid import RigidModelBuilder, create_xpbd_solver
        >>> builder = RigidModelBuilder()
        >>> builder.add_body(...)
        >>> builder.add_joint_revolute(...)
        >>> model = builder.finalize()
        >>>
        >>> # Init domain and state
        >>> solver = create_xpbd_solver(model)
        >>> rigid = RigidDomain(model, solver=solver)
        >>> rigid.create_state()
        >>>
        >>> # Simulation loop
        >>> for _ in range(steps):
        ...     contacts = CollisionPipeline.collide_rigid(rigid)
        ...     rigid.step(dt=1/60, contacts=contacts)
    """

    def __init__(
        self,
        model: RigidModel,
        solver: RigidSolver | None = None,
    ) -> None:
        """Initialize rigid body domain.

        Args:
            model: RigidModel instance.
            solver: Optional RigidSolver instance. When omitted, WanPhys
                creates its default XPBD-backed solver internally.

        Note:
            Prefer `RigidDomain(rigid_model, solver=rigid_solver)` in new
            code.
        """
        if not isinstance(model, RigidModel):
            raise TypeError("RigidDomain requires a RigidModel. Build one through RigidModelBuilder.finalize().")

        self._model: RigidModel = model
        self._solver: RigidSolver = solver if solver is not None else create_xpbd_solver(self._model)

        self._control: Any | None = None

        # Double-buffered state (lazily created)
        self._state_in: RigidState | None = None
        self._state_out: RigidState | None = None

    def _ensure_state(self) -> None:
        """Lazily create internal double-buffered states."""
        if self._state_in is None:
            self.create_state()

    @property
    def state(self) -> RigidState:
        """Current simulation state (read from the active buffer)."""
        self._ensure_state()
        return self._state_in

    @property
    def name(self) -> str:
        """Domain identifier."""
        return "rigid"

    @property
    def model(self) -> RigidModel:
        """RigidModel instance."""
        return self._model

    @property
    def solver(self) -> RigidSolver:
        """RigidSolver instance."""
        return self._solver

    @property
    def control(self) -> Any:
        """Current control input.

        Lazily created on first access. Use this to set joint targets,
        muscle activations, etc.
        """
        if self._control is None:
            self._control = self._model.control()
        return self._control

    def create_state(self) -> None:
        """Initialize internal double-buffered states from the model.

        Must be called after the domain and model are fully configured.
        After this, access the current state via ``domain.state``.
        """
        self._state_in = self._model.state()
        self._state_out = self._model.state()

    def step(
        self,
        dt: float = 1.0 / 60.0,
        contacts: Any = None,
    ) -> None:
        """Advance rigid body simulation by dt.

        Performs collision detection and physics integration using
        internal double-buffered state, then swaps buffers.

        Args:
            dt: Timestep in seconds.
            contacts: Optional DomainContacts from CollisionPipeline.  When
                None, falls back to running collision detection directly
                (backward-compatible).
        """
        self._ensure_state()

        # Use pre-computed contacts, or run collision detection
        if contacts is not None:
            raw = contacts
        else:
            from wanphys._src.collision.pipeline import CollisionPipeline

            raw = CollisionPipeline.collide_rigid(self)

        # Physics integration via RigidSolver
        self._solver.step(self._state_in, self._state_out, self.control, raw, dt)

        # Swap buffers so self.state always points to the latest result
        self._state_in, self._state_out = self._state_out, self._state_in

    def update_contacts(self, contacts: Any, state: RigidState | None = None) -> None:
        """Update a Contacts object from the active rigid solver state.

        This keeps examples and client code on the WanPhys domain/solver
        surface while Newton-backed solvers still populate contact forces.
        """
        self._ensure_state()
        self._solver.update_contacts(contacts, state or self._state_in)

    def get_max_contact_count(self, default: int | None = None) -> int | None:
        """Return active solver contact capacity when the solver exposes one."""
        return self._solver.get_max_contact_count(default=default)

    def set_joint_targets(
        self,
        positions: dict[str, float] | None = None,
        velocities: dict[str, float] | None = None,
    ) -> None:
        """Set joint target positions and/or velocities by joint key.

        Args:
            positions: Map from joint key to target position.
            velocities: Map from joint key to target velocity.
        """
        # TODO: Implement joint key lookup and target setting
        raise NotImplementedError("Joint target setting by key not yet implemented")

    def get_body_transform(self, body_idx: int, state: RigidState | None = None) -> np.ndarray:
        """Get world transform for a body.

        Args:
            body_idx: Body index.
            state: Simulation state. If None, uses internal state.

        Returns:
            Transform (position, quaternion) of the body.
        """
        if state is None:
            state = self.state
        return state.get_body_transform(body_idx)

    def get_body_velocity(self, body_idx: int, state: RigidState | None = None) -> np.ndarray:
        """Get spatial velocity for a body.

        Args:
            body_idx: Body index.
            state: Simulation state. If None, uses internal state.

        Returns:
            Spatial velocity (linear, angular) of the body.
        """
        if state is None:
            state = self.state
        return state.get_body_velocity(body_idx)

