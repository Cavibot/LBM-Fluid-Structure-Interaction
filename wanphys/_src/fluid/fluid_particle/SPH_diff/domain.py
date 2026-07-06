# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""SPHDiff domain - Weakly Compressible SPH domain implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton import Model as NewtonModel

from wanphys.core import Domain
from .model import SPHDiffModel
import warp as wp
import numpy as np
from .state import SPHDiffState
from .solver import SPHDiffSolver



from .kernels import RigidBodies, MaterialType


class SPHDiffDomain(Domain):
    """Domain implementation for SPH_diff SPHDiff baseline."""

    def __init__(
        self,
        newton_model: NewtonModel,
        sphdiff_model: SPHDiffModel | None = None,
        solver: SPHDiffSolver | None = None,
        target_time: float | None = None,
        target_timesteps: int | None = None,
        dt: float | None = None,
        task: object | None = None,
    ):
        """Initialize SPHDiff domain.

        Args:
            newton_model: Newton Model with particles (and optionally rigid bodies).
            sphdiff_model: SPHDiff configuration (optional, uses defaults if None).
            solver: SPHDiff solver instance (optional, created if None).
        """
        # Use the raw newton.Model as the model adapter (no deprecated shim).
        self._model_adapter = newton_model

        # Use provided or default SPHDiff model
        if sphdiff_model is None:
            sphdiff_model = SPHDiffModel(device=newton_model.device)
        self._sphdiff_model = sphdiff_model

        # Use provided or create default solver
        if solver is None:
            solver = SPHDiffSolver(self._model_adapter, sphdiff_model)
        self._solver = solver

        self.target_time = float(target_time) if target_time is not None else 0.0
        self.dt = float(dt) if dt is not None else 0.0

        # Backward-compatible path: if only target_timesteps is provided, infer target_time.
        if self.target_time <= 0.0 and target_timesteps is not None and self.dt > 0.0:
            self.target_time = float(target_timesteps) * self.dt

        self._step_count = 0
        self._sim_time_accum = 0.0
        self._backward_done = False
        self._request_reset = False
        self._task = task
        self._backward_round = 0
        self._last_loss_value = None
        self._state_in: SPHDiffState | None = None
        self._state_out: SPHDiffState | None = None
        self._initial_state_buffers: dict[str, wp.array] | None = None

    @property
    def name(self) -> str:
        """Domain identifier."""
        return "sphdiff"

    @property
    def model(self) -> SPHDiffModel:
        """SPHDiff model configuration."""
        return self._sphdiff_model

    @property
    def model_adapter(self) -> NewtonModel:
        """Newton model (used as adapter)."""
        return self._model_adapter

    @property
    def solver(self) -> SPHDiffSolver:
        """SPHDiff solver instance."""
        return self._solver

    @property
    def state(self) -> SPHDiffState:
        """Current simulation state (active buffer)."""
        if self._state_in is None:
            self.create_state()
        return self._state_in

    @property
    def backward_round(self) -> int:
        """Number of completed backward rounds."""
        return self._backward_round

    @property
    def last_loss_value(self):
        """Most recent scalar loss value (numpy-compatible)."""
        return self._last_loss_value



    def consume_reset_request(self) -> bool:
        """Consume and clear reset request flag (used by DiffCompositeSimulation)."""
        if self._request_reset:
            self._request_reset = False
            return True
        return False

    def on_simulation_reset(self) -> None:
        """Reset runtime counters/flags after composite reset."""
        self._step_count = 0
        self._sim_time_accum = 0.0
        self._backward_done = False
        self._request_reset = False
        if self._task is not None:
            optimizer = getattr(self._task, "optimizer", None)
            opt_var = getattr(self._task, "opt_var", None)
            if optimizer is not None and opt_var is not None:
                grads = [getattr(v, "grad", None) for v in opt_var]
                if all(g is not None for g in grads):
                    grad_norms = [float(np.linalg.norm(g.numpy())) for g in grads]
                    print(f"[SPH] opt grad norms={grad_norms}")
                    optimizer.step(grads)
                    print(f"[SPH] optimizer.step done, opt_var={[v.numpy() for v in opt_var]}")

            self._task.state.clear_grad()
            self._task.clear_grad()
            for tape in self._solver.tapes:
                tape.reset()
                tape.zero()
            self._solver.tapes = []
            # Rebuild a clean initial state for the next optimization round.
            self.create_state()
            if hasattr(self._state_in, "initialize_rigid_bodies"):
                self._state_in.initialize_rigid_bodies()
            if hasattr(self._state_out, "initialize_rigid_bodies"):
                self._state_out.initialize_rigid_bodies()
            self._task.state = self._state_in
            if hasattr(self._task, "init_targets"):
                self._task.init_targets()
        if hasattr(self._solver, "_runtime_task_initialized"):
            self._solver._runtime_task_initialized = False

    def on_reset_states(self, state_in: SPHDiffState, state_out: SPHDiffState) -> None:
        """Reinitialize per-state rigid-body buffers after composite reset."""
        for state in (state_in, state_out):
            if hasattr(state, "initialize_rigid_bodies"):
                state.initialize_rigid_bodies()

    
    def create_state(self) -> SPHDiffState:
        """Create a new SPHDiff state.

        Returns:
            SPHDiffState initialized from Newton model.
        """
        newton_state = self._model_adapter.state()

        if self._initial_state_buffers is None:
            self._initial_state_buffers = {}
            for attr_name in ("particle_q", "particle_qd", "particle_f", "body_q", "body_qd", "body_f", "joint_q", "joint_qd"):
                buffer = getattr(newton_state, attr_name, None)
                if buffer is not None:
                    self._initial_state_buffers[attr_name] = wp.clone(buffer)
        else:
            for attr_name, snapshot in self._initial_state_buffers.items():
                buffer = getattr(newton_state, attr_name, None)
                if buffer is not None and snapshot is not None:
                    wp.copy(buffer, snapshot)

        particle_x0 = wp.clone(newton_state.particle_q) if hasattr(newton_state, "particle_q") else None

        particle_mass = self._model_adapter.particle_mass
        mass_np = particle_mass.numpy() if hasattr(particle_mass, "numpy") else particle_mass
        object_id_np = np.zeros_like(mass_np, dtype=np.int32)
        object_id_np[np.abs(mass_np) == 0.0] = 1
        object_id = wp.array(object_id_np, dtype=wp.int32, device=newton_state.particle_q.device)
        state = SPHDiffState(newton_state, self._sphdiff_model, rbs=None, material_marks=None, object_id=object_id, particle_x0=particle_x0)

        n = state._state.particle_q.shape[0]
        rho_np = np.zeros(n, dtype=np.float32)
        boundary_idx_np_state = state.boundary_indices.numpy() if state.boundary_indices is not None else np.array([], dtype=np.int32)
        if boundary_idx_np_state.size > 0:
            rho_np[boundary_idx_np_state] = float(state._model.rigid_density)
        state._rho = wp.array(rho_np, dtype=wp.float32, device=state._model._device, requires_grad=True)

        self._state_in = state
        self._state_out = SPHDiffState(newton_state, self._sphdiff_model, rbs=None, material_marks=None, object_id=object_id, particle_x0=particle_x0)
        self._state_out._rho = wp.array(rho_np, dtype=wp.float32, device=state._model._device, requires_grad=True)

        return self._state_in

    def step(
        self,
        state_in: SPHDiffState,
        state_out: SPHDiffState,
        dt: float,
    ) -> None:
        """Advance SPHDiff simulation by dt.

        Args:
            state_in: Current state.
            state_out: State to write results into.
            dt: Timestep in seconds.
        """
        # Optional: collision detection for rigid-fluid coupling
        contacts = None
        if self._model_adapter.shape_count > 0:
            contacts = self._model_adapter.collide(state_in)

        # SPHDiff integration
        if hasattr(self._solver, "_runtime_task"):
            self._solver._runtime_task = self._task
        self._solver.step(state_in, state_out, dt, contacts=contacts)

    def pre_step(self, state: SPHDiffState, dt: float) -> None:
        """Pre-step hook - clear forces.

        Args:
            state: Current state.
            dt: Upcoming timestep.
        """
        state.clear_forces()

    def post_step(self, state: SPHDiffState, dt: float) -> None:
        self._step_count += 1
        self._sim_time_accum += float(dt)

        if self._task is not None and hasattr(self._task, "state"):
            self._task.state = state

        if self._backward_done:
            return

        if self.target_time <= 0.0 or self._sim_time_accum + 1.0e-12 < self.target_time:
            return

        if self._task is None:
            print("[SPH] target sim time reached but no task is bound; skip backward and request reset.")
            self._backward_done = True
            self._request_reset = True
            return

        loss = wp.zeros((1,), dtype=float, requires_grad=True)
        loss_tape = wp.Tape()
        with loss_tape:
            self._task.compute_loss(state, loss)

        loss_value = loss.numpy()
        if not np.isfinite(loss_value).all():
            print(f"[SPH] invalid loss at step={self._step_count}: {loss_value}; skip backward and request reset.")
            self._backward_done = True
            self._request_reset = True
            return

        loss_tape.backward(loss)

        if hasattr(state, "rbs") and getattr(state.rbs, "rigid_x", None) is not None and state.rbs.rigid_x.grad is not None:
            rigid_x_grad_np = state.rbs.rigid_x.grad.numpy()
            sample_idx = min(1, max(0, rigid_x_grad_np.shape[0] - 1))
        solver_tapes = getattr(self._solver, "tapes", [])
        for tape in reversed(solver_tapes):
            tape.backward()

        print(
            f"[SPH] backward triggered at t={self._sim_time_accum:.6f}s "
            f"(step={self._step_count}), loss={loss_value}"
        )
        if hasattr(self._task, "opt_rigid_v") and getattr(self._task.opt_rigid_v, "grad", None) is not None:
            print(f"[SPH] grad opt_rigid_v: {self._task.opt_rigid_v.grad.numpy()}")
        if hasattr(self._task, "opt_rigid_omega") and getattr(self._task.opt_rigid_omega, "grad", None) is not None:
            print(f"[SPH] grad opt_rigid_omega: {self._task.opt_rigid_omega.grad.numpy()}")

        self._last_loss_value = np.array(loss_value, copy=True)
        self._backward_round += 1

        self._backward_done = True
        self._request_reset = True
        print("[SPH] request composite reset after backward.")

    def get_particle_positions(self, state: SPHDiffState):
        """Get particle positions as numpy array.

        Args:
            state: Current state.

        Returns:
            Numpy array of shape (N, 3) with particle positions.
        """
        return state.particle_q.numpy()

    def get_particle_velocities(self, state: SPHDiffState):
        """Get particle velocities as numpy array.

        Args:
            state: Current state.

        Returns:
            Numpy array of shape (N, 3) with particle velocities.
        """
        return state.particle_qd.numpy()

    def get_particle_densities(self, state: SPHDiffState):
        """Get particle densities as numpy array.

        Args:
            state: Current state.

        Returns:
            Numpy array of shape (N,) with particle densities.
        """
        return state.rho.numpy()

    def get_particle_pressures(self, state: SPHDiffState):
        """Get particle pressures as numpy array.

        Args:
            state: Current state.

        Returns:
            Numpy array of shape (N,) with particle pressures.
        """
        return state.pressure.numpy()
