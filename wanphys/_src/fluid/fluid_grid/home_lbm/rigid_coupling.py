# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Midpoint partitioned schedules for the independent HOME-LBM rigid path."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .diagnostics import HomeLbmDiagnostics
from .domain import HomeLbmDomain
from .lubrication import SphereWallLubrication, SphereWallLubricationConfig
from .motion import RigidMotionPredictor
from .rigid_geometry import RigidGeometryUpdateDiagnostics
from .rigid_interface import HomeLbmRigidInterface
from .state import HomeLbmState
from .transitions import CellTransitionDiagnostics


@dataclass(frozen=True)
class HomeLbmRigidStepDiagnostics:
    """Diagnostics produced by one complete fluid-rigid coupling step."""

    maximum_lattice_displacement: float
    geometry: RigidGeometryUpdateDiagnostics
    transitions: CellTransitionDiagnostics
    fluid: HomeLbmDiagnostics
    rigid_substeps: int
    strong_coupling_iterations: int
    strong_coupling_residual: float


class HomeLbmRigidCoupling:
    """Own the verified ordering of one partitioned HOME-FSI step.

    External force arrays are treated as caller-owned inputs.  The fluid
    wrench is added only while the rigid solver advances, held constant over
    all rigid substeps, then removed by restoring the original force arrays.
    Setting ``strong_coupling_max_iterations`` above one freezes the predicted
    midpoint geometry and solves its wall-velocity/wrench fixed point with
    rollback and safeguarded Aitken relaxation.
    """

    def __init__(
        self,
        fluid_domain: HomeLbmDomain,
        rigid_domain: Any,
        body_ids: list[int] | tuple[int, ...] | None = None,
        rigid_substeps: int = 1,
        displacement_limit: float = 0.5,
        strong_coupling_max_iterations: int = 1,
        strong_coupling_tolerance: float = 1.0e-5,
        strong_coupling_relaxation: float = 0.5,
        lubrication: SphereWallLubricationConfig | None = None,
    ) -> None:
        if rigid_substeps < 1:
            raise ValueError("rigid_substeps must be at least one")
        if not 0.0 < displacement_limit <= 1.0:
            raise ValueError("displacement_limit must be in (0, 1]")
        if strong_coupling_max_iterations < 1:
            raise ValueError("strong_coupling_max_iterations must be at least one")
        if strong_coupling_tolerance <= 0.0:
            raise ValueError("strong_coupling_tolerance must be positive")
        if not 0.0 < strong_coupling_relaxation <= 1.0:
            raise ValueError("strong_coupling_relaxation must be in (0, 1]")
        if rigid_domain.model.body_count < 1:
            raise ValueError("HOME rigid coupling requires at least one rigid body")

        self.fluid_domain = fluid_domain
        self.rigid_domain = rigid_domain
        self.rigid_substeps = int(rigid_substeps)
        self.displacement_limit = float(displacement_limit)
        self.strong_coupling_max_iterations = int(strong_coupling_max_iterations)
        self.strong_coupling_tolerance = float(strong_coupling_tolerance)
        self.strong_coupling_relaxation = float(strong_coupling_relaxation)
        self.interface = HomeLbmRigidInterface(
            fluid_domain,
            rigid_domain,
            body_ids=body_ids,
        )
        self.lubrication = (
            None
            if lubrication is None
            else SphereWallLubrication(
                fluid_domain.model,
                rigid_domain,
                lubrication,
                body_ids=body_ids,
                rigid_substeps=rigid_substeps,
            )
        )
        if self.lubrication is not None:
            fluid_domain.solver.enable_wall_confined_lubrication_closure()
        self.predictor = RigidMotionPredictor(
            rigid_domain.model.body_count,
            device=fluid_domain.model._device,
        )
        self._strong_fluid_input: HomeLbmState | None = None
        self._strong_fluid_prepared: HomeLbmState | None = None
        self._strong_rigid_input: Any | None = None
        self._strong_velocity_guess: wp.array | None = None
        self._radius_bound_host = (
            self.interface.geometry.radius_bound.numpy().astype(np.float64)
        )

    def step(
        self,
        dt: float | None = None,
        contacts: object | None = None,
        contact_provider: Callable[[Any], object] | None = None,
    ) -> HomeLbmRigidStepDiagnostics:
        """Advance fluid once and rigid dynamics through configured substeps.

        ``contact_provider`` is evaluated before every rigid substep.  Passing
        fixed ``contacts`` and a provider together is rejected because stale
        contact ownership would otherwise be ambiguous.
        """

        if contacts is not None and contact_provider is not None:
            raise ValueError("pass either fixed contacts or contact_provider, not both")
        fluid_dt = self.fluid_domain.model.time_step if dt is None else float(dt)
        self.fluid_domain.model.validate_step(fluid_dt)

        rigid_state = self.rigid_domain.state
        body_q = rigid_state.body_q
        body_qd = rigid_state.body_qd
        if body_q is None or body_qd is None:
            raise RuntimeError("HOME rigid coupling requires rigid body state arrays")

        if self.strong_coupling_max_iterations > 1:
            return self._step_strong(
                fluid_dt,
                contacts,
                contact_provider,
                body_q,
                body_qd,
            )

        maximum_displacement = self.predictor.validate_lattice_displacement(
            body_qd,
            self.interface.geometry.radius_bound,
            self.fluid_domain.model.fluid_grid_cell_size,
            fluid_dt,
            limit=self.displacement_limit,
        )
        self.predictor.sample(
            body_q,
            body_qd,
            self.rigid_domain.model.body_com,
            fluid_dt,
        )
        transitions = self.interface.prepare_step(
            self.predictor.body_q_half,
            body_qd,
        )

        self.fluid_domain.step(fluid_dt)
        fluid_diagnostics = self.fluid_domain.solver.validate_state(
            self.fluid_domain.state
        )

        external_body_f = self._clone_optional(rigid_state.body_f)
        external_particle_f = self._clone_optional(rigid_state.particle_f)
        self.interface.collect_wrench(
            self.predictor.body_q_half,
            rigid_state.body_f,
        )
        if self.lubrication is not None:
            self.lubrication.add_to_rigid_forces(
                self.predictor.body_q_half,
                body_qd,
                rigid_state.body_f,
            )
        total_body_f = self._clone_optional(rigid_state.body_f)

        rigid_dt = fluid_dt / self.rigid_substeps
        for substep in range(self.rigid_substeps):
            if substep:
                self._copy_optional(self.rigid_domain.state.body_f, total_body_f)
                self._copy_optional(
                    self.rigid_domain.state.particle_f,
                    external_particle_f,
                )
            substep_contacts = (
                contact_provider(self.rigid_domain)
                if contact_provider is not None
                else contacts
            )
            self.rigid_domain.step(rigid_dt, contacts=substep_contacts)
            if self.lubrication is not None:
                self.lubrication.project_contacts(
                    self.rigid_domain.state.body_q,
                    self.rigid_domain.state.body_qd,
                )

        self._copy_optional(self.rigid_domain.state.body_f, external_body_f)
        self._copy_optional(
            self.rigid_domain.state.particle_f,
            external_particle_f,
        )
        return HomeLbmRigidStepDiagnostics(
            maximum_lattice_displacement=maximum_displacement,
            geometry=self.interface.last_geometry_update,
            transitions=transitions,
            fluid=fluid_diagnostics,
            rigid_substeps=self.rigid_substeps,
            strong_coupling_iterations=1,
            strong_coupling_residual=0.0,
        )

    def _step_strong(
        self,
        fluid_dt: float,
        contacts: object | None,
        contact_provider: Callable[[Any], object] | None,
        body_q: wp.array,
        body_qd: wp.array,
    ) -> HomeLbmRigidStepDiagnostics:
        """Solve the midpoint wall-velocity/wrench fixed point with rollback."""

        self._ensure_strong_buffers()
        assert self._strong_fluid_input is not None
        assert self._strong_fluid_prepared is not None
        assert self._strong_rigid_input is not None
        assert self._strong_velocity_guess is not None

        fluid_input = self._strong_fluid_input
        fluid_prepared = self._strong_fluid_prepared
        rigid_input = self._strong_rigid_input
        fluid_input.copy_from(self.fluid_domain.state)
        rigid_input.assign(self.rigid_domain.state)

        self.predictor.validate_lattice_displacement(
            body_qd,
            self.interface.geometry.radius_bound,
            self.fluid_domain.model.fluid_grid_cell_size,
            fluid_dt,
            limit=self.displacement_limit,
        )

        initial_qd = body_qd.numpy().astype(np.float64)
        guess = initial_qd.copy()
        self._strong_velocity_guess.assign(guess.astype(np.float32))
        self.predictor.sample(
            body_q,
            self._strong_velocity_guess,
            self.rigid_domain.model.body_com,
            fluid_dt,
        )
        final_transitions = self.interface.prepare_step(
            self.predictor.body_q_half,
            self._strong_velocity_guess,
        )
        fluid_prepared.copy_from(self.fluid_domain.state)

        previous_residual: np.ndarray | None = None
        relaxation = self.strong_coupling_relaxation
        residual_norm = float("inf")
        converged_iteration = 0
        iteration_history: list[tuple[float, float]] = []

        for iteration in range(1, self.strong_coupling_max_iterations + 1):
            if iteration > 1:
                # Geometry is frozen for the fixed-point solve and both fluid
                # buffers acquired it during the first stream-collide step.
                wp.copy(self.fluid_domain.state.moments, fluid_prepared.moments)
                self.rigid_domain.state.assign(rigid_input)
                self._strong_velocity_guess.assign(guess.astype(np.float32))
            self.interface.cut_links.set_rigid_wall_velocity(
                self.predictor.body_q_half,
                self._strong_velocity_guess,
                self.rigid_domain.model.body_com,
                self.fluid_domain.model.fluid_grid_cell_size,
                fluid_dt,
            )
            self.fluid_domain.step(fluid_dt)

            rigid_state = self.rigid_domain.state
            external_particle_f = self._clone_optional(rigid_state.particle_f)
            self.interface.collect_wrench(
                self.predictor.body_q_half,
                rigid_state.body_f,
            )
            if self.lubrication is not None:
                self.lubrication.add_to_rigid_forces(
                    self.predictor.body_q_half,
                    self._strong_velocity_guess,
                    rigid_state.body_f,
                )
            total_body_f = self._clone_optional(rigid_state.body_f)
            self._advance_rigid_substeps(
                fluid_dt,
                contacts,
                contact_provider,
                total_body_f,
                external_particle_f,
            )

            candidate_qd = self.rigid_domain.state.body_qd.numpy().astype(np.float64)
            target = 0.5 * (initial_qd + candidate_qd)
            residual = target - guess
            residual_norm = self._surface_velocity_residual(residual, fluid_dt)
            iteration_history.append((residual_norm, relaxation))
            if residual_norm <= self.strong_coupling_tolerance:
                converged_iteration = iteration
                break

            if previous_residual is not None:
                residual_delta = residual - previous_residual
                denominator = float(np.sum(residual_delta * residual_delta))
                if denominator > 1.0e-30:
                    relaxation = -relaxation * float(
                        np.sum(previous_residual * residual_delta)
                    ) / denominator
                    relaxation = float(
                        np.clip(
                            relaxation,
                            0.02,
                            self.strong_coupling_relaxation,
                        )
                    )
            guess += relaxation * residual
            previous_residual = residual

        if converged_iteration == 0:
            self._restore_strong_inputs(fluid_input, rigid_input)
            raise RuntimeError(
                "strong HOME rigid coupling failed to converge after "
                f"{self.strong_coupling_max_iterations} iterations; "
                f"surface lattice-velocity residual={residual_norm:.6e}, "
                f"tolerance={self.strong_coupling_tolerance:.6e}, "
                f"history={iteration_history}"
            )

        self._copy_optional(self.rigid_domain.state.body_f, rigid_input.body_f)
        self._copy_optional(
            self.rigid_domain.state.particle_f,
            rigid_input.particle_f,
        )
        fluid_diagnostics = self.fluid_domain.solver.validate_state(
            self.fluid_domain.state
        )
        maximum_displacement = self.predictor.validate_lattice_displacement(
            self.rigid_domain.state.body_qd,
            self.interface.geometry.radius_bound,
            self.fluid_domain.model.fluid_grid_cell_size,
            fluid_dt,
            limit=self.displacement_limit,
        )
        assert final_transitions is not None
        return HomeLbmRigidStepDiagnostics(
            maximum_lattice_displacement=maximum_displacement,
            geometry=self.interface.last_geometry_update,
            transitions=final_transitions,
            fluid=fluid_diagnostics,
            rigid_substeps=self.rigid_substeps,
            strong_coupling_iterations=converged_iteration,
            strong_coupling_residual=residual_norm,
        )

    def _ensure_strong_buffers(self) -> None:
        if self._strong_fluid_input is None:
            self._strong_fluid_input = HomeLbmState(self.fluid_domain.model)
        if self._strong_fluid_prepared is None:
            self._strong_fluid_prepared = HomeLbmState(self.fluid_domain.model)
        if self._strong_rigid_input is None:
            self._strong_rigid_input = self.rigid_domain.model.state()
        if self._strong_velocity_guess is None:
            self._strong_velocity_guess = wp.zeros(
                self.rigid_domain.model.body_count,
                dtype=wp.spatial_vector,
                device=self.fluid_domain.model._device,
            )

    def _advance_rigid_substeps(
        self,
        fluid_dt: float,
        contacts: object | None,
        contact_provider: Callable[[Any], object] | None,
        total_body_f: wp.array | None,
        external_particle_f: wp.array | None,
    ) -> None:
        rigid_dt = fluid_dt / self.rigid_substeps
        for substep in range(self.rigid_substeps):
            if substep:
                self._copy_optional(self.rigid_domain.state.body_f, total_body_f)
                self._copy_optional(
                    self.rigid_domain.state.particle_f,
                    external_particle_f,
                )
            substep_contacts = (
                contact_provider(self.rigid_domain)
                if contact_provider is not None
                else contacts
            )
            self.rigid_domain.step(rigid_dt, contacts=substep_contacts)
            if self.lubrication is not None:
                self.lubrication.project_contacts(
                    self.rigid_domain.state.body_q,
                    self.rigid_domain.state.body_qd,
                )

    def _surface_velocity_residual(
        self,
        residual: np.ndarray,
        fluid_dt: float,
    ) -> float:
        linear = np.linalg.norm(residual[:, :3], axis=1)
        angular = np.linalg.norm(residual[:, 3:], axis=1)
        physical_surface_speed = linear + angular * self._radius_bound_host
        return float(
            np.max(physical_surface_speed)
            * fluid_dt
            / self.fluid_domain.model.fluid_grid_cell_size
        )

    def _restore_strong_inputs(
        self,
        fluid_input: HomeLbmState,
        rigid_input: Any,
    ) -> None:
        self.fluid_domain.state.copy_from(fluid_input)
        self.rigid_domain.state.assign(rigid_input)
        self.interface.remapper.capture_geometry(self.fluid_domain.state)
        self.interface.last_geometry_update = self.interface.geometry.rasterize(
            self.fluid_domain.state,
            rigid_input.body_q,
        )
        self.interface.cut_links = self.interface._build_links(
            rigid_input.body_q,
            rigid_input.body_qd,
        )
        self.fluid_domain.solver.set_cut_links(self.interface.cut_links)

    @staticmethod
    def _clone_optional(array: wp.array | None) -> wp.array | None:
        return None if array is None else wp.clone(array)

    @staticmethod
    def _copy_optional(destination: wp.array | None, source: wp.array | None) -> None:
        if destination is None and source is None:
            return
        if destination is None or source is None:
            raise RuntimeError("rigid force-array ownership changed during coupling step")
        wp.copy(destination, source)
