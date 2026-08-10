# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit midpoint rigid-body schedule for HOME-Free LBM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from ..cut_link import CutLinkBuffer
from ..motion import RigidMotionPredictor
from .domain import HomeFreeDomain, HomeFreeDomainState
from .rigid_interface import (
    HomeFreeRigidDynamicStepDiagnostics,
    HomeFreeRigidInterface,
)


@dataclass(frozen=True)
class HomeFreeRigidStepDiagnostics:
    """Diagnostics for one complete explicit HOME-Free rigid step."""

    maximum_lattice_displacement: float
    interface: HomeFreeRigidDynamicStepDiagnostics
    rigid_substeps: int
    strong_coupling_iterations: int
    strong_coupling_residual: float


class HomeFreeRigidCoupling:
    """Advance one HOME-Free step and caller-owned rigid dynamics.

    The fluid wrench is present only while rigid substeps advance. External
    force arrays are restored afterwards, matching the ownership contract of
    the verified single-phase HOME coupling. Strong mode freezes midpoint
    geometry/topology and solves the wall-velocity/wrench fixed point with
    full HOME-Free and rigid rollback.
    """

    def __init__(
        self,
        fluid_domain: HomeFreeDomain,
        rigid_domain: Any,
        body_ids: list[int] | tuple[int, ...] | None = None,
        *,
        rigid_substeps: int = 1,
        displacement_limit: float = 0.5,
        strong_coupling_max_iterations: int = 1,
        strong_coupling_tolerance: float = 1.0e-5,
        strong_coupling_relaxation: float = 0.5,
        interface: HomeFreeRigidInterface | None = None,
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
            raise ValueError("HOME-Free rigid coupling requires at least one body")
        if interface is not None and (
            interface.fluid_domain is not fluid_domain
            or interface.rigid_domain is not rigid_domain
        ):
            raise ValueError("injected interface must own the coupled domains")
        self.fluid_domain = fluid_domain
        self.rigid_domain = rigid_domain
        self.rigid_substeps = int(rigid_substeps)
        self.displacement_limit = float(displacement_limit)
        self.strong_coupling_max_iterations = int(strong_coupling_max_iterations)
        self.strong_coupling_tolerance = float(strong_coupling_tolerance)
        self.strong_coupling_relaxation = float(strong_coupling_relaxation)
        self.interface = interface or HomeFreeRigidInterface(
            fluid_domain,
            rigid_domain,
            body_ids=body_ids,
        )
        self.predictor = RigidMotionPredictor(
            rigid_domain.model.body_count,
            device=fluid_domain.model._device,
        )
        self._strong_fluid_input: HomeFreeDomainState | None = None
        self._strong_fluid_prepared: HomeFreeDomainState | None = None
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
    ) -> HomeFreeRigidStepDiagnostics:
        if contacts is not None and contact_provider is not None:
            raise ValueError("pass either fixed contacts or contact_provider, not both")
        fluid_dt = self.fluid_domain.model.time_step if dt is None else float(dt)
        self.fluid_domain.model.validate_step(fluid_dt)
        rigid_state = self.rigid_domain.state
        body_q = rigid_state.body_q
        body_qd = rigid_state.body_qd
        if body_q is None or body_qd is None:
            raise RuntimeError("HOME-Free rigid coupling requires body state arrays")
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
        self.interface.prepare_step(
            self.predictor.body_q_half,
            body_qd,
        )
        interface_diagnostics = self.interface.step_prepared(
            self.predictor.body_q_half,
            fluid_dt,
        )

        external_body_f = self._clone_optional(rigid_state.body_f)
        external_particle_f = self._clone_optional(rigid_state.particle_f)
        self.interface.wrench.add_to_rigid_forces(
            rigid_state.body_f,
            self.fluid_domain.model.scaling,
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
        self._copy_optional(self.rigid_domain.state.body_f, external_body_f)
        self._copy_optional(
            self.rigid_domain.state.particle_f,
            external_particle_f,
        )
        return HomeFreeRigidStepDiagnostics(
            maximum_lattice_displacement=maximum_displacement,
            interface=interface_diagnostics,
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
    ) -> HomeFreeRigidStepDiagnostics:
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
        previous_fluid_diagnostics = self.fluid_domain._last_diagnostics
        previous_surface_diagnostics = (
            self.fluid_domain._last_surface_tension_diagnostics
        )
        input_transaction = self._capture_optional_transaction_history()

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

        try:
            self.interface.prepare_step(
                self.predictor.body_q_half,
                self._strong_velocity_guess,
            )
            fluid_prepared.copy_from(self.fluid_domain.state)
            prepared_transaction = self._capture_optional_transaction_history()
            previous_residual: np.ndarray | None = None
            relaxation = self.strong_coupling_relaxation
            residual_norm = float("inf")
            converged_iteration = 0
            iteration_history: list[tuple[float, float]] = []
            final_interface: HomeFreeRigidDynamicStepDiagnostics | None = None

            for iteration in range(1, self.strong_coupling_max_iterations + 1):
                if iteration > 1:
                    self.fluid_domain.state.copy_from(fluid_prepared)
                    self._restore_optional_transaction_history(prepared_transaction)
                    self.rigid_domain.state.assign(rigid_input)
                    self._strong_velocity_guess.assign(guess.astype(np.float32))
                self.interface.set_prepared_wall_velocity(
                    self.predictor.body_q_half,
                    self._strong_velocity_guess,
                    fluid_dt,
                )
                final_interface = self.interface.step_prepared(
                    self.predictor.body_q_half,
                    fluid_dt,
                    retain_prepared=True,
                )

                rigid_state = self.rigid_domain.state
                external_particle_f = self._clone_optional(rigid_state.particle_f)
                self.interface.wrench.add_to_rigid_forces(
                    rigid_state.body_f,
                    self.fluid_domain.model.scaling,
                )
                total_body_f = self._clone_optional(rigid_state.body_f)
                self._advance_rigid_substeps(
                    fluid_dt,
                    contacts,
                    contact_provider,
                    total_body_f,
                    external_particle_f,
                )

                candidate_qd = self.rigid_domain.state.body_qd.numpy().astype(
                    np.float64
                )
                target = 0.5 * (initial_qd + candidate_qd)
                residual = target - guess
                residual_norm = self._surface_velocity_residual(
                    residual, fluid_dt
                )
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
                raise RuntimeError(
                    "strong HOME-Free rigid coupling failed to converge after "
                    f"{self.strong_coupling_max_iterations} iterations; "
                    f"surface lattice-velocity residual={residual_norm:.6e}, "
                    f"tolerance={self.strong_coupling_tolerance:.6e}, "
                    f"history={iteration_history}"
                )

            self.interface.finish_prepared_step()
            self._copy_optional(self.rigid_domain.state.body_f, rigid_input.body_f)
            self._copy_optional(
                self.rigid_domain.state.particle_f,
                rigid_input.particle_f,
            )
            maximum_displacement = self.predictor.validate_lattice_displacement(
                self.rigid_domain.state.body_qd,
                self.interface.geometry.radius_bound,
                self.fluid_domain.model.fluid_grid_cell_size,
                fluid_dt,
                limit=self.displacement_limit,
            )
            assert final_interface is not None
            return HomeFreeRigidStepDiagnostics(
                maximum_lattice_displacement=maximum_displacement,
                interface=final_interface,
                rigid_substeps=self.rigid_substeps,
                strong_coupling_iterations=converged_iteration,
                strong_coupling_residual=residual_norm,
            )
        except Exception:
            self._restore_strong_inputs(
                fluid_input,
                rigid_input,
                previous_fluid_diagnostics,
                previous_surface_diagnostics,
            )
            self._restore_optional_transaction_history(input_transaction)
            raise

    def _ensure_strong_buffers(self) -> None:
        if self._strong_fluid_input is None:
            self._strong_fluid_input = HomeFreeDomainState(
                self.fluid_domain.model
            )
        if self._strong_fluid_prepared is None:
            self._strong_fluid_prepared = HomeFreeDomainState(
                self.fluid_domain.model
            )
        if self._strong_rigid_input is None:
            self._strong_rigid_input = self.rigid_domain.model.state()
        if self._strong_velocity_guess is None:
            self._strong_velocity_guess = wp.zeros(
                self.rigid_domain.model.body_count,
                dtype=wp.spatial_vector,
                device=self.fluid_domain.model._device,
            )

    def _capture_optional_transaction_history(self) -> object | None:
        create = getattr(self.fluid_domain, "create_transaction_history", None)
        capture = getattr(self.fluid_domain, "capture_transaction_history", None)
        if create is None or capture is None:
            return None
        history = create()
        diagnostics = capture(history)
        return history, diagnostics

    def _restore_optional_transaction_history(self, checkpoint: object | None) -> None:
        if checkpoint is None:
            return
        restore = getattr(self.fluid_domain, "restore_transaction_history", None)
        if restore is None:
            raise RuntimeError("fluid transaction-history ownership changed")
        history, diagnostics = checkpoint
        restore(history, diagnostics)

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
        fluid_input: HomeFreeDomainState,
        rigid_input: Any,
        previous_fluid_diagnostics: Any,
        previous_surface_diagnostics: Any,
    ) -> None:
        self.fluid_domain.state.copy_from(fluid_input)
        self.rigid_domain.state.assign(rigid_input)
        self.interface.abort_prepared_step()
        self.interface.last_geometry_update = self.interface.geometry.rasterize(
            self.fluid_domain.fluid_state,
            rigid_input.body_q,
        )
        if self.interface._remapper is not None:
            self.interface._remapper.capture_geometry(
                self.fluid_domain.fluid_state,
                self.fluid_domain.free_surface_state,
            )
        self.interface.cut_links = CutLinkBuffer.build_from_sdf(
            self.fluid_domain.fluid_state
        )
        self.interface.cut_links.set_rigid_wall_velocity(
            rigid_input.body_q,
            rigid_input.body_qd,
            self.rigid_domain.model.body_com,
            self.fluid_domain.model.fluid_grid_cell_size,
            self.fluid_domain.model.time_step,
        )
        self.interface.cut_links.validate_body_ids()
        self.fluid_domain.solver.set_cut_links(self.interface.cut_links)
        self.interface._impulses_corrected = False
        self.fluid_domain._last_diagnostics = previous_fluid_diagnostics
        self.fluid_domain._last_surface_tension_diagnostics = (
            previous_surface_diagnostics
        )

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
