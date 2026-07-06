import warp as wp
from typing import TYPE_CHECKING

from wanphys import rigid


from ..solver import ParticleFluidSolverBase
from .kernels import (
    sync_boundary_particles_from_object_id,
    compute_boundary_psi_from_boundary_grid,
    compute_density_object_id,
    compute_pressure_object_id,
    compute_non_pressure_forces_object_id,
    compute_rigid_force_torque_object_id,
    get_acceleration_object_id,
    kick,
    drift,
    enforce_boundary_3D_warp_object_id,
)
from .rigid_fluid_coupling import solve_rigid_body_diff, update_rigid_particle_info_diff_object_id

from .model import SPHDiffModel
from .state import SPHDiffState
from newton._src.sim.contacts import Contacts
from newton._src.sim.control import Control


if TYPE_CHECKING:
    from newton import Contacts, Control
    from .model import SPHDiffModel
    from .state import SPHDiffState



class SPHDiffSolver(ParticleFluidSolverBase):
    def __init__(
        self,
        model_adapter,
        sphdiff_model: "SPHDiffModel",
    ):
        self._model_adapter = model_adapter
        self._sphdiff_model = sphdiff_model
        self._grid = wp.HashGrid(128, 128, 128)
        self._boundary_grid = wp.HashGrid(128, 128, 128)
        self.tapes = []
        self._runtime_task = None
        self._runtime_task_initialized = False

    @property
    def model_adapter(self):
        return self._model_adapter

    @property
    def sphdiff_model(self) -> "SPHDiffModel":
        return self._sphdiff_model

    def _ensure_boundary_grid(self, state: SPHDiffState, support_radius: float) -> bool:
        if int(state.boundary_q.shape[0]) <= 0:
            return False
        self._boundary_grid.build(state.boundary_q, radius=support_radius)
        return True

    def step(
        self,
        state_in: SPHDiffState,
        state_out: SPHDiffState,
        dt: float,
        contacts: Contacts | None = None,
        control: Control | None = None,
    ) -> None:
        """
        严格对齐SimSPH_diff的step/sub_step结构和顺序。
        """
        current_tape = wp.Tape()
        self.tapes.append(current_tape)

        if self._runtime_task is not None and not self._runtime_task_initialized:
            self._runtime_task.state = state_in
            if hasattr(self._runtime_task, "init_simulation_state"):
                self._runtime_task.init_simulation_state(tape=current_tape)
            self._runtime_task_initialized = True

        model = self._model_adapter
        sphdiff = self._sphdiff_model
        boundary_body_id = state_in.boundary_body_id
        rbs = state_in.rbs
        particle_x0 = getattr(state_in, "particle_x0", None)

        if model.particle_count == 0:
            return

        dt = float(min(dt, sphdiff.max_dt))

        grid = self._grid

        grid.build(state_in.particle_q, radius=sphdiff.h)
        self._ensure_boundary_grid(state_in, sphdiff.h)

        state_out.zero_scratch_buffers()

        if state_in is not state_out:
            wp.copy(state_out.boundary_q, state_in.boundary_q)
            wp.copy(state_out.boundary_psi, state_in.boundary_psi)
            wp.copy(state_out.boundary_body_id, state_in.boundary_body_id)
            wp.copy(state_out.boundary_object_id, state_in.boundary_object_id)
            wp.copy(state_out.particle_flags, state_in.particle_flags)

        self.sub_step(state_in, state_out, dt, grid, current_tape)

        if sphdiff.bounds_enabled:
            wp.launch(
                kernel=enforce_boundary_3D_warp_object_id,
                dim=model.particle_count,
                inputs=[
                    state_out.particle_q,
                    state_out.particle_qd,
                    state_out.particle_flags,
                    sphdiff.get_bounds_min_vec3(),
                    sphdiff.get_bounds_max_vec3(),
                    sphdiff.boundary_padding,
                ],
                device=sphdiff._device,
            )
        rigid_count = state_in.num_objects
        if sphdiff.integrate_rigid and rigid_count > 0:
            with current_tape:
                wp.launch(
                    kernel=solve_rigid_body_diff,
                    dim=rigid_count,
                    inputs=[
                        rbs.rigid_x,
                        rbs.rigid_v,
                        rbs.rigid_force,
                        rbs.rigid_mass,
                        rbs.rigid_quaternion,
                        rbs.rigid_omega,
                        rbs.rigid_torque,
                        rbs.rigid_inertia0,
                        rbs.rigid_inv_inertia,
                        wp.vec3(*sphdiff.gravity),
                        dt,
                    ],
                    outputs=[
                        state_out.rbs.rigid_x,
                        state_out.rbs.rigid_v,
                        state_out.rbs.rigid_force,
                        state_out.rbs.rigid_quaternion,
                        state_out.rbs.rigid_omega,
                        state_out.rbs.rigid_torque,
                        state_out.rbs.rigid_inertia0,
                        state_out.rbs.rigid_inv_inertia,
                    ],
                    device=sphdiff._device,
                )

                wp.launch(
                    kernel=update_rigid_particle_info_diff_object_id,
                    dim=model.particle_count,
                    inputs=[
                        state_out.particle_q,
                        state_out.particle_qd,
                        particle_x0,
                        boundary_body_id,
                        rbs.rigid_rest_cm,
                        state_out.rbs.rigid_x,
                        state_out.rbs.rigid_quaternion,
                        state_out.rbs.rigid_v,
                        state_out.rbs.rigid_omega,
                    ],
                    device=sphdiff._device,
                )

            state_out.rbs.rigid_rest_cm = rbs.rigid_rest_cm
            state_out.rbs.rigid_mass = rbs.rigid_mass
            state_out.rbs.rigid_inertia = rbs.rigid_inertia
            state_out.rbs.rigid_inertia0 = rbs.rigid_inertia0
            state_out.rbs.rigid_inv_mass = rbs.rigid_inv_mass
            state_out.rbs.rigid_inv_inertia = rbs.rigid_inv_inertia

    def sub_step(
        self,
        state_in: SPHDiffState,
        state_out: SPHDiffState,
        dt: float,
        grid,
        current_tape,
    ) -> None:
        """Advance the fluid state one sub-step."""
        model = self._model_adapter
        sphdiff = self._sphdiff_model
        boundary_body_id = state_in.boundary_body_id
        has_boundary = 1 if int(state_in.boundary_q.shape[0]) > 0 else 0

        if has_boundary != 0:
            wp.launch(
                kernel=sync_boundary_particles_from_object_id,
                dim=int(state_in.boundary_q.shape[0]),
                inputs=[state_out.boundary_q, state_in.particle_q, state_in.boundary_indices],
                device=sphdiff._device,
            )

            # Rebuild boundary grid from the synced positions before recomputing psi.
            with wp.ScopedDevice(sphdiff._device):
                self._boundary_grid.build(state_out.boundary_q, radius=sphdiff.h)

            # Recompute boundary_psi from updated boundary positions so it matches original m_V semantics.
            wp.launch(
                kernel=compute_boundary_psi_from_boundary_grid,
                dim=int(state_out.boundary_q.shape[0]),
                inputs=[
                    self._boundary_grid.id if has_boundary != 0 else wp.uint64(0),
                    state_out.boundary_q,
                    state_out.boundary_psi,
                    sphdiff.h,
                    sphdiff.rest_density,
                ],
                device=sphdiff._device,
            )

        wp.launch(
            kernel=compute_density_object_id,
            dim=model.particle_count,
            inputs=[
                grid.id,
                state_in.particle_q,
                state_out.rho,
                1.0,
                sphdiff.h,
                state_in.m_V,
                sphdiff.rest_density,
                state_in.particle_flags,
                state_out.boundary_q,
                state_out.boundary_psi,
                self._boundary_grid.id if has_boundary != 0 else wp.uint64(0),
                has_boundary,
            ],
            device=sphdiff._device,
        )

        wp.launch(
            kernel=compute_pressure_object_id,
            dim=model.particle_count,
            inputs=[
                state_out.rho,
                state_out.pressure,
                state_in.particle_flags,
                sphdiff.c0,
                sphdiff.gamma,
                sphdiff.rest_density,
            ],
            device=sphdiff._device,
        )

        wp.launch(
            kernel=compute_non_pressure_forces_object_id,
            dim=model.particle_count,
            inputs=[
                grid.id,
                state_in.particle_q,
                state_in.particle_qd,
                state_out.rho,
                sphdiff.viscosity,
                sphdiff.h,
                state_in.particle_flags,
                state_in.m_V,
                sphdiff.rest_density,
                state_out.viscous_forces,
                sphdiff.surface_tension,
                sphdiff.gravity[2],
                state_out.a,
            ],
            device=sphdiff._device,
        )

        wp.launch(
            kernel=get_acceleration_object_id,
            dim=model.particle_count,
            inputs=[
                grid.id,
                state_in.particle_q,
                state_in.particle_qd,
                state_out.rho,
                state_out.pressure,
                sphdiff.c0,
                sphdiff.gamma,
                sphdiff.rest_density,
                sphdiff.gravity[2],
                1.0,
                sphdiff.h,
                state_in.particle_flags,
                state_in.m_V,
                state_out.pressure_forces,
                state_out.viscous_forces,
                state_out.debug_val,
                state_out.boundary_q,
                state_out.boundary_psi,
                self._boundary_grid.id if has_boundary != 0 else wp.uint64(0),
                state_out.boundary_object_id,
                has_boundary,
            ],
            outputs=[state_out.a],
            device=sphdiff._device,
        )

        with current_tape:
            # Clear rigid accumulators before accumulating forces/torques this sub-step
            state_in.rbs.rigid_force.zero_()
            state_in.rbs.rigid_torque.zero_()

            wp.launch(
                kernel=compute_rigid_force_torque_object_id,
                dim=model.particle_count,
                inputs=[
                    grid.id,
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_out.rho,
                    state_out.pressure,
                    sphdiff.rest_density,
                    1.0,
                    sphdiff.h,
                    state_in.m_V,
                    state_in.particle_flags,
                    state_out.debug_val,
                    state_in.rbs.rigid_x,
                    sphdiff.requires_grad,
                        state_out.boundary_q,
                        state_out.boundary_psi,
                        self._boundary_grid.id if has_boundary != 0 else wp.uint64(0),
                        state_out.boundary_object_id,
                    has_boundary,
                ],
                outputs=[state_in.rbs.rigid_force, state_in.rbs.rigid_torque, state_out.a],
            )

        wp.launch(
            kernel=kick,
            dim=model.particle_count,
            inputs=[state_out.a, dt, state_in.particle_qd, state_out.particle_qd],
            device=sphdiff._device,
        )

        wp.launch(
            kernel=drift,
            dim=model.particle_count,
            inputs=[state_in.particle_q, state_out.particle_qd, dt, state_out.particle_q],
            device=sphdiff._device,
        )

        if has_boundary != 0:
            wp.launch(
                kernel=sync_boundary_particles_from_object_id,
                dim=int(state_out.boundary_q.shape[0]),
                inputs=[state_out.boundary_q, state_out.particle_q, state_out.boundary_indices],
                device=sphdiff._device,
            )

            
