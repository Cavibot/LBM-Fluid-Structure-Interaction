from __future__ import annotations

from typing import Any, Optional

import warp as wp

from ..base import FluidGridMacSolverBase
from ..particle_sort import ParticleCellSorter
from . import kernels
from .model import FluidGridApicModel
from .state import FluidGridApicState


class FluidGridApicSolver(FluidGridMacSolverBase):
    """APIC liquid solver on MAC grid with particle-surface reconstruction."""

    def __init__(self, model: FluidGridApicModel) -> None:
        super().__init__(model)
        self.gravity = wp.vec3(0.0, 0.0, -9.8)

        self.weight_u = wp.zeros((self.nx + 1, self.ny, self.nz), dtype=float, device=self.device)
        self.weight_v = wp.zeros((self.nx, self.ny + 1, self.nz), dtype=float, device=self.device)
        self.weight_w = wp.zeros((self.nx, self.ny, self.nz + 1), dtype=float, device=self.device)
        self.solid_weight_u = wp.zeros((self.nx + 1, self.ny, self.nz), dtype=float, device=self.device)
        self.solid_weight_v = wp.zeros((self.nx, self.ny + 1, self.nz), dtype=float, device=self.device)
        self.solid_weight_w = wp.zeros((self.nx, self.ny, self.nz + 1), dtype=float, device=self.device)
        self.vel_solid_u = wp.zeros((self.nx + 1, self.ny, self.nz), dtype=float, device=self.device)
        self.vel_solid_v = wp.zeros((self.nx, self.ny + 1, self.nz), dtype=float, device=self.device)
        self.vel_solid_w = wp.zeros((self.nx, self.ny, self.nz + 1), dtype=float, device=self.device)

        self.valid_u = wp.zeros((self.nx + 1, self.ny, self.nz), dtype=wp.int32, device=self.device)
        self.valid_v = wp.zeros((self.nx, self.ny + 1, self.nz), dtype=wp.int32, device=self.device)
        self.valid_w = wp.zeros((self.nx, self.ny, self.nz + 1), dtype=wp.int32, device=self.device)

        self.valid_u_aux = wp.zeros((self.nx + 1, self.ny, self.nz), dtype=wp.int32, device=self.device)
        self.valid_v_aux = wp.zeros((self.nx, self.ny + 1, self.nz), dtype=wp.int32, device=self.device)
        self.valid_w_aux = wp.zeros((self.nx, self.ny, self.nz + 1), dtype=wp.int32, device=self.device)
        self._particle_sorter = ParticleCellSorter(self.model.particle_count, self.device, include_affine=True)

    def step(
        self,
        state_in: FluidGridApicState,
        state_out: FluidGridApicState,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        del contacts, control

        wp.copy(state_out.solid_phi, state_in.solid_phi)
        wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
        wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
        wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_v, state_in.particle_v)
        wp.copy(state_out.particle_c, state_in.particle_c)

        self._sort_particles_by_cell(state_out)
        self._rebuild_fluid_cells(state_out)
        self._particles_to_grid(state_out)
        self._extrapolate_velocity(state_out)

        self._apply_gravity(state_out, dt)
        self._compute_divergence(state_out, dt)
        self._solve_pressure(state_in, state_out, dt)
        self._project_velocity(state_out, dt)
        self._enforce_boundary(state_out)

        self._update_particle_velocity(state_out)
        self._advect_particles(state_out, dt)
        self._rebuild_fluid_cells(state_out)

    def bake_box(self, state: FluidGridApicState, center: wp.vec3, half_extents: wp.vec3) -> None:
        wp.launch(
            kernels.bake_solid_box_kernel,
            dim=self.model.resolution,
            inputs=[state.solid_phi, self.model.dh, center, half_extents],
        )

    def bake_mesh(
        self,
        state: FluidGridApicState,
        mesh: Any,
        pos: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
        rot: wp.quat = wp.quat_identity(),
        scale: float = 1.0,
    ) -> None:
        if isinstance(mesh, wp.Mesh):
            wp_mesh = mesh
        elif hasattr(mesh, "wp_mesh"):
            wp_mesh = mesh.wp_mesh
        elif hasattr(mesh, "mesh") and isinstance(mesh.mesh, wp.Mesh):
            wp_mesh = mesh.mesh
        elif hasattr(mesh, "vertices") and hasattr(mesh, "indices"):
            wp_mesh = wp.Mesh(points=mesh.vertices, indices=mesh.indices)
        else:
            raise ValueError(f"Cannot get wp.Mesh. input type: {type(mesh)}")

        wp.launch(
            kernels.bake_solid_mesh_kernel,
            dim=self.model.resolution,
            inputs=[state.solid_phi, wp_mesh.id, self.model.dh, pos, rot, scale, 1e6],
        )

    def _particles_to_grid(self, state: FluidGridApicState) -> None:
        wp.launch(
            kernels.build_u_face_weights,
            dim=(self.nx + 1, self.ny, self.nz),
            inputs=[state.solid_phi, self.solid_weight_u, self.nx],
        )
        wp.launch(
            kernels.build_v_face_weights,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[state.solid_phi, self.solid_weight_v, self.ny],
        )
        wp.launch(
            kernels.build_w_face_weights,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[state.solid_phi, self.solid_weight_w, self.nz],
        )

        wp.launch(kernels.fill_grid_face, dim=(self.nx + 1, self.ny, self.nz), inputs=[state.vel_u, 0.0])
        wp.launch(kernels.fill_grid_face, dim=(self.nx, self.ny + 1, self.nz), inputs=[state.vel_v, 0.0])
        wp.launch(kernels.fill_grid_face, dim=(self.nx, self.ny, self.nz + 1), inputs=[state.vel_w, 0.0])

        wp.launch(kernels.fill_grid_face, dim=(self.nx + 1, self.ny, self.nz), inputs=[self.weight_u, 0.0])
        wp.launch(kernels.fill_grid_face, dim=(self.nx, self.ny + 1, self.nz), inputs=[self.weight_v, 0.0])
        wp.launch(kernels.fill_grid_face, dim=(self.nx, self.ny, self.nz + 1), inputs=[self.weight_w, 0.0])

        wp.launch(
            kernels.p2g_velocity_u_apic,
            dim=self.model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_v,
                state.particle_c,
                state.vel_u,
                self.weight_u,
                self.solid_weight_u,
                self.nx,
                self.ny,
                self.nz,
                self.model.dh,
            ],
        )
        wp.launch(
            kernels.p2g_velocity_v_apic,
            dim=self.model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_v,
                state.particle_c,
                state.vel_v,
                self.weight_v,
                self.solid_weight_v,
                self.nx,
                self.ny,
                self.nz,
                self.model.dh,
            ],
        )
        wp.launch(
            kernels.p2g_velocity_w_apic,
            dim=self.model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_v,
                state.particle_c,
                state.vel_w,
                self.weight_w,
                self.solid_weight_w,
                self.nx,
                self.ny,
                self.nz,
                self.model.dh,
            ],
        )

        wp.launch(
            kernels.normalize_face_velocity_with_valid,
            dim=(self.nx + 1, self.ny, self.nz),
            inputs=[state.vel_u, self.weight_u, self.valid_u, self.solid_weight_u, 1.0e-8],
        )
        wp.launch(
            kernels.normalize_face_velocity_with_valid,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[state.vel_v, self.weight_v, self.valid_v, self.solid_weight_v, 1.0e-8],
        )
        wp.launch(
            kernels.normalize_face_velocity_with_valid,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[state.vel_w, self.weight_w, self.valid_w, self.solid_weight_w, 1.0e-8],
        )

        self._enforce_boundary(state)

    def _update_particle_velocity(self, state: FluidGridApicState) -> None:
        wp.launch(
            kernels.update_particle_velocity_apic,
            dim=self.model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_v,
                state.particle_c,
                state.vel_u,
                state.vel_v,
                state.vel_w,
                self.nx,
                self.ny,
                self.nz,
                self.model.dh,
            ],
        )

    def _sort_particles_by_cell(self, state: FluidGridApicState) -> None:
        if not bool(self.model.sort_particles_by_cell):
            return

        self._particle_sorter.reorder_qvc(
            state.particle_q,
            state.particle_v,
            state.particle_c,
            self.model.sort_particles_key_mode,
            self.nx,
            self.ny,
            self.nz,
            self.model.dh,
        )

    def _velocity_advect_kernels(self) -> tuple[Any, Any, Any]:
        return kernels.advect_u, kernels.advect_v, kernels.advect_w

    def _prepare_step(
        self,
        state_in: FluidGridApicState,
        state_out: FluidGridApicState,
        dt: float,
        contacts: Optional[Any] = None,
        control: Optional[Any] = None,
    ) -> None:
        del state_in, state_out, dt, contacts, control

    def _after_velocity_advection(
        self,
        state_in: FluidGridApicState,
        state_out: FluidGridApicState,
        dt: float,
        contacts: Optional[Any] = None,
        control: Optional[Any] = None,
    ) -> None:
        del state_in, state_out, dt, contacts, control

    def _compute_divergence(self, state: FluidGridApicState, dt: float) -> None:
        del dt
        wp.launch(
            kernels.compute_divergence_cell_type_mac,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state.vel_u,
                state.vel_v,
                state.vel_w,
                state.cell_type,
                state.solid_phi,
                self.div_array,
                self.nx,
                self.ny,
                self.nz,
                self.model.dh,
            ],
        )

    def _pressure_kernel(self) -> Any:
        return kernels.pressure_jacobi_cell_type_mac

    def _pressure_iteration_inputs(
        self,
        pressure_src: Any,
        pressure_dst: Any,
        state: FluidGridApicState,
        dt: float,
    ) -> list[Any]:
        return [
            pressure_src,
            pressure_dst,
            self.div_array,
            state.cell_type,
            self.nx,
            self.ny,
            self.nz,
            self.model.dh,
            dt,
        ]

    def _pressure_apply_operator_kernel(self) -> Any:
        return kernels.pressure_apply_operator_cell_type_mac

    def _pressure_apply_operator_inputs(
        self,
        x: Any,
        y: Any,
        state: FluidGridApicState,
    ) -> list[Any]:
        return [x, y, state.cell_type, self.nx, self.ny, self.nz]

    def _pressure_build_inv_diag_kernel(self) -> Any:
        return kernels.pressure_build_inv_diag_cell_type_mac

    def _pressure_build_inv_diag_inputs(
        self,
        inv_diag: Any,
        state: FluidGridApicState,
    ) -> list[Any]:
        return [inv_diag, state.cell_type, self.nx, self.ny, self.nz]

    def _project_velocity(self, state: FluidGridApicState, dt: float) -> None:
        wp.launch(
            kernels.project_u_cell_type,
            dim=(self.nx + 1, self.ny, self.nz),
            inputs=[state.vel_u, state.pressure, state.cell_type, self.nx, self.model.dh, dt],
        )
        wp.launch(
            kernels.project_v_cell_type,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[state.vel_v, state.pressure, state.cell_type, self.ny, self.model.dh, dt],
        )
        wp.launch(
            kernels.project_w_cell_type,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[state.vel_w, state.pressure, state.cell_type, self.nz, self.model.dh, dt],
        )

    def _enforce_boundary(self, state: FluidGridApicState) -> None:
        wp.launch(
            kernels.enforce_solid_u,
            dim=(self.nx + 1, self.ny, self.nz),
            inputs=[state.vel_u, state.solid_phi, self.nx],
        )
        wp.launch(
            kernels.enforce_solid_v,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[state.vel_v, state.solid_phi, self.ny],
        )
        wp.launch(
            kernels.enforce_solid_w,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[state.vel_w, state.solid_phi, self.nz],
        )

    def _extrapolate_velocity(self, state: FluidGridApicState) -> None:
        iterations = max(0, int(self.model.extrap_iterations))
        for _ in range(iterations):
            wp.launch(
                kernels.extrapolate_u_from_valid,
                dim=(self.nx + 1, self.ny, self.nz),
                inputs=[
                    state.vel_u,
                    self.valid_u,
                    self.u_aux,
                    self.valid_u_aux,
                    self.vel_solid_u,
                    self.solid_weight_u,
                    self.nx,
                    self.ny,
                    self.nz,
                ],
            )
            wp.launch(
                kernels.extrapolate_v_from_valid,
                dim=(self.nx, self.ny + 1, self.nz),
                inputs=[
                    state.vel_v,
                    self.valid_v,
                    self.v_aux,
                    self.valid_v_aux,
                    self.vel_solid_v,
                    self.solid_weight_v,
                    self.nx,
                    self.ny,
                    self.nz,
                ],
            )
            wp.launch(
                kernels.extrapolate_w_from_valid,
                dim=(self.nx, self.ny, self.nz + 1),
                inputs=[
                    state.vel_w,
                    self.valid_w,
                    self.w_aux,
                    self.valid_w_aux,
                    self.vel_solid_w,
                    self.solid_weight_w,
                    self.nx,
                    self.ny,
                    self.nz,
                ],
            )

            wp.copy(state.vel_u, self.u_aux)
            wp.copy(state.vel_v, self.v_aux)
            wp.copy(state.vel_w, self.w_aux)

            wp.copy(self.valid_u, self.valid_u_aux)
            wp.copy(self.valid_v, self.valid_v_aux)
            wp.copy(self.valid_w, self.valid_w_aux)

    def _advect_particles(self, state: FluidGridApicState, dt: float) -> None:
        bounds_min = wp.vec3(0.0, 0.0, 0.0)
        bounds_max = wp.vec3(self.nx * self.model.dh, self.ny * self.model.dh, self.nz * self.model.dh)

        wp.launch(
            kernels.advect_particles_in_grid_rk2,
            dim=self.model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_v,
                state.vel_u,
                state.vel_v,
                state.vel_w,
                dt,
                self.nx,
                self.ny,
                self.nz,
                self.model.dh,
                bounds_min,
                bounds_max,
                0.2 * self.model.dh,
            ],
        )
        wp.launch(
            kernels.resolve_particle_solid_collision_with_velocity,
            dim=self.model.particle_count,
            inputs=[
                state.particle_q,
                state.particle_v,
                self.vel_solid_u,
                self.vel_solid_v,
                self.vel_solid_w,
                state.solid_phi,
                self.nx,
                self.ny,
                self.nz,
                self.model.dh,
            ],
        )

    def _rebuild_fluid_cells(self, state: FluidGridApicState) -> None:
        wp.launch(
            kernels.initialize_liquid_cell_state,
            dim=(self.nx, self.ny, self.nz),
            inputs=[state.solid_phi, state.cell_type, state.density],
        )
        wp.launch(
            kernels.mark_liquid_cells_from_particles,
            dim=self.model.particle_count,
            inputs=[
                state.particle_q,
                state.solid_phi,
                state.cell_type,
                state.density,
                self.nx,
                self.ny,
                self.nz,
                self.model.dh,
            ],
        )

    def _apply_gravity(self, state: FluidGridApicState, dt: float) -> None:
        gx, gy, gz = self.gravity
        if gx != 0.0:
            wp.launch(
                kernels.apply_gravity_u,
                dim=(self.nx + 1, self.ny, self.nz),
                inputs=[state.vel_u, gx, dt],
            )
        if gy != 0.0:
            wp.launch(
                kernels.apply_gravity_v,
                dim=(self.nx, self.ny + 1, self.nz),
                inputs=[state.vel_v, gy, dt],
            )
        if gz != 0.0:
            wp.launch(
                kernels.apply_gravity_w,
                dim=(self.nx, self.ny, self.nz + 1),
                inputs=[state.vel_w, gz, dt],
            )

