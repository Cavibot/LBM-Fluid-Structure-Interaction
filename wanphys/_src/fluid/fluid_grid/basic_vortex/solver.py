# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import warp as wp
import numpy as np
from ..base import FluidGridMacSolverBase
from . import kernels
from .model import FluidGridModel
from .state import FluidGridState

from .sparse_hash_grid import SparseHashGrid, BLOCK_VOL, BLOCK_SIZE

class FluidGridSolver(FluidGridMacSolverBase):
    """Smoke-style MAC solver with density advection and vorticity confinement."""

    def __init__(self, model: FluidGridModel, max_blocks: int = 4096):
        # 0407 modify init
        #super().__init__(model)
        self.model = model
        self.nx = model.nx
        self.ny = model.ny
        self.nz = model.nz
        self.device = model._device

        self.max_blocks = max_blocks
        device = model._device
        self.domain_blocks = (
            int((self.nx + BLOCK_SIZE - 1) // BLOCK_SIZE),
            int((self.ny + BLOCK_SIZE - 1) // BLOCK_SIZE),
            int((self.nz + BLOCK_SIZE - 1) // BLOCK_SIZE),
        )
        # grid initialize and activity tracing
        self.grid = SparseHashGrid(max_blocks=self.max_blocks, device=device, domain_blocks=self.domain_blocks)
        self.active_block_coords = wp.zeros(self.max_blocks, dtype=wp.vec3i, device=device)
        self.active_count = 0

        # 1D memory pool
        pool_len = self.max_blocks * BLOCK_VOL
        self.curl_array = wp.zeros(pool_len, dtype=wp.vec3, device=device)

        self.solid_phi = wp.zeros(pool_len, dtype=float, device=device)
        self.solid_phi.fill_(1000.0)

        self.density_aux = wp.zeros(pool_len, dtype=float, device=device)

        # MAC depend on self.div_array
        self.div_array = wp.zeros(pool_len, dtype=float, device=device)

        self.external_force = wp.vec3(0.0, 0.0, 0.0)
        self._sphere_center = wp.vec3(0.0, 0.0, 0.0)
        self._sphere_radius = 0.0

        #330 test for Dynamic Grid
        self.current_time = 0.0

    def seed_emitter(self, center: wp.vec3, radius: float):
        min_x = int((center[0] - radius) / (self.model.dh * BLOCK_SIZE))
        max_x = int((center[0] + radius) / (self.model.dh * BLOCK_SIZE))
        min_y = int((center[1] - radius) / (self.model.dh * BLOCK_SIZE))
        max_y = int((center[1] + radius) / (self.model.dh * BLOCK_SIZE))
        min_z = int((center[2] - radius) / (self.model.dh * BLOCK_SIZE))
        max_z = int((center[2] + radius) / (self.model.dh * BLOCK_SIZE))

        coords = []
        for bx in range(min_x, max_x + 1):
            for by in range(min_y, max_y + 1):
                for bz in range(min_z, max_z + 1):
                    if bx >= 0 and by >= 0 and bz >= 0:
                        coords.append([bx, by, bz])

        num_new = len(coords)
        if num_new > 0:
            if self.active_count + num_new > self.max_blocks:
                num_new = self.max_blocks - self.active_count

            if num_new > 0:
                # memory protection
                new_coords_np = np.array(coords[:num_new], dtype=np.int32)
                wp.copy(
                    self.active_block_coords,
                    wp.array(new_coords_np, dtype=wp.vec3i, device=self.model._device),
                    dest_offset=self.active_count,
                    count=num_new
                )
                self.active_count += num_new

    def step(self, state_in: FluidGridState, state_out: FluidGridState, dt: float) -> None:
        # 330 test garbage recycle
        self.current_time += dt
        #  garbage recycle
        self._prune_blocks(state_in)

        # seed grid
        source_center = wp.vec3(
            0.5 * self.nx * self.model.dh,
            0.5 * self.ny * self.model.dh,
            0.0,
        )
        source_radius = 2.0
        self.seed_emitter(source_center, source_radius)
        # self.grid.update_topology(self.active_block_coords, self.active_count)
        self.grid.update_topology_boundary_driven(
            self.active_block_coords,
            self.active_count,
            state_in.density,
            state_in.vel_u,
            state_in.vel_v,
            state_in.vel_w,
            density_threshold=1.0e-10,
            velocity_threshold=1.0e-3,
        )

        self.active_count = self.grid.get_active_blocks_count()

        if self.active_count > self.max_blocks:
            self.active_count = self.max_blocks

        wp.copy(self.active_block_coords, self.grid.block_coords, count=self.active_count)

        if self.active_count == 0:
            return

        # physics step
        super().step(state_in, state_out, dt)




    def set_spherical_obstacle(self, center: Any, radius: float):
        self._sphere_center = center
        self._sphere_radius = radius
        self._bake_solid_phi()

    def add_external_force(self, force: Any):
        self.external_force += force

    def _bake_solid_phi(self):
        # self.solid_phi.fill_(1000.0)
        # if self._sphere_radius > 0.0:
        #     wp.launch(
        #         kernels.bake_solid_sphere_kernel,
        #         dim=(self.nx, self.ny, self.nz),
        #         inputs=[self.solid_phi, self.model.dh, self._sphere_center, self._sphere_radius],
        #     )
        pass

    #0403
    def _advect_velocity(self, state_in: FluidGridState, state_out: FluidGridState, dt: float) -> None:
        wp.launch(
            kernels.advect_u_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state_in.vel_u,
                state_in.vel_v,
                state_in.vel_w,
                state_out.vel_u,
                self.grid.old_hash_keys,
                self.grid.old_hash_vals,
                dt,
                self.model.dh,
                self.nx,
                self.ny,
                self.nz,
            ]
        )
        wp.launch(
            kernels.advect_v_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state_in.vel_u,
                state_in.vel_v,
                state_in.vel_w,
                state_out.vel_v,
                self.grid.old_hash_keys,
                self.grid.old_hash_vals,
                dt,
                self.model.dh,
                self.nx,
                self.ny,
                self.nz,
            ]
        )
        wp.launch(
            kernels.advect_w_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state_in.vel_u,
                state_in.vel_v,
                state_in.vel_w,
                state_out.vel_w,
                self.grid.old_hash_keys,
                self.grid.old_hash_vals,
                dt,
                self.model.dh,
                self.nx,
                self.ny,
                self.nz,
            ]
        )

    def _after_velocity_advection(
        self,
        state_in: FluidGridState,
        state_out: FluidGridState,
        dt: float,
        contacts: Optional[Any] = None,
        control: Optional[Any] = None,
    ) -> None:
        self._apply_force(state_out, dt)
        self._apply_vorticity(state_out, dt)
        if self.active_count > 0:
            wp.launch(
                kernels.inject_velocity_sparse,
                dim=self.active_count * BLOCK_VOL,
                inputs=[
                    self.active_block_coords,
                    state_out.vel_w,
                    self.nx,
                    self.model.dh
                ]
            )

    def _compute_divergence(self, state: FluidGridState, dt: float) -> None:
        wp.launch(
            kernels.compute_divergence_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state.vel_u,
                state.vel_v,
                state.vel_w,
                self.grid.hash_keys,
                self.grid.hash_vals,
                self.model.dh,
                self.div_array,
                self.nx,
                self.ny,
                self.nz
            ]
        )
    def _pressure_kernel(self) -> Any:
        # return kernels.pressure_jacobi_mac
        return None

    def _pressure_iteration_inputs(
        self,
        pressure_src: Any,
        pressure_dst: Any,
        state: FluidGridState,
        dt: float,
    ) -> list[Any]:
        # return [
        #     self.active_block_coords,
        #     self.div_array,
        #     pressure_src,
        #     pressure_dst,
        #     self.grid.hash_keys,
        #     self.grid.hash_vals,
        #     self.model.dh
        # ]
        return []

    def _pressure_apply_operator_kernel(self) -> Any:
        return kernels.pressure_apply_operator_mac

    def _velocity_advect_kernels(self) -> tuple[Any, Any, Any]:
        # for abstract check, actually intercepted by _advect_velocity
        return kernels.advect_u_mac, kernels.advect_v_mac, kernels.advect_w_mac
    def _pressure_apply_operator_inputs(
        self,
        x: Any,
        y: Any,
        state: FluidGridState,
    ) -> list[Any]:
        del state
        return [x, y, self.solid_phi, self.nx, self.ny, self.nz]

    def _pressure_build_inv_diag_kernel(self) -> Any:
        return kernels.pressure_build_inv_diag_mac

    def _pressure_build_inv_diag_inputs(
        self,
        inv_diag: Any,
        state: FluidGridState,
    ) -> list[Any]:
        del state
        return [inv_diag, self.solid_phi, self.nx, self.ny, self.nz]

    #0403
    def _project_velocity(self, state: FluidGridState, dt: float) -> None:
        wp.launch(
            kernels.project_velocity_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state.pressure,
                state.vel_u,
                state.vel_v,
                state.vel_w,
                self.grid.hash_keys,
                self.grid.hash_vals,
                self.model.dh,
                self.nx,
                self.ny,
                self.nz
            ]
        )

    # 330 ？？
    def _enforce_boundary(self, state: FluidGridState) -> None:
        # if self._sphere_radius <= 0.0:
        #     return
        #
        # wp.launch(kernels.enforce_solid_u_mac, dim=(self.nx + 1, self.ny, self.nz), inputs=[state.vel_u, self.solid_phi, self.nx])
        # wp.launch(kernels.enforce_solid_v_mac, dim=(self.nx, self.ny + 1, self.nz), inputs=[state.vel_v, self.solid_phi, self.ny])
        # wp.launch(kernels.enforce_solid_w_mac, dim=(self.nx, self.ny, self.nz + 1), inputs=[state.vel_w, self.solid_phi, self.nz])
        #def _enforce_boundary(self, state: FluidGridState) -> None:
            if self._sphere_radius <= 0.0:
                return

            # 1. dynamic SDF
            wp.launch(
                bake_solid_sphere_sparse,
                dim=self.active_count * BLOCK_VOL,
                inputs=[
                    self.active_block_coords,
                    self.solid_phi,
                    self._sphere_center,
                    self._sphere_radius,
                    self.model.dh
                ]
            )

            # 2. Dirichlet boundry
            wp.launch(
                kernels.enforce_solid_velocity_sparse,
                dim=self.active_count * BLOCK_VOL,
                inputs=[
                    state.vel_u,
                    state.vel_v,
                    state.vel_w,
                    self.solid_phi
                ]
            )

    # 330 add new hashtable
    def _after_pressure_projection(
        self,
        state_in: FluidGridState,
        state_out: FluidGridState,
        dt: float,
        contacts: Optional[Any] = None,
        control: Optional[Any] = None,
    ) -> None:
        if self.active_count == 0:
            return

        wp.launch(
            kernels.advect_scalar_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state_in.density,
                state_out.vel_u,
                state_out.vel_v,
                state_out.vel_w,
                self.density_aux,
                # self.grid.old_hash_keys,
                # self.grid.old_hash_vals,
                self.grid.hash_keys,  # <--- 传新字典
                self.grid.hash_vals,  # <--- 传新字典
                self.grid.old_hash_keys,  # <--- 传老字典
                self.grid.old_hash_vals,  # <--- 传老字典
                dt,
                self.model.dh,
                self.nx,
                self.ny,
                self.nz,
            ],
        )
        wp.copy(state_out.density, self.density_aux)

        wp.launch(
            kernels.inject_density_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state_out.density,
                self.nx,
                self.model.dh,
                self.model.smoke_source_strength if hasattr(self.model, 'smoke_source_strength') else 0.1
            ]
        )
        # 1D Sparse dissipate_scalar_sparse
        wp.launch(
            kernels.dissipate_scalar_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[state_out.density, self.model.dissipation_rate, dt],
        )

    def _apply_force(self, state: FluidGridState, dt: float):
        if self.external_force != wp.vec3(0.0):
            wp.launch(
                kernels.apply_force_sparse,
                dim=self.active_count * BLOCK_VOL,
                inputs=[state.vel_u, self.external_force.x, dt],
            )
            wp.launch(
                kernels.apply_force_sparse,
                dim=self.active_count * BLOCK_VOL,
                inputs=[state.vel_v, self.external_force.y, dt],
            )
            wp.launch(
                kernels.apply_force_sparse,
                dim=self.active_count * BLOCK_VOL,
                inputs=[state.vel_w, self.external_force.z, dt],
            )
        if self.model.buoyancy > 0.0:
            wp.launch(
                kernels.apply_buoyancy_sparse,
                dim=self.active_count * BLOCK_VOL,
                inputs=[state.vel_w, state.density, self.model.buoyancy, dt]
            )
        if self.model.damping > 0.0:
            wp.launch(
                kernels.damp_velocity_sparse,
                dim=self.active_count * BLOCK_VOL,
                inputs=[state.vel_u, state.vel_v, state.vel_w, self.model.damping, dt]
            )

    def _apply_vorticity(self, state: FluidGridState, dt: float) -> None:
        if self.model.vorticity_scale == 0.0 or self.active_count == 0:
            return

        wp.launch(
            kernels.compute_curl_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state.vel_u,
                state.vel_v,
                state.vel_w,
                self.curl_array,
                self.grid.hash_keys,
                self.grid.hash_vals,
                self.model.dh,
            ],
        )

        wp.launch(
            kernels.apply_vorticity_u_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state.vel_u,
                self.curl_array,
                self.grid.hash_keys,
                self.grid.hash_vals,
                self.model.dh,
                dt,
                self.model.vorticity_scale,
            ],
        )
        wp.launch(
            kernels.apply_vorticity_v_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state.vel_v,
                self.curl_array,
                self.grid.hash_keys,
                self.grid.hash_vals,
                self.model.dh,
                dt,
                self.model.vorticity_scale,
            ],
        )
        wp.launch(
            kernels.apply_vorticity_w_sparse,
            dim=self.active_count * BLOCK_VOL,
            inputs=[
                self.active_block_coords,
                state.vel_w,
                self.curl_array,
                self.grid.hash_keys,
                self.grid.hash_vals,
                self.model.dh,
                dt,
                self.model.vorticity_scale,
            ],
        )

    def _solve_pressure(self, state_in: FluidGridState, state_out: FluidGridState, dt: float) -> None:
        if self.active_count == 0:
            return

        state_out.pressure.fill_(0.0)
        self.density_aux.fill_(0.0)

        p_src = state_out.pressure
        p_dst = self.density_aux

        # jacob iteration (25 is randomly set)
        for _ in range(self.model.pressure_iteration):
            wp.launch(
                kernels.pressure_jacobi_sparse,
                dim=self.active_count * BLOCK_VOL,
                inputs=[
                    self.active_block_coords,
                    p_src,
                    p_dst,
                    self.div_array,
                    self.grid.hash_keys,
                    self.grid.hash_vals,
                    self.model.dh,
                    self.nx,
                    self.ny,
                    self.nz
                ]
            )
            p_src, p_dst = p_dst, p_src

        if p_src.ptr != state_out.pressure.ptr:
            wp.copy(state_out.pressure, p_src)

    def _prune_blocks(self, state: FluidGridState):
        if self.active_count == 0:
            return

        new_coords = wp.empty_like(self.active_block_coords)
        new_count = wp.zeros(1, dtype=int, device=self.model._device)

        wp.launch(
            kernels.prune_blocks_sparse,
            dim=self.active_count,
            inputs=[
                self.active_block_coords,
                state.density,
                state.vel_u,  # added velocity UVW
                state.vel_v,
                state.vel_w,
                new_coords,
                new_count
            ]
        )

        # update active count
        self.active_count = int(new_count.numpy()[0])
        wp.copy(self.active_block_coords, new_coords, count=self.active_count)


@wp.kernel
def bake_solid_sphere_sparse(
        block_coords: wp.array(dtype=wp.vec3i),
        solid_phi: wp.array(dtype=float),
        center: wp.vec3,
        radius: float,
        dh: float
):
    block_id = wp.tid() // BLOCK_VOL
    voxel_idx = wp.tid() % BLOCK_VOL
    coord = block_coords[block_id]

    vx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
    rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
    vy = rem // BLOCK_SIZE
    vz = rem % BLOCK_SIZE

    gx = coord[0] * BLOCK_SIZE + vx
    gy = coord[1] * BLOCK_SIZE + vy
    gz = coord[2] * BLOCK_SIZE + vz

    px = (wp.float32(gx) + 0.5) * dh
    py = (wp.float32(gy) + 0.5) * dh
    pz = (wp.float32(gz) + 0.5) * dh

    dx = px - center[0]
    dy = py - center[1]
    dz = pz - center[2]

    # calculate SDF
    dist = wp.sqrt(dx * dx + dy * dy + dz * dz) - radius
    solid_phi[wp.tid()] = dist
