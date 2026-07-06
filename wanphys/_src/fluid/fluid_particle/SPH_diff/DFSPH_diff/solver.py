import warp as wp
from typing import TYPE_CHECKING

from ..solver import SPHDiffSolver
from ...dfsph.kernels import compute_dfsph_alpha as dfsph_compute_dfsph_alpha
from ..kernels import (
    compute_moving_boundary_volume,
    sync_boundary_particles_from_object_id,
    compute_boundary_psi_from_boundary_grid,
    compute_density,
    compute_density_object_id,
    compute_dfsph_factor_kernel,
    compute_density_change_kernel,
    compute_density_adv_kernel,
    divergence_solve_iteration_kernel_fluid,
    divergence_solve_iteration_kernel_solid,
    pressure_solve_iteration_kernel_fluid,
    pressure_solve_iteration_kernel_solid,
    compute_density_error_kernel,
    compute_non_pressure_forces,
    kick,
    drift,
)

from .state import DFSPHDiffState

if TYPE_CHECKING:
    from ..state import SPHDiffState


class DFSPHDiffSolver(SPHDiffSolver):
    """DFSPH solver wrapping SPHDiffSolver with DFSPH-specific substep algorithm."""

    def __init__(
        self,
        model_adapter,
        sphdiff_model,
    ):
        super().__init__(model_adapter, sphdiff_model)
        self.m_max_iterations = 5
        self.max_error = 0.05
        self.max_error_V = 0.1
        self.m_max_iterations_v = 5
        self.enable_divergence_solver = True
        self.enable_pressure_solver = True

    def sub_step(
        self,
        state_in: DFSPHDiffState,
        state_out: DFSPHDiffState,
        dt: float,
        grid,
        current_tape,
    ) -> None:
        """DFSPH-specific substep following SimDFSPH_diff control flow."""
        sphdiff = self._sphdiff_model

        x_in = state_in.particle_q
        v_in = state_in.particle_qd
        x_out = state_out.particle_q
        v_out = state_out.particle_qd
        rho_out = state_out.rho
        # non-pressure acceleration (prediction target)
        a_non_p = state_out.a

        dfsph_fac = state_in.dfsph_factor
        material_marks = state_in.material_marks
        particle_flags = state_in.particle_flags
        boundary_grid_id = self._boundary_grid.id if int(state_in.boundary_q.shape[0]) > 0 else wp.uint64(0)
        particle_volume = float(state_in.m_V0)
        has_boundary = 1 if int(state_in.boundary_q.shape[0]) > 0 else 0

        # If there are boundary particles, sync their positions from rigid particle indices
        # and recompute `boundary_psi` so its values match the current boundary volume.
        if has_boundary != 0:
            wp.launch(
                kernel=sync_boundary_particles_from_object_id,
                dim=int(state_out.boundary_q.shape[0]),
                inputs=[state_out.boundary_q, state_in.particle_q, state_in.boundary_indices],
                device=sphdiff._device,
            )

            # Rebuild boundary grid from the synced positions before recomputing psi.
            with wp.ScopedDevice(sphdiff._device):
                self._boundary_grid.build(state_out.boundary_q, radius=sphdiff.h)

            wp.launch(
                kernel=compute_boundary_psi_from_boundary_grid,
                dim=int(state_out.boundary_q.shape[0]),
                inputs=[
                    boundary_grid_id,
                    state_out.boundary_q,
                    state_out.boundary_psi,
                    sphdiff.h,
                    sphdiff.rest_density,
                ],
                device=sphdiff._device,
            )

        # 1. compute density at x_in
        wp.launch(
            kernel=compute_density_object_id,
            dim=state_in.particle_count,
            inputs=[
                grid.id,
                x_in,
                rho_out,
                1.0,
                sphdiff.h,
                state_in.m_V,
                sphdiff.rest_density,
                particle_flags,
                state_out.boundary_q,
                state_out.boundary_psi,
                boundary_grid_id,
                has_boundary,
            ],
            device=sphdiff._device,
        )

        # Keep dfsph alpha launch for A/B debugging; currently disabled by request.
        # wp.launch(
        #     kernel=dfsph_compute_dfsph_alpha,
        #     dim=state_in.particle_count,
        #     inputs=[
        #         state_in.particle_count,
        #         dfsph_fac,
        #         x_in,
        #         particle_flags,
        #         grid.id,
        #         particle_volume,
        #         sphdiff.h,
        #         state_in.boundary_q,
        #         state_in.boundary_psi,
        #         boundary_grid_id,
        #         sphdiff.rest_density,
        #         has_boundary,
        #     ],
        #     device=sphdiff._device,
        # )

        wp.launch(
            kernel=compute_dfsph_factor_kernel,
            dim=state_in.particle_count,
            inputs=[
                state_in.particle_count,
                dfsph_fac,
                x_in,
                particle_flags,
                grid.id,
                particle_volume,
                sphdiff.h,
                state_out.boundary_q,
                state_out.boundary_psi,
                boundary_grid_id,
                sphdiff.rest_density,
                has_boundary,
            ],
            device=sphdiff._device,
        )

        # Clear rigid accumulators at start of sub-step to avoid stale accumulation
        state_in.rbs.rigid_force.zero_()
        state_in.rbs.rigid_torque.zero_()

        if self.enable_divergence_solver:
            with current_tape:
                density_change_buf = state_in.density_change
                wp.launch(
                    kernel=compute_density_change_kernel,
                    dim=state_in.particle_count,
                    inputs=[grid.id, x_in, v_in, material_marks, state_in.m_V, sphdiff.h, int(getattr(sphdiff, 'dim', 3))],
                    outputs=[density_change_buf],
                    device=sphdiff._device,
                )

                for _ in range(self.m_max_iterations_v):
                    wp.launch(
                        kernel=divergence_solve_iteration_kernel_fluid,
                        dim=state_in.particle_count,
                        inputs=[grid.id, x_in, density_change_buf, dfsph_fac, material_marks, state_in.m_V, sphdiff.h, dt, v_in],
                        device=sphdiff._device,
                    )

                    wp.launch(
                        kernel=divergence_solve_iteration_kernel_solid,
                        dim=state_in.particle_count,
                        inputs=[
                            grid.id,
                            x_in,
                            rho_out,
                            density_change_buf,
                            dfsph_fac,
                            material_marks,
                            state_in.m_V,
                            sphdiff.h,
                            dt,
                            state_in.object_id,
                            state_in.rbs.rigid_x,
                            v_in,
                            state_in.rbs.rigid_force,
                            state_in.rbs.rigid_torque,
                        ],
                        device=sphdiff._device,
                    )

                    wp.launch(
                        kernel=compute_density_change_kernel,
                        dim=state_in.particle_count,
                        inputs=[grid.id, x_in, v_in, material_marks, state_in.m_V, sphdiff.h, int(getattr(sphdiff, 'dim', 3))],
                        outputs=[density_change_buf],
                        device=sphdiff._device,
                    )

                    state_in.density_error_accum.zero_()
                    wp.launch(
                        kernel=compute_density_error_kernel,
                        dim=state_in.particle_count,
                        inputs=[density_change_buf, material_marks, sphdiff.rest_density, 0.0, state_in.density_error_accum],
                        device=sphdiff._device,
                    )

                    avg_err_div = state_in.density_error_accum.numpy()[0] / max(1, state_in.particle_count)
                    if avg_err_div <= (1.0 / max(1, dt)) * self.max_error_V * 0.01 * sphdiff.rest_density:
                        break

        # Divergence corrections are solver-internal; do not pass them through as rigid force.
        state_in.rbs.rigid_force.zero_()
        state_in.rbs.rigid_torque.zero_()

        wp.launch(
            kernel=compute_non_pressure_forces,
            dim=state_in.particle_count,
            inputs=[
                grid.id,
                x_in,
                v_in,
                rho_out,
                sphdiff.viscosity,
                sphdiff.h,
                material_marks,
                state_in.m_V,
                sphdiff.rest_density,
                state_out.viscous_forces,
                sphdiff.surface_tension,
                sphdiff.gravity[2],
                state_out.a,
            ],
            device=sphdiff._device,
        )

        with current_tape:
            wp.launch(
                kernel=kick,
                dim=state_in.particle_count,
                inputs=[a_non_p, dt, v_in],
                outputs=[v_out],
                device=sphdiff._device,
            )

        # 5. pressure solver loop
        if self.enable_pressure_solver:
            with current_tape:
                density_adv_buf = state_in.density_adv
                wp.launch(
                    kernel=compute_density_adv_kernel,
                    dim=state_in.particle_count,
                    inputs=[
                        grid.id,
                        x_in,
                        v_out,
                        rho_out,
                        material_marks,
                        state_in.m_V,
                        sphdiff.h,
                        dt,
                        sphdiff.rest_density,
                    ],
                    outputs=[density_adv_buf],
                    device=sphdiff._device,
                )

                for m in range(self.m_max_iterations):
                    wp.launch(
                        kernel=pressure_solve_iteration_kernel_fluid,
                        dim=state_in.particle_count,
                        inputs=[
                            grid.id,
                            x_in,
                            v_out,
                            density_adv_buf,
                            dfsph_fac,
                            material_marks,
                            state_in.m_V,
                            sphdiff.h,
                            dt,
                            v_out,
                        ],
                        device=sphdiff._device,
                    )

                    wp.launch(
                        kernel=pressure_solve_iteration_kernel_solid,
                        dim=state_in.particle_count,
                        inputs=[
                            grid.id,
                            x_in,
                            density_adv_buf,
                            dfsph_fac,
                            material_marks,
                            state_in.m_V,
                            sphdiff.h,
                            dt,
                            sphdiff.rest_density,
                            v_out,
                            state_in.object_id,
                            state_in.rbs.rigid_force,
                            state_in.rbs.rigid_torque,
                            state_in.rbs.rigid_x,
                        ],
                        device=sphdiff._device,
                    )
                    wp.launch(
                        kernel=compute_density_adv_kernel,
                        dim=state_in.particle_count,
                        inputs=[
                            grid.id,
                            x_in,
                            v_out,
                            rho_out,
                            material_marks,
                            state_in.m_V,
                            sphdiff.h,
                            dt,
                            sphdiff.rest_density,
                        ],
                        outputs=[density_adv_buf],
                        device=sphdiff._device,
                    )

                    state_in.density_error_accum.zero_()
                    wp.launch(
                        kernel=compute_density_error_kernel,
                        dim=state_in.particle_count,
                        inputs=[
                            density_adv_buf,
                            material_marks,
                            sphdiff.rest_density,
                            sphdiff.rest_density,
                            state_in.density_error_accum,
                        ],
                        device=sphdiff._device,
                    )

                    avg_err_press = state_in.density_error_accum.numpy()[0] / max(1, state_in.particle_count)
                    if avg_err_press <= self.max_error * 0.01 * sphdiff.rest_density:
                        break

        # 6. advect
        with current_tape:
            wp.launch(
                kernel=drift,
                dim=state_in.particle_count,
                inputs=[x_in, v_out, dt],
                outputs=[x_out],
                device=sphdiff._device,
            )
