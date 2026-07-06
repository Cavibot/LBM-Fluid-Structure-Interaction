# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""DFSPH solver - Divergence-Free SPH time integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..solver import ParticleFluidSolverBase
from . import kernels as df_kernels

if TYPE_CHECKING:
    from newton import Contacts
    from .model import DFSPHModel
    from .state import DFSPHState

class DFSPHSolver(ParticleFluidSolverBase):
    """Divergence-Free SPH solver."""

    def __init__(self, dfsph_model: DFSPHModel):
        self._dfsph_model = dfsph_model

        # Local hash grids for neighbor search
        self._fluid_grid = wp.HashGrid(128, 128, 128)
        self._boundary_grid = wp.HashGrid(128, 128, 128)

        # Boundary build cache
        self._boundary_built = False
        self._boundary_count_cached = -1

        # CUDA graph cache
        self._graphs = {}
        self._max_graph_cache_size = 4

    @property
    def dfsph_model(self) -> DFSPHModel:
        return self._dfsph_model

    def _ensure_boundary_grid(self, state: DFSPHState, support_radius: float, device) -> None:
        """Build/rebuild boundary grid when needed."""
        boundary_count = int(state.boundary_q.shape[0])

        if boundary_count <= 0:
            self._boundary_built = False
            self._boundary_count_cached = 0
            return

        if (not self._boundary_built) or (self._boundary_count_cached != boundary_count):
            with wp.ScopedDevice(device):
                self._boundary_grid.build(state.boundary_q, support_radius)
            self._boundary_built = True
            self._boundary_count_cached = boundary_count

    def step(
        self,
        state_in: DFSPHState,
        state_out: DFSPHState,
        dt: float,
        contacts: Contacts | None = None,
    ) -> None:
        m = self._dfsph_model
        device = m._device

        n = int(state_in.particle_q.shape[0])
        if n == 0:
            return

        # -----------------------------
        # Model compatibility helpers
        # -----------------------------
        dt_used = float(getattr(m, "fixed_dt", dt))

        support_radius = float(getattr(m, "support_radius", getattr(m, "h", 0.04)))

        if hasattr(m, "particle_mass_scalar"):
            mass_scalar = float(m.particle_mass_scalar)
        else:
            mass_scalar = float(m.particle_mass)

        if hasattr(m, "particle_volume"):
            particle_volume = float(m.particle_volume)
        else:
            particle_volume = mass_scalar / max(float(m.rest_density), 1.0e-6)

        density_error_tolerance = float(m.density_error_tolerance)

        # Copy state_in -> state_out
        if state_in is not state_out:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)

            # Boundary arrays sync
            in_bcount = int(state_in.boundary_q.shape[0])
            out_bcount = int(state_out.boundary_q.shape[0])

            if in_bcount > 0:
                if out_bcount != in_bcount:
                    state_out.boundary_q = wp.clone(state_in.boundary_q)
                    state_out.boundary_psi = wp.clone(state_in.boundary_psi)
                else:
                    wp.copy(state_out.boundary_q, state_in.boundary_q)
                    wp.copy(state_out.boundary_psi, state_in.boundary_psi)

        # Fixed dt into state container
        state_out.deltaT.fill_(dt_used)

        fluid_grid = self._fluid_grid
        has_boundary = 0
        b_grid_id = wp.uint64(0)

        def _prepare_grids():
            nonlocal has_boundary, b_grid_id

            # 1) Build fluid grid
            with wp.ScopedDevice(device):
                fluid_grid.build(state_out.particle_q, support_radius)

            # 2) Build boundary grid if needed
            self._ensure_boundary_grid(state_out, support_radius, device)
            has_boundary = 1 if self._boundary_built and int(state_out.boundary_q.shape[0]) > 0 else 0
            b_grid_id = self._boundary_grid.id if has_boundary else wp.uint64(0)

        def _launch_sequence():

            # 3) Density
            wp.launch(
                kernel=df_kernels.compute_density,
                dim=n,
                inputs=[
                    n,
                    state_out.rho,
                    state_out.particle_q,
                    m.particle_flags,
                    fluid_grid.id,
                    mass_scalar,
                    support_radius,
                    state_out.boundary_q,
                    state_out.boundary_psi,
                    b_grid_id,
                    has_boundary,
                ],
                device=device,
            )

            # 4) Alpha
            wp.launch(
                kernel=df_kernels.compute_dfsph_alpha,
                dim=n,
                inputs=[
                    n,
                    state_out.alpha,
                    state_out.particle_q,
                    m.particle_flags,
                    fluid_grid.id,
                    particle_volume,
                    support_radius,
                    state_out.boundary_q,
                    state_out.boundary_psi,
                    b_grid_id,
                    m.rest_density,
                    has_boundary,
                ],
                device=device,
            )

            # 5) Divergence solve
            wp.launch(
                kernel=df_kernels.begin_divergence_iter,
                dim=n,
                inputs=[
                    n,
                    fluid_grid.id,
                    state_out.particle_q,
                    state_out.particle_qd,
                    m.particle_flags,
                    state_out.adv_rho,
                    state_out.alpha,
                    state_out.kappa_v,
                    state_out.deltaT,
                    particle_volume,
                    support_radius,
                    state_out.boundary_q,
                    state_out.boundary_psi,
                    b_grid_id,
                    m.rest_density,
                    has_boundary,
                ],
                device=device,
            )

            for _ in range(int(m.divergence_max_iterations)):
                wp.launch(
                    kernel=df_kernels.divergence_step,
                    dim=n,
                    inputs=[
                        n,
                        fluid_grid.id,
                        state_out.particle_q,
                        state_out.particle_qd,
                        m.particle_flags,
                        state_out.adv_rho,
                        state_out.alpha,
                        state_out.kappa_v,
                        state_out.avg_density_err,
                        state_out.deltaT,
                        particle_volume,
                        support_radius,
                        state_out.boundary_q,
                        state_out.boundary_psi,
                        b_grid_id,
                        m.rest_density,
                        has_boundary,
                    ],
                    device=device,
                )

            wp.launch(
                kernel=df_kernels.end_divergence_iter,
                dim=n,
                inputs=[n, m.particle_flags, state_out.kappa_v, state_out.alpha, state_out.deltaT],
                device=device,
            )

            # 6) Non-pressure forces
            wp.launch(
                kernel=df_kernels.compute_forces_and_update_vel,
                dim=n,
                inputs=[
                    n,
                    fluid_grid.id,
                    state_out.particle_q,
                    state_out.particle_qd,
                    m.particle_flags,
                    state_out.rho,
                    mass_scalar,
                    m.get_gravity_vec3(),
                    support_radius,
                    m.viscosity,
                    state_out.deltaT,
                ],
                device=device,
            )

            # 7) Pressure solve
            state_out.density_err_prev.fill_(999999.0)

            wp.launch(
                kernel=df_kernels.begin_pressure_iter,
                dim=n,
                inputs=[
                    n,
                    fluid_grid.id,
                    state_out.particle_q,
                    state_out.particle_qd,
                    m.particle_flags,
                    state_out.adv_rho,
                    state_out.rho,
                    state_out.alpha,
                    state_out.kappa,
                    m.rest_density,
                    particle_volume,
                    support_radius,
                    state_out.deltaT,
                    state_out.boundary_q,
                    state_out.boundary_psi,
                    b_grid_id,
                    has_boundary,
                ],
                device=device,
            )

            for _ in range(int(m.pressure_max_iterations)):
                state_out.density_err_curr.zero_()

                wp.launch(
                    kernel=df_kernels.pressure_iter,
                    dim=n,
                    inputs=[
                        n,
                        fluid_grid.id,
                        state_out.particle_q,
                        state_out.particle_qd,
                        m.particle_flags,
                        state_out.adv_rho,
                        state_out.rho,
                        state_out.alpha,
                        state_out.kappa,
                        state_out.density_err_prev,
                        state_out.density_err_curr,
                        density_error_tolerance,
                        m.rest_density,
                        particle_volume,
                        support_radius,
                        state_out.deltaT,
                        state_out.boundary_q,
                        state_out.boundary_psi,
                        b_grid_id,
                        has_boundary,
                    ],
                    device=device,
                )

                wp.copy(state_out.density_err_prev, state_out.density_err_curr)

            wp.launch(
                kernel=df_kernels.end_pressure_iter,
                dim=n,
                inputs=[n, m.particle_flags, state_out.kappa, state_out.deltaT],
                device=device,
            )

            # 8) Position update
            wp.launch(
                kernel=df_kernels.update_pos,
                dim=n,
                inputs=[n, state_out.particle_q, state_out.particle_qd, m.particle_flags, state_out.deltaT],
                device=device,
            )

        # Fast path
        if not bool(m.use_graph):
            _prepare_grids()
            _launch_sequence()
            return

        # Graph cache key
        key = (
            id(state_in.particle_q),
            id(state_out.particle_q),
            round(dt_used, 12),
            int(n),
            int(fluid_grid.id),
            int(state_out.boundary_q.shape[0]),
            int(m.divergence_max_iterations),
            int(m.pressure_max_iterations),
            int(support_radius * 1e6),
        )

        _prepare_grids()

        graph = self._graphs.get(key)
        if graph is None:
            _launch_sequence()
            wp.synchronize()

            # Rebuild grids explicitly outside the captured region.
            _prepare_grids()

            wp.capture_begin(device=device)
            _launch_sequence()
            graph = wp.capture_end(device=device)
            self._graphs[key] = graph
            if len(self._graphs) > self._max_graph_cache_size:
                # Keep cache bounded to prevent graph memory growth.
                self._graphs.pop(next(iter(self._graphs)))
        else:
            wp.capture_launch(graph)
