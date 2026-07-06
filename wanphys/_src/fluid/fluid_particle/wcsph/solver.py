# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WCSPH solver - Weakly Compressible SPH time integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..solver import ParticleFluidSolverBase
from .kernels import (
    add_vec3,
    clear_vec3,
    compute_density_only,
    compute_density_pressure,
    resolve_soft_contacts_inelastic,
    compute_sph_forces,
    compute_xsph_delta_v,
    integrate_semi_implicit,
)

if TYPE_CHECKING:
    from newton import Contacts
    from wanphys._src.rigid.model import RigidModel
    from wanphys._src.rigid.state import RigidState
    from .model import WCSPHModel
    from .state import WCSPHState


class WCSPHSolver(ParticleFluidSolverBase):
    """Weakly Compressible SPH solver.

    Implements the WCSPH algorithm:
    1. Build neighbor hash grid
    2. Clear external forces (optional)
    3. Compute density and pressure (Tait EOS)
    4. Compute SPH forces (pressure + viscosity)
    5. Accumulate total forces (external + SPH)
    6. Semi-implicit integration (kick-drift)
    7. rigid-fluid contacts
    8. XSPH velocity smoothing (optional)

    Example:
        >>> wcsph_model = WCSPHModel(h=0.08, rest_density=1000.0, ...)
        >>> solver = WCSPHSolver(wcsph_model)
        >>> solver.step(state_in, state_out, dt=1/60)
    """

    def __init__(self, wcsph_model: WCSPHModel):
        """Initialize WCSPH solver.

        Args:
            wcsph_model: WCSPH configuration with particle data.
        """
        self._wcsph_model = wcsph_model

        # Local hash grid for neighbor search
        self._grid = wp.HashGrid(128, 128, 128)

        # CUDA graph cache for performance
        self._graphs = {}
        self._max_graph_cache_size = 4

    @property
    def wcsph_model(self) -> WCSPHModel:
        """WCSPH configuration."""
        return self._wcsph_model

    def resolve_contacts(
        self,
        state: WCSPHState,
        contacts: Contacts | None,
        rigid_state: RigidState,
        rigid_model: RigidModel,
    ) -> None:
        """统一处理 rigid-fluid contacts"""
        if contacts is None:
            return
        wp.launch(
            resolve_soft_contacts_inelastic,
            dim=contacts.soft_contact_max,
            inputs=[
                state.particle_q,
                state.particle_qd,
                self._wcsph_model.particle_radius,
                self._wcsph_model.particle_flags,
                rigid_state.body_q,
                rigid_model.shape_body,
                contacts.soft_contact_count,
                contacts.soft_contact_particle,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                self._wcsph_model.contact_margin,
                self._wcsph_model.contact_max_push_frac,
                self._wcsph_model.contact_vel_damp,
            ],
            device=self._wcsph_model._device
        )
    def step(
        self,
        state_in: WCSPHState,
        state_out: WCSPHState,
        dt: float,
        contacts: Contacts | None = None,
    ) -> None:
        """Advance WCSPH simulation by one timestep.

        Args:
            state_in: Current state.
            state_out: State to write results into.
            dt: Timestep in seconds.
        """
        model = self._wcsph_model
        wcsph = model

        if model.particle_count == 0:
            return

        # Clamp timestep for stability
        dt = float(min(dt, 1.0 / 200.0))

        grid = self._grid

        def _launch_sequence():
            # 1. Build neighbor grid
            with wp.ScopedDevice(wcsph._device):
                grid.build(state_in.particle_q, radius=wcsph.h)

            # 2. Clear external forces (optional)
            if wcsph.clear_external_forces:
                wp.launch(
                    kernel=clear_vec3,
                    dim=model.particle_count,
                    inputs=[state_in.particle_f],
                    device=wcsph._device,
                )

            # 3. Compute density + pressure
            wp.launch(
                kernel=compute_density_pressure,
                dim=model.particle_count,
                inputs=[
                    grid.id,
                    state_in.particle_q,
                    model.particle_mass,
                    model.particle_flags,
                    wcsph.h,
                    wcsph.rest_density,
                    wcsph.poly6_coef,
                    wcsph.c0,
                    wcsph.gamma,
                ],
                outputs=[state_out._rho, state_out._pressure],
                device=wcsph._device,
            )

            # 4. Compute SPH forces
            wp.launch(
                kernel=compute_sph_forces,
                dim=model.particle_count,
                inputs=[
                    grid.id,
                    state_in.particle_q,
                    state_in.particle_qd,
                    model.particle_mass,
                    model.particle_flags,
                    wcsph.h,
                    wcsph.viscosity,
                    wcsph.spiky_grad_coef,
                    wcsph.visc_lap_coef,
                    state_out._rho,
                    state_out._pressure,
                ],
                outputs=[state_out._f_sph],
                device=wcsph._device,
            )

            # 5. Total force = external + SPH
            wp.launch(
                kernel=add_vec3,
                dim=model.particle_count,
                inputs=[state_in.particle_f, state_out._f_sph],
                outputs=[state_out._f_total],
                device=wcsph._device,
            )

            # 6. Semi-implicit integration
            wp.launch(
                kernel=integrate_semi_implicit,
                dim=model.particle_count,
                inputs=[
                    state_in.particle_q,
                    state_in.particle_qd,
                    state_out._f_total,
                    model.particle_mass,
                    model.particle_flags,
                    wcsph.get_gravity_vec3(),
                    dt,
                    wcsph.max_velocity,
                ],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=wcsph._device,
            )
            # 7. Unified rigid-fluid contact projection (optional)
            # if contacts is not None:
            #     self.resolve_contacts(state_out, contacts)
            # 8. XSPH velocity smoothing (optional)
            if wcsph.xsph_c > 0.0:
                # Rebuild grid with updated positions
                with wp.ScopedDevice(wcsph._device):
                    grid.build(state_out.particle_q, radius=wcsph.h)

                # Recompute density
                wp.launch(
                    kernel=compute_density_only,
                    dim=model.particle_count,
                    inputs=[
                        grid.id,
                        state_out.particle_q,
                        model.particle_mass,
                        model.particle_flags,
                        wcsph.h,
                        wcsph.rest_density,
                        wcsph.poly6_coef,
                    ],
                    outputs=[state_out._rho],
                    device=wcsph._device,
                )

                # Compute XSPH velocity correction
                wp.launch(
                    kernel=compute_xsph_delta_v,
                    dim=model.particle_count,
                    inputs=[
                        grid.id,
                        state_out.particle_q,
                        state_out.particle_qd,
                        model.particle_mass,
                        model.particle_flags,
                        wcsph.h,
                        wcsph.xsph_c,
                        wcsph.poly6_coef,
                        state_out._rho,
                    ],
                    outputs=[state_out._dv_xsph],
                    device=wcsph._device,
                )

                # Apply velocity correction
                wp.launch(
                    kernel=add_vec3,
                    dim=model.particle_count,
                    inputs=[state_out.particle_qd, state_out._dv_xsph],
                    outputs=[state_out.particle_qd],
                    device=wcsph._device,
                )

        # Fast path: no graph capture
        if not wcsph.use_graph:
            _launch_sequence()
            return

        # Graph key for caching
        key = (
            id(state_in.particle_q),
            id(state_out.particle_q),
            round(dt, 12),
            float(wcsph.xsph_c),
            bool(wcsph.clear_external_forces),
            int(grid.id),
        )

        graph = self._graphs.get(key)
        if graph is None:
            # Warmup before capture
            _launch_sequence()
            wp.synchronize()

            wp.capture_begin(device=wcsph._device)
            _launch_sequence()
            graph = wp.capture_end(device=wcsph._device)
            self._graphs[key] = graph
            if len(self._graphs) > self._max_graph_cache_size:
                # Keep cache bounded to prevent graph memory growth.
                self._graphs.pop(next(iter(self._graphs)))

        wp.capture_launch(graph)
