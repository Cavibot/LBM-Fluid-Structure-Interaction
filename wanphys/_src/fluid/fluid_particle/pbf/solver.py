# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
# pyright: reportInvalidTypeForm=false

"""PBF solver - Position Based Fluids time integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..solver import ParticleFluidSolverBase
from .kernels import (
    integrate_gravity,
    calculate_lambdas,
    calculate_position_update,
    calculate_vorticity,
    clamp_aabb_bounds,
    enforce_aabb_bounds,
    solve_boundary_contacts,
    update_velocity,
    apply_position_deltas,
    apply_vorticity_confinement_and_XSPH_viscosity,
)

if TYPE_CHECKING:
    from newton import Contacts
    from wanphys._src.rigid.model import RigidModel
    from wanphys._src.rigid.state import RigidState
    from .model import PBFModel
    from .state import PBFState


class PBFSolver(ParticleFluidSolverBase):
    """Position Based Fluids solver.

    Implements the PBF algorithm from Macklin & Müller (2013):
    1. Backup positions for velocity update
    2. Apply gravity (integrate particles)
    3. Build neighbor hash grid
    4. Iterative constraint solving:
       - Calculate constraint multipliers (λ)
       - Calculate position corrections (Δp)
       - Resolve boundary contacts
       - Apply position corrections
    5. Update velocities from position change
    6. Apply vorticity confinement and XSPH viscosity

    The solver uses Newton's Model for particle management and collision detection,
    wrapped behind the WanPhys API.

    Example:
        >>> # In the dam-break example, PBFSolver is used via PBFDomain.
        >>> pbf_domain = PBFDomain(fluid_model.particle_q, fluid_model.particle_qd, pbf_model)
        >>> pbf_domain.create_state()
        >>>
        >>> pbf_domain.pre_step(sim_dt)
        >>> contacts = CollisionPipeline.collide_rigid_fluid(
        ...     rigid_domain,
        ...     pbf_domain,
        ...     rigid_state=rigid_domain.state,
        ...     fluid_state=pbf_domain.state,
        ... )
        >>> pbf_domain.step(sim_dt, contacts=contacts, rigid_state=rigid_domain.state, rigid_model=rigid_domain.model)
        >>> pbf_domain.post_step(sim_dt)
    """

    def __init__(self, pbf_model: PBFModel):
        """Initialize PBF solver.

        Args:
            pbf_model: PBF configuration parameters.
        """
        self._pbf_model = pbf_model

        # Pre-compute rest density array (all particles same density)
        self._particle_count = self._pbf_model.particle_count
        device = self._pbf_model._device

        self._particle_rest_density = self._pbf_model.rest_density
        if isinstance(self._pbf_model.particle_radius, wp.array):
            radius_np = self._pbf_model.particle_radius.numpy()
            if radius_np.size == 0:
                raise ValueError("particle_radius array is empty")
            self._particle_radius_scalar = float(radius_np[0])
        else:
            self._particle_radius_scalar = float(self._pbf_model.particle_radius)
        
        self._grid = wp.HashGrid(128, 128, 128, device=device)

        # CUDA graph capture
        self._graphs = {}
        self._max_graph_cache_size = 4

    @property
    def pbf_model(self) -> PBFModel:
        """PBF configuration."""
        return self._pbf_model

    def step(
        self,
        state_in: PBFState,
        state_out: PBFState,
        dt: float,
        contacts: Contacts | None = None,
        rigid_state: RigidState | None = None,
        rigid_model: RigidModel | None = None,
    ) -> None:
        """Advance PBF simulation by one timestep.

        Args:
            state_in: Current state.
            state_out: State to write results into.
            dt: Timestep in seconds.
            contacts: Collision contacts from Newton (for boundary handling).
        """
        pbf = self._pbf_model

        if self._particle_count == 0:
            return

        particle_count = int(self._particle_count)
        has_boundary_contacts = (
            contacts is not None
            and rigid_state is not None
            and rigid_model is not None
        )

        contact_key = (
            int(getattr(contacts, "soft_contact_max", 0)),
            id(getattr(contacts, "soft_contact_count", None)),
            id(getattr(contacts, "soft_contact_particle", None)),
            id(getattr(contacts, "soft_contact_shape", None)),
            id(getattr(contacts, "soft_contact_body_pos", None)),
            id(getattr(contacts, "soft_contact_body_vel", None)),
            id(getattr(contacts, "soft_contact_normal", None)),
        ) if has_boundary_contacts else (0, 0, 0, 0, 0, 0, 0)

        graph_key = (
            id(state_in.particle_q),
            id(state_out.particle_q),
            round(float(dt), 12),
            particle_count,
            int(self._grid.id),
            int(pbf.iterations),
            1 if has_boundary_contacts else 0,
            contact_key,
        )

        def _prepare_predicted_state():
            # Backup positions before projection; used for velocity reconstruction.
            wp.copy(state_out._particle_q_init, state_in.particle_q)

            # Semi-implicit integration with gravity.
            self._integrate_particles(state_in, state_out, dt)

            # Rebuild hash grid for neighbor queries on predicted positions.
            if self._grid:
                self._grid.build(
                    points=state_out.particle_q,
                    radius=pbf.support_radius,
                )

        def _solve_sequence():
            self._execute_solver_iterations(
                state_out,
                dt,
                contacts,
                rigid_state,
                rigid_model,
            )

        if not bool(pbf.use_graph):
            _prepare_predicted_state()
            _solve_sequence()
            return

        with wp.ScopedTimer("simulate", False):
            # Always prepare predicted positions and rebuild grid outside graph capture.
            _prepare_predicted_state()

            graph = self._graphs.get(graph_key)
            if graph is None:
                # Warm up once to avoid capturing first-use allocations.
                _solve_sequence()
                wp.synchronize()

                # Re-prepare state because warmup mutated state_out.
                _prepare_predicted_state()

                wp.capture_begin(device=pbf._device)
                _solve_sequence()
                graph = wp.capture_end(device=pbf._device)
                self._graphs[graph_key] = graph
                if len(self._graphs) > self._max_graph_cache_size:
                    # Keep cache bounded to prevent graph memory growth.
                    self._graphs.pop(next(iter(self._graphs)))

            wp.capture_launch(graph)

    def _integrate_particles(
        self,
        state_in: PBFState,
        state_out: PBFState,
        dt: float,
    ) -> None:
        """Apply gravity to particle velocities and positions."""
        pbf = self._pbf_model

        gx, gy, gz = pbf.gravity
        gravity = wp.vec3(gx, gy, gz)

        wp.launch(
            kernel=integrate_gravity,
            dim=pbf.particle_count,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                pbf.particle_flags,
                gravity,
                dt,
                pbf.max_velocity,
            ],
            outputs=[
                state_out.particle_q,
                state_out.particle_qd,
            ],
            device=pbf._device,
        )

    def _execute_solver_iterations(
        self,
        state: PBFState,
        dt: float,
        contacts=None,
        rigid_state=None,
        rigid_model=None,
    ) -> None:
        """Execute PBF constraint iterations."""
        pbf = self._pbf_model

        particle_q = state.particle_q
        particle_qd = state.particle_qd

        has_boundary_contacts = (
            contacts is not None
            and rigid_state is not None
            and rigid_model is not None
        )

        if has_boundary_contacts:
            # Clear previous-step rigid reaction forces before this step's accumulation.
            rigid_state.body_f.zero_()

        for _ in range(pbf.iterations):
            # 1. Lambdas
            wp.launch(
                kernel=calculate_lambdas,
                dim=pbf.particle_count,
                inputs=[
                    self._grid.id,
                    particle_q,
                    pbf.particle_flags,
                    pbf.particle_mass,
                    self._particle_rest_density,
                    pbf.support_radius,
                    pbf.support_radius_sq,
                    pbf.poly6_coef,
                    pbf.spiky_grad_coef,
                    pbf.relaxation_parameter,
                ],
                outputs=[state._particle_lambdas],
                device=pbf._device,
            )

            # 2. Delta Position
            wp.launch(
                kernel=calculate_position_update,
                dim=pbf.particle_count,
                inputs=[
                    self._grid.id,
                    particle_q,
                    pbf.particle_flags,
                    pbf.particle_mass,
                    state._particle_lambdas,
                    self._particle_rest_density,
                    pbf.support_radius,
                    pbf.support_radius_sq,
                    pbf.poly6_coef,
                    pbf.spiky_grad_coef,
                    pbf.artificial_pressure_k,
                    pbf.artificial_pressure_n,
                    pbf.d_q,
                ],
                outputs=[state._particle_position_deltas],
                device=pbf._device,
            )

            # 3. Boundary Contacts
            if has_boundary_contacts:
                wp.launch(
                    kernel=solve_boundary_contacts,
                    dim=contacts.soft_contact_max,
                    inputs=[
                        particle_q,
                        state.particle_qd,
                        pbf.particle_flags,
                        self._particle_radius_scalar,
                        pbf.particle_mass,
                        rigid_state.body_q,
                        rigid_state.body_qd,
                        rigid_model.body_com,
                        contacts.soft_contact_count,
                        contacts.soft_contact_particle,
                        contacts.soft_contact_shape,
                        rigid_model.shape_body,
                        contacts.soft_contact_body_pos,
                        contacts.soft_contact_body_vel,
                        contacts.soft_contact_normal,
                        pbf.boundary_friction,
                        dt,
                    ],
                    outputs=[state._particle_position_deltas, rigid_state.body_f],
                    device=pbf._device,
                )

            # 4. Apply
            wp.launch(
                kernel=apply_position_deltas,
                dim=pbf.particle_count,
                inputs=[
                    self._grid.id,
                    particle_q,
                    pbf.particle_flags,
                    state._particle_position_deltas,
                ],
                device=pbf._device,
            )

        # Reconstruct velocity from corrected positions.
        if pbf.particle_count:
            wp.launch(
                kernel=update_velocity,
                dim=pbf.particle_count,
                inputs=[
                    self._grid.id,
                    particle_q,
                    state._particle_q_init,
                    pbf.particle_flags,
                    dt,
                ],
                outputs=[particle_qd],
                device=pbf._device,
            )

        # Apply vorticity confinement and XSPH viscosity post-process.
        with wp.ScopedTimer("vorticity confinement and XSPH viscosity", False):
            if pbf.particle_count:
                wp.launch(
                    kernel=calculate_vorticity,
                    dim=pbf.particle_count,
                    inputs=[
                        self._grid.id,
                        particle_q,
                        particle_qd,
                        pbf.particle_flags,
                        pbf.particle_mass,
                        self._particle_rest_density,
                        pbf.support_radius,
                        pbf.support_radius_sq,
                        pbf.spiky_grad_coef,
                    ],
                    outputs=[state._particle_vorticities],
                    device=pbf._device,
                )

                wp.launch(
                    kernel=apply_vorticity_confinement_and_XSPH_viscosity,
                    dim=pbf.particle_count,
                    inputs=[
                        self._grid.id,
                        particle_q,
                        particle_qd,
                        pbf.particle_flags,
                        pbf.particle_mass,
                        self._particle_rest_density,
                        state._particle_vorticities,
                        pbf.support_radius,
                        pbf.support_radius_sq,
                        pbf.poly6_coef,
                        pbf.spiky_grad_coef,
                        dt,
                        pbf.vorticity_coefficient,
                        pbf.xsph_c,
                    ],
                    device=pbf._device,
                )
