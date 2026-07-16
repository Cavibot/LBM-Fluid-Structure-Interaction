# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 FullF/HOME stream-physics-collide-encode pipeline."""

from __future__ import annotations

from typing import Any

import warp as wp

from ..base import FluidGridSolverBase
from . import collisions, encoding, kernels, streaming
from .model import LbmModel
from .state import FullFLbmState, HomeLbmState, LbmState, LbmStateBase


class LbmSolver(FluidGridSolverBase):
    """D3Q19 LBM solver with pluggable persistent encoding and collision.

    The solver owns temporary arrays for macroscopic moments that are
    reused across steps.  Distribution functions and visualisation fields
    live on :class:`LbmState`.

    Parameters
    ----------
    model:
        Static LBM configuration (grid size, *τ*, body force, …).
    """

    def __init__(self, model: LbmModel) -> None:
        self.model: LbmModel = model
        self.nx: int = int(model.nx)
        self.ny: int = int(model.ny)
        self.nz: int = int(model.nz)
        self.device: wp.Device = model._device

        # Stride for flat-array indexing: nx * ny * nz
        self._stride: int = self.nx * self.ny * self.nz

        # SC force stride: recompute only every N steps, reuse cached force
        # between updates.  Configured by LbmModel.sc_force_stride.
        self._step_count: int = 0
        self._sc_stride: int = int(model.sc_force_stride)

        # ---- Boundary condition arrays (synced from model) ------------------
        self._bc_types = wp.zeros(6, dtype=wp.int32, device=self.device)
        self._bc_vel_x = wp.zeros(6, dtype=float, device=self.device)
        self._bc_vel_y = wp.zeros(6, dtype=float, device=self.device)
        self._bc_vel_z = wp.zeros(6, dtype=float, device=self.device)
        self._sync_bc_from_model()

        # ---- Solver-owned temporary macroscopic fields --------------------
        self._rho: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )
        self._ux: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )
        self._uy: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )
        self._uz: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )

        # ---- Solver-owned temporary force arrays (Shan-Chen interaction) ---
        self._fx: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )
        self._fy: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )
        self._fz: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )

        self._collision_name = model.resolved_collision
        backend_types = {
            "srt": collisions.SrtCollision,
            "trt": collisions.TrtCollision,
            "home_nocm": collisions.HomeNocmMrtCollision,
        }
        self._collision_backend = backend_types[self._collision_name]()
        self._collision_kind = self._collision_backend.input_kind

        # ---- Explicit-population scratch ---------------------------------
        self._f_star = None
        self._f_post = None
        if self._collision_kind == "population":
            self._f_star = wp.zeros(19 * self._stride, dtype=float, device=self.device)
            if model.encoding == "home":
                self._f_post = wp.zeros(19 * self._stride, dtype=float, device=self.device)

        # ---- Encoded-moment scratch (rho, rho*u, rho*S) -------------------
        shape = (self.nx, self.ny, self.nz)
        self._moments: tuple[wp.array3d, ...] = ()
        self._post_moments: tuple[wp.array3d, ...] = ()
        if self._collision_kind == "moments":
            self._moments = tuple(
                wp.zeros(shape, dtype=float, device=self.device) for _ in range(10)
            )
            if model.encoding == "fullf":
                self._post_moments = tuple(
                    wp.zeros(shape, dtype=float, device=self.device) for _ in range(10)
                )
        if self._collision_name == "home_nocm" and any(
            g != 0.0 for g in (model.gravity_x, model.gravity_y, model.gravity_z)
        ):
            raise NotImplementedError("HOME-NOCM forcing is not implemented in this iteration")
        if model.encoding == "home" and any(
            g != 0.0 for g in (model.gravity_x, model.gravity_y, model.gravity_z)
        ):
            raise NotImplementedError("HOME encoding with prescribed force is not implemented")
        unsupported_bc = [t for t in model.bc_types if t not in (0, 3)]
        if unsupported_bc and not model.has_moving_walls:
            raise NotImplementedError(
                "The post-collision pipeline currently supports only periodic "
                "and static halfway bounce-back boundaries"
            )

    def create_state(self, requires_grad: bool = False) -> LbmStateBase:
        """Create the concrete persistent state selected by the model."""
        if self.model.encoding == "home":
            return HomeLbmState(self.model, requires_grad=requires_grad)
        return FullFLbmState(self.model, requires_grad=requires_grad)

    # ------------------------------------------------------------------
    # Boundary condition helpers
    # ------------------------------------------------------------------

    def _sync_bc_from_model(self) -> None:
        """Copy BC parameters from the model to device arrays."""
        import numpy as np

        wp.copy(
            self._bc_types,
            wp.array(np.array(self.model.bc_types, dtype=np.int32), dtype=wp.int32, device=self.device),
        )
        wp.copy(
            self._bc_vel_x,
            wp.array(np.array([v[0] for v in self.model.bc_velocity], dtype=np.float32), dtype=float, device=self.device),
        )
        wp.copy(
            self._bc_vel_y,
            wp.array(np.array([v[1] for v in self.model.bc_velocity], dtype=np.float32), dtype=float, device=self.device),
        )
        wp.copy(
            self._bc_vel_z,
            wp.array(np.array([v[2] for v in self.model.bc_velocity], dtype=np.float32), dtype=float, device=self.device),
        )

    def set_boundary_condition(
        self,
        face: int,
        bc_type: int,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Set the boundary condition on one face of the domain.

        Parameters
        ----------
        face:
            Face index: 0=xmin, 1=xmax, 2=ymin, 3=ymax, 4=zmin, 5=zmax.
        bc_type:
            0 = bounce-back, 1 = Zou-He velocity inlet, 2 = convective outflow.
        velocity:
            Prescribed velocity ``(ux, uy, uz)`` in lattice units.  Only
            used when *bc_type* == 1 (Zou-He).
        """
        if bc_type not in (0, 3) and not self.model.has_moving_walls:
            raise NotImplementedError(
                "The post-collision pipeline currently supports only periodic "
                "and static halfway bounce-back boundaries"
            )
        types = list(self.model.bc_types)
        vels = list(self.model.bc_velocity)
        types[face] = bc_type
        vels[face] = tuple(velocity)
        self.model.bc_types = tuple(types)  # type: ignore[assignment]
        self.model.bc_velocity = tuple(vels)  # type: ignore[assignment]
        self._sync_bc_from_model()

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        state_in: LbmStateBase,
        state_out: LbmStateBase,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Advance ``post-collision -> stream -> collide -> post-collision``."""
        if self.model.has_moving_walls:
            if not isinstance(state_in, FullFLbmState) or not isinstance(state_out, FullFLbmState):
                raise NotImplementedError("HOME encoding does not support rigid/moving-wall coupling")
            self._step_legacy(state_in, state_out, dt, contacts, control)
            return

        del contacts, control, dt
        self._copy_boundary_fields(state_in, state_out)
        px, py, pz = self.model._periodic_ints

        # 1. transport: encoding selects provider, collision selects collector
        if self._collision_kind == "population":
            assert self._f_star is not None
            self._stream_to_populations(state_in, px, py, pz)
            # 2. physics: local moments, then force/hydrodynamic closure
            wp.launch(
                kernels.compute_moments_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    self._f_star, self._stride, self.nx, self.ny, self.nz,
                    self._rho, self._ux, self._uy, self._uz,
                ],
                device=self.device,
            )
            self._prepare_population_physics(state_out)
            if self.model.use_regularization and self.model.omega_reg > 0.0:
                wp.launch(
                    kernels.reg_trt_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[
                        self._f_star, self._rho, self._ux, self._uy, self._uz,
                        float(self.model.omega_reg), px, py, pz,
                        self.nx, self.ny, self.nz, self._stride,
                    ],
                    device=self.device,
                )
            # 3. EPC collision
            if isinstance(state_out, FullFLbmState):
                output_f = state_out.f_post
            else:
                assert self._f_post is not None
                output_f = self._f_post
            omega_even, omega_odd = self._collision_backend.relaxation_rates(self.model)
            wp.launch(
                collisions.epc_collision_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    self._f_star, self._rho, self._ux, self._uy, self._uz,
                    output_f, omega_even, omega_odd,
                    self.ny, self.nz, self._stride,
                ],
                device=self.device,
            )
            self._apply_population_force(output_f)
            # 4. encode
            if isinstance(state_out, HomeLbmState):
                self._encode_populations_to_home(output_f, state_out)
        else:
            self._stream_to_moments(state_in, px, py, pz)
            m = self._moments
            if isinstance(state_out, HomeLbmState):
                collision_target_moments = state_out.kinetic_fields
            else:
                collision_target_moments = self._post_moments
            wp.launch(
                streaming.moment_velocity_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[m[0], m[1], m[2], m[3], self._ux, self._uy, self._uz],
                device=self.device,
            )
            wp.launch(
                collisions.home_nocm_collision_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[*m, *collision_target_moments, float(self.model.omega)],
                device=self.device,
            )
            if isinstance(state_out, FullFLbmState):
                pm = self._post_moments
                wp.launch(
                    encoding.home_to_populations_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[*pm, state_out.f_post, self.ny, self.nz, self._stride],
                    device=self.device,
                )
            wp.copy(self._rho, m[0])

        self._write_observables(state_out)

    def _copy_boundary_fields(self, state_in: LbmStateBase, state_out: LbmStateBase) -> None:
        for name in ("solid_phi", "solid_body_id", "vel_solid_u", "vel_solid_v", "vel_solid_w"):
            wp.copy(getattr(state_out, name), getattr(state_in, name))

    def _home_inputs(self, state: HomeLbmState) -> list[wp.array]:
        return list(state.kinetic_fields)

    def _stream_to_populations(self, state: LbmStateBase, px: int, py: int, pz: int) -> None:
        assert self._f_star is not None
        common = [state.solid_phi, self._f_star, px, py, pz, self.nx, self.ny, self.nz, self._stride]
        if isinstance(state, FullFLbmState):
            kernel = streaming.stream_fullf_to_populations_kernel
            inputs = [state.f_post, *common]
        elif isinstance(state, HomeLbmState):
            kernel = streaming.stream_home_to_populations_kernel
            inputs = [*self._home_inputs(state), *common]
        else:
            raise TypeError(f"Unsupported LBM state type: {type(state).__name__}")
        wp.launch(kernel, dim=(self.nx, self.ny, self.nz), inputs=inputs, device=self.device)

    def _stream_to_moments(self, state: LbmStateBase, px: int, py: int, pz: int) -> None:
        tail = [state.solid_phi, *self._moments, px, py, pz, self.nx, self.ny, self.nz]
        if isinstance(state, FullFLbmState):
            kernel = streaming.stream_fullf_to_moments_kernel
            inputs = [state.f_post, *tail, self._stride]
        elif isinstance(state, HomeLbmState):
            kernel = streaming.stream_home_to_moments_kernel
            inputs = [*self._home_inputs(state), *tail]
        else:
            raise TypeError(f"Unsupported LBM state type: {type(state).__name__}")
        wp.launch(kernel, dim=(self.nx, self.ny, self.nz), inputs=inputs, device=self.device)

    def _prepare_population_physics(self, state_out: LbmStateBase) -> None:
        G_sc = float(self.model.G)
        if G_sc == 0.0:
            return
        self._step_count += 1
        if self._step_count % self._sc_stride == 1:
            px, py, pz = self.model._periodic_ints
            wp.launch(
                kernels.compute_shan_chen_force_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    self._rho, state_out.solid_phi, self._fx, self._fy, self._fz,
                    G_sc, int(self.model.psi_type), float(self.model.psi_ref),
                    float(self.model.sc_solid_psi_scale), float(self.model.sc_boundary_psi),
                    float(self.model.cs_a), float(self.model.cs_b), float(self.model.cs_T),
                    int(self.model.sc_homogeneous_early_out),
                    float(self.model.sc_homogeneous_rel_tol), px, py, pz,
                    self.nx, self.ny, self.nz,
                ],
                device=self.device,
            )
        wp.launch(
            kernels.apply_velocity_shift_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                self._ux, self._uy, self._uz, self._rho,
                self._fx, self._fy, self._fz,
                float(self.model.gravity_x), float(self.model.gravity_y), float(self.model.gravity_z),
                float(self.model.tau), self.nx, self.ny, self.nz,
            ],
            device=self.device,
        )

    def _apply_population_force(self, f_post: wp.array) -> None:
        gx, gy, gz = float(self.model.gravity_x), float(self.model.gravity_y), float(self.model.gravity_z)
        if self.model.G == 0.0 and (gx != 0.0 or gy != 0.0 or gz != 0.0):
            wp.launch(
                kernels.apply_guo_force_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[f_post, gx, gy, gz, float(self.model.omega), self.nx, self.ny, self.nz, self._stride],
                device=self.device,
            )
        if self.model.G != 0.0:
            wp.launch(
                kernels.restore_physical_velocity_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    self._ux, self._uy, self._uz, self._rho,
                    self._fx, self._fy, self._fz, gx, gy, gz,
                    float(self.model.tau), self.nx, self.ny, self.nz,
                ],
                device=self.device,
            )

    def _encode_populations_to_home(self, f_post: wp.array, state: HomeLbmState) -> None:
        wp.launch(
            encoding.populations_to_home_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[f_post, *state.kinetic_fields, self.ny, self.nz, self._stride],
            device=self.device,
        )

    def _write_observables(self, state_out: LbmStateBase) -> None:
        wp.copy(state_out.density, self._rho)
        wp.copy(state_out.velocity_x, self._ux)
        wp.copy(state_out.velocity_y, self._uy)
        wp.copy(state_out.velocity_z, self._uz)
        if self.model.G != 0.0:
            wp.copy(state_out.force_x, self._fx)
            wp.copy(state_out.force_y, self._fy)
            wp.copy(state_out.force_z, self._fz)
        else:
            state_out.force_x.zero_()
            state_out.force_y.zero_()
            state_out.force_z.zero_()
        wp.launch(kernels.moments_to_mac_u_kernel, dim=(self.nx + 1, self.ny, self.nz), inputs=[self._ux, state_out.vel_u, self.nx], device=self.device)
        wp.launch(kernels.moments_to_mac_v_kernel, dim=(self.nx, self.ny + 1, self.nz), inputs=[self._uy, state_out.vel_v, self.ny], device=self.device)
        wp.launch(kernels.moments_to_mac_w_kernel, dim=(self.nx, self.ny, self.nz + 1), inputs=[self._uz, state_out.vel_w, self.nz], device=self.device)

    # Temporary compatibility path for moving-wall/momentum-exchange coupling.
    def _step_legacy(
        self,
        state_in: LbmState,
        state_out: LbmState,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Advance the LBM simulation by one lattice timestep.

        The physical *dt* is accepted for API compatibility but ignored
        internally – the LBM always uses ``dt = 1`` in lattice units.
        """
        del contacts, control, dt  # LBM dt = 1 lattice unit

        # ---- 0. Copy persistent fields (skipped when walls are static) ---
        if self.model.has_moving_walls:
            wp.copy(state_out.solid_phi, state_in.solid_phi)
            wp.copy(state_out.solid_body_id, state_in.solid_body_id)
            wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
            wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
            wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)

        # ---- 1. Compute macroscopic moments (rho, u) from f ----------------
        wp.launch(
            kernels.compute_moments_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_in.f,
                self._stride,
                self.nx,
                self.ny,
                self.nz,
                self._rho,
                self._ux,
                self._uy,
                self._uz,
            ],
        )

        # ---- 2. Body-force correction (gravity + optional Shan-Chen) ----
        gx: float = float(self.model.gravity_x)
        gy: float = float(self.model.gravity_y)
        gz: float = float(self.model.gravity_z)
        G_sc: float = float(self.model.G)
        px: int = int(self.model._periodic_ints[0])
        py: int = int(self.model._periodic_ints[1])
        pz: int = int(self.model._periodic_ints[2])

        if G_sc != 0.0:
            # --- 2a. Compute Shan-Chen interaction force (strided) ----------
            self._step_count += 1
            if self._step_count % self._sc_stride == 1:
                wp.launch(
                    kernels.compute_shan_chen_force_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[
                        self._rho,
                        state_out.solid_phi,
                        self._fx,
                        self._fy,
                        self._fz,
                        G_sc,
                        int(self.model.psi_type),
                        float(self.model.psi_ref),
                        float(self.model.sc_solid_psi_scale),
                        float(self.model.sc_boundary_psi),
                        float(self.model.cs_a),
                        float(self.model.cs_b),
                        float(self.model.cs_T),
                        int(self.model.sc_homogeneous_early_out),
                        float(self.model.sc_homogeneous_rel_tol),
                        px, py, pz,
                        self.nx,
                        self.ny,
                        self.nz,
                    ],
                )
            # --- 2b. Velocity shift (SC + gravity): u_eq = u + τ₋·(F/ρ + g) --
            # The velocity shift uses the shear relaxation time τ (not τ₊)
            # because the SC interaction force acts on momentum (odd modes),
            # whose relaxation is controlled by ω = 1/τ.  Using τ₊ would
            # amplify the shift when τ₊ > τ (i.e. whenever TRT is active),
            # pushing u_eq beyond the low-Mach limit at sharp interfaces.
            # TRT ghost-mode damping in collision is unaffected by this choice.
            wp.launch(
                kernels.apply_velocity_shift_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    self._ux,
                    self._uy,
                    self._uz,
                    self._rho,
                    self._fx,
                    self._fy,
                    self._fz,
                    gx,
                    gy,
                    gz,
                    self.model.tau,
                    self.nx,
                    self.ny,
                    self.nz,
                ],
            )
        # (gravity-only Guo force was here pre-collision — moved to step 5a below)

        # ---- 3. Regularization filter (pre-collision) ------------------------
        if self.model.use_regularization and self.model.omega_reg > 0.0:
            wp.launch(
                kernels.reg_trt_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state_in.f,
                    self._rho, self._ux, self._uy, self._uz,
                    float(self.model.omega_reg),
                    px, py, pz,
                    self.nx, self.ny, self.nz, self._stride,
                ],
            )
        # ---- 4. BGK/TRT collide + stream (applies relaxation ONCE) ----------
        wp.launch(
            kernels.collide_stream_bounceback_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_in.f, self._rho, self._ux, self._uy, self._uz,
                state_out.solid_phi,
                state_out.vel_solid_u, state_out.vel_solid_v, state_out.vel_solid_w,
                state_out.f,
                self.model.omega_plus, self.model.omega_minus,
                int(self.model.has_moving_walls),
                px, py, pz,
                self.nx, self.ny, self.nz, self._stride,
            ],
        )

        # ---- 5. Apply non-bounce-back boundary conditions -----------------
        # Skip entirely when all 6 faces are bounce-back (the common case).
        if any(t != 0 for t in self.model.bc_types):
            wp.launch(
                kernels.apply_boundary_conditions_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state_out.f,
                    self._rho,
                    self._ux,
                    self._uy,
                    self._uz,
                    self._bc_types,
                    self._bc_vel_x,
                    self._bc_vel_y,
                    self._bc_vel_z,
                    self.nx,
                    self.ny,
                    self.nz,
                    self._stride,
                ],
            )

        # ---- 5a. Guo body force (post-collision, gravity-only path) -------
        # Guo forcing with coefficient (1-ω/2) is designed for post-collision
        # application.  Applied here after collision-stream and boundary
        # conditions, only when no Shan-Chen interaction is active (G==0).
        if G_sc == 0.0 and (gx != 0.0 or gy != 0.0 or gz != 0.0):
            wp.launch(
                kernels.apply_guo_force_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state_out.f,
                    gx,
                    gy,
                    gz,
                    self.model.omega,
                    self.nx,
                    self.ny,
                    self.nz,
                    self._stride,
                ],
            )

        # ---- 5b. Restore physical velocity (reverse SC velocity shift) ----
        # apply_velocity_shift_kernel overwrote self._ux/_uy/_uz with the
        # equilibrium velocity u_eq = u + τ₋·(F/ρ + g).  The collision
        # and boundary-condition kernels have consumed u_eq; now reverse
        # the shift so that state_out.velocity_* and MAC-face velocities
        # report the true physical velocity.
        # Uses the same τ to exactly undo the forward shift.
        if G_sc != 0.0:
            wp.launch(
                kernels.restore_physical_velocity_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    self._ux,
                    self._uy,
                    self._uz,
                    self._rho,
                    self._fx,
                    self._fy,
                    self._fz,
                    gx,
                    gy,
                    gz,
                    self.model.tau,
                    self.nx,
                    self.ny,
                    self.nz,
                ],
            )

        # ---- 6. Copy macroscopic fields to state_out ----------------------
        wp.copy(state_out.density, self._rho)
        wp.copy(state_out.velocity_x, self._ux)
        wp.copy(state_out.velocity_y, self._uy)
        wp.copy(state_out.velocity_z, self._uz)
        if G_sc != 0.0:
            wp.copy(state_out.force_x, self._fx)
            wp.copy(state_out.force_y, self._fy)
            wp.copy(state_out.force_z, self._fz)

        # ---- 7. Populate MAC-face velocities (visualisation / coupling) ---
        wp.launch(
            kernels.moments_to_mac_u_kernel,
            dim=(self.nx + 1, self.ny, self.nz),
            inputs=[self._ux, state_out.vel_u, self.nx],
        )
        wp.launch(
            kernels.moments_to_mac_v_kernel,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[self._uy, state_out.vel_v, self.ny],
        )
        wp.launch(
            kernels.moments_to_mac_w_kernel,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[self._uz, state_out.vel_w, self.nz],
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_equilibrium(
        self,
        state: LbmStateBase,
        rho0: float = 1.0,
        u0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Initialise *state.f* to the equilibrium for uniform (ρ₀, u₀).

        Also sets the macroscopic fields on *state* so that
        ``state.density`` / ``state.velocity_*`` are consistent from the
        first step.

        Parameters
        ----------
        state:
            LBM state whose ``f`` array will be overwritten.
        rho0:
            Uniform initial density (default 1.0).
        u0:
            Uniform initial velocity ``(ux, uy, uz)`` in lattice units.
        """
        u0x, u0y, u0z = u0

        if isinstance(state, FullFLbmState):
            wp.launch(
                kernels.initialize_equilibrium_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state.f_post, rho0, u0x, u0y, u0z,
                    self.nx, self.ny, self.nz, self._stride,
                ],
                device=self.device,
            )
        elif isinstance(state, HomeLbmState):
            wp.launch(
                encoding.initialize_home_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[*state.kinetic_fields, rho0, u0x, u0y, u0z],
                device=self.device,
            )
        else:
            raise TypeError(f"Unsupported LBM state type: {type(state).__name__}")

        # Populate macroscopic fields for consistency
        state.density.fill_(rho0)
        state.velocity_x.fill_(u0x)
        state.velocity_y.fill_(u0y)
        state.velocity_z.fill_(u0z)
