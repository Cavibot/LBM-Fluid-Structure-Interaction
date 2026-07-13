# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Distribution Lattice Boltzmann method – BGK/TRT solver pipeline."""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import warp as wp

from ..base import FluidGridSolverBase
from . import kernels
from . import kernels_q
from .core.pipeline import LbmStepControl, StepStats
from .model import LbmModel
from .backends.moment.home_fp32_ref.bridge import HomeFp32VofBridge
from .phases.shan_chen import MacroscopicBuffers, ShanChenPhase
from .phases.vof_sharp import VofSharpPhase
from .state import LbmState


def _resolve_step_control(control: Any | None) -> LbmStepControl:
    if isinstance(control, LbmStepControl):
        return control
    return LbmStepControl(collect_stats=False)


class LbmSolver(FluidGridSolverBase):
    """Distribution LBM solver (D3Q19 / D3Q27) with Guo forcing and halfway BB.

    The solver owns temporary arrays for macroscopic moments that are
    reused across steps.  Distribution functions and visualisation fields
    live on :class:`LbmState`.

    Shan-Chen multiphase forcing is delegated to :class:`ShanChenPhase`.
    Sharp free-surface VOF is delegated to :class:`VofSharpPhase`.

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
        self.num_dirs: int = int(model.num_dirs)

        # Stride for flat-array indexing: nx * ny * nz
        self._stride: int = self.nx * self.ny * self.nz
        self._use_q_kernels: bool = self.num_dirs != 19

        spec = model.lattice_spec
        self._cx = wp.array(
            np.asarray(spec.cx, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._cy = wp.array(
            np.asarray(spec.cy, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._cz = wp.array(
            np.asarray(spec.cz, dtype=np.int32), dtype=wp.int32, device=self.device
        )
        self._w = wp.array(
            np.asarray(spec.weights, dtype=np.float32), dtype=float, device=self.device
        )
        self._opp = wp.array(
            np.asarray(spec.opposite, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )

        self._shan_chen: ShanChenPhase = ShanChenPhase(model)
        self._vof_sharp: VofSharpPhase = VofSharpPhase(model)
        self._home_fp32: HomeFp32VofBridge | None = (
            HomeFp32VofBridge(model) if model.lbm_backend == "home_fp32" else None
        )

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

    @property
    def macroscopic_buffers(self) -> MacroscopicBuffers:
        """Temporary rho/velocity/force fields used during :meth:`step`."""
        return MacroscopicBuffers(
            rho=self._rho,
            ux=self._ux,
            uy=self._uy,
            uz=self._uz,
            fx=self._fx,
            fy=self._fy,
            fz=self._fz,
        )

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
        state_in: LbmState,
        state_out: LbmState,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> StepStats | None:
        """Advance the LBM simulation by one lattice timestep.

        The physical *dt* is accepted for API compatibility but ignored
        internally – the LBM always uses ``dt = 1`` in lattice units.

        Returns
        -------
        StepStats or None
            Timing breakdown when *control* requests stats collection.
        """
        del contacts, dt  # LBM dt = 1 lattice unit

        step_control = _resolve_step_control(control)
        collect_stats = bool(step_control.collect_stats)
        stats = StepStats() if collect_stats else None
        t_step = time.perf_counter() if collect_stats else 0.0

        # ---- Optional HOME-FREE moment backend (H5) -----------------------
        if self._home_fp32 is not None:
            return self._step_home_fp32(
                state_in, state_out, stats, collect_stats, t_step,
            )

        gx: float = float(self.model.gravity_x)
        gy: float = float(self.model.gravity_y)
        gz: float = float(self.model.gravity_z)
        G_sc: float = float(self.model.G)
        px: int = int(self.model._periodic_ints[0])
        py: int = int(self.model._periodic_ints[1])
        pz: int = int(self.model._periodic_ints[2])
        buffers = self.macroscopic_buffers

        # ---- 0. Copy persistent fields --------------------------------
        wp.copy(state_out.solid_phi, state_in.solid_phi)
        wp.copy(state_out.solid_body_id, state_in.solid_body_id)
        wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
        wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
        wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)

        use_vof = self._vof_sharp.enabled
        if use_vof:
            # Keep phase fields coherent during the step (force / stream).
            wp.copy(state_out.phi, state_in.phi)
            wp.copy(state_out.cell_type, state_in.cell_type)

        # ---- 1. Compute macroscopic moments (rho, u) from f ----------------
        t0 = time.perf_counter() if collect_stats else 0.0
        if use_vof:
            self._vof_sharp.compute_moments(
                state_in, buffers, self.nx, self.ny, self.nz, self._stride
            )
        elif self._use_q_kernels:
            wp.launch(
                kernels_q.compute_moments_q_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state_in.f,
                    self._cx,
                    self._cy,
                    self._cz,
                    self.num_dirs,
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
        else:
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
        if stats is not None:
            stats.ms_moments = (time.perf_counter() - t0) * 1000.0

        # ---- 2. Phase pre-collision (Shan-Chen or VOF gravity shift) ------
        if self._shan_chen.enabled:
            t0 = time.perf_counter() if collect_stats else 0.0
            self._shan_chen.pre_collision(
                buffers,
                state_out.solid_phi,
                self.nx,
                self.ny,
                self.nz,
            )
            if stats is not None:
                stats.ms_phase = (time.perf_counter() - t0) * 1000.0
        elif use_vof and (gx != 0.0 or gy != 0.0 or gz != 0.0):
            t0 = time.perf_counter() if collect_stats else 0.0
            self._vof_sharp.apply_gravity_shift(
                buffers,
                state_in.cell_type,
                gx,
                gy,
                gz,
                self.nx,
                self.ny,
                self.nz,
            )
            if stats is not None:
                stats.ms_phase = (time.perf_counter() - t0) * 1000.0

        # ---- 3. Regularization filter (pre-collision) -------------------
        # Skip under VOF: gas cells have zero DFs and must not be filtered.
        # Skip under D3Q27: reg_trt_kernel is D3Q19-only (model already rejects).
        if (
            (not use_vof)
            and (not self._use_q_kernels)
            and self.model.use_regularization
            and self.model.omega_reg > 0.0
        ):
            t0 = time.perf_counter() if collect_stats else 0.0
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
            if stats is not None:
                stats.ms_regularization = (time.perf_counter() - t0) * 1000.0

        # ---- 4. BGK/TRT collide + stream --------------------------------
        t0 = time.perf_counter() if collect_stats else 0.0
        if use_vof:
            self._vof_sharp.collide_stream(
                state_in,
                state_out,
                buffers,
                self.nx,
                self.ny,
                self.nz,
                self._stride,
            )
        elif self._use_q_kernels:
            wp.launch(
                kernels_q.collide_stream_bounceback_q_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state_in.f,
                    self._rho,
                    self._ux,
                    self._uy,
                    self._uz,
                    state_out.solid_phi,
                    state_out.vel_solid_u,
                    state_out.vel_solid_v,
                    state_out.vel_solid_w,
                    state_out.f,
                    self._cx,
                    self._cy,
                    self._cz,
                    self._w,
                    self._opp,
                    self.model.omega_plus,
                    self.model.omega_minus,
                    self.num_dirs,
                    px,
                    py,
                    pz,
                    self.nx,
                    self.ny,
                    self.nz,
                    self._stride,
                ],
            )
        else:
            wp.launch(
                kernels.collide_stream_bounceback_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state_in.f, self._rho, self._ux, self._uy, self._uz,
                    state_out.solid_phi,
                    state_out.vel_solid_u, state_out.vel_solid_v, state_out.vel_solid_w,
                    state_out.f,
                    self.model.omega_plus, self.model.omega_minus,
                    px, py, pz,
                    self.nx, self.ny, self.nz, self._stride,
                ],
            )
        if stats is not None:
            stats.ms_collision = (time.perf_counter() - t0) * 1000.0

        # ---- 5. Apply non-bounce-back boundary conditions -----------------
        t0 = time.perf_counter() if collect_stats else 0.0
        # Q-generic Zou-He / outflow (D3Q19 + D3Q27, including body diagonals).
        wp.launch(
            kernels_q.apply_boundary_conditions_q_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_out.f,
                self._bc_types,
                self._bc_vel_x,
                self._bc_vel_y,
                self._bc_vel_z,
                self._cx,
                self._cy,
                self._cz,
                self._w,
                self._opp,
                self.num_dirs,
                self.nx,
                self.ny,
                self.nz,
                self._stride,
            ],
        )
        if stats is not None:
            stats.ms_bc = (time.perf_counter() - t0) * 1000.0

        # ---- 5a. Guo body force (single-phase path only) ------------------
        # VOF uses pre-collision velocity shift instead (avoids stale cell_type
        # and matches the SC gravity path that dam-break already tunes around).
        if (not use_vof) and G_sc == 0.0 and (gx != 0.0 or gy != 0.0 or gz != 0.0):
            t0 = time.perf_counter() if collect_stats else 0.0
            if self._use_q_kernels:
                wp.launch(
                    kernels_q.apply_guo_force_q_kernel,
                    dim=(self.nx, self.ny, self.nz),
                    inputs=[
                        state_out.f,
                        self._cx,
                        self._cy,
                        self._cz,
                        self._w,
                        gx,
                        gy,
                        gz,
                        self.model.omega,
                        self.num_dirs,
                        self.nx,
                        self.ny,
                        self.nz,
                        self._stride,
                    ],
                )
            else:
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
            if stats is not None:
                stats.ms_body_force = (time.perf_counter() - t0) * 1000.0

        # ---- 5b. Phase post-collision (SC restore / VOF φ update) ---------
        if self._shan_chen.enabled:
            t0 = time.perf_counter() if collect_stats else 0.0
            self._shan_chen.post_collision(
                buffers,
                state_out,
                self.nx,
                self.ny,
                self.nz,
            )
            if stats is not None:
                stats.ms_phase += (time.perf_counter() - t0) * 1000.0
        elif use_vof:
            t0 = time.perf_counter() if collect_stats else 0.0
            self._vof_sharp.update_interface(
                state_in,
                state_out,
                buffers,
                self.nx,
                self.ny,
                self.nz,
                self._stride,
            )
            # Refresh macros after fill/empty / new-interface init.
            self._vof_sharp.compute_moments(
                state_out, buffers, self.nx, self.ny, self.nz, self._stride
            )
            if stats is not None:
                stats.ms_phase += (time.perf_counter() - t0) * 1000.0

        # ---- 6–7. Export macroscopic + MAC fields -----------------------
        t0 = time.perf_counter() if collect_stats else 0.0
        wp.copy(state_out.density, self._rho)
        wp.copy(state_out.velocity_x, self._ux)
        wp.copy(state_out.velocity_y, self._uy)
        wp.copy(state_out.velocity_z, self._uz)

        wp.launch(
            kernels.compute_pressure_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[self._rho, state_out.pressure],
        )

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
        if stats is not None:
            stats.ms_export = (time.perf_counter() - t0) * 1000.0
            stats.ms_total = (time.perf_counter() - t_step) * 1000.0
            stats.with_num_cells(self._stride)
            if step_control.stats_out is not None:
                step_control.stats_out.ms_total = stats.ms_total
                step_control.stats_out.ms_moments = stats.ms_moments
                step_control.stats_out.ms_phase = stats.ms_phase
                step_control.stats_out.ms_regularization = stats.ms_regularization
                step_control.stats_out.ms_collision = stats.ms_collision
                step_control.stats_out.ms_bc = stats.ms_bc
                step_control.stats_out.ms_body_force = stats.ms_body_force
                step_control.stats_out.ms_export = stats.ms_export

        return stats

    def _step_home_fp32(
        self,
        state_in: LbmState,
        state_out: LbmState,
        stats: StepStats | None,
        collect_stats: bool,
        t_step: float,
    ) -> StepStats | None:
        """Moment-encoded HOME-FREE VOF step (``lbm_backend='home_fp32'``)."""
        assert self._home_fp32 is not None
        wp.copy(state_out.solid_phi, state_in.solid_phi)
        wp.copy(state_out.solid_body_id, state_in.solid_body_id)
        wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
        wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
        wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)

        t0 = time.perf_counter() if collect_stats else 0.0
        self._home_fp32.ensure_from_state(state_in)
        self._home_fp32.step(state_out)
        self._home_fp32.copy_kappa_to(self._vof_sharp._kappa)
        if stats is not None:
            stats.ms_collision = (time.perf_counter() - t0) * 1000.0

        # Keep solver macro buffers in sync for any consumers.
        wp.copy(self._rho, state_out.density)
        wp.copy(self._ux, state_out.velocity_x)
        wp.copy(self._uy, state_out.velocity_y)
        wp.copy(self._uz, state_out.velocity_z)

        t0 = time.perf_counter() if collect_stats else 0.0
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
        self._vof_sharp.update_visual_field(
            state_out, self.nx, self.ny, self.nz,
        )
        if stats is not None:
            stats.ms_export = (time.perf_counter() - t0) * 1000.0
            stats.ms_total = (time.perf_counter() - t_step) * 1000.0
            stats.with_num_cells(self._stride)
        return stats

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_equilibrium(
        self,
        state: LbmState,
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

        if self._use_q_kernels:
            wp.launch(
                kernels_q.initialize_equilibrium_q_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state.f,
                    self._cx,
                    self._cy,
                    self._cz,
                    self._w,
                    rho0,
                    u0x,
                    u0y,
                    u0z,
                    self.num_dirs,
                    self.nx,
                    self.ny,
                    self.nz,
                    self._stride,
                ],
            )
        else:
            wp.launch(
                kernels.initialize_equilibrium_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[
                    state.f,
                    rho0,
                    u0x,
                    u0y,
                    u0z,
                    self.nx,
                    self.ny,
                    self.nz,
                    self._stride,
                ],
            )

        # Populate macroscopic fields for consistency
        state.density.fill_(rho0)
        state.velocity_x.fill_(u0x)
        state.velocity_y.fill_(u0y)
        state.velocity_z.fill_(u0z)
        state.pressure.fill_(rho0 / 3.0)
        self._shan_chen.reset()
        self._vof_sharp.reset()
        if self._home_fp32 is not None:
            self._home_fp32.reset()
