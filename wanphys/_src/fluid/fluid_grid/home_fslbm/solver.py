# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM solver pipeline.

Implements :class:`HomeFslbmSolver`, the two-phase solver that mirrors the
reference implementation's ``coupling()`` → ``mrSolver3DGpu()`` call
structure (``mrSolver3D.h:121-127``).

Phase 1 (coupling) — bubble CCL + tagging + g_handle (CMR-MRT gas).
Phase 2 (mrSolver3DGpu) — disjoining / atmosphere + fluid + free-surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from ..base import FluidGridSolverBase
from . import constants as C
from . import kernels_bubble
from . import kernels_fluid
from . import kernels_foam
from . import kernels_gas
from . import kernels_surface
from .model import HomeFslbmModel
from .state import HomeFslbmState


class HomeFslbmSolver(FluidGridSolverBase):
    """HOME-FSLBM two-phase solver.

    Parameters
    ----------
    model:
        Static HOME-FSLBM configuration.
    """

    def __init__(self, model: HomeFslbmModel) -> None:
        self.model: HomeFslbmModel = model
        self.nx: int = int(model.nx)
        self.ny: int = int(model.ny)
        self.nz: int = int(model.nz)
        self.device: wp.Device = model._device

        # Stride for flat array indexing
        self._stride: int = self.nx * self.ny * self.nz

        # Step counter (for conditional kernels like clear_inlet)
        self._step_count: int = 0

        # Solver-owned temporary arrays
        self._div_array: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )

        # Turbulence pre-pass buffers
        self._small_bubble_mark: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=wp.uint8, device=self.device
        )
        self._near_small_bubble: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=wp.uint8, device=self.device
        )
        self._small_bubble_any = wp.zeros(1, dtype=wp.int32, device=self.device)

        # GPU-side merge/split flags (ref: mrFlow3D merge_flag / split_flag)
        self._merge_flag_gpu = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._split_flag_gpu = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._label_num_gpu = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._bubble_count_gpu = wp.zeros(1, dtype=wp.int32, device=self.device)
        # Scratch for MergeSplitDetector host readback
        self._det_merge = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._det_split = wp.zeros(1, dtype=wp.int32, device=self.device)

        # Direction arrays for kernel (warp-compatible, avoid Python list subscript)
        self._cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._w3d = wp.array(np.array(C.W, dtype=np.float32), dtype=float, device=self.device)
        self._opposite = wp.array(np.array(C.OPPOSITE, dtype=np.int32), dtype=wp.int32, device=self.device)

        self._max_bubbles: int = int(model.max_bubbles)

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _buffers_aliased(state_in: HomeFslbmState, state_out: HomeFslbmState) -> bool:
        return state_in.shares_buffers_with(state_out)

    @staticmethod
    def _mirror_bubble_list_handles(
        state_in: HomeFslbmState,
        state_out: HomeFslbmState,
    ) -> None:
        """Mirror ``bubble_list_swap`` refs onto ``state_in`` when buffers are aliased."""
        state_in.bubble_volume = state_out.bubble_volume
        state_in.bubble_label_volume = state_out.bubble_label_volume
        state_in.bubble_init_volume = state_out.bubble_init_volume
        state_in.bubble_label_init_volume = state_out.bubble_label_init_volume

    def _copy_persistent_fields(
        self,
        state_in: HomeFslbmState,
        state_out: HomeFslbmState,
    ) -> None:
        """Copy in→out for independent double-buffer unit tests."""
        wp.copy(state_out.g_mom, state_in.g_mom)
        wp.copy(state_out.g_mom_post, state_in.g_mom_post)
        wp.copy(state_out.c_value, state_in.c_value)
        wp.copy(state_out.src, state_in.src)
        wp.copy(state_out.delta_g, state_in.delta_g)
        wp.copy(state_out.tag_matrix, state_in.tag_matrix)
        wp.copy(state_out.previous_tag, state_in.previous_tag)
        wp.copy(state_out.previous_merge_tag, state_in.previous_merge_tag)
        wp.copy(state_out.label_matrix, state_in.label_matrix)
        wp.copy(state_out.input_matrix, state_in.input_matrix)
        wp.copy(state_out.merge_detector, state_in.merge_detector)
        wp.copy(state_out.islet, state_in.islet)
        wp.copy(state_out.disjoin_force, state_in.disjoin_force)
        wp.copy(state_out.bubble_volume, state_in.bubble_volume)
        wp.copy(state_out.bubble_init_volume, state_in.bubble_init_volume)
        wp.copy(state_out.bubble_rho, state_in.bubble_rho)
        wp.copy(state_out.bubble_label_init_volume, state_in.bubble_label_init_volume)
        wp.copy(state_out.bubble_label_volume, state_in.bubble_label_volume)
        state_out.merge_flag = state_in.merge_flag
        state_out.split_flag = state_in.split_flag
        state_out.label_num = state_in.label_num
        state_out.bubble_count = state_in.bubble_count
        wp.copy(state_out.flag, state_in.flag)
        wp.copy(state_out.phi, state_in.phi)
        wp.copy(state_out.delta_phi, state_in.delta_phi)
        wp.copy(state_out.mass, state_in.mass)
        wp.copy(state_out.massex, state_in.massex)
        wp.copy(state_out.solid_phi, state_in.solid_phi)
        wp.copy(state_out.solid_body_id, state_in.solid_body_id)
        wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
        wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
        wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)

    @staticmethod
    def _swap_moment_buffers(state: HomeFslbmState) -> None:
        """Ping-pong ``f_mom`` / ``f_mom_post`` via pointer exchange (no memcpy).

        After ``step()``, the active moments live in ``f_mom``; ``f_mom_post``
        is scratch for the next collide write.
        """
        state.f_mom, state.f_mom_post = state.f_mom_post, state.f_mom

    @staticmethod
    def _swap_gas_buffers(state: HomeFslbmState) -> None:
        state.g_mom, state.g_mom_post = state.g_mom_post, state.g_mom

    def _update_turbulence_mask(self, state: HomeFslbmState, dim: tuple[int, int, int]) -> None:
        """Build dilated small-bubble mask for eddy-viscosity (ref cu:1001-1027)."""
        tf = float(self.model.turbulence_factor)
        tr = int(self.model.turbulence_radius)
        if tf <= 0.0 or tr <= 0:
            self._near_small_bubble.zero_()
            return
        self._small_bubble_any.zero_()
        wp.launch(
            kernels_fluid.mark_small_bubble_kernel,
            dim=dim,
            inputs=[
                state.tag_matrix,
                state.bubble_volume,
                self._small_bubble_mark,
                self._small_bubble_any,
                self.nx,
                self.ny,
                self.nz,
            ],
            device=self.device,
        )
        # Skip O(N³ × radius³) dilate when no small bubbles were marked.
        if int(self._small_bubble_any.numpy()[0]) == 0:
            self._near_small_bubble.zero_()
            return
        wp.launch(
            kernels_fluid.dilate_near_small_bubble_kernel,
            dim=dim,
            inputs=[
                self._small_bubble_mark,
                self._near_small_bubble,
                tr,
                self.nx,
                self.ny,
                self.nz,
            ],
            device=self.device,
        )

    def _seed_gpu_bookkeeping_from_host(self, state_in: HomeFslbmState) -> None:
        """Seed solver GPU counters from host (independent double-buffer tests only)."""
        self._merge_flag_gpu.fill_(int(state_in.merge_flag))
        self._split_flag_gpu.fill_(int(state_in.split_flag))
        self._bubble_count_gpu.fill_(int(state_in.bubble_count))
        self._label_num_gpu.fill_(int(max(state_in.label_num, 0)))

    def _sync_host_bubble_bookkeeping(
        self,
        state_in: HomeFslbmState,
        state_out: HomeFslbmState,
        *,
        aliased: bool,
    ) -> None:
        count = int(self._bubble_count_gpu.numpy()[0])
        label_num = int(self._label_num_gpu.numpy()[0])
        if aliased:
            state_in.bubble_count = count
            state_in.label_num = label_num
        state_out.bubble_count = count
        state_out.label_num = label_num

    def _sync_host_merge_flags(
        self,
        state_in: HomeFslbmState,
        state_out: HomeFslbmState,
        *,
        aliased: bool,
        merge_flag: int,
        split_flag: int,
    ) -> None:
        if aliased:
            state_in.merge_flag = merge_flag
            state_in.split_flag = split_flag
        state_out.merge_flag = merge_flag
        state_out.split_flag = split_flag

    # ------------------------------------------------------------------
    # Bubble initialisation
    # ------------------------------------------------------------------

    def init_bubbles(self, state: HomeFslbmState) -> None:
        """Run reference ``InitBubble`` on ``state`` (CCL + create labels)."""
        kernels_bubble.init_bubbles(
            state,
            self._label_num_gpu,
            self._bubble_count_gpu,
            self._merge_flag_gpu,
            self._split_flag_gpu,
        )
        state.bubble_count = int(self._bubble_count_gpu.numpy()[0])
        state.label_num = int(self._label_num_gpu.numpy()[0])

    # ------------------------------------------------------------------
    # Main step — two-phase pipeline
    # ------------------------------------------------------------------

    def step(
        self,
        state_in: HomeFslbmState,
        state_out: HomeFslbmState,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Advance the HOME-FSLBM simulation by one timestep."""
        del contacts, control, dt  # LBM dt = 1 lattice unit

        self._step_count += 1
        aliased = self._buffers_aliased(state_in, state_out)
        state = state_out
        # When aliased, in/out share buffers — mutate through either handle.
        if not aliased:
            self._copy_persistent_fields(state_in, state_out)
            self._seed_gpu_bookkeeping_from_host(state_in)

        dim = (self.nx, self.ny, self.nz)

        # ------------------------------------------------------------------
        # Phase 1: coupling() — bubbles + g_handle
        # GPU merge/split flags persist across steps (surface_3 → split_flag_gpu).
        # ------------------------------------------------------------------

        wp.launch(
            kernels_bubble.get_tag_kernel,
            dim=dim,
            inputs=[
                state.tag_matrix,
                state.previous_merge_tag,
                state.merge_detector,
                self._cx, self._cy, self._cz,
                self.nx, self.ny, self.nz,
            ],
        )
        wp.launch(
            kernels_bubble.assign_tag_kernel,
            dim=dim,
            inputs=[
                state.tag_matrix,
                state.previous_merge_tag,
                state.merge_detector,
                self.nx, self.ny, self.nz,
            ],
        )
        wp.launch(
            kernels_bubble.recheck_merge_kernel,
            dim=dim,
            inputs=[
                state.tag_matrix,
                state.merge_detector,
                self._merge_flag_gpu,
                self._cx, self._cy, self._cz,
                self.nx, self.ny, self.nz,
            ],
        )

        wp.launch(
            kernels_bubble.bubble_volume_update_kernel,
            dim=dim,
            inputs=[
                state.delta_phi,
                state.tag_matrix,
                state.previous_tag,
                state.bubble_volume,
                self.nx, self.ny, self.nz,
            ],
        )
        wp.launch(
            kernels_bubble.bubble_rho_update_kernel,
            dim=self._max_bubbles,
            inputs=[
                state.bubble_volume,
                state.bubble_init_volume,
                state.bubble_rho,
                self._bubble_count_gpu,
            ],
        )

        self._sync_host_bubble_bookkeeping(state_in, state_out, aliased=aliased)

        wp.launch(
            kernels_bubble.merge_split_detector_kernel,
            dim=1,
            inputs=[
                self._merge_flag_gpu,
                self._split_flag_gpu,
                self._det_merge,
                self._det_split,
            ],
        )
        # Single host sync for merge/split branch (avoid mid-pipeline stalls).
        merge_flag = int(self._det_merge.numpy()[0])
        split_flag = int(self._det_split.numpy()[0])

        if merge_flag > 0 or split_flag > 0:
            self._handle_merge_split(state)
            if aliased:
                self._mirror_bubble_list_handles(state_in, state_out)
            wp.launch(
                kernels_bubble.clear_detector_kernel,
                dim=dim,
                inputs=[
                    state.merge_detector,
                    self._merge_flag_gpu,
                    self._split_flag_gpu,
                    self.nx, self.ny, self.nz,
                ],
            )
            self._sync_host_bubble_bookkeeping(state_in, state_out, aliased=aliased)
            merge_flag = 0
            split_flag = 0

        # ---- g_handle: reconstruction -> CMR stream-collide -> volume_g ----
        if self.model.enable_gas:
            wp.launch(
                kernels_gas.g_reconstruction_kernel,
                dim=dim,
                inputs=[
                    state.g_mom,
                    state_in.f_mom,
                    state.flag,
                    state.tag_matrix,
                    state.bubble_rho,
                    state.delta_g,
                    self._cx, self._cy, self._cz, self._opposite,
                    float(self.model.henry_constant),
                    self.nx, self.ny, self.nz,
                    self._stride,
                ],
            )
            wp.launch(
                kernels_gas.g_stream_collide_kernel,
                dim=dim,
                inputs=[
                    state.g_mom,
                    state.g_mom_post,
                    state_in.f_mom,
                    state.flag,
                    state.src,
                    state.c_value,
                    state.islet,
                    self._cx, self._cy, self._cz,
                    self.nx, self.ny, self.nz,
                    self._stride,
                ],
            )
            wp.launch(
                kernels_gas.bubble_volume_g_update_kernel,
                dim=dim,
                inputs=[
                    state.delta_g,
                    state.phi,
                    state.flag,
                    state.tag_matrix,
                    state.bubble_init_volume,
                    self.nx, self.ny, self.nz,
                ],
            )
            self._swap_gas_buffers(state)
            if aliased:
                state_in.g_mom = state.g_mom
                state_in.g_mom_post = state.g_mom_post
            wp.launch(
                kernels_bubble.bubble_rho_update_kernel,
                dim=self._max_bubbles,
                inputs=[
                    state.bubble_volume,
                    state.bubble_init_volume,
                    state.bubble_rho,
                    self._bubble_count_gpu,
                ],
            )
            self._sync_host_bubble_bookkeeping(state_in, state_out, aliased=aliased)

        # ------------------------------------------------------------------
        # Phase 2: mrSolver3DGpu() — foam + fluid + free-surface
        # ------------------------------------------------------------------

        if self.model.enable_disjoin:
            wp.launch(
                kernels_foam.calculate_disjoint_kernel,
                dim=dim,
                inputs=[
                    state.flag,
                    state.phi,
                    state.mass,
                    state.massex,
                    state_in.f_mom,
                    state.tag_matrix,
                    state.disjoin_force,
                    self._cx, self._cy, self._cz, self._opposite,
                    self.nx, self.ny, self.nz,
                    self._stride,
                ],
            )
        else:
            state.disjoin_force.zero_()

        if self.model.clear_inlet_enabled:
            wp.launch(
                kernels_bubble.clear_inlet_kernel,
                dim=dim,
                inputs=[
                    state.islet,
                    state.flag,
                    state.phi,
                    state.mass,
                    state.massex,
                    state_in.f_mom,
                    self.nx, self.ny, self.nz,
                    self._stride,
                ],
            )

        wp.launch(
            kernels_foam.atmosphere_rho_update_kernel,
            dim=dim,
            inputs=[
                state.tag_matrix,
                state.bubble_volume,
                state.bubble_rho,
                self.nx, self.ny, self.nz,
            ],
        )
        wp.launch(
            kernels_foam.atmosphere_volme_update_kernel,
            dim=1,
            inputs=[
                state.bubble_volume,
                state.bubble_init_volume,
                state.bubble_rho,
                self._bubble_count_gpu,
            ],
        )

        wp.launch(
            kernels_fluid.add_gravity_kernel,
            dim=dim,
            inputs=[
                state.force_x,
                state.force_y,
                state.force_z,
                float(self.model.gravity_x),
                float(self.model.gravity_y),
                float(self.model.gravity_z),
            ],
        )

        self._update_turbulence_mask(state, dim)

        kernels_fluid.launch_stream_collide_bvh(
            dim=dim,
            device=self.device,
            f_mom=state_in.f_mom,
            flag=state.flag,
            phi=state.phi,
            tag_matrix=state.tag_matrix,
            disjoin_force=state.disjoin_force,
            islet=state.islet,
            bubble_volume=state.bubble_volume,
            bubble_init_volume=state.bubble_init_volume,
            bubble_rho=state.bubble_rho,
            solid_phi=state.solid_phi,
            solid_body_id=state.solid_body_id,
            vel_solid_u=state.vel_solid_u,
            vel_solid_v=state.vel_solid_v,
            vel_solid_w=state.vel_solid_w,
            mass=state.mass,
            massex=state.massex,
            delta_g=state.delta_g,
            delta_phi=state.delta_phi,
            force_x=state.force_x,
            force_y=state.force_y,
            force_z=state.force_z,
            c_value=state.c_value,
            src=state.src,
            f_mom_post=state.f_mom_post,
            near_small_bubble=self._near_small_bubble,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            stride=self._stride,
            omega=float(self.model.omega),
            surface_tension=float(self.model.surface_tension),
            henry_constant=float(self.model.henry_constant),
            disjoin_factor=float(self.model.disjoin_factor),
            turbulence_factor=float(self.model.turbulence_factor),
            turbulence_radius=int(self.model.turbulence_radius),
            atmosphere_open=int(self.model.atmosphere_open),
            px=int(self.model._periodic_ints[0]),
            py=int(self.model._periodic_ints[1]),
            pz=int(self.model._periodic_ints[2]),
            cx=self._cx,
            cy=self._cy,
            cz=self._cz,
            w3d=self._w3d,
            opposite=self._opposite,
        )

        wp.launch(
            kernels_foam.reset_disjoin_force_kernel,
            dim=dim,
            inputs=[
                state.disjoin_force,
                state.massex,
                self.nx, self.ny, self.nz,
            ],
        )

        wp.launch(
            kernels_surface.surface_1_kernel,
            dim=dim,
            inputs=[
                state.flag,
                self._cx, self._cy, self._cz,
                self.nx, self.ny, self.nz,
            ],
        )
        wp.launch(
            kernels_surface.surface_2_kernel,
            dim=dim,
            inputs=[
                state.f_mom_post,
                state.flag,
                state.c_value,
                state.g_mom,
                state.islet,
                state.merge_detector,
                self._cx, self._cy, self._cz, self._w3d,
                self.nx, self.ny, self.nz,
                self._stride,
            ],
        )
        self._split_flag_gpu.zero_()

        wp.launch(
            kernels_surface.surface_3_kernel,
            dim=dim,
            inputs=[
                state.f_mom_post,
                state.flag,
                state.mass,
                state.massex,
                state.phi,
                state.tag_matrix,
                state.previous_tag,
                state.islet,
                state.delta_phi,
                state.delta_g,
                state.g_mom,
                self._split_flag_gpu,
                self._cx, self._cy, self._cz,
                self.nx, self.ny, self.nz,
                self._stride,
            ],
        )

        # Post-step: f_mom_post → f_mom via pointer ping-pong (ref step2Kernel
        # memcpy). Active moments are in ``f_mom`` after this swap.
        state.f_mom, state.f_mom_post = state.f_mom_post, state.f_mom
        if aliased:
            state_in.f_mom = state.f_mom
            state_in.f_mom_post = state.f_mom_post

        self._sync_host_bubble_bookkeeping(state_in, state_out, aliased=aliased)
        self._sync_host_merge_flags(
            state_in,
            state_out,
            aliased=aliased,
            merge_flag=merge_flag,
            split_flag=split_flag,
        )

    def _handle_merge_split(self, state: HomeFslbmState) -> None:
        """Conditional CCL rebuild after merge/split detection."""
        kernels_bubble.handle_merge_split(
            state,
            self._label_num_gpu,
            self._bubble_count_gpu,
        )

    # ------------------------------------------------------------------
    # Initialisation helper
    # ------------------------------------------------------------------

    def initialize_equilibrium(
        self,
        state: HomeFslbmState,
        rho0: float = 1.0,
        u0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Initialise moments to equilibrium (quiescent or uniform flow)."""
        ux, uy, uz = u0

        N = self._stride

        f_mom_data = np.zeros((C.NUM_MOMENTS, N), dtype=np.float32)
        f_mom_data[C.M_RHO, :] = rho0
        f_mom_data[C.M_UX, :] = ux
        f_mom_data[C.M_UY, :] = uy
        f_mom_data[C.M_UZ, :] = uz
        f_mom_data[C.M_SXX, :] = ux * ux
        f_mom_data[C.M_SXY, :] = ux * uy
        f_mom_data[C.M_SXZ, :] = ux * uz
        f_mom_data[C.M_SYY, :] = uy * uy
        f_mom_data[C.M_SYZ, :] = uy * uz
        f_mom_data[C.M_SZZ, :] = uz * uz

        wp.copy(state.f_mom, wp.array(f_mom_data.flatten(), dtype=float, device=self.device))
        wp.copy(state.f_mom_post, state.f_mom)
