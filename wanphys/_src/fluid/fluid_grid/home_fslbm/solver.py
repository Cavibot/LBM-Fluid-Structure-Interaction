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
        """Advance the HOME-FSLBM simulation by one timestep.

        Follows the two-phase structure from ``mrSolver3D::mlIterateCouplingGpu``
        (``mrSolver3D.h:121-127``):

        1. Phase 1 — coupling(): bubble tagging + volume/rho + conditional CCL
        2. Phase 2 — mrSolver3DGpu(): fluid + free-surface subsystem
        """
        del contacts, control, dt  # LBM dt = 1 lattice unit

        self._step_count += 1

        # ------------------------------------------------------------------
        # Phase 1: coupling() — bubbles + g_handle
        # ------------------------------------------------------------------
        # Seed GPU flags from previous step (surface_3 may have set split_flag)
        self._merge_flag_gpu.fill_(int(state_in.merge_flag))
        self._split_flag_gpu.fill_(int(state_in.split_flag))
        self._bubble_count_gpu.fill_(int(state_in.bubble_count))
        self._label_num_gpu.fill_(int(max(state_in.label_num, 0)))

        # Copy bubble / gas persistent fields into state_out first so all
        # Phase-1 kernels mutate the outgoing buffer.
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

        # Also need flag/phi/delta_phi early for bubble volume update
        wp.copy(state_out.flag, state_in.flag)
        wp.copy(state_out.phi, state_in.phi)
        wp.copy(state_out.delta_phi, state_in.delta_phi)
        wp.copy(state_out.mass, state_in.mass)
        wp.copy(state_out.massex, state_in.massex)

        dim = (self.nx, self.ny, self.nz)

        # get_tag → assign_tag → recheck_merge
        wp.launch(
            kernels_bubble.get_tag_kernel,
            dim=dim,
            inputs=[
                state_out.tag_matrix,
                state_out.previous_merge_tag,
                state_out.merge_detector,
                self._cx, self._cy, self._cz,
                self.nx, self.ny, self.nz,
            ],
        )
        wp.launch(
            kernels_bubble.assign_tag_kernel,
            dim=dim,
            inputs=[
                state_out.tag_matrix,
                state_out.previous_merge_tag,
                state_out.merge_detector,
                self.nx, self.ny, self.nz,
            ],
        )
        wp.launch(
            kernels_bubble.recheck_merge_kernel,
            dim=dim,
            inputs=[
                state_out.tag_matrix,
                state_out.merge_detector,
                self._merge_flag_gpu,
                self._cx, self._cy, self._cz,
                self.nx, self.ny, self.nz,
            ],
        )

        # update_bubble: volume → rho → merge/split detect
        wp.launch(
            kernels_bubble.bubble_volume_update_kernel,
            dim=dim,
            inputs=[
                state_out.delta_phi,
                state_out.tag_matrix,
                state_out.previous_tag,
                state_out.bubble_volume,
                self.nx, self.ny, self.nz,
            ],
        )
        bc = int(state_out.bubble_count)
        if bc > 0:
            wp.launch(
                kernels_bubble.bubble_rho_update_kernel,
                dim=bc,
                inputs=[
                    state_out.bubble_volume,
                    state_out.bubble_init_volume,
                    state_out.bubble_rho,
                    bc,
                ],
            )

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
        merge_flag = int(self._det_merge.numpy()[0])
        split_flag = int(self._det_split.numpy()[0])
        state_out.merge_flag = merge_flag
        state_out.split_flag = split_flag

        if merge_flag > 0 or split_flag > 0:
            self._handle_merge_split(state_out)
            wp.launch(
                kernels_bubble.clear_detector_kernel,
                dim=dim,
                inputs=[
                    state_out.merge_detector,
                    self._merge_flag_gpu,
                    self._split_flag_gpu,
                    self.nx, self.ny, self.nz,
                ],
            )
            state_out.merge_flag = 0
            state_out.split_flag = 0

        # ---- g_handle: reconstruction -> CMR stream-collide -> volume_g ----
        # Fluid moments for gas advection come from state_in (pre-fluid step).
        if self.model.enable_gas:
            wp.launch(
                kernels_gas.g_reconstruction_kernel,
                dim=dim,
                inputs=[
                    state_out.g_mom,
                    state_in.f_mom,
                    state_out.flag,
                    state_out.tag_matrix,
                    state_out.bubble_rho,
                    state_out.delta_g,
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
                    state_out.g_mom,
                    state_out.g_mom_post,
                    state_in.f_mom,
                    state_out.flag,
                    state_out.src,
                    state_out.c_value,
                    state_out.islet,
                    self._cx, self._cy, self._cz,
                    self.nx, self.ny, self.nz,
                    self._stride,
                ],
            )
            wp.launch(
                kernels_gas.bubble_volume_g_update_kernel,
                dim=dim,
                inputs=[
                    state_out.delta_g,
                    state_out.phi,
                    state_out.flag,
                    state_out.tag_matrix,
                    state_out.bubble_init_volume,
                    self.nx, self.ny, self.nz,
                ],
            )
            # mrSolver3D_g_step2Kernel: gMom <- gMomPost
            wp.copy(state_out.g_mom, state_out.g_mom_post)
            bc = int(state_out.bubble_count)
            if bc > 0:
                wp.launch(
                    kernels_bubble.bubble_rho_update_kernel,
                    dim=bc,
                    inputs=[
                        state_out.bubble_volume,
                        state_out.bubble_init_volume,
                        state_out.bubble_rho,
                        bc,
                    ],
                )

        # ------------------------------------------------------------------
        # Phase 2: mrSolver3DGpu() — foam + fluid + free-surface
        # ------------------------------------------------------------------
        wp.copy(state_out.force_x, state_in.force_x)
        wp.copy(state_out.force_y, state_in.force_y)
        wp.copy(state_out.force_z, state_in.force_z)
        wp.copy(state_out.f_mom_post, state_in.f_mom_post)

        # ---- Copy solid coupling fields ----
        wp.copy(state_out.solid_phi, state_in.solid_phi)
        wp.copy(state_out.solid_body_id, state_in.solid_body_id)
        wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
        wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
        wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)

        # Disjoining pressure (before stream_collide)
        if self.model.enable_disjoin:
            wp.launch(
                kernels_foam.calculate_disjoint_kernel,
                dim=dim,
                inputs=[
                    state_out.flag,
                    state_out.phi,
                    state_out.mass,
                    state_out.massex,
                    state_in.f_mom,
                    state_out.tag_matrix,
                    state_out.disjoin_force,
                    self._cx, self._cy, self._cz, self._opposite,
                    self.nx, self.ny, self.nz,
                    self._stride,
                ],
            )
        else:
            state_out.disjoin_force.zero_()

        # Optional clear_inlet (disabled by default)
        if self.model.clear_inlet_enabled:
            wp.launch(
                kernels_bubble.clear_inlet_kernel,
                dim=dim,
                inputs=[
                    state_out.islet,
                    state_out.flag,
                    state_out.phi,
                    state_out.mass,
                    state_out.massex,
                    state_in.f_mom,  # stream_collide reads state_in.f_mom
                    self.nx, self.ny, self.nz,
                    self._stride,
                ],
            )

        # Atmosphere (always launched; no-op without large open bubbles)
        wp.launch(
            kernels_foam.atmosphere_rho_update_kernel,
            dim=dim,
            inputs=[
                state_out.tag_matrix,
                state_out.bubble_volume,
                state_out.bubble_rho,
                self.nx, self.ny, self.nz,
            ],
        )
        if bc > 0:
            wp.launch(
                kernels_foam.atmosphere_volme_update_kernel,
                dim=1,
                inputs=[
                    state_out.bubble_volume,
                    state_out.bubble_init_volume,
                    state_out.bubble_rho,
                    bc,
                ],
            )

        # ---- Assign gravity into body-force arrays (no accumulation) ----
        gx: float = float(self.model.gravity_x)
        gy: float = float(self.model.gravity_y)
        gz: float = float(self.model.gravity_z)
        wp.launch(
            kernels_fluid.add_gravity_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_out.force_x,
                state_out.force_y,
                state_out.force_z,
                gx, gy, gz,
            ],
        )

        # ---- Launch stream_collide_bvh (THE single kernel, Audit item B4) ----
        wp.launch(
            kernels_fluid.stream_collide_bvh_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_in.f_mom,
                state_out.flag,
                state_out.phi,
                state_out.tag_matrix,
                state_out.disjoin_force,
                state_out.islet,
                state_out.bubble_volume,
                state_out.bubble_init_volume,
                state_out.bubble_rho,
                state_out.solid_phi,
                state_out.solid_body_id,
                state_out.vel_solid_u,
                state_out.vel_solid_v,
                state_out.vel_solid_w,
                state_out.mass,
                state_out.massex,
                state_out.delta_g,
                state_out.delta_phi,
                state_out.force_x,
                state_out.force_y,
                state_out.force_z,
                state_out.c_value,
                state_out.src,
                state_out.f_mom_post,
                self.nx,
                self.ny,
                self.nz,
                self._stride,
                float(self.model.omega),
                float(self.model.surface_tension),
                float(self.model.henry_constant),
                float(self.model.disjoin_factor),
                float(self.model.turbulence_factor),
                int(self.model.turbulence_radius),
                int(self.model.atmosphere_open),
                int(self.model._periodic_ints[0]),
                int(self.model._periodic_ints[1]),
                int(self.model._periodic_ints[2]),
                self._cx,
                self._cy,
                self._cz,
                self._w3d,
                self._opposite,
            ],
        )

        # Clear disjoin force + massex after collide (before surface)
        wp.launch(
            kernels_foam.reset_disjoin_force_kernel,
            dim=dim,
            inputs=[
                state_out.disjoin_force,
                state_out.massex,
                self.nx, self.ny, self.nz,
            ],
        )

        # ---- Surface marker propagation (Phase 2) ----
        wp.launch(
            kernels_surface.surface_1_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_out.flag,
                self._cx, self._cy, self._cz,
                self.nx, self.ny, self.nz,
            ],
        )
        wp.launch(
            kernels_surface.surface_2_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_out.f_mom_post,
                state_out.flag,
                state_out.c_value,
                state_out.g_mom,
                state_out.islet,
                state_out.merge_detector,
                self._cx, self._cy, self._cz, self._w3d,
                self.nx, self.ny, self.nz,
                self._stride,
            ],
        )
        # Do NOT zero split_flag here — surface_3 accumulates split signals
        # for the *next* coupling step. Clear happens in clear_detector after
        # a merge/split CCL pass (and at InitBubble).
        # Preserve any unresolved merge_flag on GPU; surface_3 only touches split.
        self._split_flag_gpu.zero_()

        wp.launch(
            kernels_surface.surface_3_kernel,
            dim=(self.nx, self.ny, self.nz),
            inputs=[
                state_out.f_mom_post,
                state_out.flag,
                state_out.mass,
                state_out.massex,
                state_out.phi,
                state_out.tag_matrix,
                state_out.previous_tag,
                state_out.islet,
                state_out.delta_phi,
                state_out.delta_g,
                state_out.g_mom,
                self._split_flag_gpu,
                self._cx, self._cy, self._cz,
                self.nx, self.ny, self.nz,
                self._stride,
            ],
        )

        # Pull split_flag for next-step coupling (merge_flag already on host)
        state_out.split_flag = int(self._split_flag_gpu.numpy()[0])
        state_out.merge_flag = int(self._merge_flag_gpu.numpy()[0])

        # ---- Post-step swap: f_mom_post → f_mom (gMom already swapped in Phase 1) ----
        wp.launch(
            kernels_fluid.swap_moments_kernel,
            dim=self._stride,
            inputs=[
                state_out.f_mom,
                state_out.f_mom_post,
                self._stride,
            ],
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
        """Initialise moments to equilibrium (quiescent or uniform flow).

        Parameters
        ----------
        state:
            Target state to initialise.
        rho0:
            Uniform initial density.
        u0:
            Uniform initial velocity ``(ux, uy, uz)``.
        """
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
