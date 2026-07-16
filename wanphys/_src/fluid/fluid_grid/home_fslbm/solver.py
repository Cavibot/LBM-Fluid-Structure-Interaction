# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM solver pipeline.

Implements :class:`HomeFslbmSolver`, the two-phase solver that mirrors the
reference implementation's ``coupling()`` → ``mrSolver3DGpu()`` call
structure (``mrSolver3D.h:121-127``).

Phase 1 (coupling) — bubble CCL + gas subsystem (deferred to stages 3-4).
Phase 2 (mrSolver3DGpu) — fluid + free-surface subsystem (this stage).
"""

from __future__ import annotations

from typing import Any, Optional

import warp as wp

from ..base import FluidGridSolverBase
from . import constants as C
from . import kernels_fluid
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
        # (moment swap and other merge-requiring ops benefit from
        # solver-owned scratch space rather than per-kernel allocations)
        self._div_array: wp.array3d = wp.zeros(
            (self.nx, self.ny, self.nz), dtype=float, device=self.device
        )

        # Direction arrays for kernel (warp-compatible, avoid Python list subscript)
        import numpy as np
        self._cx = wp.array(np.array(C.CX, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._cy = wp.array(np.array(C.CY, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._cz = wp.array(np.array(C.CZ, dtype=np.int32), dtype=wp.int32, device=self.device)
        self._w3d = wp.array(np.array(C.W, dtype=np.float32), dtype=float, device=self.device)
        self._opposite = wp.array(np.array(C.OPPOSITE, dtype=np.int32), dtype=wp.int32, device=self.device)

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

        1. Phase 1 — coupling(): bubble CCL + gas subsystem
        2. Phase 2 — mrSolver3DGpu(): fluid + free-surface subsystem
        """
        del contacts, control, dt  # LBM dt = 1 lattice unit

        self._step_count += 1

        # ------------------------------------------------------------------
        # Phase 1: coupling() — bubbles + gas (deferred to stages 3-4)
        # ------------------------------------------------------------------
        # In the reference code, coupling() handles:
        #   get_tag_kernel → assign_tag_kernel → recheck_merge_kernel
        #   → bubble_volume_update → bubble_rho_update
        #   → g_reconstruction → g_stream_collide → bubble_volume_g_update
        #   → mrSolver3D_g_step2Kernel → bubble_rho_update
        #
        # Phase 1 NO-OP: copy gas moments from state_in to state_out.
        wp.copy(state_out.g_mom, state_in.g_mom)
        wp.copy(state_out.g_mom_post, state_in.g_mom_post)
        wp.copy(state_out.c_value, state_in.c_value)
        wp.copy(state_out.src, state_in.src)
        wp.copy(state_out.delta_g, state_in.delta_g)

        # ---- Bubble fields: copy persistent data ----
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

        # ------------------------------------------------------------------
        # Phase 2: mrSolver3DGpu() — fluid + free-surface
        # ------------------------------------------------------------------
        # In the reference code, mrSolver3DGpu() handles:
        #   calculate_disjoint
        #   [conditional] clear_inlet
        #   atmosphere_rho_update_kernel
        #   atmosphere_volme_update_kernel
        #   stream_collide_bvh ★ (single kernel, inline turbulence + free-surface)
        #   ResetDisjoinForce
        #   surface_1 → surface_2 → surface_3 (marker propagation + mass redist)
        #   mrSolver3D_step2Kernel (fMom swap)
        #
        # For Phase 1 (this stage), we implement:
        #   - Copy force / solid / mass arrays from state_in
        #   - stream_collide_bvh_kernel (single monolithic kernel)
        #   - Swap f_mom_post → f_mom (Phase 2 post-step)
        #   - surface kernels and other post-processing deferred to stage 2

        # ---- Copy persistent fields from state_in ----
        wp.copy(state_out.flag, state_in.flag)
        wp.copy(state_out.mass, state_in.mass)
        wp.copy(state_out.massex, state_in.massex)
        wp.copy(state_out.phi, state_in.phi)
        wp.copy(state_out.force_x, state_in.force_x)
        wp.copy(state_out.force_y, state_in.force_y)
        wp.copy(state_out.force_z, state_in.force_z)
        wp.copy(state_out.delta_phi, state_in.delta_phi)

        # ---- Copy solid coupling fields ----
        wp.copy(state_out.solid_phi, state_in.solid_phi)
        wp.copy(state_out.solid_body_id, state_in.solid_body_id)
        wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
        wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
        wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)

        # ---- Inject gravity into body-force arrays ----
        gx: float = float(self.model.gravity_x)
        gy: float = float(self.model.gravity_y)
        gz: float = float(self.model.gravity_z)
        if gx != 0.0 or gy != 0.0 or gz != 0.0:
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
                # Read: current moments
                state_in.f_mom,
                state_out.flag,          # read/write (islet→TYPE_F)
                state_out.phi,           # read/write (interface placeholder)
                state_out.tag_matrix,    # read (bubble tag for gas pressure)
                state_out.disjoin_force, # read
                state_out.islet,         # read (may be set by Phase 1)

                # Read: bubble properties
                state_out.bubble_volume,
                state_out.bubble_init_volume,
                state_out.bubble_rho,

                # Read: solid coupling
                state_out.solid_phi,
                state_out.solid_body_id,
                state_out.vel_solid_u,
                state_out.vel_solid_v,
                state_out.vel_solid_w,

                # Read+write: accumulators
                state_out.mass,
                state_out.massex,
                state_out.delta_g,
                state_out.delta_phi,
                state_out.force_x,
                state_out.force_y,
                state_out.force_z,
                state_out.c_value,
                state_out.src,

                # Write: post-collision moments
                state_out.f_mom_post,

                # Parameters
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

        # ---- Post-step swap: f_mom_post → f_mom ----
        # Reference: ``mrSolver3D_step2Kernel`` — copy post-collision moments
        # into the current-moment arrays for the next step.
        wp.launch(
            kernels_fluid.swap_moments_kernel,
            dim=self._stride,
            inputs=[
                state_out.f_mom,
                state_out.f_mom_post,
                state_out.g_mom,
                state_out.g_mom_post,
                self._stride,
                self._stride,  # gas_stride (same for 3D grid)
            ],
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
        import numpy as np

        ux, uy, uz = u0

        # Build equilibrium stress tensor for uniform density/velocity
        # Π_xx^eq = c_s² + u_x², Π_xy^eq = u_x·u_y, …
        pi_xx_eq = C.CS2 + ux * ux
        pi_yy_eq = C.CS2 + uy * uy
        pi_zz_eq = C.CS2 + uz * uz
        pi_xy_eq = ux * uy
        pi_xz_eq = ux * uz
        pi_yz_eq = uy * uz

        N = self._stride

        # Host-side arrays for init.
        # Reference fMom storage convention: velocity + half-force correction
        # for [1..3]; traceless stress S_ab = Pi_ab/rho - cs²*delta_ab for [4..9].
        # Equilibrium: S_xx_eq = ux², S_xy_eq = ux*uy, etc.
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