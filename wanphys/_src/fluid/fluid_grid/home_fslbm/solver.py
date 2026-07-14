# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM solver pipeline — stream+collide + MomSwap.

Reference:
    [REF] Home-FSLBM inc/3D/gpu/mrLbmSolverGpu3D.cu — mrSolver3DGpu.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from ..base import FluidGridSolverBase
from .constants import (
    CS2,
    TYPE_F, TYPE_G, TYPE_S,
    MAX_VELOCITY,
    C_X, C_Y, C_Z, W, OPPOSITE,
    HERMITE_COEFFS,
)
from .kernels import (
    stream_collide_kernel,
    flag_domain_boundary_kernel,
    initialize_equilibrium_kernel,
)
from .model import HomeFSLbmModel
from .state import HomeFSLbmState


# ---------------------------------------------------------------------------
# Helper kernel: copy 1D flat flag → 3D flag  (avoids numpy scramble)
# ---------------------------------------------------------------------------
@wp.kernel
def _copy_flat_to_3d_kernel(
    dst: wp.array3d(dtype=wp.int32),
    src: wp.array(dtype=wp.int32),
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if i < nx and j < ny and k < nz:
        idx = k * ny * nx + j * nx + i
        dst[i, j, k] = src[idx]


@wp.kernel
def _copy_3d_to_flat_kernel(
    dst: wp.array(dtype=wp.int32),
    src: wp.array3d(dtype=wp.int32),
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if i < nx and j < ny and k < nz:
        idx = k * ny * nx + j * nx + i
        dst[idx] = src[i, j, k]


class HomeFSLbmSolver(FluidGridSolverBase):
    """HOME-FSLBM solver with fused stream+collide on GPU.

    Stage 2: full single-phase LBM pipeline:
        1. stream_collide_kernel (fused stream + central-moment MRT)
        2. MomSwap (double-buffer pointer exchange)

    Parameters
    ----------
    model:
        Static HOME-FSLBM configuration.
    """

    def __init__(self, model: HomeFSLbmModel) -> None:
        self.model: HomeFSLbmModel = model
        self.nx: int = int(model.nx)
        self.ny: int = int(model.ny)
        self.nz: int = int(model.nz)
        self.device: wp.Device = model._device
        self._total_num: int = self.nx * self.ny * self.nz

        # Pre-compute tau and forces
        self._tau: float = float(model.tau)
        self._gx: float = float(model.gravity_x)
        self._gy: float = float(model.gravity_y)
        self._gz: float = float(model.gravity_z)
        self._max_vel: float = float(model.max_velocity)
        self._flag_flat: wp.array | None = None

        # Convert bc_types to GPU int array (0=bounce_back, 1=periodic)
        self._bc_array: wp.array = self._make_bc_array(model.bc_types)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_bc_array(bc_types: tuple) -> wp.array:
        """Convert bc_types tuple to wp.array of ints.

        Order: (-x, +x, -y, +y, -z, +z).
        0 = bounce_back, 1 = periodic.
        """
        data = []
        for bc in bc_types:
            bc_str = str(bc).strip().lower()
            if bc_str in ("periodic",):
                data.append(1)
            else:
                data.append(0)  # default: bounce_back
        return wp.array(dtype=wp.int32, shape=(6,), data=data)

    # ------------------------------------------------------------------
    # DomainSolver protocol
    # ------------------------------------------------------------------

    def step(
        self,
        state_in: HomeFSLbmState,
        state_out: HomeFSLbmState,
        dt: float,
        **kwargs: Any,
    ) -> None:
        """Advance simulation by *dt*.

        Args:
            state_in: Current state (read from ``.f_mom``).
            state_out: Next state (write target ``.f_mom_post``).
            dt: Timestep in seconds (ignored, LBM uses lattice dt=1).
            **kwargs: Ignored.
        """
        total_num = self._total_num
        nx, ny, nz = self.nx, self.ny, self.nz

        # Lazy-init the 1D flat flag from state.flag (3D) if not already set.
        if self._flag_flat is None:
            self._flag_flat = wp.empty(total_num, dtype=wp.int32, device=self.device)
            wp.launch(
                _copy_3d_to_flat_kernel,
                dim=(nx, ny, nz),
                inputs=[self._flag_flat, state_in.flag, nx, ny, nz],
            )

        # ---- Step 1: Fused stream + collide ----
        # Uses self._flag_flat (built once in initialize_state, or lazily above).
        # Flag is static — no need to rebuild each step.
        wp.launch(
            stream_collide_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state_in.f_mom,
                state_out.f_mom_post,
                state_out.mass,
                state_out.phi,
                self._flag_flat,
                state_out.force_x,
                state_out.force_y,
                state_out.force_z,
                total_num, nx, ny, nz,
                self._tau,
                self._gx, self._gy, self._gz,
                self._max_vel,
                self._bc_array,
                HERMITE_COEFFS,
                W,
                C_X, C_Y, C_Z,
                OPPOSITE,
            ],
        )

        # Copy flag and scalar fields that are not written by kernel
        wp.copy(state_out.flag, state_in.flag)
        wp.copy(state_out.massex, state_in.massex)
        wp.copy(state_out.disjoin_force, state_in.disjoin_force)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_state(
        self,
        state: HomeFSLbmState,
        rho0: float = 1.0,
        u0_x: float = 0.0,
        u0_y: float = 0.0,
        u0_z: float = 0.0,
    ) -> None:
        """Initialise a state with equilibrium moments and flag domain faces.

        All interior cells are set to TYPE_F with uniform density and velocity.
        Domain faces with bounce_back are marked TYPE_S.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        total_num = self._total_num

        # Build a 1D flag array with all TYPE_F then mark boundary faces
        flag_flat = wp.full(
            total_num, TYPE_F, dtype=wp.int32, device=self.device
        )

        # Mark domain boundary faces as TYPE_S where bounce_back
        wp.launch(
            flag_domain_boundary_kernel,
            dim=(nx, ny, nz),
            inputs=[flag_flat, nx, ny, nz, self._bc_array],
        )

        # Store the 1D flag for kernel use (flag is static, never changes)
        self._flag_flat = flag_flat

        # Copy flat flag to state's 3D flag array via a proper Warp kernel
        # (avoids fragile numpy reshape / flatten which can scramble data)
        wp.launch(
            _copy_flat_to_3d_kernel,
            dim=(nx, ny, nz),
            inputs=[state.flag, flag_flat, nx, ny, nz],
        )

        # Initialise moments to equilibrium
        wp.launch(
            initialize_equilibrium_kernel,
            dim=(nx, ny, nz),
            inputs=[
                state.f_mom,
                state.mass,
                state.phi,
                flag_flat,
                total_num, nx, ny, nz,
                rho0, u0_x, u0_y, u0_z,
            ],
        )

        # Also initialise f_mom_post the same way
        wp.copy(state.f_mom_post, state.f_mom)

    # ------------------------------------------------------------------
    # Bound-check / mass-sum helpers
    # ------------------------------------------------------------------

    def total_mass(self, state: HomeFSLbmState) -> float:
        """Return sum of density across all non-solid cells (for diagnostics)."""
        total = 0.0
        f_mom_np = state.f_mom.numpy()
        flag_np = state.flag.numpy()
        total_num = self._total_num
        for idx in range(total_num):
            i = idx % self.nx
            j = (idx // self.nx) % self.ny
            k = idx // (self.nx * self.ny)
            if (flag_np[i, j, k] & TYPE_S) == 0:
                total += float(f_mom_np[idx])
        return total

    def total_momentum(self, state: HomeFSLbmState) -> tuple:
        """Return sum of momentum (rho*u) across all non-solid cells."""
        mx, my, mz = 0.0, 0.0, 0.0
        f_mom_np = state.f_mom.numpy()
        flag_np = state.flag.numpy()
        total_num = self._total_num
        for idx in range(total_num):
            i = idx % self.nx
            j = (idx // self.nx) % self.ny
            k = idx // (self.nx * self.ny)
            if (flag_np[i, j, k] & TYPE_S) == 0:
                mx += float(f_mom_np[total_num + idx])
                my += float(f_mom_np[2 * total_num + idx])
                mz += float(f_mom_np[3 * total_num + idx])
        return (mx, my, mz)

    def max_velocity(self, state: HomeFSLbmState) -> float:
        """Return maximum velocity magnitude across all non-solid cells."""
        max_u = 0.0
        f_mom_np = state.f_mom.numpy()
        flag_np = state.flag.numpy()
        total_num = self._total_num
        for idx in range(total_num):
            i = idx % self.nx
            j = (idx // self.nx) % self.ny
            k = idx // (self.nx * self.ny)
            if (flag_np[i, j, k] & TYPE_S) == 0:
                ux = float(f_mom_np[total_num + idx])
                uy = float(f_mom_np[2 * total_num + idx])
                uz = float(f_mom_np[3 * total_num + idx])
                mag_sq = ux * ux + uy * uy + uz * uz
                if mag_sq > max_u:
                    max_u = mag_sq
        return max_u**0.5
