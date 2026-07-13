# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shan-Chen pseudo-potential multiphase model (diffuse interface)."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .. import kernels
from ..model import LbmModel
from ..state import LbmState


@dataclass
class MacroscopicBuffers:
    """Solver-owned macroscopic fields passed into phase hooks."""

    rho: wp.array3d
    ux: wp.array3d
    uy: wp.array3d
    uz: wp.array3d
    fx: wp.array3d
    fy: wp.array3d
    fz: wp.array3d


class ShanChenPhase:
    """Shan-Chen interaction force and velocity-shift forcing.

    Encapsulates the pre-collision SC path (force compute + velocity shift)
    and the post-collision restore/copy path so the solver main chain stays
    collision-stream focused.
    """

    def __init__(self, model: LbmModel) -> None:
        self.model: LbmModel = model
        self._step_count: int = 0
        self._force_stride: int = int(model.sc_force_stride)

    @property
    def enabled(self) -> bool:
        """``True`` when Shan-Chen interaction is active (``G != 0``)."""
        return float(self.model.G) != 0.0

    def reset(self) -> None:
        """Reset internal stride counter (e.g. after domain re-init)."""
        self._step_count = 0

    def pre_collision(
        self,
        buffers: MacroscopicBuffers,
        solid_phi: wp.array3d,
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        """Compute SC force (strided) and apply velocity shift before collision."""
        if not self.enabled:
            return

        px, py, pz = self.model._periodic_ints
        gx = float(self.model.gravity_x)
        gy = float(self.model.gravity_y)
        gz = float(self.model.gravity_z)
        G_sc = float(self.model.G)

        self._step_count += 1
        if self._step_count % self._force_stride == 1:
            wp.launch(
                kernels.compute_shan_chen_force_kernel,
                dim=(nx, ny, nz),
                inputs=[
                    buffers.rho,
                    solid_phi,
                    buffers.fx,
                    buffers.fy,
                    buffers.fz,
                    G_sc,
                    int(self.model.psi_type),
                    float(self.model.psi_ref),
                    float(self.model.sc_solid_psi_scale),
                    float(self.model.sc_boundary_psi),
                    int(self.model.sc_homogeneous_early_out),
                    float(self.model.sc_homogeneous_rel_tol),
                    int(px),
                    int(py),
                    int(pz),
                    nx,
                    ny,
                    nz,
                ],
            )

        wp.launch(
            kernels.apply_velocity_shift_kernel,
            dim=(nx, ny, nz),
            inputs=[
                buffers.ux,
                buffers.uy,
                buffers.uz,
                buffers.rho,
                buffers.fx,
                buffers.fy,
                buffers.fz,
                gx,
                gy,
                gz,
                self.model.tau,
                nx,
                ny,
                nz,
            ],
        )

    def post_collision(
        self,
        buffers: MacroscopicBuffers,
        state_out: LbmState,
        nx: int,
        ny: int,
        nz: int,
    ) -> None:
        """Reverse velocity shift and copy SC forces to output state."""
        if not self.enabled:
            return

        gx = float(self.model.gravity_x)
        gy = float(self.model.gravity_y)
        gz = float(self.model.gravity_z)

        wp.launch(
            kernels.restore_physical_velocity_kernel,
            dim=(nx, ny, nz),
            inputs=[
                buffers.ux,
                buffers.uy,
                buffers.uz,
                buffers.rho,
                buffers.fx,
                buffers.fy,
                buffers.fz,
                gx,
                gy,
                gz,
                self.model.tau,
                nx,
                ny,
                nz,
            ],
        )

        wp.copy(state_out.force_x, buffers.fx)
        wp.copy(state_out.force_y, buffers.fy)
        wp.copy(state_out.force_z, buffers.fz)
