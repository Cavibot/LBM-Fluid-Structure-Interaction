# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P3 fixed-atmosphere, zero-surface-tension boundary dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from .surface_kernels import (
    complete_gas_to_interface_fullf_kernel,
    restore_gas_fullf_state_kernel,
)

if TYPE_CHECKING:
    from ..state import FullFLbmState


class VofSurfaceBoundary:
    """Complete Eq. (11) links and preserve invalid GAS storage."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        periodic: tuple[int, int, int],
        atmosphere_pressure: float,
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.device = device
        self.periodic = tuple(int(value) for value in periodic)
        self.atmosphere_pressure = float(atmosphere_pressure)
        self.rho_g = 3.0 * self.atmosphere_pressure

    def complete_fullf(
        self,
        state_in: FullFLbmState,
        f_star: wp.array,
    ) -> None:
        """Overwrite only gas-to-interface links in streamed populations."""

        if state_in.vof is None:
            raise ValueError("P3 surface completion requires state.vof storage")
        nx, ny, nz = self.shape
        px, py, pz = self.periodic
        wp.launch(
            complete_gas_to_interface_fullf_kernel,
            dim=self.shape,
            inputs=[
                state_in.f_post,
                state_in.velocity_x,
                state_in.velocity_y,
                state_in.velocity_z,
                state_in.vof.cell_type,
                f_star,
                self.rho_g,
                px,
                py,
                pz,
                nx,
                ny,
                nz,
                nx * ny * nz,
            ],
            device=self.device,
        )

    def restore_gas(
        self,
        state_in: FullFLbmState,
        state_out: FullFLbmState,
    ) -> None:
        """Restore GAS storage after the generic all-cell LBM kernels."""

        if state_in.vof is None:
            raise ValueError("P3 GAS restoration requires state.vof storage")
        nx, ny, nz = self.shape
        wp.launch(
            restore_gas_fullf_state_kernel,
            dim=self.shape,
            inputs=[
                state_in.f_post,
                state_in.density,
                state_in.velocity_x,
                state_in.velocity_y,
                state_in.velocity_z,
                state_in.force_x,
                state_in.force_y,
                state_in.force_z,
                state_in.vof.cell_type,
                state_out.f_post,
                state_out.density,
                state_out.velocity_x,
                state_out.velocity_y,
                state_out.velocity_z,
                state_out.force_x,
                state_out.force_y,
                state_out.force_z,
                ny,
                nz,
                nx * ny * nz,
            ],
            device=self.device,
        )


__all__ = ["VofSurfaceBoundary"]
