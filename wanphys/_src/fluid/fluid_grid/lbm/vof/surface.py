# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P3/P6 fixed-atmosphere free-surface boundary dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .geometry import validate_authoritative_geometry
from .surface_kernels import (
    complete_gas_to_interface_fullf_kernel,
    restore_gas_fullf_state_kernel,
    restore_gas_home_kinetic_kernel,
    restore_gas_macroscopic_state_kernel,
)

if TYPE_CHECKING:
    from ..state import FullFLbmState, LbmStateBase


class VofSurfaceBoundary:
    """Complete Eq. (11) links and preserve invalid GAS storage."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        periodic: tuple[int, int, int],
        atmosphere_pressure: float,
        surface_tension: float,
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.device = device
        self.periodic = tuple(int(value) for value in periodic)
        self.atmosphere_pressure = float(atmosphere_pressure)
        self.surface_tension = float(surface_tension)
        self.rho_g = 3.0 * self.atmosphere_pressure

    def validate_pressure_state(self, state_in: LbmStateBase) -> None:
        """Fail before population writes when P6 geometry/pressure is invalid."""

        if state_in.vof is None:
            raise ValueError("P6 surface completion requires state.vof storage")
        validate_authoritative_geometry(state_in.vof)
        if self.surface_tension == 0.0:
            return
        curvature = np.asarray(state_in.vof.curvature.numpy())
        cell_type = np.asarray(state_in.vof.cell_type.numpy())
        interface = cell_type == 1
        rho_g = 3.0 * (
            self.atmosphere_pressure
            - 2.0 * self.surface_tension * curvature[interface]
        )
        if not np.all(np.isfinite(rho_g)) or np.any(rho_g <= 0.0):
            minimum = float(np.min(rho_g)) if rho_g.size else float("inf")
            raise ValueError(
                "P6 Eq. (12) requires finite positive rho_g on every "
                f"INTERFACE cell; minimum={minimum}"
            )

    def complete_fullf(
        self,
        state_in: FullFLbmState,
        f_star: wp.array,
    ) -> None:
        """Overwrite only gas-to-interface links in streamed populations."""

        if state_in.vof is None:
            raise ValueError("P3 surface completion requires state.vof storage")
        return self.complete_populations(state_in, state_in.f_post, f_star)

    def complete_populations(
        self,
        state_in: LbmStateBase,
        logical_f_post_n: wp.array,
        f_star: wp.array,
    ) -> None:
        """Complete Eq. (11) from an encoding-independent logical population."""

        if state_in.vof is None:
            raise ValueError("P7 surface completion requires state.vof storage")
        self.validate_pressure_state(state_in)
        nx, ny, nz = self.shape
        px, py, pz = self.periodic
        wp.launch(
            complete_gas_to_interface_fullf_kernel,
            dim=self.shape,
            inputs=[
                logical_f_post_n,
                state_in.velocity_x,
                state_in.velocity_y,
                state_in.velocity_z,
                state_in.vof.cell_type,
                state_in.vof.curvature,
                f_star,
                self.rho_g,
                self.atmosphere_pressure,
                self.surface_tension,
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
        state_in: LbmStateBase,
        state_out: LbmStateBase,
    ) -> None:
        """Restore GAS storage after the generic all-cell LBM kernels."""

        if state_in.vof is None:
            raise ValueError("P3 GAS restoration requires state.vof storage")
        from ..state import FullFLbmState, HomeLbmState

        if isinstance(state_in, FullFLbmState) and isinstance(
            state_out, FullFLbmState
        ):
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
            return
        if not isinstance(state_in, HomeLbmState) or not isinstance(
            state_out, HomeLbmState
        ):
            raise TypeError("P7 GAS restore requires matching FullF or HOME states")
        wp.launch(
            restore_gas_home_kinetic_kernel,
            dim=self.shape,
            inputs=[
                *state_in.kinetic_fields,
                state_in.vof.cell_type,
                *state_out.kinetic_fields,
            ],
            device=self.device,
        )
        wp.launch(
            restore_gas_macroscopic_state_kernel,
            dim=self.shape,
            inputs=[
                state_in.density,
                state_in.velocity_x,
                state_in.velocity_y,
                state_in.velocity_z,
                state_in.force_x,
                state_in.force_y,
                state_in.force_z,
                state_in.vof.cell_type,
                state_out.density,
                state_out.velocity_x,
                state_out.velocity_y,
                state_out.velocity_z,
                state_out.force_x,
                state_out.force_y,
                state_out.force_z,
            ],
            device=self.device,
        )


__all__ = ["VofSurfaceBoundary"]
