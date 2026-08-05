# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Observation-only adapter from Shan-Chen density to a VOF-shaped debug mock."""

from __future__ import annotations

import warp as wp

from .kernels.debug import density_to_debug_fill_fraction_kernel
from ..solver.geometry import InterfaceGeometry
from .kernels.classification import classify_bounded_fill_fraction_kernel
from ..state import DebugMockScToVofState


class DebugMockScToVofObserver:
    """Populate an isolated debug mock without feeding back into LBM physics."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        rho_gas: float,
        rho_liquid: float,
        epsilon: float,
        periodic: tuple[int, int, int],
    ) -> None:
        self.shape = shape
        self.device = device
        self.rho_gas = float(rho_gas)
        self.inverse_density_span = 1.0 / (float(rho_liquid) - float(rho_gas))
        self.epsilon = float(epsilon)
        self.geometry = InterfaceGeometry(shape, device, periodic)
        self._epoch = 0

    def update_from_density(
        self,
        density: wp.array3d,
        solid_phi: wp.array3d,
        debug_mock: DebugMockScToVofState,
        epoch: int | None = None,
    ) -> int:
        """Refresh one state and return the epoch assigned to it.

        Passing an epoch lets callers refresh both sides of a freshly mirrored
        double buffer as one logical observation.
        """

        if epoch is None:
            epoch = self._epoch
            self._epoch += 1
        wp.launch(
            density_to_debug_fill_fraction_kernel,
            dim=self.shape,
            inputs=[
                density,
                debug_mock.phi,
                self.rho_gas,
                self.inverse_density_span,
            ],
            device=self.device,
        )
        wp.launch(
            classify_bounded_fill_fraction_kernel,
            dim=self.shape,
            inputs=[
                debug_mock.phi,
                solid_phi,
                debug_mock.cell_type,
                self.epsilon,
                1.0 - self.epsilon,
            ],
            device=self.device,
        )
        debug_mock.epoch = epoch
        self.geometry.compute_normal(debug_mock, solid_phi)
        return epoch


__all__ = [
    "DebugMockScToVofObserver",
    "density_to_debug_fill_fraction_kernel",
]
