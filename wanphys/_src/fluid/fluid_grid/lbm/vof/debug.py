# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Observation-only adapter from Shan-Chen density to reusable VOF state."""

from __future__ import annotations

import warp as wp

from .geometry import InterfaceGeometry
from .kernels import classify_bounded_fill_fraction_kernel
from .state import VofGridState


@wp.kernel
def density_to_debug_fill_fraction_kernel(
    density: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    rho_gas: float,
    inverse_density_span: float,
) -> None:
    i, j, k = wp.tid()
    value = (density[i, j, k] - rho_gas) * inverse_density_span
    phi[i, j, k] = wp.clamp(value, 0.0, 1.0)


class DebugVofObserver:
    """Populate diagnostic VOF fields without feeding back into LBM physics."""

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

    def update(
        self,
        density: wp.array3d,
        solid_phi: wp.array3d,
        vof: VofGridState,
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
                vof.phi,
                self.rho_gas,
                self.inverse_density_span,
            ],
            device=self.device,
        )
        wp.launch(
            classify_bounded_fill_fraction_kernel,
            dim=self.shape,
            inputs=[
                vof.phi,
                solid_phi,
                vof.cell_type,
                self.epsilon,
                1.0 - self.epsilon,
            ],
            device=self.device,
        )
        vof.epoch = epoch
        self.geometry.compute_normal(vof, solid_phi)
        return epoch


__all__ = ["DebugVofObserver", "density_to_debug_fill_fraction_kernel"]
