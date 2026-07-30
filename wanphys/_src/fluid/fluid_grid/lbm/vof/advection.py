# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Host dispatch and scratch ownership for P2 VOF mass transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .advection_kernels import advect_vof_mass_fullf_fixed_topology_kernel
from .initialization import (
    validate_initial_topology,
    validate_initialized_density,
    validate_no_solid_cells,
)

if TYPE_CHECKING:
    from ..state import FullFLbmState, LbmStateBase


@dataclass(frozen=True)
class VofMassTransportResult:
    """Solver-owned provisional P2 arrays, valid until the next compute call."""

    mass_tmp: wp.array
    phi_tmp: wp.array
    mass_delta: wp.array


def validate_p2_transport_input(
    state: LbmStateBase,
    *,
    periodic: tuple[bool, bool, bool],
) -> None:
    """Validate the initialized fixed-topology state consumed by P2."""

    from ..state import FullFLbmState

    if not isinstance(state, FullFLbmState):
        raise NotImplementedError(
            "P2 authoritative VOF mass transport supports FullF only; "
            "HOME transport is deferred to P7"
        )
    if state.vof is None:
        raise ValueError("P2 VOF mass transport requires state.vof storage")

    validate_no_solid_cells(state.solid_phi)
    validate_initialized_density(state.density)

    populations = np.asarray(state.f_post.numpy())
    if not np.all(np.isfinite(populations)):
        raise ValueError("P2 FullF populations must be finite")

    mass = np.asarray(state.vof.mass.numpy())
    phi = np.asarray(state.vof.phi.numpy())
    cell_type = np.asarray(state.vof.cell_type.numpy())
    if not np.all(np.isfinite(mass)):
        raise ValueError("P2 input mass must be finite")
    if not np.all(np.isfinite(phi)):
        raise ValueError("P2 input phi must be finite")
    if np.any(phi < 0.0) or np.any(phi > 1.0):
        raise ValueError("P2 input phi must lie in [0, 1]")
    legal = np.isin(cell_type, np.array([0, 1, 2], dtype=np.uint8))
    if not np.all(legal):
        raise ValueError("P2 input cell_type contains an unknown value")
    validate_initial_topology(cell_type, periodic=periodic)

    density = np.asarray(state.density.numpy())
    if not np.allclose(mass, density * phi, atol=1.0e-6, rtol=1.0e-5):
        raise ValueError("P2 input must satisfy mass ~= density * phi")


class VofMassTransport:
    """Compute P2 fixed-topology FullF mass exchange without state mutation."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        periodic: tuple[int, int, int],
    ) -> None:
        self.shape = tuple(int(value) for value in shape)
        self.device = device
        self.periodic = tuple(int(value) for value in periodic)
        self._mass_tmp = wp.zeros(self.shape, dtype=float, device=device)
        self._phi_tmp = wp.zeros(self.shape, dtype=float, device=device)
        self._mass_delta = wp.zeros(self.shape, dtype=float, device=device)
        self.result = VofMassTransportResult(
            mass_tmp=self._mass_tmp,
            phi_tmp=self._phi_tmp,
            mass_delta=self._mass_delta,
        )

    def compute_fullf(
        self,
        state: FullFLbmState,
        *,
        validate: bool = True,
    ) -> VofMassTransportResult:
        """Return provisional mass/phi at fixed topology using ``density^n``."""

        from ..state import FullFLbmState

        if not isinstance(state, FullFLbmState):
            raise NotImplementedError(
                "P2 authoritative VOF mass transport supports FullF only; "
                "HOME transport is deferred to P7"
            )
        if tuple(int(value) for value in state.res) != self.shape:
            raise ValueError(
                f"P2 state shape {state.res} does not match transport {self.shape}"
            )
        if state.vof is None:
            raise ValueError("P2 VOF mass transport requires state.vof storage")
        if validate:
            validate_p2_transport_input(
                state,
                periodic=tuple(bool(value) for value in self.periodic),
            )

        nx, ny, nz = self.shape
        px, py, pz = self.periodic
        wp.launch(
            advect_vof_mass_fullf_fixed_topology_kernel,
            dim=self.shape,
            inputs=[
                state.f_post,
                state.density,
                state.vof.mass,
                state.vof.phi,
                state.vof.cell_type,
                self._mass_tmp,
                self._phi_tmp,
                self._mass_delta,
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
        return self.result


__all__ = [
    "VofMassTransport",
    "VofMassTransportResult",
    "validate_p2_transport_input",
]
