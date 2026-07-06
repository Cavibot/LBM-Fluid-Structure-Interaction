# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 Lattice Boltzmann method – time-varying simulation state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from wanphys._src.core.domain import DomainState

if TYPE_CHECKING:
    from .model import LbmModel


class LbmState(DomainState):
    """GPU-resident state for a D3Q19 LBM simulation.

    Stores 19 distribution functions in a single flat array and exposes
    cell-centred macroscopic fields (density, velocity, pressure) together
    with MAC staggered velocity fields for visualisation and rigid-body
    coupling compatibility.

    Parameters
    ----------
    model:
        Static LBM configuration.
    requires_grad:
        If ``True``, allocate arrays with gradient tracking enabled.
    """

    def __init__(self, model: LbmModel, requires_grad: bool = False) -> None:
        self.model: LbmModel = model
        nx: int = int(model.nx)
        ny: int = int(model.ny)
        nz: int = int(model.nz)
        self.res: tuple[int, int, int] = (nx, ny, nz)
        self.device: wp.Device = model._device
        self.requires_grad: bool = requires_grad

        # Stride for flattening (nx, ny, nz) → 1D
        self._stride: int = nx * ny * nz

        # ---- Distribution functions (flat: 19 × N) -----------------------
        self.f: wp.array = wp.zeros(
            19 * self._stride,
            dtype=float,
            device=self.device,
            requires_grad=requires_grad,
        )

        # ---- Cell-centred macroscopic fields ------------------------------
        self.density: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.pressure: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.velocity_x: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.velocity_y: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.velocity_z: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )

        # ---- MAC staggered velocity (for visualisation / coupling) --------
        self.vel_u: wp.array3d = wp.zeros(
            (nx + 1, ny, nz), dtype=float, device=self.device
        )
        self.vel_v: wp.array3d = wp.zeros(
            (nx, ny + 1, nz), dtype=float, device=self.device
        )
        self.vel_w: wp.array3d = wp.zeros(
            (nx, ny, nz + 1), dtype=float, device=self.device
        )
        self.vel_solid_u: wp.array3d = wp.zeros(
            (nx + 1, ny, nz), dtype=float, device=self.device
        )
        self.vel_solid_v: wp.array3d = wp.zeros(
            (nx, ny + 1, nz), dtype=float, device=self.device
        )
        self.vel_solid_w: wp.array3d = wp.zeros(
            (nx, ny, nz + 1), dtype=float, device=self.device
        )

        # ---- Solid boundary -------------------------------------------------
        self.solid_phi: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.solid_phi.fill_(1000.0)
        self.solid_body_id: wp.array3d = wp.full(
            (nx, ny, nz), -1, dtype=wp.int32, device=self.device
        )

        # ---- Shan-Chen / total body force (for visualisation / debug) ----
        self.force_x: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.force_y: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )
        self.force_z: wp.array3d = wp.zeros(
            (nx, ny, nz), dtype=float, device=self.device
        )

    # ------------------------------------------------------------------
    # DomainState protocol
    # ------------------------------------------------------------------

    def clear_forces(self) -> None:
        """LBM has no accumulated forces – no-op for protocol compliance."""
        pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Zero all fields and reset solid SDF to a large value."""
        self.f.zero_()
        self.density.zero_()
        self.pressure.zero_()
        self.velocity_x.zero_()
        self.velocity_y.zero_()
        self.velocity_z.zero_()
        self.vel_u.zero_()
        self.vel_v.zero_()
        self.vel_w.zero_()
        self.vel_solid_u.zero_()
        self.vel_solid_v.zero_()
        self.vel_solid_w.zero_()
        self.solid_phi.fill_(1000.0)
        self.solid_body_id.fill_(-1)
        self.force_x.zero_()
        self.force_y.zero_()
        self.force_z.zero_()

    def clone(self) -> "LbmState":
        """Deep-copy all GPU arrays into a new :class:`LbmState`."""
        new_state: LbmState = LbmState(self.model, requires_grad=self.requires_grad)
        wp.copy(new_state.f, self.f)
        wp.copy(new_state.density, self.density)
        wp.copy(new_state.pressure, self.pressure)
        wp.copy(new_state.velocity_x, self.velocity_x)
        wp.copy(new_state.velocity_y, self.velocity_y)
        wp.copy(new_state.velocity_z, self.velocity_z)
        wp.copy(new_state.vel_u, self.vel_u)
        wp.copy(new_state.vel_v, self.vel_v)
        wp.copy(new_state.vel_w, self.vel_w)
        wp.copy(new_state.vel_solid_u, self.vel_solid_u)
        wp.copy(new_state.vel_solid_v, self.vel_solid_v)
        wp.copy(new_state.vel_solid_w, self.vel_solid_w)
        wp.copy(new_state.solid_phi, self.solid_phi)
        wp.copy(new_state.solid_body_id, self.solid_body_id)
        wp.copy(new_state.force_x, self.force_x)
        wp.copy(new_state.force_y, self.force_y)
        wp.copy(new_state.force_z, self.force_z)
        return new_state
