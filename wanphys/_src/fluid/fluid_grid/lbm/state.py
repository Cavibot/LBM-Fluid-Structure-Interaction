# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent D3Q19 kinetic-state encodings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from wanphys._src.core.domain import DomainState

if TYPE_CHECKING:
    from .model import LbmModel


class LbmStateBase(DomainState):
    """Fields shared by FullF and HOME persistent kinetic states."""

    def __init__(self, model: LbmModel, requires_grad: bool = False) -> None:
        self.model = model
        nx, ny, nz = int(model.nx), int(model.ny), int(model.nz)
        self.res = (nx, ny, nz)
        self.device = model._device
        self.requires_grad = requires_grad
        self._stride = nx * ny * nz

        self.density = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self.velocity_x = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self.velocity_y = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self.velocity_z = wp.zeros((nx, ny, nz), dtype=float, device=self.device)

        self.vel_u = wp.zeros((nx + 1, ny, nz), dtype=float, device=self.device)
        self.vel_v = wp.zeros((nx, ny + 1, nz), dtype=float, device=self.device)
        self.vel_w = wp.zeros((nx, ny, nz + 1), dtype=float, device=self.device)
        self.vel_solid_u = wp.zeros((nx + 1, ny, nz), dtype=float, device=self.device)
        self.vel_solid_v = wp.zeros((nx, ny + 1, nz), dtype=float, device=self.device)
        self.vel_solid_w = wp.zeros((nx, ny, nz + 1), dtype=float, device=self.device)

        self.solid_phi = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self.solid_phi.fill_(1000.0)
        self.solid_body_id = wp.full(
            (nx, ny, nz), -1, dtype=wp.int32, device=self.device
        )

        self.force_x = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self.force_y = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        self.force_z = wp.zeros((nx, ny, nz), dtype=float, device=self.device)
        # Unified physical force density F = rho*g + F_sc + future providers.

    def clear_forces(self) -> None:
        """LBM has no accumulated force buffer in the DomainState sense."""

    def _clear_common(self) -> None:
        for field in (
            self.density, self.velocity_x, self.velocity_y, self.velocity_z,
            self.vel_u, self.vel_v, self.vel_w,
            self.vel_solid_u, self.vel_solid_v, self.vel_solid_w,
            self.force_x, self.force_y, self.force_z,
        ):
            field.zero_()
        self.solid_phi.fill_(1000.0)
        self.solid_body_id.fill_(-1)

    def _copy_common_to(self, target: "LbmStateBase") -> None:
        for name in (
            "density", "velocity_x", "velocity_y", "velocity_z",
            "vel_u", "vel_v", "vel_w",
            "vel_solid_u", "vel_solid_v", "vel_solid_w",
            "solid_phi", "solid_body_id", "force_x", "force_y", "force_z",
        ):
            wp.copy(getattr(target, name), getattr(self, name))


class FullFLbmState(LbmStateBase):
    """Post-collision FullF state with all 19 D3Q19 populations."""

    def __init__(self, model: LbmModel, requires_grad: bool = False) -> None:
        super().__init__(model, requires_grad)
        self.f_post = wp.zeros(
            19 * self._stride,
            dtype=float,
            device=self.device,
            requires_grad=requires_grad,
        )

    @property
    def f(self) -> wp.array:
        """Compatibility alias for the post-collision population buffer."""
        return self.f_post

    def clear(self) -> None:
        self.f_post.zero_()
        self._clear_common()

    def clone(self) -> "FullFLbmState":
        target = FullFLbmState(self.model, self.requires_grad)
        wp.copy(target.f_post, self.f_post)
        self._copy_common_to(target)
        return target


class HomeLbmState(LbmStateBase):
    """Post-collision HOME state storing density-weighted Hermite moments.

    The ten persistent fields are ``rho``, ``rho*u`` and the six symmetric
    components of ``rho*S``.  No full population array is retained.
    """

    def __init__(self, model: LbmModel, requires_grad: bool = False) -> None:
        super().__init__(model, requires_grad)
        shape = self.res

        def make() -> wp.array3d:
            return wp.zeros(
                shape,
                dtype=float,
                device=self.device,
                requires_grad=requires_grad,
            )

        self.rho = make()
        self.rho_u_x, self.rho_u_y, self.rho_u_z = make(), make(), make()
        self.rho_s_xx, self.rho_s_yy, self.rho_s_zz = make(), make(), make()
        self.rho_s_xy, self.rho_s_xz, self.rho_s_yz = make(), make(), make()

    @property
    def kinetic_fields(self) -> tuple[wp.array3d, ...]:
        return (
            self.rho,
            self.rho_u_x, self.rho_u_y, self.rho_u_z,
            self.rho_s_xx, self.rho_s_yy, self.rho_s_zz,
            self.rho_s_xy, self.rho_s_xz, self.rho_s_yz,
        )

    def clear(self) -> None:
        for field in self.kinetic_fields:
            field.zero_()
        self._clear_common()

    def clone(self) -> "HomeLbmState":
        target = HomeLbmState(self.model, self.requires_grad)
        for dst, src in zip(target.kinetic_fields, self.kinetic_fields):
            wp.copy(dst, src)
        self._copy_common_to(target)
        return target


# Backward-compatible public name.  The default encoding remains FullF.
LbmState = FullFLbmState
