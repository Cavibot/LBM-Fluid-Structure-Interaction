# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Authoritative VOF state and isolated Shan-Chen debug observation state."""

from __future__ import annotations

import warp as wp

from .contracts import VofCellType


class VofGridState:
    """Authoritative conservative VOF state: mass, phi, and cell type.

    P1 introduces only the three persistent authoritative fields.  No
    derived geometry cache (normal, PLIC, curvature), epoch, active mask,
    density/moment copy or transition scratch is stored here.  The canonical
    empty state after construction and :meth:`clear` is all-GAS with zero
    mass and zero phi.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
    ) -> None:
        self.shape = shape
        self.device = device
        self.mass = wp.zeros(shape, dtype=float, device=device)
        self.phi = wp.zeros(shape, dtype=float, device=device)
        self.cell_type = wp.full(
            shape,
            int(VofCellType.GAS),
            dtype=wp.uint8,
            device=device,
        )

    def clear(self) -> None:
        """Reset to the canonical all-GAS empty state."""
        self.mass.zero_()
        self.phi.zero_()
        self.cell_type.fill_(int(VofCellType.GAS))

    def copy_to(self, target: VofGridState) -> None:
        """Copy the three authoritative arrays into *target*."""
        if target.shape != self.shape:
            raise ValueError(
                "VofGridState shape mismatch: "
                f"source {self.shape} != target {target.shape}"
            )
        wp.copy(target.mass, self.mass)
        wp.copy(target.phi, self.phi)
        wp.copy(target.cell_type, self.cell_type)

    def clone(self) -> VofGridState:
        """Return an independent deep copy of this state."""
        target = VofGridState(self.shape, self.device)
        self.copy_to(target)
        return target


class DebugMockScToVofState:
    """Non-conservative VOF-shaped debug data derived from Shan-Chen density.

    This state has no mass, is not authoritative, and must never influence the
    LBM solver or the separate :class:`VofGridState`.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        requires_grad: bool = False,
    ) -> None:
        self.shape = shape
        self.device = device
        self.requires_grad = requires_grad
        self.phi = wp.zeros(
            shape,
            dtype=float,
            device=device,
            requires_grad=requires_grad,
        )
        self.cell_type = wp.full(
            shape,
            int(VofCellType.GAS),
            dtype=wp.uint8,
            device=device,
        )
        self.normal = wp.zeros(
            shape,
            dtype=wp.vec3,
            device=device,
            requires_grad=requires_grad,
        )
        self.epoch = -1
        self.normal_valid_epoch = -1

    def clear(self) -> None:
        self.phi.zero_()
        self.cell_type.fill_(int(VofCellType.GAS))
        self.normal.zero_()
        self.epoch = -1
        self.normal_valid_epoch = -1

    def copy_to(self, target: DebugMockScToVofState) -> None:
        wp.copy(target.phi, self.phi)
        wp.copy(target.cell_type, self.cell_type)
        wp.copy(target.normal, self.normal)
        target.epoch = self.epoch
        target.normal_valid_epoch = self.normal_valid_epoch

    def clone(self) -> DebugMockScToVofState:
        target = DebugMockScToVofState(
            self.shape,
            self.device,
            requires_grad=self.requires_grad,
        )
        self.copy_to(target)
        return target


__all__ = ["DebugMockScToVofState", "VofGridState"]
