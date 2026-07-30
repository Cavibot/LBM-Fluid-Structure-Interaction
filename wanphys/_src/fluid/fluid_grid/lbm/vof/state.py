# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Authoritative VOF state and isolated Shan-Chen debug observation state."""

from __future__ import annotations

import warp as wp

from .contracts import VofCellType


class VofGridState:
    """Authoritative VOF state, pending mass transfer, and derived geometry.

    ``mass`` is the committed resident liquid mass. ``pending_excess`` is a
    signed, one-step delayed transfer scheduled for
    ``pending_receiver_count`` D3Q19 INTERFACE neighbors.  Their sum is the
    closed-domain conservative ledger.  ``phi`` remains strictly bounded for
    geometry and mass-flux weights.

    P6 adds an epoch-scoped geometry cache: liquid-to-gas ``normal``,
    centered unit-cube ``plic_offset`` and mean ``curvature``.
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
        self.pending_excess = wp.zeros(shape, dtype=float, device=device)
        self.pending_receiver_count = wp.zeros(
            shape,
            dtype=wp.uint8,
            device=device,
        )
        self.cell_type = wp.full(
            shape,
            int(VofCellType.GAS),
            dtype=wp.uint8,
            device=device,
        )
        self.normal = wp.zeros(shape, dtype=wp.vec3, device=device)
        self.plic_offset = wp.zeros(shape, dtype=float, device=device)
        self.curvature = wp.zeros(shape, dtype=float, device=device)
        self.epoch = -1
        self.geometry_epoch = -1
        self.reference_mass = 0.0

    def clear(self) -> None:
        """Reset to the canonical all-GAS empty state."""
        self.mass.zero_()
        self.phi.zero_()
        self.pending_excess.zero_()
        self.pending_receiver_count.zero_()
        self.cell_type.fill_(int(VofCellType.GAS))
        self.normal.zero_()
        self.plic_offset.zero_()
        self.curvature.zero_()
        self.epoch = -1
        self.geometry_epoch = -1
        self.reference_mass = 0.0

    def copy_to(self, target: VofGridState) -> None:
        """Copy conservative fields, derived geometry and epochs into target."""
        if target.shape != self.shape:
            raise ValueError(
                "VofGridState shape mismatch: "
                f"source {self.shape} != target {target.shape}"
            )
        wp.copy(target.mass, self.mass)
        wp.copy(target.phi, self.phi)
        wp.copy(target.pending_excess, self.pending_excess)
        wp.copy(
            target.pending_receiver_count,
            self.pending_receiver_count,
        )
        wp.copy(target.cell_type, self.cell_type)
        wp.copy(target.normal, self.normal)
        wp.copy(target.plic_offset, self.plic_offset)
        wp.copy(target.curvature, self.curvature)
        target.epoch = self.epoch
        target.geometry_epoch = self.geometry_epoch
        target.reference_mass = self.reference_mass

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
