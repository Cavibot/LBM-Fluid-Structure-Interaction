# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Observation-only state derived from Shan-Chen density."""

from __future__ import annotations

import warp as wp

from .contracts import VofCellType


class DebugMockScToVofState:
    """Non-conservative VOF-shaped debug data derived from Shan-Chen density.

    This state has no mass, is not authoritative, and must never influence the
    LBM solver.  A formal ``VofGridState`` will be introduced separately in P1.
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


__all__ = ["DebugMockScToVofState"]
