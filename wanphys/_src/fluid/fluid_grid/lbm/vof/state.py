# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent VOF grid state and derived interface-geometry cache."""

from __future__ import annotations

import warp as wp

from .contracts import VofCellType


class GeometryState:
    """Derived interface geometry associated with one VOF state epoch."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        device: wp.Device,
        requires_grad: bool = False,
    ) -> None:
        self.normal = wp.zeros(
            shape,
            dtype=wp.vec3,
            device=device,
            requires_grad=requires_grad,
        )
        self.valid_epoch = -1

    def clear(self) -> None:
        self.normal.zero_()
        self.valid_epoch = -1

    def copy_to(self, target: GeometryState) -> None:
        wp.copy(target.normal, self.normal)
        target.valid_epoch = self.valid_epoch


class VofGridState:
    """Persistent liquid fill fraction, cell type, and geometry cache.

    ``mass`` is intentionally absent in the observation-only first phase.  It
    will be added when conservative VOF transport becomes authoritative.
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
        self.geometry = GeometryState(shape, device, requires_grad=requires_grad)
        self.epoch = -1

    def clear(self) -> None:
        self.phi.zero_()
        self.cell_type.fill_(int(VofCellType.GAS))
        self.geometry.clear()
        self.epoch = -1

    def copy_to(self, target: VofGridState) -> None:
        wp.copy(target.phi, self.phi)
        wp.copy(target.cell_type, self.cell_type)
        self.geometry.copy_to(target.geometry)
        target.epoch = self.epoch

    def clone(self) -> VofGridState:
        target = VofGridState(
            self.shape,
            self.device,
            requires_grad=self.requires_grad,
        )
        self.copy_to(target)
        return target


__all__ = ["GeometryState", "VofGridState"]
