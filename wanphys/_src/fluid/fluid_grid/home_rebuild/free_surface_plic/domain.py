# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Standalone grid and wall ownership for isolated geometric VOF."""

from __future__ import annotations

import numpy as np
import warp as wp


class PlicDomain:
    """Own a PLIC cell grid without allocating or importing HOME/FSL state."""

    def __init__(
        self,
        resolution: tuple[int, int, int],
        *,
        device: str = "cpu",
        solid_mask: np.ndarray | None = None,
        closed_axes: tuple[bool, bool, bool] = (True, True, True),
    ) -> None:
        if len(resolution) != 3 or any(int(value) < 1 for value in resolution):
            raise ValueError("PLIC resolution must contain three positive values")
        if len(closed_axes) != 3:
            raise ValueError("closed_axes must contain three values")
        self.res = tuple(int(value) for value in resolution)
        self.closed_axes = tuple(bool(value) for value in closed_axes)
        self._device = wp.get_device(device)
        if solid_mask is None:
            solid = np.zeros(self.res, dtype=bool)
        else:
            solid = np.asarray(solid_mask, dtype=bool)
            if solid.shape != self.res:
                raise ValueError("PLIC solid_mask must match the grid")
            solid = solid.copy()
        for axis, closed in enumerate(self.closed_axes):
            if not closed:
                continue
            if self.res[axis] < 3:
                raise ValueError("a closed PLIC axis requires at least three cells")
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = 0
            upper[axis] = -1
            solid[tuple(lower)] = True
            solid[tuple(upper)] = True
        self.host = solid
        self.device = wp.array(
            solid.astype(np.int32), dtype=wp.int32, device=self._device
        )
        self.model = self

    @classmethod
    def closed_box(
        cls, resolution: tuple[int, int, int], *, device: str = "cpu"
    ) -> PlicDomain:
        return cls(resolution, device=device)

    @classmethod
    def periodic(
        cls,
        resolution: tuple[int, int, int],
        *,
        device: str = "cpu",
    ) -> PlicDomain:
        return cls(
            resolution,
            device=device,
            closed_axes=(False, False, False),
        )

    @property
    def solid_cell_count(self) -> int:
        return int(np.count_nonzero(self.host))
