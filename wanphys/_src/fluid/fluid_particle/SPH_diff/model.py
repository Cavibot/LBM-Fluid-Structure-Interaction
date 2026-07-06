# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH_diff model - static configuration for differentiable WCSPH baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import warp as wp


@dataclass
class SPHDiffModel:
    rigid_density: float = 1000.0
    """Static configuration for SPH_diff simulation.

    This is the WCSPH-first baseline model for differentiable migration.
    DFSPH can be introduced later via model/solver extension.
    """

    h: float = 1.0
    rest_density: float = 1000.0
    viscosity: float = 0.1
    c0: float = 10000.0
    gamma: float = 7.0
    surface_tension: float = 0.0

    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    bounds_min: Tuple[float, float, float] | None = (-1.0, 0.0, -1.0)
    bounds_max: Tuple[float, float, float] | None = (1.0, 2.0, 1.0)

    boundary_padding: float = 0.0
    bounds_restitution: float = 0.1
    bounds_damping: float = 0.02

    integrate_rigid: bool = True
    requires_grad: bool = True

    max_dt: float = 1.0 / 240.0
    fixed_dt: float = 0.002
    """Compatibility timestep used by inherited DFSPH state buffers."""
    # Stability
    max_velocity: float = 100.0
    """Maximum particle velocity to prevent instability."""


    device: str | wp.Device | None = None
    _device: wp.Device = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.device is None:
            self._device = wp.get_device()
        elif isinstance(self.device, str):
            self._device = wp.get_device(self.device)
        else:
            self._device = self.device

    @property
    def bounds_enabled(self) -> bool:
        return self.bounds_min is not None and self.bounds_max is not None

    def get_bounds_min_vec3(self):
        if self.bounds_min is None:
            return wp.vec3(0.0, 0.0, 0.0)
        return wp.vec3(*self.bounds_min)

    def get_bounds_max_vec3(self):
        if self.bounds_max is None:
            return wp.vec3(0.0, 0.0, 0.0)
        return wp.vec3(*self.bounds_max)
