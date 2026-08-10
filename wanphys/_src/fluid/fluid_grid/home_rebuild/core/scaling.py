# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical-to-lattice scaling required by the isolated HOME core."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LatticeScaling:
    cell_size: float
    time_step: float
    reference_density: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_size", self.cell_size),
            ("time_step", self.time_step),
            ("reference_density", self.reference_density),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")

    @property
    def velocity_unit(self) -> float:
        return self.cell_size / self.time_step

    @property
    def acceleration_unit(self) -> float:
        return self.cell_size / (self.time_step * self.time_step)

    @property
    def kinematic_viscosity_unit(self) -> float:
        return self.cell_size * self.cell_size / self.time_step

    @property
    def pressure_unit(self) -> float:
        return self.reference_density * self.cell_size**2 / self.time_step**2

    @property
    def surface_tension_unit(self) -> float:
        return self.pressure_unit * self.cell_size

    def acceleration_to_lattice(self, value: float) -> float:
        return value / self.acceleration_unit

    def viscosity_to_lattice(self, value: float) -> float:
        return value / self.kinematic_viscosity_unit

    def surface_tension_to_lattice(self, value: float) -> float:
        return value / self.surface_tension_unit

    def surface_tension_to_physical(self, value: float) -> float:
        return value * self.surface_tension_unit
