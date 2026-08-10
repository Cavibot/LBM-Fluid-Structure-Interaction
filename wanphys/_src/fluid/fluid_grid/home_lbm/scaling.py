# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit conversion between physical and lattice units."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LatticeScaling:
    """Physical-to-lattice conversion for one uniform-grid simulation.

    ``reference_density`` maps lattice density one to physical mass density.
    The derived force unit is ``rho_ref * dx**4 / dt**2`` in three dimensions.
    """

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
    def mass_unit(self) -> float:
        return self.reference_density * self.cell_size**3

    @property
    def momentum_unit(self) -> float:
        return self.mass_unit * self.velocity_unit

    @property
    def angular_momentum_unit(self) -> float:
        return self.momentum_unit * self.cell_size

    @property
    def force_unit(self) -> float:
        return self.reference_density * self.cell_size**4 / self.time_step**2

    @property
    def pressure_unit(self) -> float:
        return self.reference_density * self.cell_size**2 / self.time_step**2

    @property
    def surface_tension_unit(self) -> float:
        return self.pressure_unit * self.cell_size

    @property
    def torque_unit(self) -> float:
        return self.force_unit * self.cell_size

    def velocity_to_lattice(self, value: float) -> float:
        return value / self.velocity_unit

    def velocity_to_physical(self, value: float) -> float:
        return value * self.velocity_unit

    def acceleration_to_lattice(self, value: float) -> float:
        return value / self.acceleration_unit

    def acceleration_to_physical(self, value: float) -> float:
        return value * self.acceleration_unit

    def viscosity_to_lattice(self, value: float) -> float:
        return value / self.kinematic_viscosity_unit

    def viscosity_to_physical(self, value: float) -> float:
        return value * self.kinematic_viscosity_unit

    def force_to_lattice(self, value: float) -> float:
        return value / self.force_unit

    def force_to_physical(self, value: float) -> float:
        return value * self.force_unit

    def pressure_to_lattice(self, value: float) -> float:
        return value / self.pressure_unit

    def pressure_to_physical(self, value: float) -> float:
        return value * self.pressure_unit

    def surface_tension_to_lattice(self, value: float) -> float:
        return value / self.surface_tension_unit

    def surface_tension_to_physical(self, value: float) -> float:
        return value * self.surface_tension_unit

    def torque_to_lattice(self, value: float) -> float:
        return value / self.torque_unit

    def torque_to_physical(self, value: float) -> float:
        return value * self.torque_unit
