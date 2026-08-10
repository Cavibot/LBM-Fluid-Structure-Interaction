# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Configuration contract for the isolated periodic HOME core."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...base import FluidGridModelBase
from .scaling import LatticeScaling


@dataclass
class HomeCoreModel(FluidGridModelBase):
    time_step: float = 1.0
    reference_density: float = 1.0
    kinematic_viscosity: float = 1.0 / 6.0
    max_lattice_speed: float = 0.2
    body_acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    periodic: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.kinematic_viscosity) or self.kinematic_viscosity <= 0.0:
            raise ValueError("kinematic_viscosity must be finite and positive")
        if not math.isfinite(self.max_lattice_speed) or not 0.0 < self.max_lattice_speed < 1.0 / math.sqrt(3.0):
            raise ValueError("max_lattice_speed must be positive and below the lattice sound speed")
        if len(self.body_acceleration) != 3 or not all(math.isfinite(v) for v in self.body_acceleration):
            raise ValueError("body_acceleration must contain three finite components")
        if tuple(self.periodic) != (True, True, True):
            raise ValueError("HomeCoreModel is periodic-only; boundary conditions belong to a composition layer")
        if not 0.0 < self.shear_omega < 2.0:
            raise ValueError(f"physical scaling produces invalid shear omega {self.shear_omega}")

    @property
    def scaling(self) -> LatticeScaling:
        return LatticeScaling(
            cell_size=float(self.fluid_grid_cell_size),
            time_step=float(self.time_step),
            reference_density=float(self.reference_density),
        )

    @property
    def lattice_viscosity(self) -> float:
        return self.scaling.viscosity_to_lattice(self.kinematic_viscosity)

    @property
    def shear_omega(self) -> float:
        return 1.0 / (0.5 + 3.0 * self.lattice_viscosity)

    @property
    def lattice_acceleration(self) -> tuple[float, float, float]:
        return tuple(self.scaling.acceleration_to_lattice(v) for v in self.body_acceleration)

    def validate_step(self, dt: float) -> None:
        if not math.isclose(dt, self.time_step, rel_tol=1.0e-9, abs_tol=1.0e-12):
            raise ValueError(
                f"HOME core step dt={dt} does not match configured time_step={self.time_step}"
            )
