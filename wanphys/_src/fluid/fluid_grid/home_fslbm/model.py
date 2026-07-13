# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM static model configuration.

Reference:
    [REF] Home-FSLBM inc/3D/cpu/mrFlow3D.h — mrFlow3D::Create
    [P4] Wang et al. 2025 — free-surface LBM with moment encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..base import FluidGridModelBase


@dataclass
class HomeFSLbmModel(FluidGridModelBase):
    """Static configuration for a HOME-FSLBM free-surface fluid simulation.

    Stores grid properties, relaxation parameters, body forces, and
    domain boundary condition types.

    Parameters
    ----------
    fluid_grid_res:
        Grid resolution (nx, ny, nz) in cells.
    fluid_grid_cell_size:
        Physical size of one cell in metres.  In lattice units the cell
        size is always 1; this parameter is used for physical-unit output
        only.
    tau:
        BGK relaxation time.  Kinematic viscosity in lattice units:
        ``nu = c_s^2 * (tau - 0.5)`` with ``c_s^2 = 1/3``.
        Must be > 0.5.
    gravity_x, gravity_y, gravity_z:
        Body-force components in lattice units (force per unit mass).
    bc_types:
        Six-element tuple of boundary condition types for the domain faces
        in order (-x, +x, -y, +y, -z, +z).  Currently only ``"bounce_back"``
        is supported for the free-surface solver.
    max_velocity:
        Hard clamp on velocity magnitude in lattice units.
        [REF] normalizing_clamp(u, 0.4).
    device:
        CUDA device identifier string (``None`` = default device).
    """

    # ---- Relaxation / viscosity -----------------------------------------
    tau: float = 0.55
    """BGK relaxation time.  ``nu = (1/3) * (tau - 0.5)``."""

    # ---- Body force (lattice units, per unit mass) ----------------------
    gravity_x: float = 0.0
    """Gravity x-component in lattice units."""
    gravity_y: float = 0.0
    """Gravity y-component in lattice units."""
    gravity_z: float = 0.0
    """Gravity z-component in lattice units."""

    # ---- Domain boundary conditions --------------------------------------
    bc_types: tuple = ("bounce_back",) * 6
    """Boundary condition per face (-x, +x, -y, +y, -z, +z).
    Supported values: ``"bounce_back"``, ``"periodic"``."""

    # ---- Velocity safety clamp -------------------------------------------
    max_velocity: float = 0.4
    """Maximum allowed velocity magnitude in lattice units.
    [REF] normalizing_clamp in stream_collide_bvh."""

    # ---- Derived ---------------------------------------------------------
    @property
    def omega(self) -> float:
        """Collision frequency ``omega = 1 / tau``."""
        return 1.0 / self.tau

    @property
    def viscosity(self) -> float:
        """Kinematic viscosity in lattice units ``nu = cs2 * (tau - 0.5)``."""
        return (1.0 / 3.0) * (self.tau - 0.5)
