# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM static model configuration.

Defines :class:`HomeFslbmModel`, the dataclass holding all time-invariant
simulation parameters: grid geometry, NOCM-MRT collision parameters,
free-surface / bubble / foam settings, and boundary conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..base import FluidGridModelBase


@dataclass
class HomeFslbmModel(FluidGridModelBase):
    """Static configuration for a HOME-FSLBM free-surface simulation.

    Inherits grid properties (``fluid_grid_res``, ``fluid_grid_cell_size``,
    ``device``) from :class:`FluidGridModelBase`.

    Parameters
    ----------
    omega:
        NOCM-MRT relaxation frequency (ω = 1/τ).  Must satisfy ω > 0
        (equivalent to τ > 0.5 for positive kinematic viscosity).
        Ref: paper-5 Eq.(21).
    gravity_x / gravity_y / gravity_z:
        External body force components (lattice units, per unit mass).
        Corresponds to ``param->gx/gy/gz`` in ``MLFluidParam3D``.
    surface_tension:
        Surface tension coefficient (``def_6_sigma``).  The D3Q27 stencil
        requires a factor of 6 relative to the physical σ.
        Ref: paper-4 Eq.(12).
    gas_omega:
        D3Q7 CMR-MRT base relaxation frequency.  The actual collision uses
        7 independent relaxation rates ``GAS_CMR_S`` (see :mod:`constants`).
        Ref: paper-4 Eq.(34).
    henry_constant:
        Henry's law constant ``K_h`` for dissolved gas exchange at the
        free-surface interface.  Ref: paper-4 Eq.(38).
    disjoin_factor:
        Disjoining pressure strength multiplier.  Prevents bubble
        coalescence when films thin below the grid scale.
        Ref: paper-4 Eq.(40).
    turbulence_factor:
        Eddy-viscosity multiplier ν_e = factor · ‖S‖_F.
        Ref: paper-4 Sec.5.1.
    turbulence_radius:
        Neighbourhood search radius for eddy-viscosity computation.
        ±3 → 6×6×6 = 216 neighbours.  Ref: ``stream_collide_bvh:1001``.
        *Audit item S6 — corrected from 4 → 3.*
    max_bubbles:
        Maximum number of tracked bubbles (``mlBubble3D::max_bubble_count``).
        Ref: ``mrFlow3D.h:25``.
    atmosphere_open:
        If ``True``, the top boundary is open to the atmosphere, enabling
        ``atmosphere_rho_update_kernel`` and ``atmosphere_volme_update_kernel``.
    bc_types:
        Per-face boundary condition types (length 6: xmin, xmax, ymin, ymax,
        zmin, zmax).  0 = bounce-back, 1 = Zou-He velocity inlet,
        2 = convective outflow, 3 = periodic.
    bc_periodic:
        Per-axis periodic flags ``(px, py, pz)``.
    bc_velocity:
        Per-face prescribed velocity for Zou-He inlets ``((ux,uy,uz), …)``.
    initial_density:
        Uniform initial density ρ₀ for equilibrium initialisation.
    """
    # ---- NOCM-MRT collision -----------------------------------------------
    omega: float = 1.0
    """NOCM-MRT relaxation frequency ω = 1/τ."""

    # ---- External body force -----------------------------------------------
    gravity_x: float = 0.0
    gravity_y: float = 0.0
    gravity_z: float = 0.0

    # ---- Free-surface parameters ------------------------------------------
    surface_tension: float = 6.0 * 4e-3
    """Surface tension σ × 6 (D3Q27 scaling)."""

    # ---- Dissolved gas model -----------------------------------------------
    gas_omega: float = 1.0
    """D3Q7 CMR-MRT base relaxation frequency."""
    henry_constant: float = 1e-3
    """Henry's law constant K_h."""

    # ---- Foam / disjoining pressure ----------------------------------------
    disjoin_factor: float = 0.032
    """Disjoining pressure strength multiplier."""

    # ---- Turbulence model --------------------------------------------------
    turbulence_factor: float = 4.0
    """Eddy-viscosity multiplier ν_e = factor · ‖S‖_F."""
    turbulence_radius: int = 3
    """Neighbourhood search radius (±3 → 6×6×6 = 216 neighbours)."""

    # ---- Bubble tracking ---------------------------------------------------
    max_bubbles: int = 65536
    """Maximum tracked bubble count."""

    # ---- Atmosphere --------------------------------------------------------
    atmosphere_open: bool = False
    """Enable open-atmosphere top boundary."""

    # ---- Boundary conditions -----------------------------------------------
    bc_types: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    """Per-face BC types: 0=bounce-back, 1=Zou-He inlet, 2=outflow, 3=periodic."""
    bc_periodic: tuple[bool, bool, bool] = (False, False, False)
    """Per-axis periodic flags."""
    bc_velocity: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    """Per-face prescribed inlet velocities (lattice units)."""

    # ---- Initial condition -------------------------------------------------
    initial_density: float = 1.0
    """Uniform initial density ρ₀."""

    # ---- Derived properties ------------------------------------------------

    @property
    def tau(self) -> float:
        """Relaxation time τ = 1 / ω."""
        return 1.0 / self.omega

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity ν = c_s² · (τ − 0.5) = (1/3)·(τ − 0.5)."""
        return (1.0 / 3.0) * (self.tau - 0.5)

    @property
    def _periodic_ints(self) -> tuple[int, int, int]:
        """Periodic flags as ints ``(px, py, pz)`` for kernel passing."""
        return (
            int(self.bc_periodic[0]),
            int(self.bc_periodic[1]),
            int(self.bc_periodic[2]),
        )

    # ---- Validation --------------------------------------------------------

    def __post_init__(self) -> None:
        super().__post_init__()

        # Stability guard: τ > 0.5  ⇔  ω > 0.0
        if self.omega <= 0.0:
            raise ValueError(
                f"omega must be > 0 for positive viscosity (got {self.omega})"
            )

        # Grid resolution
        nx, ny, nz = self.nx, self.ny, self.nz
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError(
                f"Grid dimensions must be positive (got {nx}×{ny}×{nz})"
            )

        # BC consistency: periodic flag on an axis implies all faces on that
        # axis should be BC_PERIODIC.
        _bc = list(self.bc_types)
        if self.bc_periodic[0]:
            if _bc[0] != 3 or _bc[1] != 3:
                raise ValueError(
                    "X-axis is periodic but bc_types[0] or [1] is not BC_PERIODIC"
                )
        if self.bc_periodic[1]:
            if _bc[2] != 3 or _bc[3] != 3:
                raise ValueError(
                    "Y-axis is periodic but bc_types[2] or [3] is not BC_PERIODIC"
                )
        if self.bc_periodic[2]:
            if _bc[4] != 3 or _bc[5] != 3:
                raise ValueError(
                    "Z-axis is periodic but bc_types[4] or [5] is not BC_PERIODIC"
                )

    # ---- Convenience factories ---------------------------------------------

    @classmethod
    def default_32(cls, device: Optional[str] = None) -> "HomeFslbmModel":
        """Return a model with a 32³ grid and sensible defaults."""
        return cls(
            fluid_grid_res=(32, 32, 32),
            fluid_grid_cell_size=1.0,
            omega=1.0,
            device=device,
        )

    @classmethod
    def dam_break_64(cls, device: Optional[str] = None) -> "HomeFslbmModel":
        """Return a model configured for a 64³ dam-break scenario."""
        return cls(
            fluid_grid_res=(64, 64, 64),
            fluid_grid_cell_size=1.0,
            omega=1.0,
            gravity_y=-0.001,
            surface_tension=6.0 * 4e-3,
            atmosphere_open=False,
            device=device,
        )
