# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 Lattice Boltzmann method – static model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..base import FluidGridModelBase


@dataclass
class LbmModel(FluidGridModelBase):
    """Static configuration for a D3Q19 LBM fluid simulation.

    Inherits grid properties (``fluid_grid_res``, ``fluid_grid_cell_size``,
    ``device``) from :class:`FluidGridModelBase`.  LBM-specific parameters
    control the BGK collision operator and external body forces.

    Notes
    -----
    The LBM timestep is **always 1 in lattice units**.  The physical
    timestep ``dt`` passed to :meth:`LbmSolver.step` is accepted for API
    compatibility but ignored internally.  Physical-to-lattice conversion
    uses the cell size ``dh`` (see :meth:`cell_size`).

    Kinematic viscosity (lattice units):
        ``nu = c_s² · (τ - 0.5) = (1/3) · (τ - 0.5)``.
    """

    # ---- BGK collision ---------------------------------------------------
    tau: float = 0.55
    """BGK relaxation time.  Must be > 0.5 for positive viscosity.
    ``omega = 1 / tau``.  Larger *tau* → more viscous."""

    # ---- External body force (lattice units, per unit mass) --------------
    gravity_x: float = 0.0
    """Gravity / body-force x-component in lattice units."""
    gravity_y: float = 0.0
    """Gravity / body-force y-component in lattice units."""
    gravity_z: float = 0.0
    """Gravity / body-force z-component in lattice units."""

    # ---- Shan-Chen multiphase interaction --------------------------------
    G: float = 0.0
    """Shan-Chen interaction strength.  Negative values produce attraction
    between fluid particles, driving phase separation below a critical
    threshold.  ``G = 0`` disables the interaction (single-phase mode).
    Typical values: -4.0 to -6.0 for PSI_EXP, -0.3 to -0.5 for PSI_RHO."""
    psi_type: int = 0
    """Pseudopotential type: ``PSI_RHO = 0`` (ψ = ρ), ``PSI_EXP = 1``
    (ψ = 1 − exp(−ρ / ψ_ref))."""
    psi_ref: float = 1.0
    """Reference density for the exponential pseudopotential (PSI_EXP)."""
    sc_wall_G: float = 0.0
    """(deprecated) Former explicit wall-force strength.  No longer used;
    kept for backward-compatible construction only.  Wetting is now
    controlled via ``sc_solid_psi_scale`` and ``sc_boundary_psi``.
    """
    sc_solid_psi_scale: float = 1.0
    """Virtual-density scale for solid-body neighbours (``solid_phi < 0``).

    The solid neighbour contributes ψ_solid = ψ_c × sc_solid_psi_scale
    to the unified SC force sum.  Values > 1 make solids appear denser
    (repelling liquid); values < 1 make solids appear lighter
    (attracting liquid).  ``1.0`` is the neutral mirror closure.
    """
    sc_boundary_psi: float = -1.0
    """Fixed pseudopotential for domain-boundary (out-of-bounds) neighbours.

    When >= 0, out-of-bounds neighbours use this fixed ψ value instead of
    the mirror closure ``ψ_c × sc_solid_psi_scale``.  This gives domain
    walls a genuine physical character:
      - ``~0.61`` — neutral wall (ρ ≈ 0.95, 90° contact angle)
      - ``~0.1``  — hydrophobic (gas-like wall)
      - ``~0.83`` — hydrophilic (water-like wall)

    When < 0 (default), the legacy mirror closure is used for backwards
    compatibility.  This parameter has no effect on solid-body neighbours
    (``solid_phi < 0``), which always use the mirror closure.
    """

    sc_force_stride: int = 2
    """Number of solver steps between Shan-Chen force recomputations.

    ``1`` recomputes the interaction force every step and is the safest
    accuracy setting.  Larger values reuse the cached force between updates
    for performance; the default preserves the current optimized behavior.
    """
    sc_homogeneous_early_out: bool = False
    """Skip full Shan-Chen force evaluation in locally homogeneous bulk cells."""
    sc_homogeneous_rel_tol: float = 0.15
    """Relative density tolerance for homogeneous-region early-out checks."""

    # ---- TRT (Two-Relaxation-Time) collision ---------------------------
    lambda_trt: float = 0.0
    """TRT "magic" parameter Lambda = (tau_plus - 0.5)(tau_minus - 0.5).

    ``0.0`` disables TRT (pure BGK).  D3Q19 recommended value: ``0.1875``
    (= 3/16), which gives optimal stability for Poiseuille flow with
    halfway bounce-back.

    Valid range: ``0.0 <= lambda_trt <= (tau - 0.5)^2``.  Values outside
    this range produce ``tau_plus <= 0.5``, which is unstable.
    """
    use_regularization: bool = False
    omega_reg: float = 0.5
    """Regularization blend factor for the TRT collision.

    ``1.0`` means full regularization (replace all non-equilibrium with
    2nd-order Hermite projection), ``0.0`` means no regularization
    (keep original post-collision distributions), and intermediate values
    blend: ``f_out = omega_reg * f_reg + (1 - omega_reg) * f_in``.
    """

    # ---- Carnahan-Starling EOS parameters (psi_type = PSI_CS = 2) ---------
    cs_a: float = 0.5
    """CS EOS attraction parameter *a*.  Controls the depth of the potential
    well.  Typical values: 0.25–1.0.  Larger *a* → stronger phase separation,
    larger density ratio."""
    cs_b: float = 4.0
    """CS EOS covolume parameter *b*.  Together with *cs_a* determines the
    critical temperature ``T_c = 0.3773·a / b``.  ``b = 4`` is standard
    for LBM, giving η = ρ (simplified packing fraction)."""
    cs_T: float = 0.07
    """CS EOS reduced temperature.  Must be below T_c for phase separation.
    Lower T → larger density ratio, sharper interface.  Typical: 0.04–0.09.
    With a=0.5, b=4 → T_c ≈ 0.0943 → T=0.07 gives ρ_l/ρ_g ≈ 100:1."""

    @property
    def omega_minus(self) -> float:
        """Shear relaxation frequency ω₋ = 1/τ."""
        return 1.0 / self.tau

    @property
    def tau_minus(self) -> float:
        """Shear relaxation time τ₋ = τ (alias for ``tau``)."""
        return self.tau

    @property
    def tau_plus(self) -> float:
        """Energy (even-mode) relaxation time τ₊ = 1/ω₊.

        For TRT mode (``lambda_trt > 0``):
            τ₊ = λ / (τ - 0.5) + 0.5
        For BGK mode (``lambda_trt == 0``):
            τ₊ = τ.

        This controls even-mode TRT relaxation in the collision operator.
        The current Shan-Chen velocity-shift path intentionally uses τ₋
        (``tau``) as a stability choice for strong-interface flows.
        """
        if self.lambda_trt <= 0.0:
            return self.tau
        return self.lambda_trt / (self.tau - 0.5) + 0.5

    @property
    def omega_plus(self) -> float:
        """Energy relaxation frequency ω₊ = 1/τ₊.

        Derived from Lambda and omega_minus via:
            Λ = (1/ω₊ - 0.5)(1/ω₋ - 0.5)
        If Lambda=0 (BGK mode), returns ω₋.

        Raises ValueError via __post_init__ if lambda_trt produces τ₊ <= 0.5.
        """
        return 1.0 / self.tau_plus

    # ---- Initial condition -----------------------------------------------
    initial_density: float = 1.0
    """Uniform initial density (ρ₀) for equilibrium initialisation."""
    initial_velocity_x: float = 0.0
    """Uniform initial velocity x-component."""
    initial_velocity_y: float = 0.0
    """Uniform initial velocity y-component."""
    initial_velocity_z: float = 0.0
    """Uniform initial velocity z-component."""

    # ---- Derived ----------------------------------------------------------
    @property
    def omega(self) -> float:
        """BGK relaxation frequency ω = 1 / τ."""
        return 1.0 / self.tau

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity ν in lattice units: (1/3)·(τ - 0.5)."""
        return (1.0 / 3.0) * (self.tau - 0.5)

    @property
    def _periodic_ints(self) -> tuple[int, int, int]:
        """Periodic flags as ints ``(px, py, pz)`` for kernel passing."""
        return (
            int(self.bc_periodic[0]),
            int(self.bc_periodic[1]),
            int(self.bc_periodic[2]),
        )

    # ---- Boundary conditions (per face) -----------------------------------
    # bc_types[FACE] and bc_velocity[FACE] — set after construction.
    # Default: all faces are bounce-back.
    bc_types: tuple[int, int, int, int, int, int] = (
        0, 0, 0, 0, 0, 0,
    )
    """Boundary condition type for each face.
    Indices: 0=xmin, 1=xmax, 2=ymin, 3=ymax, 4=zmin, 5=zmax.
    0 = bounce-back, 1 = Zou-He velocity, 2 = convective outflow,
    3 = periodic (must be paired on both faces of an axis).
    """

    bc_velocity: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    """Prescribed velocity (ux, uy, uz) for Zou-He faces [lattice units]."""

    bc_periodic: tuple[bool, bool, bool] = (False, False, False)
    """Per-axis periodicity flags ``(x, y, z)``.

    When an axis is periodic, both the min and max faces of that axis
    are treated as periodic boundaries: out-of-bounds neighbour lookups
    wrap to the opposite face during streaming and Shan-Chen force
    computation.

    This is independent of ``bc_types`` — periodic streaming is handled
    inside the collide-stream kernel, not by the boundary-condition
    kernel.  Faces whose ``bc_types`` entry is ``BC_PERIODIC`` (3) are
    treated as periodic regardless of this tuple; conversely this tuple
    can enable periodicity without setting every face to type 3.

    Typical use: ``bc_periodic = (True, False, False)`` for a channel
    that is periodic in *x* with walls in *y* and *z*.
    """

    has_moving_walls: bool = False
    """Whether solid boundaries may move (e.g. rigid-body coupling).

    When ``False`` (default), the moving-wall bounce-back correction
    short-circuits to zero and per-step solid-field copies are skipped
    because the fields are static after initialisation.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.tau <= 0.5:
            raise ValueError(
                f"LBM relaxation time tau must be > 0.5 for stability, "
                f"got tau = {self.tau}"
            )
        if self.lambda_trt < 0.0:
            raise ValueError(
                f"TRT magic parameter lambda_trt must be >= 0, "
                f"got lambda_trt = {self.lambda_trt}"
            )
        if self.lambda_trt > 0.0:
            tau_plus = self.lambda_trt / (self.tau - 0.5) + 0.5
            if tau_plus <= 0.5:
                raise ValueError(
                    f"TRT lambda_trt={self.lambda_trt} produces tau_plus={tau_plus:.4f} "
                    f"<= 0.5, which is unstable.  Reduce lambda_trt or increase tau. "
                    f"Valid range: 0 <= lambda_trt <= (tau - 0.5)^2 = {(self.tau - 0.5)**2:.4f}"
                )
        if not (0.0 <= self.omega_reg <= 1.0):
            raise ValueError(
                f"Regularization blend factor omega_reg must be in [0, 1], "
                f"got omega_reg = {self.omega_reg}"
            )
        if self.psi_type not in (0, 1, 2):
            raise ValueError(
                f"Shan-Chen psi_type must be 0 (PSI_RHO), 1 (PSI_EXP), "
                f"or 2 (PSI_CS), got psi_type = {self.psi_type}"
            )
        if self.psi_type == 2:
            if self.cs_a <= 0.0:
                raise ValueError(
                    f"CS EOS attraction parameter cs_a must be > 0, "
                    f"got cs_a = {self.cs_a}"
                )
            if self.cs_b <= 0.0:
                raise ValueError(
                    f"CS EOS covolume parameter cs_b must be > 0, "
                    f"got cs_b = {self.cs_b}"
                )
            if self.cs_T <= 0.0:
                raise ValueError(
                    f"CS EOS temperature cs_T must be > 0, "
                    f"got cs_T = {self.cs_T}"
                )
        if self.psi_ref <= 0.0:
            raise ValueError(
                f"Shan-Chen psi_ref must be > 0, got psi_ref = {self.psi_ref}"
            )
        if self.sc_force_stride < 1:
            raise ValueError(
                f"Shan-Chen sc_force_stride must be >= 1, "
                f"got sc_force_stride = {self.sc_force_stride}"
            )
        if self.sc_homogeneous_rel_tol < 0.0:
            raise ValueError(
                f"Shan-Chen sc_homogeneous_rel_tol must be >= 0, "
                f"got sc_homogeneous_rel_tol = {self.sc_homogeneous_rel_tol}"
            )
        # ---- Periodic boundary validation -----------------------------------
        if len(self.bc_periodic) != 3:
            raise ValueError(
                f"bc_periodic must be a tuple of 3 bools (x, y, z), "
                f"got {self.bc_periodic!r}"
            )
        # Check that bc_types consistent with bc_periodic: if a face is
        # BC_PERIODIC (3), its paired face must also be BC_PERIODIC.
        axis_pairs: tuple[tuple[int, int], ...] = ((0, 1), (2, 3), (4, 5))
        axis_names: tuple[str, ...] = ("x", "y", "z")
        for axis_idx, (f_min, f_max) in enumerate(axis_pairs):
            bc_min: int = int(self.bc_types[f_min])
            bc_max: int = int(self.bc_types[f_max])
            is_periodic: bool = bool(self.bc_periodic[axis_idx])
            min_is_per: bool = (bc_min == 3)
            max_is_per: bool = (bc_max == 3)
            if min_is_per != max_is_per:
                raise ValueError(
                    f"Periodic BC (type 3) must be symmetric on the "
                    f"{axis_names[axis_idx]}-axis: face {f_min} is type "
                    f"{bc_min} but face {f_max} is type {bc_max}."
                )
            if is_periodic and (bc_min == 1 or bc_max == 1):
                raise ValueError(
                    f"Periodic {axis_names[axis_idx]}-axis cannot coexist "
                    f"with Zou-He velocity inlet on the same axis."
                )
