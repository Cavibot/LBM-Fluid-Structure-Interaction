# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Lattice Boltzmann method – static model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..base import FluidGridModelBase
from .core.lattice import LatticeSpec, get_lattice_spec


@dataclass
class LbmModel(FluidGridModelBase):
    """Static configuration for a distribution LBM fluid simulation.

    Inherits grid properties (``fluid_grid_res``, ``fluid_grid_cell_size``,
    ``device``) from :class:`FluidGridModelBase`.  LBM-specific parameters
    control the BGK/TRT collision operator and external body forces.

    Notes
    -----
    The LBM timestep is **always 1 in lattice units**.  The physical
    timestep ``dt`` passed to :meth:`LbmSolver.step` is accepted for API
    compatibility but ignored internally.  Physical-to-lattice conversion
    uses the cell size ``dh`` (see :meth:`cell_size`).

    Kinematic viscosity (lattice units):
        ``nu = c_s² · (τ - 0.5) = (1/3) · (τ - 0.5)``.

    D3Q27 currently supports ``phase_mode`` ``none`` / ``vof_sharp`` with
    bounce-back, Zou-He, convective outflow, and periodic faces.
    Shan-Chen force and regularization remain D3Q19-only until ported.
    """

    # ---- Lattice discretization ------------------------------------------
    lattice: str = "D3Q19"
    """Velocity lattice name: ``D3Q19`` | ``D3Q27``."""

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

    # ---- Phase interface mode --------------------------------------------
    phase_mode: str = "none"
    """Interface model: ``none`` | ``shan_chen`` | ``vof_sharp``.

    - ``none``: single-phase (optionally with Guo gravity when ``G == 0``).
    - ``shan_chen``: diffuse-interface SC (requires ``G != 0``).
    - ``vof_sharp``: FSLBM-style free-surface VOF (``G`` must be 0).
    """

    # ---- VOF sharp free-surface (FSLBM) ----------------------------------
    vof_epsilon: float = 1.0e-4
    """φ fill/empty threshold for L/I/G reclassification."""
    vof_rho_gas: float = 1.0
    """Atmospheric gas density ρ_atm = p_g / c_s² in free-surface BC."""
    vof_gamma: float = 0.0
    """Surface tension γ. With PLIC curvature κ: ρ_g = ρ_atm - 6 γ κ.
    ``0`` disables curvature (constant ρ_g = vof_rho_gas)."""
    vof_kappa_smooth: int = 2
    """Jacobi smoothing passes on PLIC κ before applying surface tension.
    ``0`` uses raw κ. 1–3 typically damp lattice-scale surface noise."""
    vof_home_fs_filter: bool = True
    """HOME-FREE Eq. 11 with filtered ``\\bar f_ī`` (3rd-order Hermite, Eq. 16).

    When True, unknown gas→interface populations use
    ``f_i* = f_i^eq(ρ_g,u) + f_ī^eq(ρ_g,u) − \\bar f_ī`` where ``\\bar f_ī``
    is reconstructed from the interface cell moments (ρ,u,S).
    When False, classic FSLBM uses the raw opposite post-collide ``f_ī``.
    """
    vof_wall_wetting: float = 0.0
    """HOME vertical-wall Young bias added to PLIC κ on wall *interface* cells.

    ``0`` disables (recommended). Negative (e.g. ``-0.3``) → mild hydrophobic.
    Strong values or inset/corner amplification peel the pool off the walls
    into a frustum (四棱台) free surface — avoid for dam-break visuals.
    """
    vof_wall_film_drain: bool = False
    """HOME mass-conserving subgrid drain of stagnant vertical-wall films.

    Moves thin dry-inward vertical-wall rim mass into ``massex`` for the next
    fused gather (surface3 inventory rule). Never evaporates. Gated by
    ``vof_wall_film_u_max``. Prefer ``vof_wall_wetting`` first; keep film off
    unless rim coats remain.
    """
    vof_wall_film_phi_max: float = 0.35
    """Drain candidates with φ below this (also: dry-inward rim coats)."""
    vof_wall_film_u_max: float = 0.015
    """Only drain wall films with |u| below this (near-quiescent gate)."""
    vof_wall_film_edge_only: bool = True
    """If True, film drain acts only on vertical domain edges/corners (2 walls).

    Face-only films are left to wetting; this targets the tall corner spikes
    that inflate ``z_p2p`` without stripping the whole wall meniscus.
    """
    vof_home_fill_empty: bool = False
    """Softer Home-FSLBM fill/empty (TechRep-05-4 style mass gates).

    Fill: ``mass>ρ`` or (no gas neighbor and ``mass>0.99ρ``).
    Empty: ``mass<0`` or (no fluid neighbor and ``mass<0.1ρ``).
    Bare TYPE_NO_G/F without mass gates evaporates crests and thrash-cuts the pool.
    """
    vof_home_wall_eq: bool = False
    """Wall pull uses Home ``f^eq(ρ, u_wall)`` instead of HOME Eq.24 stress retain."""
    vof_seal_fg: bool = True
    """After surface_3, face-only liquid–gas seal → interface. Home relies on
    surface_1/2 only; set False for closer Home topology."""
    vof_quiet_fill: bool = False
    """Late-time high→low free-surface leveling (host, once per frame).

    When armed, peels mass from columns with ``z_top ≥ median+1`` and deposits
    onto lower face-neighbor columns. Does not drain pool bottom.
    """
    vof_quiet_fill_rate: float = 0.2
    """Fraction of donor *top-cell* mass moved per leveling pass."""
    vof_quiet_fill_u_max: float = 0.025
    """Arm leveling only when mean |u| is below this (example gate)."""
    vof_orphan_reabsorb: bool = True
    """When quiet-level is armed, reabsorb disconnected airborne liquid blobs."""
    vof_orphan_max_cells: int = 96
    """Orphan components with ≤ this many wet cells are folded into the pool."""
    vof_orphan_height_margin: int = 3
    """Also fold components whose min_z exceeds main-pool median + this margin."""

    # ---- Solver backend -------------------------------------------------
    lbm_backend: str = "dist"
    """Fluid advance backend: ``dist`` (distribution LBM) | ``home_fp32``.

    ``home_fp32`` is the moment-encoded HOME-FREE VOF path (H4/H5, no quant).
    Requires ``phase_mode='vof_sharp'``. Visualisation still uses ``LbmState``
    macros (ρ,u,φ); distribution ``f`` is unused.
    """

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

    def __post_init__(self) -> None:
        super().__post_init__()
        # Resolve lattice early so downstream state allocation validates.
        self._lattice_spec: LatticeSpec = get_lattice_spec(self.lattice)
        object.__setattr__(self, "lattice", self._lattice_spec.name)
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
        mode = str(self.phase_mode).lower()
        if mode not in ("none", "shan_chen", "vof_sharp"):
            raise ValueError(
                f"phase_mode must be 'none', 'shan_chen', or 'vof_sharp', "
                f"got {self.phase_mode!r}"
            )
        object.__setattr__(self, "phase_mode", mode)
        backend = str(self.lbm_backend).lower()
        if backend not in ("dist", "home_fp32"):
            raise ValueError(
                f"lbm_backend must be 'dist' or 'home_fp32', got {self.lbm_backend!r}"
            )
        object.__setattr__(self, "lbm_backend", backend)
        if backend == "home_fp32" and mode != "vof_sharp":
            raise ValueError(
                "lbm_backend='home_fp32' currently requires phase_mode='vof_sharp'"
            )
        if mode == "shan_chen" and float(self.G) == 0.0:
            raise ValueError("phase_mode='shan_chen' requires G != 0")
        if mode == "vof_sharp" and float(self.G) != 0.0:
            raise ValueError("phase_mode='vof_sharp' requires G == 0 (no Shan-Chen)")
        if self._lattice_spec.name == "D3Q27":
            if mode == "shan_chen":
                raise ValueError(
                    "lattice='D3Q27' does not yet support phase_mode='shan_chen' "
                    "(Shan-Chen force kernel is D3Q19-only)"
                )
            if self.use_regularization and self.omega_reg > 0.0:
                raise ValueError(
                    "lattice='D3Q27' does not yet support regularization "
                    "(reg_trt_kernel is D3Q19-only); set use_regularization=False"
                )
        if self.vof_epsilon <= 0.0:
            raise ValueError(f"vof_epsilon must be > 0, got {self.vof_epsilon}")
        if self.vof_rho_gas <= 0.0:
            raise ValueError(f"vof_rho_gas must be > 0, got {self.vof_rho_gas}")
        if self.vof_gamma < 0.0:
            raise ValueError(f"vof_gamma must be >= 0, got {self.vof_gamma}")
        if int(self.vof_kappa_smooth) < 0:
            raise ValueError(
                f"vof_kappa_smooth must be >= 0, got {self.vof_kappa_smooth}"
            )
        if float(self.vof_wall_film_phi_max) <= 0.0:
            raise ValueError(
                f"vof_wall_film_phi_max must be > 0, got {self.vof_wall_film_phi_max}"
            )
        if float(self.vof_wall_film_u_max) <= 0.0:
            raise ValueError(
                f"vof_wall_film_u_max must be > 0, got {self.vof_wall_film_u_max}"
            )
        if self.psi_type not in (0, 1):
            raise ValueError(
                f"Shan-Chen psi_type must be 0 (PSI_RHO) or 1 (PSI_EXP), "
                f"got psi_type = {self.psi_type}"
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

    @property
    def lattice_spec(self) -> LatticeSpec:
        """Resolved :class:`LatticeSpec` for this model."""
        return self._lattice_spec

    @property
    def num_dirs(self) -> int:
        """Number of discrete velocities (distribution functions per node)."""
        return int(self._lattice_spec.num_dirs)
