# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Showcase-only empirical sphere FSI (buoyancy / push / drag / EMA).

**Not** part of the HOME-FREE VOF solver or ``GridLbmRigidCoupling`` core path
(raster → Eq.24 walls → reconstructed-link ME → rigid). Enable only via demo
flags such as ``--showcase-fsi`` / ``--empirical-fsi``. Never wire this into
coupling or the HOME bridge defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.sphere_buoyancy_warp import (
    apply_sphere_buoyancy_forces_gpu,
    ensure_buoyancy_scratch,
)

# Sample just outside the sphere: interior cells are forced to gas by FSI mask.
DEFAULT_SHELL_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (1.25, 0.0, 0.0),
    (-1.25, 0.0, 0.0),
    (0.0, 1.25, 0.0),
    (0.0, -1.25, 0.0),
    (0.0, 0.0, 1.25),
    (0.0, 0.0, -1.25),
    (0.9, 0.9, 0.0),
    (0.9, -0.9, 0.0),
    (-0.9, 0.9, 0.0),
    (-0.9, -0.9, 0.0),
    (0.9, 0.0, 0.9),
    (0.9, 0.0, -0.9),
    (-0.9, 0.0, 0.9),
    (-0.9, 0.0, -0.9),
    (0.0, 0.9, 0.9),
    (0.0, 0.9, -0.9),
    (0.0, -0.9, 0.9),
    (0.0, -0.9, -0.9),
)


@dataclass
class EmpiricalSphereFsiConfig:
    """Tunables for shell-sample buoyancy + fluid chase + linear drag."""

    buoyancy_scale: float = 1.0
    push_rate: float = 8.0
    drag_xy: float = 4.0
    drag_z: float = 12.0
    late_pool_push_scale: float = 0.12
    ema_alpha: float = 0.05
    dsub_cap: float = 0.015
    phi_wet: float = 0.25
    offsets: tuple[tuple[float, float, float], ...] = field(
        default_factory=lambda: DEFAULT_SHELL_OFFSETS
    )


@dataclass
class EmpiricalSphereFsiPlugin:
    """Stateful wrapper: scratch buffers + apply each substep."""

    device: str
    body_ids: tuple[int, ...]
    densities: tuple[float, ...]
    radius: float
    volume: float
    rho_liquid: float
    gravity_abs: float
    dh: float
    nx: int
    ny: int
    nz: int
    config: EmpiricalSphereFsiConfig = field(default_factory=EmpiricalSphereFsiConfig)
    scratch: dict | None = None

    def ensure_scratch(self) -> dict:
        self.scratch = ensure_buoyancy_scratch(
            device=self.device,
            offsets_xyz=self.config.offsets,
            body_ids=self.body_ids,
            densities=self.densities,
            scratch=self.scratch,
        )
        return self.scratch

    def apply(
        self,
        *,
        phi,
        cell,
        solid,
        ux,
        uy,
        uz,
        body_q,
        body_qd,
        body_f_apply,
        vel_scale: float,
        late_pool_armed: bool = False,
        sync_submerged: bool = False,
    ) -> dict[int, float]:
        """Atomic-add empirical forces onto ``body_f`` via ``body_f_apply``."""
        scratch = self.ensure_scratch()
        push = float(self.config.push_rate)
        if late_pool_armed:
            push *= float(self.config.late_pool_push_scale)
        return apply_sphere_buoyancy_forces_gpu(
            phi=phi,
            cell=cell,
            solid=solid,
            ux=ux,
            uy=uy,
            uz=uz,
            body_q=body_q,
            body_qd=body_qd,
            body_f_apply=body_f_apply,
            radius=float(self.radius),
            dh=float(self.dh),
            nx=int(self.nx),
            ny=int(self.ny),
            nz=int(self.nz),
            scratch=scratch,
            volume=float(self.volume),
            rho_liquid=float(self.rho_liquid),
            gravity_abs=float(self.gravity_abs),
            buoyancy_scale=float(self.config.buoyancy_scale),
            push_rate=push,
            drag_xy=float(self.config.drag_xy),
            drag_z=float(self.config.drag_z),
            vel_scale=float(vel_scale),
            phi_wet=float(self.config.phi_wet),
            ema_alpha=float(self.config.ema_alpha),
            dsub_cap=float(self.config.dsub_cap),
            sync_submerged=bool(sync_submerged),
        )
