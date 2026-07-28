# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Example-only φ-volume Archimedes buoyancy plugin (no push / drag).

Opt-in alternative to ``_home_vof_empirical_sphere_fsi`` for demos that want
buoyancy closer to displaced-liquid volume without shell chase / linear drag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.phi_volume_buoyancy_warp import (
    apply_phi_volume_buoyancy_gpu,
    ensure_phi_volume_scratch,
    fibonacci_shell_offsets,
)


@dataclass
class PhiVolumeBuoyancyConfig:
    buoyancy_scale: float = 1.0
    ema_alpha: float = 0.08
    dsub_cap: float = 0.04
    phi_wet: float = 0.05
    n_dirs: int = 48
    radii: tuple[float, ...] = (1.05, 1.15)
    offsets: tuple[tuple[float, float, float], ...] | None = None


@dataclass
class PhiVolumeBuoyancyPlugin:
    device: str
    body_ids: tuple[int, ...]
    radius: float
    volume: float
    rho_liquid: float
    gravity_abs: float
    dh: float
    nx: int
    ny: int
    nz: int
    config: PhiVolumeBuoyancyConfig = field(default_factory=PhiVolumeBuoyancyConfig)
    scratch: dict | None = None

    def _offsets(self) -> tuple[tuple[float, float, float], ...]:
        if self.config.offsets is not None:
            return self.config.offsets
        return fibonacci_shell_offsets(
            int(self.config.n_dirs), radii=tuple(self.config.radii)
        )

    def ensure_scratch(self) -> dict:
        self.scratch = ensure_phi_volume_scratch(
            device=self.device,
            offsets_xyz=self._offsets(),
            body_ids=self.body_ids,
            scratch=self.scratch,
        )
        return self.scratch

    def apply(
        self,
        *,
        phi,
        cell,
        solid,
        body_q,
        body_f_apply,
        sync_submerged: bool = False,
    ) -> dict[int, float]:
        scratch = self.ensure_scratch()
        return apply_phi_volume_buoyancy_gpu(
            phi=phi,
            cell=cell,
            solid=solid,
            body_q=body_q,
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
            phi_wet=float(self.config.phi_wet),
            ema_alpha=float(self.config.ema_alpha),
            dsub_cap=float(self.config.dsub_cap),
            sync_submerged=bool(sync_submerged),
        )
