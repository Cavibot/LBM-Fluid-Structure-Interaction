# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Buoyancy as a **state force** from hydrostatic pressure (not ME, not ρVg).

With uniform liquid density the continuum definition is

    p = ρ · |g| · depth_below_free_surface
    F = ∮ −p n dA     (wet surface; atmospheric gauge)

Displaced-volume ``ρ V g s`` is only the closed-form integral of that pressure
field for a fully/partially submerged body. This module integrates pressure
on surface samples so the force comes from **压差**, not an explicit drain
volume. Link ME (Eq.32) stays for hydrodynamic impact under Guo + ρ≈1.

Typical use (after fluid + ME, before rigid step)::

    buoy = ArchimedesBuoyancy(device=..., body_ids=(...), radius=R)
    buoy.apply(
        phi=state.phi, cell=state.cell_type, solid=state.solid_phi,
        body_q=rigid.body_q, body_f_apply=rigid.apply_body_forces,
        dh=dh, grid_shape=(nx, ny, nz),
        rho_liquid=1.0, gravity_abs=abs(g_rigid_z),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.pressure_buoyancy_warp import (
    apply_pressure_buoyancy_gpu,
    ensure_pressure_buoyancy_scratch,
    fibonacci_sphere_dirs,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.phi_volume_buoyancy_warp import (
    apply_phi_volume_buoyancy_gpu,
    ensure_phi_volume_scratch,
    fibonacci_shell_offsets,
)


Method = Literal["pressure", "volume"]


@dataclass
class ArchimedesBuoyancyConfig:
    """Tunables for buoyancy state force."""

    method: Method = "pressure"
    """``pressure`` = ∮ −p n dA (default); ``volume`` = legacy ρ V g s."""
    scale: float = 1.0
    """Overall multiplier (1.0 = physical)."""
    phi_wet: float = 0.05
    """Minimum φ counted as wet."""
    ema_alpha: float = 0.08
    """EMA on submerged diagnostic (0 = raw)."""
    dsub_cap: float = 0.04
    """Max |Δs| per apply when EMA is active."""
    n_samples: int = 96
    """Fibonacci surface samples (pressure) or per-shell count (volume)."""
    sample_radius_scale: float = 1.02
    """Sample slightly outside SDF (pressure method)."""
    shell_radii: tuple[float, ...] = (1.08, 1.18)
    """Legacy volume-method shell radii in units of R."""


@dataclass
class ArchimedesBuoyancyResult:
    """Diagnostics from the last ``apply``."""

    submerged: dict[int, float] = field(default_factory=dict)
    forces: dict[int, tuple[float, float, float]] = field(default_factory=dict)


class ArchimedesBuoyancy:
    """Reusable buoyancy applicator for equal-radius sphere bodies.

    Default: hydrostatic pressure surface integral. Owns GPU scratch.
    """

    def __init__(
        self,
        *,
        device: wp.context.Device | str,
        body_ids: tuple[int, ...] | list[int],
        radius: float,
        volume: float | None = None,
        config: ArchimedesBuoyancyConfig | None = None,
    ) -> None:
        import math

        ids = tuple(int(b) for b in body_ids)
        if not ids:
            raise ValueError("body_ids must be non-empty")
        r = float(radius)
        if r <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.device = device
        self.body_ids = ids
        self.radius = r
        self.volume = (
            float(volume)
            if volume is not None
            else (4.0 / 3.0) * math.pi * r * r * r
        )
        self.config = config if config is not None else ArchimedesBuoyancyConfig()
        self._dirs = fibonacci_sphere_dirs(int(self.config.n_samples))
        self._vol_offsets = fibonacci_shell_offsets(
            int(self.config.n_samples),
            radii=tuple(float(x) for x in self.config.shell_radii),
        )
        self._scratch: dict[str, Any] | None = None
        self.last_result = ArchimedesBuoyancyResult()

    def apply(
        self,
        *,
        phi: wp.array,
        cell: wp.array,
        solid: wp.array,
        body_q: wp.array,
        body_f_apply: Any,
        dh: float,
        grid_shape: tuple[int, int, int],
        rho_liquid: float,
        gravity_abs: float,
        scale: float | None = None,
        sync_diagnostics: bool = True,
    ) -> ArchimedesBuoyancyResult:
        """Accumulate buoyancy onto ``body_f_apply``; return diagnostics."""
        nx, ny, nz = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
        cfg = self.config
        buoy_scale = float(cfg.scale if scale is None else scale)
        g_abs = abs(float(gravity_abs))

        if cfg.method == "volume":
            self._scratch = ensure_phi_volume_scratch(
                device=self.device,
                offsets_xyz=self._vol_offsets,
                body_ids=self.body_ids,
                scratch=self._scratch if self._scratch and "offsets" in self._scratch else None,
            )
            apply_phi_volume_buoyancy_gpu(
                phi=phi,
                cell=cell,
                solid=solid,
                body_q=body_q,
                body_f_apply=body_f_apply,
                radius=self.radius,
                dh=float(dh),
                nx=nx,
                ny=ny,
                nz=nz,
                scratch=self._scratch,
                volume=self.volume,
                rho_liquid=float(rho_liquid),
                gravity_abs=g_abs,
                buoyancy_scale=buoy_scale,
                phi_wet=float(cfg.phi_wet),
                ema_alpha=float(cfg.ema_alpha),
                dsub_cap=float(cfg.dsub_cap),
                sync_submerged=False,
            )
        else:
            self._scratch = ensure_pressure_buoyancy_scratch(
                device=self.device,
                dirs_xyz=self._dirs,
                body_ids=self.body_ids,
                nx=nx,
                ny=ny,
                nz=nz,
                scratch=self._scratch if self._scratch and "dirs" in self._scratch else None,
            )
            apply_pressure_buoyancy_gpu(
                phi=phi,
                cell=cell,
                solid=solid,
                body_q=body_q,
                body_f_apply=body_f_apply,
                radius=self.radius,
                dh=float(dh),
                nx=nx,
                ny=ny,
                nz=nz,
                scratch=self._scratch,
                rho_liquid=float(rho_liquid),
                gravity_abs=g_abs,
                buoyancy_scale=buoy_scale,
                phi_wet=float(cfg.phi_wet),
                ema_alpha=float(cfg.ema_alpha),
                dsub_cap=float(cfg.dsub_cap),
                sample_radius_scale=float(cfg.sample_radius_scale),
                sync_submerged=False,
            )

        result = ArchimedesBuoyancyResult()
        if sync_diagnostics and self._scratch is not None:
            result = self.read_diagnostics()
        else:
            self.last_result = result
        return result if sync_diagnostics else self.last_result

    def read_diagnostics(self) -> ArchimedesBuoyancyResult:
        """Host read of last GPU submerged / force buffers (no apply)."""
        scratch = self._scratch
        if scratch is None:
            return ArchimedesBuoyancyResult()
        sub = scratch["submerged"].numpy()
        forces = scratch["forces"].numpy()
        ids = scratch["body_ids_host"]
        result = ArchimedesBuoyancyResult()
        for i, body_id in enumerate(ids):
            bid = int(body_id)
            result.submerged[bid] = float(sub[i])
            f = forces[i]
            result.forces[bid] = (float(f[0]), float(f[1]), float(f[2]))
        self.last_result = result
        return result


def apply_archimedes_buoyancy(
    *,
    buoy: ArchimedesBuoyancy,
    phi: wp.array,
    cell: wp.array,
    solid: wp.array,
    body_q: wp.array,
    body_f_apply: Any,
    dh: float,
    grid_shape: tuple[int, int, int],
    rho_liquid: float,
    gravity_abs: float,
    scale: float | None = None,
    sync_diagnostics: bool = True,
) -> ArchimedesBuoyancyResult:
    """Functional wrapper around :meth:`ArchimedesBuoyancy.apply`."""
    return buoy.apply(
        phi=phi,
        cell=cell,
        solid=solid,
        body_q=body_q,
        body_f_apply=body_f_apply,
        dh=dh,
        grid_shape=grid_shape,
        rho_liquid=rho_liquid,
        gravity_abs=gravity_abs,
        scale=scale,
        sync_diagnostics=sync_diagnostics,
    )


__all__ = [
    "ArchimedesBuoyancy",
    "ArchimedesBuoyancyConfig",
    "ArchimedesBuoyancyResult",
    "apply_archimedes_buoyancy",
    "fibonacci_sphere_dirs",
    "fibonacci_shell_offsets",
]
