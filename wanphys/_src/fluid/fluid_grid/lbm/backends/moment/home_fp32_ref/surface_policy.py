# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Late-pool free-surface policy for HOME-FREE VOF (outside examples).

Owns the dam-break quiet-pool workflow that used to live only in
``fluid_grid_lbm_dambreak_vof.py``:

1. After the bore settles (``|u|`` small, ``t ≥ arm_after_t``), arm quiet fill.
2. Each armed frame: orphan reabsorb + integer high→low peel (climbers / Δz≥2).
3. When integer surface is flat enough or leveling times out: stop host level,
   top up at most ``max(0, V₀ − Σmass)``.

``home_faithful=True`` disables host leveling / orphan / topup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.benchmark.metrics import (
    collect_surface_height_map,
)

if TYPE_CHECKING:
    from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bridge import (
        HomeFp32VofBridge,
    )
    from wanphys._src.fluid.fluid_grid.lbm.state import LbmState


@dataclass
class LatePoolEvent:
    """One diagnostic event from :meth:`HomeVofLatePoolController.after_frame`."""

    kind: str
    message: str
    payload: dict[str, float | int] = field(default_factory=dict)


@dataclass
class LatePoolFrameStats:
    """Lightweight inventory / motion snapshot for logging."""

    vol_display: float = 0.0
    mass_sum: float = 0.0
    vol0: float = 0.0
    mean_speed: float = 0.0
    liquid_cells: int = 0
    interface_cells: int = 0
    level_flag: str = "-"  # "L" | "T" | "-"
    height_mean: float | None = None
    height_rms: float | None = None
    height_p2p: float | None = None
    height_robust_p2p: float | None = None


class HomeVofLatePoolController:
    """Shared late-pool surface controller for any ``home_fp32`` dam-break case."""

    def __init__(
        self,
        bridge: HomeFp32VofBridge,
        *,
        home_faithful: bool = False,
        arm_after_t: float = 10.0,
        height_every_frames: int = 30,
        level_max_frames: int = 300,
        robust_p2p_stop: float = 1.0,
    ) -> None:
        self.bridge = bridge
        self.home_faithful = bool(home_faithful)
        self.arm_after_t = float(arm_after_t)
        self.height_every_frames = max(1, int(height_every_frames))
        self.level_max_frames = max(1, int(level_max_frames))
        self.robust_p2p_stop = float(robust_p2p_stop)

        self.vol0: float = 0.0
        self.quiet_armed: bool = False
        self.topup_done: bool = False
        self.level_on_frame: int = -1
        self._height_map: Any | None = None
        # Orphan scan is host-side and expensive; after topup only run sparsely.
        self.orphan_every_frames: int = 30
        self.orphan_after_topup_every_frames: int = 120

    @property
    def model(self):
        return self.bridge.model

    def reset(self) -> None:
        self.vol0 = 0.0
        self.quiet_armed = False
        self.topup_done = False
        self.level_on_frame = -1
        self._height_map = None
        self.model.vof_quiet_fill = False

    def set_reference_volume_from_state(self, state: LbmState) -> float:
        """Record ``V₀`` from display inventory ``Σ φ·ρ`` on liquid/interface."""
        phi = state.phi.numpy()
        ctype = state.cell_type.numpy()
        rho = state.density.numpy()
        self.vol0 = float(
            np.nansum(np.where(ctype > 0, phi * np.maximum(rho, 0.0), 0.0))
        )
        return self.vol0

    def measure_mass(self, state: LbmState) -> tuple[float, float]:
        """Return ``(display_Σφρ, GPU_Σmass)`` (GPU falls back to display)."""
        phi = state.phi.numpy()
        ctype = state.cell_type.numpy()
        rho = state.density.numpy()
        vol = float(np.nansum(np.where(ctype > 0, phi * np.maximum(rho, 0.0), 0.0)))
        mass = vol
        g = self.bridge._gpu
        if g is not None:
            mass = float(np.nansum(g.mass.numpy()))
        return vol, mass

    def mean_liquid_speed(self, state: LbmState) -> float:
        ctype = state.cell_type.numpy()
        rho = state.density.numpy()
        wet = (ctype > 0) & np.isfinite(rho)
        if not wet.any():
            return 0.0
        vx = state.velocity_x.numpy()
        vy = state.velocity_y.numpy()
        vz = state.velocity_z.numpy()
        return float(np.sqrt(vx[wet] ** 2 + vy[wet] ** 2 + vz[wet] ** 2).mean())

    def after_frame(
        self,
        state: LbmState,
        *,
        sim_time: float,
        frame_count: int,
        mean_speed: float | None = None,
        log_every: int = 30,
        update_visual: Any | None = None,
    ) -> tuple[list[LatePoolEvent], LatePoolFrameStats]:
        """Run late-pool host ops once per viewer frame (after LBM substeps)."""
        events: list[LatePoolEvent] = []
        did_host = False
        speed = (
            float(mean_speed)
            if mean_speed is not None
            else self.mean_liquid_speed(state)
        )

        # Keep solid SDF coherent for FSI after host edits.
        from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_warp import (
            home_vof_apply_solid_mask_kernel,
            sync_solids_from_lbm_state,
        )

        buf = self.bridge._ensure_gpu()
        sync_solids_from_lbm_state(buf, state)

        # Orphan reabsorb: every N frames while leveling; rare after topup.
        orphan_period = (
            self.orphan_after_topup_every_frames
            if self.topup_done
            else self.orphan_every_frames
        )
        run_orphan = (
            not self.home_faithful
            and self.quiet_armed
            and bool(self.model.vof_orphan_reabsorb)
            and (frame_count % max(1, orphan_period) == 0)
        )
        if run_orphan:
            moved, n_orph = self.bridge.reabsorb_orphans(state)
            did_host = True
            if n_orph > 0 and frame_count % max(1, log_every) == 0:
                events.append(
                    LatePoolEvent(
                        "orphan",
                        f"[orphan] reabsorbed {n_orph} blobs mass={moved:.2f}",
                        {"n_orphans": n_orph, "mass": float(moved)},
                    )
                )

        if (
            not self.home_faithful
            and self.quiet_armed
            and bool(self.model.vof_quiet_fill)
        ):
            # Integer-cell peel for climbers / Δz≥2.
            self.bridge.level_high_to_low(state)
            did_host = True

        # Arm quiet fill when the pool is nearly quiescent.
        if (
            not self.home_faithful
            and not self.quiet_armed
            and sim_time >= self.arm_after_t
            and speed < float(self.model.vof_quiet_fill_u_max)
        ):
            self.model.vof_quiet_fill = True
            self.quiet_armed = True
            self.level_on_frame = int(frame_count)
            events.append(
                LatePoolEvent(
                    "level_on",
                    f"[level ON] t={sim_time:.1f}s |u|={speed:.4f} "
                    f"(host Δz≥2 peel / orphan / topup)",
                    {"sim_time": float(sim_time), "speed": float(speed)},
                )
            )

        vol, mass_sum = self.measure_mass(state)
        ctype = state.cell_type.numpy()
        stats = LatePoolFrameStats(
            vol_display=vol,
            mass_sum=mass_sum,
            vol0=float(self.vol0),
            mean_speed=speed,
            liquid_cells=int((ctype == 2).sum()),
            interface_cells=int((ctype == 1).sum()),
            level_flag=(
                "L"
                if self.model.vof_quiet_fill
                else ("T" if self.topup_done else "-")
            ),
        )

        # Height gate + optional topup (evaluate on height sample frames only).
        if (
            not self.home_faithful
            and sim_time >= self.arm_after_t
            and frame_count % self.height_every_frames == 0
        ):
            phi = state.phi.numpy()
            self._height_map = collect_surface_height_map(phi, ctype)
            hm = self._height_map
            if hm is not None:
                stats.height_mean = float(hm.mean)
                stats.height_rms = float(hm.rms)
                stats.height_p2p = float(hm.p2p)
                stats.height_robust_p2p = float(hm.robust_p2p)
                level_frames = max(0, int(frame_count) - int(self.level_on_frame))
                # Do not arm+stop on the same height sample: give peel a few
                # frames before host peel/topup winds down.
                min_level_frames = max(self.height_every_frames, 1)
                should_stop = self.quiet_armed and level_frames >= min_level_frames and (
                    float(hm.robust_p2p) <= self.robust_p2p_stop + 1.0e-6
                    or level_frames >= self.level_max_frames
                )
                if should_stop:
                    if self.model.vof_quiet_fill:
                        self.model.vof_quiet_fill = False
                        why = (
                            f"z_rob={hm.robust_p2p:.2f}"
                            if float(hm.robust_p2p) <= self.robust_p2p_stop + 1.0e-6
                            else f"timeout frames={level_frames}"
                        )
                        events.append(
                            LatePoolEvent(
                                "level_off",
                                f"[level OFF] t={sim_time:.1f}s {why}",
                                {
                                    "robust_p2p": float(hm.robust_p2p),
                                    "level_frames": int(level_frames),
                                },
                            )
                        )
                    if not self.topup_done:
                        if bool(self.model.vof_orphan_reabsorb):
                            moved, n_orph = self.bridge.reabsorb_orphans(state)
                            did_host = True
                            if n_orph > 0:
                                events.append(
                                    LatePoolEvent(
                                        "orphan",
                                        f"[orphan] pre-topup {n_orph} blobs "
                                        f"mass={moved:.2f}",
                                        {
                                            "n_orphans": n_orph,
                                            "mass": float(moved),
                                        },
                                    )
                                )
                        _, mass_sum = self.measure_mass(state)
                        budget = max(0.0, float(self.vol0) - float(mass_sum))
                        invented = self.bridge.topup_with_budget(budget, state)
                        did_host = True
                        self.topup_done = True
                        _, mass_sum = self.measure_mass(state)
                        stats.mass_sum = mass_sum
                        stats.level_flag = "T"
                        events.append(
                            LatePoolEvent(
                                "topup",
                                f"[topup] invented={invented:.2f} "
                                f"(budget=|Δ|={budget:.2f}) "
                                f"mass={mass_sum:.1f} "
                                f"(Δ→{mass_sum - self.vol0:+.1f})",
                                {
                                    "invented": float(invented),
                                    "budget": float(budget),
                                    "mass": float(mass_sum),
                                },
                            )
                        )
        elif self._height_map is not None:
            hm = self._height_map
            stats.height_mean = float(hm.mean)
            stats.height_rms = float(hm.rms)
            stats.height_p2p = float(hm.p2p)
            stats.height_robust_p2p = float(hm.robust_p2p)

        if did_host:
            nx, ny, nz = buf.shape
            wp.launch(
                home_vof_apply_solid_mask_kernel,
                dim=(nx, ny, nz),
                inputs=[
                    buf.solid_phi,
                    buf.rho,
                    buf.ux,
                    buf.uy,
                    buf.uz,
                    buf.sxx,
                    buf.syy,
                    buf.szz,
                    buf.sxy,
                    buf.sxz,
                    buf.syz,
                    buf.mass,
                    buf.phi,
                    buf.cell_type,
                ],
                device=buf.device,
            )
            self.bridge.sync_to_state(state)
            if update_visual is not None:
                update_visual(state)

        return events, stats
