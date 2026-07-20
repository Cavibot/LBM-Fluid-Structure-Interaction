# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""TRT FSLBM / VOF sharp free-surface dam-break (no Shan-Chen).

Default: distribution LBM (``lbm_backend=dist``). Optional moment HOME-FREE (GPU):

    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof \\
        --backend home --n 48

Default home path: pure HOME-FREE VOF (no host ``level ON``).
Opt-in: ``--height-eq`` (solver free-surface IF leveling; gradual),
``--late-pool``.

Closer to Home-FSLBM fill/empty + walls (no host leveling/topup):

    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof \\
        --backend home --n 96 --home-faithful

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.benchmark.metrics import (
    collect_interface_roughness,
)
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer

# ---------------------------------------------------------------------------
N: int = 64
DH: float = 0.02
LATTICE: str = "D3Q27"
BACKEND: str = "dist"  # "dist" | "home"

# Stability-first dam-break (paper Fig.8: BGK/TRT FSLBM is fragile).
# lambda_trt: D3Q19-tuned; keep mild until D3Q27 retuning.
TAU: float = 0.58
LAMBDA_TRT: float = 0.015
GRAVITY: float = -0.0012

DAM_X_FRAC: float = 0.25
FILL_Z_FRAC: float = 0.5
RHO_LIQUID: float = 1.0
VOF_RHO_GAS: float = 1.0
VOF_EPSILON: float = 1.0e-3
VOF_GAMMA: float = 1.5e-3
VOF_KAPPA_SMOOTH: int = 2

SSFR_THRESHOLD: float = 0.2
RAY_MARCH_STEPS: int = 800

FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 8

# Free-surface height map (per-column max z) starts after this sim time.
HEIGHT_MAP_START_T: float = 10.0
HEIGHT_MAP_EVERY_FRAMES: int = 30


class VofDamBreak:
    def __init__(
        self,
        viewer: FluidViewerGL,
        *,
        backend: str = BACKEND,
        n: int = N,
        home_faithful: bool = False,
        bubble_pressure: bool = False,
        g_scale: float = 1.0,
        enable_late_pool: bool = False,
        enable_height_eq: bool = False,
    ):
        self.viewer = viewer
        viewer._paused = True
        self._backend = "home_fp32" if backend in ("home", "home_fp32") else "dist"
        self._n = int(n)
        self._home_faithful = bool(home_faithful) and self._backend == "home_fp32"
        self._bubble_pressure = bool(bubble_pressure) and self._backend == "home_fp32"
        self._g_scale = max(float(g_scale), 1.0e-6)
        # Late-pool host (level ON / orphan / topup) is opt-in — default is
        # pure solver (no host wipe).
        self._disable_late_pool = (not bool(enable_late_pool)) or self._home_faithful
        self._enable_height_eq = bool(enable_height_eq) and not self._home_faithful
        self._height_eq_armed = False
        self._height_eq_arm_after_t = 8.0
        self._height_eq_calls = 0
        self._last_height_eq: dict | None = None
        # Lattice |g| must scale with column height H~n/2. Fixed gz across N
        # makes Fr∝√(gH) blow up: n=128 @ gz=-0.002 → ρ_max≳1.3, wall foam,
        # and visible mass drain. Keep g·H ≈ const vs n_ref=48.
        n_ref = 48
        if self._backend == "dist":
            gravity = GRAVITY * self._g_scale
            tau = TAU
            gamma = VOF_GAMMA
            self._substeps = SIM_SUBSTEPS
            wall_wetting = 0.0
            wall_film = False
            quiet_fill = False
            home_fill_empty = False
            home_wall_eq = False
            seal_fg = True
        else:
            # Conservative late-pool params. g∝1/n keeps Fr similar across N.
            # Baseline late-pool params. Strong wall κ wetting peels liquid off
            # the walls into a frustum (四棱台); keep vof_wall_wetting=0.
            # See docs/wanphys/lbm_home_fslbm_one_cell_limit_zh.md.
            gravity = -0.0020 * (float(n_ref) / float(self._n)) * self._g_scale
            tau = 0.51
            # Home GPU def_6_sigma≈0.024 → γ≈0.004; baseline wanphys uses 1.5e-3.
            gamma = 4.0e-3 if self._home_faithful else 1.5e-3
            # Stronger g → more violent bore; give a few extra substeps.
            self._substeps = max(SIM_SUBSTEPS, 12 if self._g_scale <= 1.5 else 16)
            wall_wetting = 0.0
            wall_film = False
            quiet_fill = False  # armed later when |u| is small (unless faithful)
            home_fill_empty = self._home_faithful
            home_wall_eq = self._home_faithful
            seal_fg = not self._home_faithful

        self.model = LbmModel(
            fluid_grid_res=(self._n, self._n, self._n),
            fluid_grid_cell_size=DH,
            lattice=LATTICE,
            tau=tau,
            G=0.0,
            phase_mode="vof_sharp",
            lbm_backend=self._backend,
            vof_rho_gas=VOF_RHO_GAS,
            vof_epsilon=VOF_EPSILON,
            vof_gamma=gamma,
            vof_kappa_smooth=VOF_KAPPA_SMOOTH,
            vof_wall_wetting=wall_wetting if self._backend == "home_fp32" else 0.0,
            vof_wall_film_drain=wall_film if self._backend == "home_fp32" else False,
            vof_wall_film_phi_max=0.95,
            vof_wall_film_u_max=0.02,
            vof_wall_film_edge_only=True,
            vof_home_fill_empty=home_fill_empty,
            vof_home_wall_eq=home_wall_eq,
            vof_seal_fg=seal_fg,
            vof_quiet_fill=False,
            vof_quiet_fill_rate=0.35,
            vof_quiet_fill_u_max=0.025,
            vof_orphan_reabsorb=False,
            vof_orphan_max_cells=max(96, self._n),
            vof_orphan_height_margin=3,
            vof_height_eq=False,
            vof_height_eq_rate=0.08,
            vof_height_eq_u_max=0.04,
            vof_height_eq_dh_cap=0.08,
            vof_height_eq_every=12,
            vof_bubble_pressure=self._bubble_pressure,
            # Disjoint only by default with --bubble-pressure.
            # Eddy viscosity is very costly (6³ scan/cell) and over-damps the pool.
            vof_bubble_disjoint=self._bubble_pressure,
            vof_bubble_small_sigma=False,
            vof_bubble_eddy=False,
            lambda_trt=LAMBDA_TRT,
            initial_density=RHO_LIQUID,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=gravity,
        )
        print(
            f"VOF Dam-Break: {self._n}^3, lattice={self.model.lattice}, "
            f"backend={self.model.lbm_backend}, tau={self.model.tau}, "
            f"gz={gravity} (g_scale={self._g_scale}), gamma={gamma}, substeps={self._substeps}, "
            f"phase_mode=vof_sharp, "
            f"collide={'NOCM-HOME' if self._backend == 'home_fp32' else 'TRT/dist'}, "
            f"wall_wetting={self.model.vof_wall_wetting}, "
            f"film_drain={self.model.vof_wall_film_drain} "
            f"(edge_only={self.model.vof_wall_film_edge_only}), "
            f"home_faithful={self._home_faithful} "
            f"(fill_empty={home_fill_empty}, wall_eq={home_wall_eq}, seal={seal_fg}), "
            f"bubble_pressure={self._bubble_pressure}, "
            f"height_eq={self._enable_height_eq} "
            f"(arm_t>={self._height_eq_arm_after_t}), "
            f"late_pool={not self._disable_late_pool}"
        )

        self.domain = LbmDomain(self.model)
        self.domain.create_state()
        self._late_pool = None
        if self.domain.solver._home_fp32 is not None and not self._disable_late_pool:
            self._late_pool = self.domain.solver._home_fp32.configure_late_pool(
                home_faithful=self._home_faithful,
                arm_after_t=HEIGHT_MAP_START_T,
                height_every_frames=HEIGHT_MAP_EVERY_FRAMES,
                level_max_frames=300,
            )
        self.sim_dt = FRAME_DT / max(self._substeps, 1)
        self.sim_time = 0.0

        dam_x = int(self._n * DAM_X_FRAC)
        fill_z = int(self._n * FILL_Z_FRAC)
        state = self.domain.state
        if self.domain.solver._home_fp32 is not None:
            self.domain.solver._home_fp32.seed_dam_break(
                state, dam_x=dam_x, fill_z=fill_z, rho_liquid=RHO_LIQUID
            )
            out = self.domain._state_out
            self.domain.solver._home_fp32.sync_to_state(out)
        else:
            self.domain.solver._vof_sharp.seed_dam_break_column(
                state, dam_x=dam_x, fill_z=fill_z, rho_liquid=RHO_LIQUID
            )
            out = self.domain._state_out
            for name in (
                "f",
                "density",
                "phi",
                "cell_type",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "pressure",
                "solid_phi",
                "solid_body_id",
            ):
                wp.copy(getattr(out, name), getattr(state, name))
        self.domain.solver._vof_sharp.update_visual_field(
            state, self._n, self._n, self._n
        )
        wp.synchronize_device(self.model._device)

        phi = state.phi.numpy()
        ctype = state.cell_type.numpy()
        rho = state.density.numpy()
        mass0 = float(np.nansum(np.where(ctype > 0, phi * np.maximum(rho, 0.0), 0.0)))
        print(
            f"  liquid={int((ctype == 2).sum())}  interface={int((ctype == 1).sum())}  "
            f"gas={int((ctype == 0).sum())}  vol0={mass0:.1f}"
        )
        self._vol0 = mass0

        target_gz = float(self.model.gravity_z)
        self.model.gravity_z = 0.0
        # Longer ramp at fine grids: same lattice steps of ramp were too abrupt
        # when the column is taller in cells.
        ramp = 40 if self._n <= 64 else max(40, self._n // 2)
        for s in range(ramp):
            self.model.gravity_z = target_gz * float(s + 1) / float(ramp)
            self.domain.step(self.sim_dt)
        self.model.gravity_z = target_gz
        wp.synchronize_device(self.model._device)
        st = self.domain.state
        phi = st.phi.numpy()
        ctype = st.cell_type.numpy()
        rho = st.density.numpy()
        self._vol0 = float(
            np.nansum(np.where(ctype > 0, phi * np.maximum(rho, 0.0), 0.0))
        )
        if self._late_pool is not None:
            self._late_pool.vol0 = float(self._vol0)

        self.ssfr = ScreenSpaceFluidRenderer(
            viewer=viewer,
            max_particles=1,
            particle_radius=0.01,
            device=self.model._device,
        )
        viewer.register_post_render_callback(lambda v: self.ssfr.render(v))
        self.frame_count = 0
        self._last_ms = 0.0
        print("Controls: [Space] unpause  [R] reset  [mouse] orbit")

    def step(self):
        t0 = time.perf_counter()
        home = self.domain.solver._home_fp32
        # Arm solver free-surface leveling after splash (flag on LbmModel;
        # operator runs inside HomeFp32VofBridge.step each lattice step).
        if (
            self._enable_height_eq
            and home is not None
            and self.sim_time >= self._height_eq_arm_after_t
            and not self._height_eq_armed
        ):
            self.model.vof_height_eq = True
            self._height_eq_armed = True
            print(
                f"[height-eq ON] t={self.sim_time:.1f}s "
                f"solver IF-φ level α={self.model.vof_height_eq_rate} "
                f"|Δφ|≤{self.model.vof_height_eq_dh_cap} "
                f"every={self.model.vof_height_eq_every} "
                f"(interface cells only, no bulk rewrite)",
                file=sys.stderr,
                flush=True,
            )
        for _ in range(self._substeps):
            self.domain.step(self.sim_dt)
        if home is not None and self.model.vof_height_eq:
            self._last_height_eq = dict(getattr(home, "_last_height_eq_stats", {}) or {})
            self.domain.solver._vof_sharp.update_visual_field(
                self.domain.state, self._n, self._n, self._n
            )
        events = []
        pool_stats = None
        if self._late_pool is not None:
            events, pool_stats = self._late_pool.after_frame(
                self.domain.state,
                sim_time=self.sim_time + FRAME_DT,
                frame_count=self.frame_count + 1,
                update_visual=lambda st: self.domain.solver._vof_sharp.update_visual_field(
                    st, self._n, self._n, self._n
                ),
            )
            for ev in events:
                print(ev.message, file=sys.stderr, flush=True)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT
        self.frame_count += 1
        # Track Σφρ (display) and GPU Σmass (true VOF inventory).
        if self.frame_count % 30 == 0:
            state = self.domain.state
            phi = state.phi.numpy()
            ctype = state.cell_type.numpy()
            rho = state.density.numpy()
            if pool_stats is not None:
                vol = pool_stats.vol_display
                mass_sum = pool_stats.mass_sum
                speed = pool_stats.mean_speed
                liquid = pool_stats.liquid_cells
                interface = pool_stats.interface_cells
                fill_flag = pool_stats.level_flag
            else:
                mass_est = np.where(ctype > 0, phi * np.maximum(rho, 0.0), 0.0)
                vol = float(np.nansum(mass_est))
                mass_sum = vol
                liquid = int((ctype == 2).sum())
                interface = int((ctype == 1).sum())
                wet = (ctype > 0) & np.isfinite(rho)
                vx_f = state.velocity_x.numpy()
                vy_f = state.velocity_y.numpy()
                vz_f = state.velocity_z.numpy()
                speed = (
                    float(np.sqrt(vx_f[wet] ** 2 + vy_f[wet] ** 2 + vz_f[wet] ** 2).mean())
                    if wet.any()
                    else 0.0
                )
                fill_flag = "-"
            wet = (ctype > 0) & np.isfinite(rho)
            rho_finite = rho[np.isfinite(rho)]
            vx_f = state.velocity_x.numpy()
            vz_f = state.velocity_z.numpy()
            vx = float(vx_f[wet].mean()) if wet.any() else 0.0
            vz = float(vz_f[wet].mean()) if wet.any() else 0.0
            rho_max = float(rho_finite.max()) if rho_finite.size else float("nan")
            bub_note = ""
            if home is not None and bool(self.model.vof_bubble_pressure):
                bs = home.last_bubble_stats()
                bub_note = (
                    f" bub={bs.get('n_bubbles', 0)}"
                    f"(trap={bs.get('n_trapped', 0)}"
                    f",ρmax={float(bs.get('rho_max_bubble', 1.0)):.3f}"
                    f",ccl={int(bs.get('did_ccl', 0))})"
                )
            # Continuous free-surface height h=k+φ (NOT integer z with φ>0.5).
            from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.height_eq import (
                continuous_surface_height,
            )

            h_cont = continuous_surface_height(phi, ctype)
            mid = h_cont.shape[0] // 2
            left = np.isfinite(h_cont) & (np.arange(h_cont.shape[0])[:, None] < mid)
            right = np.isfinite(h_cont) & (np.arange(h_cont.shape[0])[:, None] >= mid)
            h_a = float(np.nanmean(h_cont[left])) if left.any() else 0.0
            h_b = float(np.nanmean(h_cont[right])) if right.any() else 0.0
            finite = np.isfinite(h_cont)
            corn_vals = []
            if finite.any():
                nxy = h_cont.shape[0]
                for sl in (
                    h_cont[:2, :2],
                    h_cont[:2, -2:],
                    h_cont[-2:, :2],
                    h_cont[-2:, -2:],
                ):
                    if np.isfinite(sl).any():
                        corn_vals.append(float(np.nanmax(sl)))
            corn_max = int(max(corn_vals)) if corn_vals else 0
            kappa = None
            vof = self.domain.solver._vof_sharp
            if float(self.model.vof_gamma) > 0.0:
                kappa = vof._kappa.numpy()
            rough = collect_interface_roughness(phi, ctype, kappa=kappa)
            # Continuous height map stats (Σφ thickness already in rough;
            # z̄ here = mean of k+φ).
            if finite.any():
                hc = h_cont[finite]
                height_note = (
                    f" h̄={float(hc.mean()):.3f} "
                    f"h_std={float(hc.std()):.3f} "
                    f"h_p2p*={float(hc.max() - hc.min()):.2f}"
                )
            else:
                height_note = ""
            heq_note = ""
            if self._last_height_eq:
                heq_note = (
                    f" H*={self._last_height_eq.get('H_star', 0):.3f}"
                    f" φ*={self._last_height_eq.get('phi_star', 0):.3f}"
                    f" Δm_heq={self._last_height_eq.get('mass_delta', 0):+.2f}"
                )
            print(
                f"[t={self.sim_time:.1f}s] L={liquid} I={interface} "
                f"vol={vol:.1f} (Δ={vol-self._vol0:+.1f}) mass={mass_sum:.1f} "
                f"rho_max={rho_max:.3f}{bub_note} "
                f"v=({vx:+.4f},{vz:+.4f}) |u|={speed:.4f} "
                f"hA={h_a:.3f} hB={h_b:.3f} corn={corn_max} "
                f"h_rms={rough.height_rms:.3f} h_p2p={rough.height_p2p:.2f} "
                f"κ_rms={rough.kappa_rms:.3f}{height_note}{heq_note} "
                f"lvl={fill_flag} sim={self._last_ms:.0f}ms "
                f"backend={self.model.lbm_backend}",
                file=sys.stderr,
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.ssfr.available:
            self.ssfr.set_density_field(
                density=self.domain.solver._vof_sharp.visual_field,
                grid_origin=(0, 0, 0),
                cell_size=DH,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()


def main():
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=("dist", "home", "home_fp32"), default=BACKEND)
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument(
        "--home-faithful",
        action="store_true",
        help=(
            "Closer Home-FSLBM dynamics: soft TYPE_NO_G/F fill-empty "
            "(mass-gated), wall f^eq, stronger γ, no F–G seal / quiet level / topup."
        ),
    )
    parser.add_argument(
        "--bubble-pressure",
        action="store_true",
        help=(
            "Enable §4.2 trapped-gas pressure (host CCL + ρ=V₀/V on G∪I). "
            "Default off; headspace touching top stays ρ=1."
        ),
    )
    parser.add_argument(
        "--g-scale",
        type=float,
        default=1.0,
        help="Multiply baseline lattice gravity (home: |gz|≈0.002·48/n). Try 2–4 to probe one-cell pinning.",
    )
    parser.add_argument(
        "--late-pool",
        action="store_true",
        help="Opt-in host quiet-level / orphan / topup (default off).",
    )
    parser.add_argument(
        "--height-eq",
        action="store_true",
        help=(
            "Enable solver IF-φ leveling (existing interface cells only). "
            "Arms after t=8. Omit to roll back."
        ),
    )
    parser.add_argument(
        "--disable-late-pool",
        action="store_true",
        help=argparse.SUPPRESS,  # kept for old scripts; now the default
    )
    pre_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    viewer, args = init_fluid_viewer()
    newton.examples.run(
        VofDamBreak(
            viewer,
            backend=pre_args.backend,
            n=pre_args.n,
            home_faithful=pre_args.home_faithful,
            bubble_pressure=pre_args.bubble_pressure,
            g_scale=float(pre_args.g_scale),
            enable_late_pool=bool(pre_args.late_pool) and not bool(pre_args.disable_late_pool),
            enable_height_eq=bool(pre_args.height_eq),
        ),
        args,
    )


if __name__ == "__main__":
    main()
