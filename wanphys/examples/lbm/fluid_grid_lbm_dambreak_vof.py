# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""TRT FSLBM / VOF sharp free-surface dam-break (no Shan-Chen).

Default: distribution LBM (``lbm_backend=dist``). Optional moment HOME-FREE (GPU):

    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof \\
        --backend home --n 48

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
    collect_surface_height_map,
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
    ):
        self.viewer = viewer
        viewer._paused = True
        self._backend = "home_fp32" if backend in ("home", "home_fp32") else "dist"
        self._n = int(n)
        self._home_faithful = bool(home_faithful) and self._backend == "home_fp32"
        # Lattice |g| must scale with column height H~n/2. Fixed gz across N
        # makes Fr∝√(gH) blow up: n=128 @ gz=-0.002 → ρ_max≳1.3, wall foam,
        # and visible mass drain. Keep g·H ≈ const vs n_ref=48.
        n_ref = 48
        if self._backend == "dist":
            gravity = GRAVITY
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
            gravity = -0.0020 * (float(n_ref) / float(self._n))
            tau = 0.51
            # Home GPU def_6_sigma≈0.024 → γ≈0.004; baseline wanphys uses 1.5e-3.
            gamma = 4.0e-3 if self._home_faithful else 1.5e-3
            self._substeps = max(SIM_SUBSTEPS, 12)
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
            lambda_trt=LAMBDA_TRT,
            initial_density=RHO_LIQUID,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=gravity,
        )
        print(
            f"VOF Dam-Break: {self._n}^3, lattice={self.model.lattice}, "
            f"backend={self.model.lbm_backend}, tau={self.model.tau}, "
            f"gz={gravity}, gamma={gamma}, substeps={self._substeps}, "
            f"phase_mode=vof_sharp, "
            f"collide={'NOCM-HOME' if self._backend == 'home_fp32' else 'TRT/dist'}, "
            f"wall_wetting={self.model.vof_wall_wetting}, "
            f"film_drain={self.model.vof_wall_film_drain} "
            f"(edge_only={self.model.vof_wall_film_edge_only}), "
            f"home_faithful={self._home_faithful} "
            f"(fill_empty={home_fill_empty}, wall_eq={home_wall_eq}, seal={seal_fg})"
        )

        self.domain = LbmDomain(self.model)
        self.domain.create_state()
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

        self.ssfr = ScreenSpaceFluidRenderer(
            viewer=viewer,
            max_particles=1,
            particle_radius=0.01,
            device=self.model._device,
        )
        viewer.register_post_render_callback(lambda v: self.ssfr.render(v))
        self.frame_count = 0
        self._last_ms = 0.0
        self._height_map = None  # SurfaceHeightMap after t >= HEIGHT_MAP_START_T
        self._quiet_fill_armed = False
        self._topup_done = False
        self._level_on_frame = -1
        # ~5s of leveling at 60fps logging / 0.5s steps? frame_count advances
        # once per FRAME_DT; with FRAME_DT=1/60 and log every 30 → use 300 frames ≈5s.
        self._level_max_frames = 300
        print("Controls: [Space] unpause  [R] reset  [mouse] orbit")

    def step(self):
        t0 = time.perf_counter()
        for _ in range(self._substeps):
            self.domain.step(self.sim_dt)
        # High→low leveling: once per frame after substeps (never from below).
        home = self.domain.solver._home_fp32
        if (
            home is not None
            and self._quiet_fill_armed
            and bool(self.model.vof_quiet_fill)
        ):
            home.level_high_to_low(self.domain.state)
            self.domain.solver._vof_sharp.update_visual_field(
                self.domain.state, self._n, self._n, self._n
            )
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
            mass_est = np.where(ctype > 0, phi * np.maximum(rho, 0.0), 0.0)
            vol = float(np.nansum(mass_est))
            mass_sum = vol
            if home is not None and home._gpu is not None:
                mass_sum = float(np.nansum(home._gpu.mass.numpy()))
            liquid = ctype == 2
            interface = ctype == 1
            wet = (ctype > 0) & np.isfinite(rho)
            rho_finite = rho[np.isfinite(rho)]
            vx_f = state.velocity_x.numpy()
            vy_f = state.velocity_y.numpy()
            vz_f = state.velocity_z.numpy()
            vx = float(vx_f[wet].mean()) if wet.any() else 0.0
            vz = float(vz_f[wet].mean()) if wet.any() else 0.0
            speed = (
                float(np.sqrt(vx_f[wet] ** 2 + vy_f[wet] ** 2 + vz_f[wet] ** 2).mean())
                if wet.any()
                else 0.0
            )
            rho_max = float(rho_finite.max()) if rho_finite.size else float("nan")
            z_top = np.zeros(ctype.shape[:2], dtype=np.int32)
            for k in range(ctype.shape[2]):
                wet_k = (ctype[:, :, k] > 0) & (phi[:, :, k] > 0.5)
                z_top[wet_k] = k
            h_a = float(z_top[-2, :].mean()) if z_top.size else 0.0
            h_b = float(z_top[1, :].mean()) if z_top.size else 0.0
            nxy = z_top.shape[0]
            corn_max = int(
                max(
                    z_top[:2, :2].max(),
                    z_top[:2, -2:].max(),
                    z_top[-2:, :2].max(),
                    z_top[-2:, -2:].max(),
                )
            ) if nxy >= 2 else 0
            # Near-quiescent: arm high→low leveling (peel tops, raise lows).
            # --home-faithful: no host leveling/topup; watch pure FSLBM only.
            if (
                self._backend == "home_fp32"
                and not self._home_faithful
                and self.sim_time >= HEIGHT_MAP_START_T
                and speed < float(self.model.vof_quiet_fill_u_max)
                and not self._quiet_fill_armed
            ):
                self.model.vof_quiet_fill = True
                self._quiet_fill_armed = True
                self._level_on_frame = self.frame_count
                print(
                    f"  [level ON] t={self.sim_time:.1f}s |u|={speed:.4f} "
                    f"(pool band high→low; climb peeled separately)",
                    file=sys.stderr,
                    flush=True,
                )
            kappa = None
            vof = self.domain.solver._vof_sharp
            if float(self.model.vof_gamma) > 0.0:
                kappa = vof._kappa.numpy()
            rough = collect_interface_roughness(phi, ctype, kappa=kappa)
            height_note = ""
            if self.sim_time >= HEIGHT_MAP_START_T:
                if self.frame_count % HEIGHT_MAP_EVERY_FRAMES == 0:
                    self._height_map = collect_surface_height_map(phi, ctype)
                hm = self._height_map
                if hm is not None:
                    height_note = (
                        f" z̄={hm.mean:.2f} z_rms={hm.rms:.3f} "
                        f"z_p2p={hm.p2p:.2f} z_rob={hm.robust_p2p:.2f}"
                    )
                    # Stop on robust p2p (p95−p05), not full max−min — climb
                    # spikes inflate z_p2p forever on finer grids (n=96).
                    # Also bail if leveling ran too long without converging.
                    level_frames = max(0, self.frame_count - self._level_on_frame)
                    should_stop = (
                        hm.robust_p2p <= 1.0 + 1e-6
                        or level_frames >= self._level_max_frames
                    )
                    if self._quiet_fill_armed and should_stop:
                        if self.model.vof_quiet_fill:
                            self.model.vof_quiet_fill = False
                            why = (
                                f"z_rob={hm.robust_p2p:.2f}"
                                if hm.robust_p2p <= 1.0 + 1e-6
                                else f"timeout frames={level_frames}"
                            )
                            print(
                                f"  [level OFF] t={self.sim_time:.1f}s "
                                f"{why} (O(1) leftover / stop gate)",
                                file=sys.stderr,
                                flush=True,
                            )
                        if home is not None and not self._topup_done:
                            # Over-budget: fill every low column to target height.
                            invented = home.topup_with_budget(
                                float("inf"), self.domain.state
                            )
                            self.domain.solver._vof_sharp.update_visual_field(
                                self.domain.state, self._n, self._n, self._n
                            )
                            self._topup_done = True
                            if home._gpu is not None:
                                mass_sum = float(np.nansum(home._gpu.mass.numpy()))
                            print(
                                f"  [topup] invented={invented:.2f} "
                                f"(unbounded) "
                                f"mass={mass_sum:.1f} (Δ→{mass_sum-self._vol0:+.1f})",
                                file=sys.stderr,
                                flush=True,
                            )
            fill_flag = "L" if self.model.vof_quiet_fill else ("T" if self._topup_done else "-")
            print(
                f"[t={self.sim_time:.1f}s] L={liquid.sum()} I={interface.sum()} "
                f"vol={vol:.1f} (Δ={vol-self._vol0:+.1f}) mass={mass_sum:.1f} "
                f"rho_max={rho_max:.3f} "
                f"v=({vx:+.4f},{vz:+.4f}) |u|={speed:.4f} "
                f"hA={h_a:.1f} hB={h_b:.1f} corn={corn_max} "
                f"h_rms={rough.height_rms:.3f} h_p2p={rough.height_p2p:.2f} "
                f"κ_rms={rough.kappa_rms:.3f}{height_note} "
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
    pre_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    viewer, args = init_fluid_viewer()
    newton.examples.run(
        VofDamBreak(
            viewer,
            backend=pre_args.backend,
            n=pre_args.n,
            home_faithful=pre_args.home_faithful,
        ),
        args,
    )


if __name__ == "__main__":
    main()
