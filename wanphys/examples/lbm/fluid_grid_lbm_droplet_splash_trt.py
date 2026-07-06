# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""TRT + Shan-Chen pseudopotential droplet impact on a thin liquid film.

Stability-first splash parameters: Re~2100, We~120, h*=0.2.
tau=0.58 (nu=0.027), G=-5.0 (dr~18:1), U=0.55, D=100, pool=20 cells.
The pool is pre-equilibrated under gravity before the droplet is added.

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom
"""

from __future__ import annotations

import sys
import time

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel, LbmState
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer

# ---------------------------------------------------------------------------
N: int = 200
DH: float = 0.02

# ---- Physics: SC pseudopotential (splash-tuned, stability-first) ---------
# Re ~ 2100, We ~ 120 — above splash thresholds (Re>2000, We>100).
# tau=0.58 (nu=0.027): conservative viscosity for multiphase stability.
# G=-5.0: sharp phase separation (dr ~ 18:1), proven stable.
TAU: float = 0.55
LAMBDA_TRT: float = 0.03
G_SC: float = -5.0                   # sharp interface, dr ~ 18:1
PSI_TYPE: int = 1                    # psi = 1 - exp(-rho)
PSI_REF: float = 1.0
SC_BOUNDARY_PSI: float = -1.0        # mirror closure: ψ_wall = ψ_fluid (no artificial force)
GRAVITY: float = -0.01
OMEGA_REG: float = 0.0
KAPPA: float = 0.05                  # placeholder (not wired into solver yet)

# ---- Droplet + thin liquid film -----------------------------------------
# h* = pool / (2*radius) = 20/100 = 0.2 — standard thin-film splash regime.
RHO_LIQUID: float = 1.8
RHO_GAS: float = 0.1
POOL_HEIGHT: int = 20                # 0.2 x D = thin film
DROPLET_RADIUS: int = 40
DROPLET_CX_FRAC: float = 0.50
DROPLET_CY_FRAC: float = 0.50
DROPLET_CZ_FRAC: float = 0.75
DROPLET_VZ_INIT: float = -0.55       # impact velocity for splash
TRANSITION_WIDTH: float = 2.5

SSFR_THRESHOLD: float = 0.7
RAY_MARCH_STEPS: int = 2000

FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 3
POOL_EQUIL_STEPS: int = 800          # long equilibration to damp standing waves
GRAVITY_RAMP_STEPS: int = 200        # smooth gravity ramp to avoid shock


# ---------------------------------------------------------------------------
# Warp kernels -- same as dambreak pattern
# ---------------------------------------------------------------------------


@wp.kernel
def _init_pool(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    pool_h: int,
    rho_high: float,
    rho_low: float,
    transition_w: float,
    seed: int,
    nx: int, ny: int, nz: int, stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    pool_w = 0.5 + 0.5 * wp.tanh((float(pool_h) - float(k)) / transition_w)
    rho = rho_low + (rho_high - rho_low) * pool_w
    n = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001)
    n = n * wp.cos(float(i * 419 + j * 233 + k * 577 + seed) * 0.0013)
    rho = wp.max(rho + n * 0.005 * rho, 0.01)
    density[i, j, k] = rho
    f[0 * stride + idx] = (1.0 / 3.0) * rho
    f[1 * stride + idx] = (1.0 / 18.0) * rho; f[2 * stride + idx] = (1.0 / 18.0) * rho
    f[3 * stride + idx] = (1.0 / 18.0) * rho; f[4 * stride + idx] = (1.0 / 18.0) * rho
    f[5 * stride + idx] = (1.0 / 18.0) * rho; f[6 * stride + idx] = (1.0 / 18.0) * rho
    f[7 * stride + idx] = (1.0 / 36.0) * rho; f[8 * stride + idx] = (1.0 / 36.0) * rho
    f[9 * stride + idx] = (1.0 / 36.0) * rho; f[10 * stride + idx] = (1.0 / 36.0) * rho
    f[11 * stride + idx] = (1.0 / 36.0) * rho; f[12 * stride + idx] = (1.0 / 36.0) * rho
    f[13 * stride + idx] = (1.0 / 36.0) * rho; f[14 * stride + idx] = (1.0 / 36.0) * rho
    f[15 * stride + idx] = (1.0 / 36.0) * rho; f[16 * stride + idx] = (1.0 / 36.0) * rho
    f[17 * stride + idx] = (1.0 / 36.0) * rho; f[18 * stride + idx] = (1.0 / 36.0) * rho


@wp.kernel
def _add_droplet(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    cx: int, cy: int, cz: int, droplet_r: int,
    rho_high: float, rho_low: float,
    transition_w: float, vz_init: float, seed: int,
    nx: int, ny: int, nz: int, stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    dx = float(i) - float(cx); dy = float(j) - float(cy); dz = float(k) - float(cz)
    dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
    drop_w = 0.5 + 0.5 * wp.tanh((float(droplet_r) - dist) / transition_w)
    rho_drop = rho_low + (rho_high - rho_low) * drop_w
    rho_cur = density[i, j, k]
    if rho_drop <= rho_cur:
        return
    n = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001)
    n = n * wp.cos(float(i * 419 + j * 233 + k * 577 + seed) * 0.0013)
    rho = wp.max(rho_drop + n * 0.005 * rho_drop, 0.01)
    density[i, j, k] = rho
    vx = 0.0; vy = 0.0; vz = vz_init; usq = vz * vz
    f[0 * stride + idx] = (1.0 / 3.0) * rho * (1.0 - 1.5 * usq)
    f[1 * stride + idx] = (1.0 / 18.0) * rho * (1.0 - 1.5 * usq)
    f[2 * stride + idx] = (1.0 / 18.0) * rho * (1.0 - 1.5 * usq)
    f[3 * stride + idx] = (1.0 / 18.0) * rho * (1.0 - 1.5 * usq)
    f[4 * stride + idx] = (1.0 / 18.0) * rho * (1.0 - 1.5 * usq)
    f[5 * stride + idx] = (1.0 / 18.0) * rho * (1.0 + 3.0 * vz + 4.5 * vz * vz - 1.5 * usq)
    f[6 * stride + idx] = (1.0 / 18.0) * rho * (1.0 - 3.0 * vz + 4.5 * vz * vz - 1.5 * usq)
    f[7 * stride + idx] = (1.0 / 36.0) * rho * (1.0 - 1.5 * usq)
    f[8 * stride + idx] = (1.0 / 36.0) * rho * (1.0 - 1.5 * usq)
    f[9 * stride + idx] = (1.0 / 36.0) * rho * (1.0 - 1.5 * usq)
    f[10 * stride + idx] = (1.0 / 36.0) * rho * (1.0 - 1.5 * usq)
    f[11 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * vz + 3.0 * vz * vz)
    f[12 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * vz + 3.0 * vz * vz)
    f[13 * stride + idx] = (1.0 / 36.0) * rho * (1.0 - 3.0 * vz + 3.0 * vz * vz)
    f[14 * stride + idx] = (1.0 / 36.0) * rho * (1.0 - 3.0 * vz + 3.0 * vz * vz)
    f[15 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * vz + 3.0 * vz * vz)
    f[16 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * vz + 3.0 * vz * vz)
    f[17 * stride + idx] = (1.0 / 36.0) * rho * (1.0 - 3.0 * vz + 3.0 * vz * vz)
    f[18 * stride + idx] = (1.0 / 36.0) * rho * (1.0 - 3.0 * vz + 3.0 * vz * vz)


class TrtDropletSplash:
    """Droplet impact with TRT + Shan-Chen pseudopotential (splash-tuned).

    tau=0.58, G=-5.0, psi_type=1 (exp): Re~2100, We~120 — above splash thresholds.
    h* = pool / D = 0.2 — standard thin-film splash regime.
    """

    def __init__(self, viewer: FluidViewerGL) -> None:
        self.viewer: FluidViewerGL = viewer
        viewer._paused = True

        self.model: LbmModel = LbmModel(
            fluid_grid_res=(N, N, N), fluid_grid_cell_size=DH,
            tau=TAU, G=G_SC, psi_type=PSI_TYPE, psi_ref=PSI_REF,
            sc_boundary_psi=SC_BOUNDARY_PSI,
            lambda_trt=LAMBDA_TRT, use_regularization=True, omega_reg=OMEGA_REG,
            gravity_x=0.0, gravity_y=0.0, gravity_z=GRAVITY,
        )
        n: int = int(self.model.nx)
        nu: float = (TAU - 0.5) / 3.0
        D: int = 2 * DROPLET_RADIUS
        u_imp: float = abs(DROPLET_VZ_INIT)
        print(f"TRT Splash: {n}^3, tau={TAU} (nu={nu:.4f}), G={G_SC}")
        print(f"  Re={u_imp * D / nu:.0f}  h*={POOL_HEIGHT / D:.2f}  r={DROPLET_RADIUS}  vz={DROPLET_VZ_INIT}")
        print(f"  Re={u_imp * D / nu:.0f}  h*={POOL_HEIGHT / D:.2f}  r={DROPLET_RADIUS}  vz={DROPLET_VZ_INIT}")

        self.domain: LbmDomain = LbmDomain(self.model)
        self.domain.create_state()
        self.sim_dt: float = FRAME_DT / SIM_SUBSTEPS
        self.sim_time: float = 0.0

        self._init_pool_only()
        self._equilibrate_pool()
        self._add_droplet_on_pool()

        self.ssfr: ScreenSpaceFluidRenderer = ScreenSpaceFluidRenderer(
            viewer=viewer, max_particles=1, particle_radius=0.01,
            device=self.model._device)
        viewer.register_post_render_callback(lambda v: self.ssfr.render(v))
        self.frame_count: int = 0; self._last_ms: float = 0.0
        print("Controls: [Space] unpause  [R] reset  [mouse] orbit  [scroll] zoom")

    def _init_pool_only(self) -> None:
        state = self.domain.state; n = int(self.model.nx); stride = n * n * n
        wp.launch(_init_pool, dim=(n, n, n),
                  inputs=[state.f, state.density, POOL_HEIGHT, RHO_LIQUID, RHO_GAS,
                          TRANSITION_WIDTH, 42, n, n, n, stride])
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)

    def _equilibrate_pool(self) -> None:
        """Smoothly ramp gravity and let the pool settle to hydrostatic equilibrium.

        A long ramp + settling phase is essential to prevent standing waves
        from forming between the bounce-back bottom wall and the free surface.
        With short equilibration, pressure waves reflect off the bottom and
        create regular grid-scale protrusions on the liquid surface.
        """
        target_gz: float = float(self.model.gravity_z)
        self.model.gravity_z = 0.0
        ramp: int = min(GRAVITY_RAMP_STEPS, POOL_EQUIL_STEPS // 3)

        print(f"  [Equil] ramping gravity over {ramp} steps...")
        for s in range(ramp):
            self.model.gravity_z = target_gz * float(s + 1) / float(ramp)
            self.domain.step(self.sim_dt)

        self.model.gravity_z = target_gz
        settle: int = POOL_EQUIL_STEPS - ramp
        print(f"  [Equil] settling over {settle} steps at full gravity...")
        for s in range(settle):
            self.domain.step(self.sim_dt)
            # Periodic surface check to monitor convergence
            if (s + 1) % 200 == 0:
                wp.synchronize_device(self.model._device)
                rho_np = self.domain.state.density.numpy()
                liq: np.ndarray = rho_np > SSFR_THRESHOLD
                zf: np.ndarray = liq.mean(axis=(0, 1))
                sz: int = int(np.argmax(zf < 0.5)) if np.any(zf < 0.5) else 0
                print(f"    step {s + 1}: surface_z={sz}, rho=[{rho_np.min():.3f},{rho_np.max():.3f}]")

        wp.synchronize_device(self.model._device)
        rho_np = self.domain.state.density.numpy()
        liq = rho_np > SSFR_THRESHOLD
        zf = liq.mean(axis=(0, 1))
        sz = int(np.argmax(zf < 0.5)) if np.any(zf < 0.5) else 0

        # Check for standing waves: compute xy-plane density std at each z
        rho_std_z: np.ndarray = rho_np.std(axis=(0, 1))
        max_std: float = float(rho_std_z.max())
        print(f"  [Equil] surface_z={sz}, liquid={liq.sum()}, "
              f"rho=[{rho_np.min():.3f},{rho_np.max():.3f}], "
              f"max_xy_std={max_std:.4f}")
        if max_std > 0.05:
            print(f"  [Equil] WARNING: large horizontal density variation ({max_std:.4f}) "
                  f"— standing waves may persist. Consider more equilibration steps.")

    def _add_droplet_on_pool(self) -> None:
        state = self.domain.state; n = int(self.model.nx); stride = n * n * n
        cx = int(n * DROPLET_CX_FRAC); cy = int(n * DROPLET_CY_FRAC); cz = int(n * DROPLET_CZ_FRAC)
        wp.launch(_add_droplet, dim=(n, n, n),
                  inputs=[state.f, state.density, cx, cy, cz, DROPLET_RADIUS,
                          RHO_LIQUID, RHO_GAS, TRANSITION_WIDTH, DROPLET_VZ_INIT, 43, n, n, n, stride])
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)

    def step(self) -> None:
        t0 = time.perf_counter()
        for _ in range(SIM_SUBSTEPS):
            self.domain.step(self.sim_dt)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT; self.frame_count += 1
        if self.frame_count % 30 == 0:
            rho_np = self.domain.state.density.numpy()
            if np.any(~np.isfinite(rho_np)):
                print(f"[t={self.sim_time:.1f}s] *** DIVERGED ***", file=sys.stderr, flush=True)
                self.viewer._paused = True; return
            liq = rho_np > SSFR_THRESHOLD; lc = int(liq.sum())
            pool = int((rho_np[:, :, :POOL_HEIGHT] > SSFR_THRESHOLD).sum())
            splash = int((rho_np[:, :, POOL_HEIGHT:] > SSFR_THRESHOLD).sum())
            if liq.any():
                com = np.argwhere(liq).mean(axis=0)
                vx = float(self.domain.state.velocity_x.numpy()[liq].mean())
                vy = float(self.domain.state.velocity_y.numpy()[liq].mean())
                vz = float(self.domain.state.velocity_z.numpy()[liq].mean())
                vm = float(np.sqrt(self.domain.state.velocity_x.numpy()[liq]**2
                       + self.domain.state.velocity_y.numpy()[liq]**2
                       + self.domain.state.velocity_z.numpy()[liq]**2).max())
                print(f"[t={self.sim_time:.1f}s] liquid={lc} (pool={pool}, splash={splash}) "
                      f"COM=({com[0]:.0f},{com[1]:.0f},{com[2]:.0f}) v=({vx:+.3f},{vy:+.3f},{vz:+.3f}) "
                      f"v_max={vm:.3f} sim={self._last_ms:.0f}ms", file=sys.stderr, flush=True)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self.ssfr.available:
            self.ssfr.set_density_field(density=self.domain.state.density,
                grid_origin=(0.0, 0.0, 0.0), cell_size=DH,
                threshold=SSFR_THRESHOLD, max_steps=RAY_MARCH_STEPS)
        self.viewer.end_frame()


def main() -> None:
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
    viewer, args = init_fluid_viewer()
    newton.examples.run(TrtDropletSplash(viewer), args)


if __name__ == "__main__":
    main()
