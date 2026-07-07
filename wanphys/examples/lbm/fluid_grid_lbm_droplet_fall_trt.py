# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""TRT + Shan-Chen (PSI_EXP) droplet free-fall onto a liquid pool.

A spherical liquid droplet is released from height above a pre-equilibrated
pool.  Gravity accelerates the droplet downward until it impacts the liquid
surface, producing crown splash, crater formation, and eventual coalescence.

Uses the standard exponential pseudopotential (PSI_EXP) with the TRT
collision operator and regularisation.  Proven stable parameter set.

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
# Grid
# ---------------------------------------------------------------------------
N: int = 128
DH: float = 0.02

# ---------------------------------------------------------------------------
# Physics: TRT + PSI_EXP (standard, proven stable)
# ---------------------------------------------------------------------------
TAU: float = 0.55
LAMBDA_TRT: float = 0.03
G_SC: float = -5.0
PSI_TYPE: int = 1                    # PSI_EXP: ψ = 1 - exp(-ρ)
PSI_REF: float = 1.0
SC_BOUNDARY_PSI: float = 0.1         # hydrophobic wall
GRAVITY: float = -0.005
OMEGA_REG: float = 0.0

# ---------------------------------------------------------------------------
# Droplet + pool geometry
# ---------------------------------------------------------------------------
RHO_LIQUID: float = 1.8
RHO_GAS: float = 0.1
POOL_HEIGHT: int = 30
DROPLET_RADIUS: int = 30
DROPLET_CX_FRAC: float = 0.50
DROPLET_CY_FRAC: float = 0.50
DROPLET_CZ_FRAC: float = 0.72
DROPLET_VZ_INIT: float = 0.0         # gravity-driven
TRANSITION_WIDTH: float = 2.5

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
SSFR_THRESHOLD: float = 0.7
RAY_MARCH_STEPS: int = 600           # with step_fraction=0.25, ~500 steps cover the diagonal
RAY_MARCH_STEP_FRACTION: float = 0.25  # cell_diag multiplier (was 0.15; 0.25 is ~1.7x faster)

# ---------------------------------------------------------------------------
# Time-stepping
# ---------------------------------------------------------------------------
FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 2
POOL_EQUIL_STEPS: int = 800
GRAVITY_RAMP_STEPS: int = 200


# ---------------------------------------------------------------------------
# Warp kernels
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
    h = (i * 1664525 + j * 1013904223 + k * 22695477 + seed * 1103515245) & 0x7FFFFFFF
    n = (float(h) / 2147483648.0) - 1.0
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
    h = (i * 1664525 + j * 1013904223 + k * 22695477 + seed * 1103515245) & 0x7FFFFFFF
    n = (float(h) / 2147483648.0) - 1.0
    rho = wp.max(rho_drop + n * 0.005 * rho_drop, 0.01)
    density[i, j, k] = rho
    vz = vz_init; usq = vz * vz
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


class TrtDropletFall:
    """Droplet free-fall onto a liquid pool with TRT + PSI_EXP.

    A spherical droplet (r=30) is released above a pre-equilibrated pool.
    Gravity-driven impact produces crown splash and crater formation.
    """

    def __init__(self, viewer: FluidViewerGL) -> None:
        self.viewer: FluidViewerGL = viewer
        viewer._paused = True

        self.model: LbmModel = LbmModel(
            fluid_grid_res=(N, N, N),
            fluid_grid_cell_size=DH,
            tau=TAU,
            G=G_SC,
            psi_type=PSI_TYPE,
            psi_ref=PSI_REF,
            sc_boundary_psi=SC_BOUNDARY_PSI,
            lambda_trt=LAMBDA_TRT,
            use_regularization=True,
            omega_reg=OMEGA_REG,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=GRAVITY,
            bc_periodic=(True, True, False),
        )
        n: int = int(self.model.nx)
        nu: float = (TAU - 0.5) / 3.0
        D: int = 2 * DROPLET_RADIUS
        fall_dist_cells: float = float(int(n * DROPLET_CZ_FRAC) - POOL_HEIGHT - DROPLET_RADIUS)
        v_impact_est: float = float(np.sqrt(2.0 * abs(GRAVITY) * max(fall_dist_cells, 1.0)))

        print(f"TRT Droplet Fall (PSI_EXP): {n}^3, tau={TAU} (nu={nu:.4f}), G={G_SC}")
        print(f"  tau_plus={self.model.tau_plus:.3f}, tau_minus={self.model.tau_minus:.3f}")
        print(f"  omega_reg={OMEGA_REG}, use_regularization=True")
        print(f"  rho_l={RHO_LIQUID}, rho_g={RHO_GAS}, gz={GRAVITY}")
        print(f"  Pool: {POOL_HEIGHT} cells  Droplet: r={DROPLET_RADIUS}  "
              f"fall_dist~{fall_dist_cells:.0f} cells  v_impact~{v_impact_est:.2f}")

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
        self.frame_count: int = 0; self._last_ms: float = 0.0; self._last_render_ms: float = 0.0
        self._step_times: list[float] = []; self._render_times: list[float] = []
        print("Controls: [Space] unpause  [R] reset  [mouse] orbit  [scroll] zoom")

    def _init_pool_only(self) -> None:
        state: LbmState = self.domain.state; n: int = int(self.model.nx); stride: int = n * n * n
        wp.launch(_init_pool, dim=(n, n, n),
                  inputs=[state.f, state.density, POOL_HEIGHT, RHO_LIQUID, RHO_GAS,
                          TRANSITION_WIDTH, 42, n, n, n, stride])
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)

    def _equilibrate_pool(self) -> None:
        target_gz: float = float(self.model.gravity_z)
        self.model.gravity_z = 0.0
        ramp: int = GRAVITY_RAMP_STEPS
        for s in range(ramp):
            self.model.gravity_z = target_gz * float(s + 1) / float(ramp)
            self.domain.step(self.sim_dt)
        self.model.gravity_z = target_gz
        settle: int = POOL_EQUIL_STEPS - ramp
        for _ in range(settle):
            self.domain.step(self.sim_dt)
        wp.synchronize_device(self.model._device)

    def _add_droplet_on_pool(self) -> None:
        state: LbmState = self.domain.state; n: int = int(self.model.nx); stride: int = n * n * n
        cx: int = int(n * DROPLET_CX_FRAC); cy: int = int(n * DROPLET_CY_FRAC)
        cz: int = int(n * DROPLET_CZ_FRAC)
        wp.launch(_add_droplet, dim=(n, n, n),
                  inputs=[state.f, state.density, cx, cy, cz, DROPLET_RADIUS,
                          RHO_LIQUID, RHO_GAS, TRANSITION_WIDTH, DROPLET_VZ_INIT, 43,
                          n, n, n, stride])
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)

    def step(self) -> None:
        t0: float = time.perf_counter()
        for _ in range(SIM_SUBSTEPS):
            self.domain.step(self.sim_dt)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self._step_times.append(self._last_ms)
        self.sim_time += FRAME_DT; self.frame_count += 1
        if self.frame_count % 30 == 0:
            rho_np: np.ndarray = self.domain.state.density.numpy()
            if np.any(~np.isfinite(rho_np)):
                print(f"[t={self.sim_time:.1f}s] *** DIVERGED ***", file=sys.stderr, flush=True)
                self.viewer._paused = True; return
            liq: np.ndarray = rho_np > SSFR_THRESHOLD
            # rolling averages
            avg_step: float = sum(self._step_times[-30:]) / max(len(self._step_times[-30:]), 1) if self._step_times else self._last_ms
            avg_render: float = sum(self._render_times[-30:]) / max(len(self._render_times[-30:]), 1) if self._render_times else self._last_render_ms
            if liq.any():
                com: np.ndarray = np.argwhere(liq).mean(axis=0)
                vm: float = float(np.sqrt(
                    self.domain.state.velocity_x.numpy()[liq]**2
                    + self.domain.state.velocity_y.numpy()[liq]**2
                    + self.domain.state.velocity_z.numpy()[liq]**2).max())
                print(f"[t={self.sim_time:.1f}s] liquid={liq.sum()} "
                      f"COM=({com[0]:.0f},{com[1]:.0f},{com[2]:.0f}) "
                      f"v_max={vm:.3f} step={self._last_ms:.0f}ms(avg={avg_step:.0f}ms) "
                      f"render={self._last_render_ms:.0f}ms(avg={avg_render:.0f}ms)",
                      file=sys.stderr, flush=True)
            else:
                print(f"[t={self.sim_time:.1f}s] step={self._last_ms:.0f}ms(avg={avg_step:.0f}ms) "
                      f"render={self._last_render_ms:.0f}ms(avg={avg_render:.0f}ms)",
                      file=sys.stderr, flush=True)

    def render(self) -> None:
        t0: float = time.perf_counter()
        self.viewer.begin_frame(self.sim_time)
        if self.ssfr.available:
            self.ssfr.set_density_field(density=self.domain.state.density,
                grid_origin=(0.0, 0.0, 0.0), cell_size=DH,
                threshold=SSFR_THRESHOLD, max_steps=RAY_MARCH_STEPS,
                step_fraction=RAY_MARCH_STEP_FRACTION)
        self.viewer.end_frame()
        self._last_render_ms = (time.perf_counter() - t0) * 1000.0
        self._render_times.append(self._last_render_ms)


def main() -> None:
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
    viewer, args = init_fluid_viewer()
    newton.examples.run(TrtDropletFall(viewer), args)


if __name__ == "__main__":
    main()
