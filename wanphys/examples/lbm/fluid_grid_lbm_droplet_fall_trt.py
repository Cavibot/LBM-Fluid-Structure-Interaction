# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""TRT + Shan-Chen pseudopotential droplet free-fall onto a deep liquid pool.

A spherical liquid droplet is released from height above a pre-equilibrated
deep pool.  Gravity accelerates the droplet downward until it impacts the
liquid surface, producing crown splash, crater formation, and eventual
coalescence.

Physics: TRT (Two-Relaxation-Time) collision damps ghost modes at the
interface, enabling the strong gravity and sharp density gradients needed
for realistic splash dynamics.  The Shan-Chen pseudopotential (PSI_EXP)
drives phase separation with a liquid-to-gas density ratio of ~18:1.

The pool is pre-equilibrated under a smooth gravity ramp before the droplet
is added, preventing standing-wave artefacts at the free surface.

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
# Physics: TRT + Shan-Chen pseudopotential
# ---------------------------------------------------------------------------
# tau = 0.53 → ν = (0.53 - 0.5) / 3 = 0.01 (lower viscosity for faster dynamics)
# lambda_trt = 0.03 → τ₊ = 0.03/0.03 + 0.5 = 1.5 (even-mode ghost damping)
#   Λ scales with (τ−0.5): must reduce when τ → 0.5, otherwise τ₊ explodes.
#   D3Q19 optimal Λ = 3/16 = 0.1875 only applies when τ ≫ 0.5 (e.g. τ ≈ 1.0).
# omega_reg = 0.5 → blended regularized TRT: odd parts preserved, even parts
#   projected onto 2nd-order Hermite basis for extra stability at the interface.
# With D ≈ 100 cells, impact U ≈ 1.0 → Re ≈ 10000
TAU: float = 0.55
LAMBDA_TRT: float = 0.03
G_SC: float = -5.0
PSI_TYPE: int = 1                    # PSI_EXP: ψ = 1 - exp(-ρ)
PSI_REF: float = 1.0
SC_BOUNDARY_PSI: float = 0.0        # mirror closure: ψ_wall = ψ_fluid
GRAVITY: float = -0.01               # stronger gravity for harder droplet impact
OMEGA_REG: float = 0.5               # reg-TRT: odd preserved, even regularized

# ---------------------------------------------------------------------------
# Droplet + pool geometry
# ---------------------------------------------------------------------------
RHO_LIQUID: float = 1.8
RHO_GAS: float = 0.1
POOL_HEIGHT: int = 30                 # moderate pool depth
DROPLET_RADIUS: int = 30             # large droplet for harder impact
DROPLET_CX_FRAC: float = 0.50
DROPLET_CY_FRAC: float = 0.50
DROPLET_CZ_FRAC: float = 0.7        # closer to surface, less deformation in flight
DROPLET_VZ_INIT: float = -0.3        # initial downward velocity for concentrated impact
TRANSITION_WIDTH: float = 2.5

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
SSFR_THRESHOLD: float = 0.7
RAY_MARCH_STEPS: int = 2000

# ---------------------------------------------------------------------------
# Time-stepping
# ---------------------------------------------------------------------------
FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 4
POOL_EQUIL_STEPS: int = 1000         # longer equilibration for the deeper pool
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
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Initialise a flat liquid pool with a smooth tanh interface at ``pool_h``."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    pool_w = 0.5 + 0.5 * wp.tanh((float(pool_h) - float(k)) / transition_w)
    rho = rho_low + (rho_high - rho_low) * pool_w
    n = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001)
    n = n * wp.cos(float(i * 419 + j * 233 + k * 577 + seed) * 0.0013)
    rho = wp.max(rho + n * 0.005 * rho, 0.01)
    density[i, j, k] = rho
    f[0 * stride + idx] = (1.0 / 3.0) * rho
    f[1 * stride + idx] = (1.0 / 18.0) * rho
    f[2 * stride + idx] = (1.0 / 18.0) * rho
    f[3 * stride + idx] = (1.0 / 18.0) * rho
    f[4 * stride + idx] = (1.0 / 18.0) * rho
    f[5 * stride + idx] = (1.0 / 18.0) * rho
    f[6 * stride + idx] = (1.0 / 18.0) * rho
    f[7 * stride + idx] = (1.0 / 36.0) * rho
    f[8 * stride + idx] = (1.0 / 36.0) * rho
    f[9 * stride + idx] = (1.0 / 36.0) * rho
    f[10 * stride + idx] = (1.0 / 36.0) * rho
    f[11 * stride + idx] = (1.0 / 36.0) * rho
    f[12 * stride + idx] = (1.0 / 36.0) * rho
    f[13 * stride + idx] = (1.0 / 36.0) * rho
    f[14 * stride + idx] = (1.0 / 36.0) * rho
    f[15 * stride + idx] = (1.0 / 36.0) * rho
    f[16 * stride + idx] = (1.0 / 36.0) * rho
    f[17 * stride + idx] = (1.0 / 36.0) * rho
    f[18 * stride + idx] = (1.0 / 36.0) * rho


@wp.kernel
def _add_droplet(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    cx: int,
    cy: int,
    cz: int,
    droplet_r: int,
    rho_high: float,
    rho_low: float,
    transition_w: float,
    vz_init: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Superpose a spherical liquid droplet with equilibrium distributions.

    The droplet is only written where its density exceeds the current local
    density, so it overwrites gas cells without disturbing the pool below.
    """
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    dx = float(i) - float(cx)
    dy = float(j) - float(cy)
    dz = float(k) - float(cz)
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
    vx = 0.0
    vy = 0.0
    vz = vz_init
    usq = vz * vz
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
    """Droplet impact onto a liquid pool with TRT + Shan-Chen.

    A large spherical droplet (r = 50 cells) is given an initial downward
    velocity (vz = -0.3) above a pre-equilibrated 50-cell-deep pool.
    Strong gravity (gz = -0.01) accelerates it further, producing a
    concentrated high-momentum impact with crown splash and crater formation.

    Parameters
    ----------
    viewer : FluidViewerGL
        The interactive OpenGL viewer for rendering.
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
            bc_periodic=(True, True, False),  # X,Y periodic (infinite pool), Z walls
        )
        n: int = int(self.model.nx)
        nu: float = (TAU - 0.5) / 3.0
        D: int = 2 * DROPLET_RADIUS
        # Estimate impact velocity: v_impact ≈ sqrt(vz_init² + 2 * |g| * fall_dist)
        fall_dist_cells: float = float(int(n * DROPLET_CZ_FRAC) - POOL_HEIGHT - DROPLET_RADIUS)
        v_impact_est: float = float(np.sqrt(DROPLET_VZ_INIT ** 2 + 2.0 * abs(GRAVITY) * max(fall_dist_cells, 1.0)))
        re_est: float = v_impact_est * float(D) / nu
        print(f"TRT Droplet Fall: {n}³, tau={TAU} (nu={nu:.4f}), lambda_trt={LAMBDA_TRT}, G={G_SC}")
        print(f"  tau_plus={self.model.tau_plus:.3f}, tau_minus={self.model.tau_minus:.3f}")
        print(f"  omega_plus={self.model.omega_plus:.3f}, omega_minus={self.model.omega_minus:.3f}")
        print(f"  omega_reg={OMEGA_REG}, use_regularization=True")
        print(f"  bc: X,Y periodic, Z bounce-back (infinite pool, no side walls)")
        print(f"  Pool: {POOL_HEIGHT} cells  Droplet: r={DROPLET_RADIUS}  "
              f"release z={int(n * DROPLET_CZ_FRAC)}  fall_dist≈{fall_dist_cells:.0f} cells")
        print(f"  v_impact≈{v_impact_est:.3f}  Re≈{re_est:.0f}  "
              f"D={D} cells  drop_surface_gap≈{int(n * DROPLET_CZ_FRAC) - DROPLET_RADIUS - POOL_HEIGHT} cells")

        self.domain: LbmDomain = LbmDomain(self.model)
        self.domain.create_state()
        self.sim_dt: float = FRAME_DT / SIM_SUBSTEPS
        self.sim_time: float = 0.0

        self._init_pool_only()
        self._equilibrate_pool()
        self._add_droplet_on_pool()

        self.ssfr: ScreenSpaceFluidRenderer = ScreenSpaceFluidRenderer(
            viewer=viewer,
            max_particles=1,
            particle_radius=0.01,
            device=self.model._device,
        )
        viewer.register_post_render_callback(lambda v: self.ssfr.render(v))
        self.frame_count: int = 0
        self._last_ms: float = 0.0
        print("Controls: [Space] unpause  [R] reset  [mouse] orbit  [scroll] zoom")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_pool_only(self) -> None:
        """Initialise the deep liquid pool at rest (zero velocity)."""
        state: LbmState = self.domain.state
        n: int = int(self.model.nx)
        stride: int = n * n * n
        wp.launch(
            _init_pool,
            dim=(n, n, n),
            inputs=[
                state.f,
                state.density,
                POOL_HEIGHT,
                RHO_LIQUID,
                RHO_GAS,
                TRANSITION_WIDTH,
                42,
                n,
                n,
                n,
                stride,
            ],
        )
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)

    def _equilibrate_pool(self) -> None:
        """Smoothly ramp gravity and let the pool settle to hydrostatic equilibrium.

        A long ramp + settling phase is essential to prevent standing waves
        from forming between the bounce-back bottom wall and the free surface.
        Without sufficient equilibration, pressure waves reflect off the bottom
        and create grid-scale protrusions on the liquid surface — for a deep
        pool these take longer to dissipate.
        """
        target_gz: float = float(self.model.gravity_z)
        self.model.gravity_z = 0.0
        ramp: int = min(GRAVITY_RAMP_STEPS, POOL_EQUIL_STEPS // 3)

        print(f"  [Equil] ramping gravity over {ramp} steps (deep pool)...")
        for s in range(ramp):
            self.model.gravity_z = target_gz * float(s + 1) / float(ramp)
            self.domain.step(self.sim_dt)

        self.model.gravity_z = target_gz
        settle: int = POOL_EQUIL_STEPS - ramp
        print(f"  [Equil] settling over {settle} steps at full gravity...")
        for s in range(settle):
            self.domain.step(self.sim_dt)
            if (s + 1) % 200 == 0:
                wp.synchronize_device(self.model._device)
                rho_np: np.ndarray = self.domain.state.density.numpy()
                liq: np.ndarray = rho_np > SSFR_THRESHOLD
                zf: np.ndarray = liq.mean(axis=(0, 1))
                sz: int = int(np.argmax(zf < 0.5)) if np.any(zf < 0.5) else 0
                print(f"    step {s + 1}: surface_z={sz}, rho=[{rho_np.min():.3f},{rho_np.max():.3f}]")

        wp.synchronize_device(self.model._device)
        rho_np = self.domain.state.density.numpy()
        liq = rho_np > SSFR_THRESHOLD
        zf = liq.mean(axis=(0, 1))
        sz = int(np.argmax(zf < 0.5)) if np.any(zf < 0.5) else 0

        rho_std_z: np.ndarray = rho_np.std(axis=(0, 1))
        max_std: float = float(rho_std_z.max())
        print(f"  [Equil] surface_z={sz}, liquid={liq.sum()}, "
              f"rho=[{rho_np.min():.3f},{rho_np.max():.3f}], "
              f"max_xy_std={max_std:.4f}")
        if max_std > 0.05:
            print(f"  [Equil] WARNING: large horizontal density variation ({max_std:.4f}) "
                  f"— standing waves may persist. Consider more equilibration steps.")

    def _add_droplet_on_pool(self) -> None:
        """Superpose the droplet above the equilibrated pool."""
        state: LbmState = self.domain.state
        n: int = int(self.model.nx)
        stride: int = n * n * n
        cx: int = int(n * DROPLET_CX_FRAC)
        cy: int = int(n * DROPLET_CY_FRAC)
        cz: int = int(n * DROPLET_CZ_FRAC)
        wp.launch(
            _add_droplet,
            dim=(n, n, n),
            inputs=[
                state.f,
                state.density,
                cx,
                cy,
                cz,
                DROPLET_RADIUS,
                RHO_LIQUID,
                RHO_GAS,
                TRANSITION_WIDTH,
                DROPLET_VZ_INIT,
                43,
                n,
                n,
                n,
                stride,
            ],
        )
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one frame (``SIM_SUBSTEPS`` LBM steps)."""
        t0: float = time.perf_counter()
        for _ in range(SIM_SUBSTEPS):
            self.domain.step(self.sim_dt)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self._print_status()

    def _print_status(self) -> None:
        """Print diagnostic information every 30 frames."""
        rho_np: np.ndarray = self.domain.state.density.numpy()
        if np.any(~np.isfinite(rho_np)):
            print(f"[t={self.sim_time:.1f}s] *** DIVERGED ***", file=sys.stderr, flush=True)
            self.viewer._paused = True
            return
        liq: np.ndarray = rho_np > SSFR_THRESHOLD
        lc: int = int(liq.sum())
        pool: int = int((rho_np[:, :, :POOL_HEIGHT] > SSFR_THRESHOLD).sum())
        splash: int = int((rho_np[:, :, POOL_HEIGHT:] > SSFR_THRESHOLD).sum())
        if liq.any():
            com: np.ndarray = np.argwhere(liq).mean(axis=0)
            vx: float = float(self.domain.state.velocity_x.numpy()[liq].mean())
            vy: float = float(self.domain.state.velocity_y.numpy()[liq].mean())
            vz: float = float(self.domain.state.velocity_z.numpy()[liq].mean())
            vm: float = float(np.sqrt(
                self.domain.state.velocity_x.numpy()[liq] ** 2
                + self.domain.state.velocity_y.numpy()[liq] ** 2
                + self.domain.state.velocity_z.numpy()[liq] ** 2
            ).max())
            # Track the droplet separately: liquid above pool + 5-cell margin
            droplet_mask: np.ndarray = (rho_np > SSFR_THRESHOLD) & (
                np.arange(rho_np.shape[2])[None, None, :] > POOL_HEIGHT + 5
            )
            droplet_cells: int = int(droplet_mask.sum())
            droplet_z_min: float = -1.0
            if droplet_cells > 0:
                droplet_z_min = float(np.argwhere(droplet_mask)[:, 2].min())
            print(
                f"[t={self.sim_time:.1f}s] liquid={lc} (pool={pool}, splash={splash}) "
                f"droplet={droplet_cells} cells, droplet_z_min={droplet_z_min:.0f} "
                f"COM=({com[0]:.0f},{com[1]:.0f},{com[2]:.0f}) "
                f"v=({vx:+.3f},{vy:+.3f},{vz:+.3f}) v_max={vm:.3f} "
                f"sim={self._last_ms:.0f}ms",
                file=sys.stderr,
                flush=True,
            )

    def render(self) -> None:
        """Render the current frame via screen-space ray-marching."""
        self.viewer.begin_frame(self.sim_time)
        if self.ssfr.available:
            self.ssfr.set_density_field(
                density=self.domain.state.density,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=DH,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()


def main() -> None:
    """Run the TRT droplet free-fall visual demo."""
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer

    viewer, args = init_fluid_viewer()
    newton.examples.run(TrtDropletFall(viewer), args)


if __name__ == "__main__":
    main()
