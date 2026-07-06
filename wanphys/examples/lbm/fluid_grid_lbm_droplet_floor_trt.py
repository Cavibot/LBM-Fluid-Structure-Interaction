# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""TRT + Shan-Chen pseudopotential droplet impact on a dry solid floor.

A spherical liquid droplet is released from height in a gas-filled domain.
Gravity accelerates it downward until it hits the bounce-back bottom wall,
spreading into a liquid lamella and ejecting secondary droplets — the
canonical "droplet-on-floor" impact benchmark.

Physics: TRT (Two-Relaxation-Time) collision with regularisation damps ghost
modes at the liquid-gas interface.  The Shan-Chen pseudopotential (PSI_EXP)
drives phase separation at ~18:1 density ratio.  The bottom wall is a
bounce-back boundary; X and Y are periodic (infinite horizontal domain).

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
# tau = 0.55 → ν = (0.55 - 0.5) / 3 ≈ 0.0167
# lambda_trt = 0.03 → τ₊ = 0.03/0.05 + 0.5 = 1.1 (mild ghost damping)
# omega_reg = 0.5 → blended regularised TRT for interface stability
# G = -6.0 → sharper interface, stronger effective surface tension
# sc_boundary_psi = 0.1 → hydrophobic (gas-like) wall — prevents the floor
#   from acting as a nucleation site and condensing liquid before impact.
# Gravity-driven fall: uniform body force accelerates every cell together.
TAU: float = 0.55
LAMBDA_TRT: float = 0.03
G_SC: float = -6.0                   # stronger phase separation → sharper interface
PSI_TYPE: int = 1                    # PSI_EXP: ψ = 1 - exp(-ρ)
PSI_REF: float = 1.0
SC_BOUNDARY_PSI: float = 0.1         # gas-like wall: hydrophobic, no liquid nucleation
GRAVITY: float = -0.005              # moderate gravity — less deformation in flight
OMEGA_REG: float = 0.5               # reg-TRT blend

# ---------------------------------------------------------------------------
# Droplet geometry
# ---------------------------------------------------------------------------
RHO_LIQUID: float = 1.8
RHO_GAS: float = 0.1
DROPLET_RADIUS: int = 35             # larger droplet resists stretching better
DROPLET_CX_FRAC: float = 0.50
DROPLET_CY_FRAC: float = 0.50
DROPLET_CZ_FRAC: float = 0.82        # released high — gravity does the work
DROPLET_VZ_INIT: float = 0.0         # start from rest; gravity accelerates uniformly
TRANSITION_WIDTH: float = 2.5

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
SSFR_THRESHOLD: float = 0.7
RAY_MARCH_STEPS: int = 1600

# ---------------------------------------------------------------------------
# Time-stepping
# ---------------------------------------------------------------------------
FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 4                # 4 lattice steps per frame


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _init_gas_uniform(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    rho_gas: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Fill the entire domain with low-density gas at rest equilibrium."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    n = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001)
    n = n * wp.cos(float(i * 419 + j * 233 + k * 577 + seed) * 0.0013)
    rho = wp.max(rho_gas + n * 0.005 * rho_gas, 0.01)
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
    """Superpose a spherical liquid droplet over the gas background.

    Only writes cells whose droplet density exceeds the current local density,
    so the droplet cleanly overwrites gas without needing a pool mask.
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


class TrtDropletFloor:
    """Droplet impact on a dry solid floor with TRT + Shan-Chen.

    A spherical liquid droplet is released from height in a gas-filled
    domain.  It free-falls under gravity (gz = -0.01) and impacts the
    bounce-back bottom wall, spreading radially into a thin lamella.

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
            bc_periodic=(True, True, False),  # infinite horizontal, solid floor/ceiling
        )
        n: int = int(self.model.nx)
        nu: float = (TAU - 0.5) / 3.0
        D: int = 2 * DROPLET_RADIUS
        fall_dist_cells: float = float(int(n * DROPLET_CZ_FRAC) - DROPLET_RADIUS)
        v_impact_est: float = float(np.sqrt(DROPLET_VZ_INIT ** 2 + 2.0 * abs(GRAVITY) * max(fall_dist_cells, 1.0)))
        re_est: float = v_impact_est * float(D) / nu

        print(f"TRT Droplet-on-Floor (gravity-driven): {n}³, tau={TAU} (nu={nu:.4f}), lambda_trt={LAMBDA_TRT}, G={G_SC}")
        print(f"  tau_plus={self.model.tau_plus:.3f}, tau_minus={self.model.tau_minus:.3f}")
        print(f"  omega_plus={self.model.omega_plus:.3f}, omega_minus={self.model.omega_minus:.3f}")
        print(f"  omega_reg={OMEGA_REG}, use_regularization=True")
        print(f"  bc: X,Y periodic, Z bounce-back (dry floor at z=0)")
        print(f"  Droplet: r={DROPLET_RADIUS}  release z={int(n * DROPLET_CZ_FRAC)}  "
              f"fall_dist≈{fall_dist_cells:.0f} cells")
        print(f"  v_impact≈{v_impact_est:.3f}  Re≈{re_est:.0f}  D={D} cells")

        self.domain: LbmDomain = LbmDomain(self.model)
        self.domain.create_state()
        self.sim_dt: float = FRAME_DT / SIM_SUBSTEPS
        self.sim_time: float = 0.0

        self._init_gas_background()
        self._add_droplet_in_gas()

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

    def _init_gas_background(self) -> None:
        """Fill the entire domain with low-density gas at rest."""
        state: LbmState = self.domain.state
        n: int = int(self.model.nx)
        stride: int = n * n * n
        wp.launch(
            _init_gas_uniform,
            dim=(n, n, n),
            inputs=[state.f, state.density, RHO_GAS, 42, n, n, n, stride],
        )
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)
        print(f"  Gas background: rho={RHO_GAS} uniform")

    def _add_droplet_in_gas(self) -> None:
        """Superpose the liquid droplet onto the gas background."""
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
                state.f, state.density,
                cx, cy, cz, DROPLET_RADIUS,
                RHO_LIQUID, RHO_GAS, TRANSITION_WIDTH, DROPLET_VZ_INIT,
                43, n, n, n, stride,
            ],
        )
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)
        rho_np: np.ndarray = state.density.numpy()
        liquid_cells: int = int((rho_np > SSFR_THRESHOLD).sum())
        print(f"  Droplet added: {liquid_cells} liquid cells")

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one frame."""
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
        # Track liquid near the floor (z < 5 cells) as "spread"
        floor_spread: int = int((rho_np[:, :, :5] > SSFR_THRESHOLD).sum())
        if liq.any():
            com: np.ndarray = np.argwhere(liq).mean(axis=0)
            vz: float = float(self.domain.state.velocity_z.numpy()[liq].mean())
            vm: float = float(np.sqrt(
                self.domain.state.velocity_x.numpy()[liq] ** 2
                + self.domain.state.velocity_y.numpy()[liq] ** 2
                + self.domain.state.velocity_z.numpy()[liq] ** 2
            ).max())
            # Droplet cells: liquid above z > 5
            droplet_mask: np.ndarray = (rho_np > SSFR_THRESHOLD) & (
                np.arange(rho_np.shape[2])[None, None, :] > 5
            )
            droplet_cells: int = int(droplet_mask.sum())
            droplet_z_min: float = float(np.argwhere(droplet_mask)[:, 2].min()) if droplet_cells > 0 else -1.0
            print(
                f"[t={self.sim_time:.1f}s] liquid={lc} floor_spread={floor_spread} "
                f"droplet={droplet_cells} cells, z_min={droplet_z_min:.0f} "
                f"COM=({com[0]:.0f},{com[1]:.0f},{com[2]:.0f}) "
                f"vz={vz:+.3f} v_max={vm:.3f} "
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
    """Run the TRT droplet-on-floor visual demo."""
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer

    viewer, args = init_fluid_viewer()
    newton.examples.run(TrtDropletFloor(viewer), args)


if __name__ == "__main__":
    main()
