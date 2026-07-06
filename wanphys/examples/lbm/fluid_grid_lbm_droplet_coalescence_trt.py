# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Droplet collision and coalescence with TRT + Shan-Chen pseudopotential.

Two liquid droplets are placed close together in a gas-filled domain.  A
small initial velocity pushes them toward each other.  When their interfaces
touch, the SC interaction force drives rapid coalescence: a liquid bridge
forms, grows, and the merged droplet oscillates briefly before settling into
a single spherical droplet.

This is a pure demonstration of surface-tension-driven coalescence — the SC
force naturally minimizes interfacial area, pulling the two droplets together
the moment their diffuse interfaces overlap.

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
# Physics: TRT + Shan-Chen
# ---------------------------------------------------------------------------
TAU: float = 0.55
LAMBDA_TRT: float = 0.03
G_SC: float = -5.0
PSI_TYPE: int = 1                    # PSI_EXP
PSI_REF: float = 1.0
SC_BOUNDARY_PSI: float = -1.0        # mirror closure (no walls matter here)
GRAVITY: float = 0.0                 # no gravity — pure coalescence
OMEGA_REG: float = 0.5

# ---------------------------------------------------------------------------
# Droplet geometry — two droplets side by side
# ---------------------------------------------------------------------------
RHO_LIQUID: float = 1.8
RHO_GAS: float = 0.1
DROPLET_RADIUS: int = 22             # each droplet
DROPLET_GAP: int = 8                 # gap between droplet surfaces
DROPLET_CY_FRAC: float = 0.50
DROPLET_CZ_FRAC: float = 0.50
# Centres placed symmetrically about domain centre, separated by 2*r + gap
APPROACH_VX: float = 0.03            # small approach velocity (each droplet)
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
SIM_SUBSTEPS: int = 10


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _init_gas(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    rho_gas: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Fill domain with uniform low-density gas."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    h = (i * 1664525 + j * 1013904223 + k * 22695477 + seed * 1103515245) & 0x7FFFFFFF
    n = (float(h) / 2147483648.0) - 1.0
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
def _add_droplet_moving(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    cx: int,
    cy: int,
    cz: int,
    droplet_r: int,
    rho_high: float,
    rho_low: float,
    transition_w: float,
    vx: float,
    vy: float,
    vz: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Superpose a spherical droplet with a given velocity (equilibrium distribution)."""
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
    h = (i * 1664525 + j * 1013904223 + k * 22695477 + seed * 1103515245) & 0x7FFFFFFF
    n = (float(h) / 2147483648.0) - 1.0
    rho = wp.max(rho_drop + n * 0.005 * rho_drop, 0.01)
    density[i, j, k] = rho

    # D3Q19 equilibrium with velocity (vx, vy, vz)
    usq = vx * vx + vy * vy + vz * vz
    f[0 * stride + idx] = (1.0 / 3.0) * rho * (1.0 - 1.5 * usq)
    f[1 * stride + idx] = (1.0 / 18.0) * rho * (1.0 + 3.0 * vx + 4.5 * vx * vx - 1.5 * usq)
    f[2 * stride + idx] = (1.0 / 18.0) * rho * (1.0 - 3.0 * vx + 4.5 * vx * vx - 1.5 * usq)
    f[3 * stride + idx] = (1.0 / 18.0) * rho * (1.0 + 3.0 * vy + 4.5 * vy * vy - 1.5 * usq)
    f[4 * stride + idx] = (1.0 / 18.0) * rho * (1.0 - 3.0 * vy + 4.5 * vy * vy - 1.5 * usq)
    f[5 * stride + idx] = (1.0 / 18.0) * rho * (1.0 + 3.0 * vz + 4.5 * vz * vz - 1.5 * usq)
    f[6 * stride + idx] = (1.0 / 18.0) * rho * (1.0 - 3.0 * vz + 4.5 * vz * vz - 1.5 * usq)
    f[7 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (vx + vy) + 4.5 * (vx + vy) * (vx + vy) - 1.5 * usq)
    f[8 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (-vx + vy) + 4.5 * (-vx + vy) * (-vx + vy) - 1.5 * usq)
    f[9 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (vx - vy) + 4.5 * (vx - vy) * (vx - vy) - 1.5 * usq)
    f[10 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (-vx - vy) + 4.5 * (-vx - vy) * (-vx - vy) - 1.5 * usq)
    f[11 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (vx + vz) + 4.5 * (vx + vz) * (vx + vz) - 1.5 * usq)
    f[12 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (-vx + vz) + 4.5 * (-vx + vz) * (-vx + vz) - 1.5 * usq)
    f[13 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (vx - vz) + 4.5 * (vx - vz) * (vx - vz) - 1.5 * usq)
    f[14 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (-vx - vz) + 4.5 * (-vx - vz) * (-vx - vz) - 1.5 * usq)
    f[15 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (vy + vz) + 4.5 * (vy + vz) * (vy + vz) - 1.5 * usq)
    f[16 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (-vy + vz) + 4.5 * (-vy + vz) * (-vy + vz) - 1.5 * usq)
    f[17 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (vy - vz) + 4.5 * (vy - vz) * (vy - vz) - 1.5 * usq)
    f[18 * stride + idx] = (1.0 / 36.0) * rho * (1.0 + 3.0 * (-vy - vz) + 4.5 * (-vy - vz) * (-vy - vz) - 1.5 * usq)


class TrtDropletCoalescence:
    """Binary droplet collision and coalescence with TRT + Shan-Chen.

    Two identical liquid droplets are placed side-by-side with a small
    approach velocity.  When their diffuse SC interfaces overlap, the
    interaction force drives rapid merging into a single droplet.

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
            bc_periodic=(True, True, True),  # fully periodic
        )
        n: int = int(self.model.nx)
        mid: int = n // 2
        r: int = DROPLET_RADIUS
        half_sep: int = r + DROPLET_GAP // 2
        self.cx1: int = mid - half_sep
        self.cx2: int = mid + half_sep
        self.cy: int = int(n * DROPLET_CY_FRAC)
        self.cz: int = int(n * DROPLET_CZ_FRAC)

        print(f"TRT Droplet Coalescence: {n}³, tau={TAU}, G={G_SC}")
        print(f"  tau_plus={self.model.tau_plus:.3f}, tau_minus={self.model.tau_minus:.3f}")
        print(f"  omega_reg={OMEGA_REG}, use_regularization=True")
        print(f"  Droplets: r={r}, gap={DROPLET_GAP} cells, approach_vx=±{APPROACH_VX}")
        print(f"  centres: ({self.cx1},{self.cy},{self.cz})  ({self.cx2},{self.cy},{self.cz})")
        print(f"  bc: fully periodic, gravity=0")

        self.domain: LbmDomain = LbmDomain(self.model)
        self.domain.create_state()
        self.sim_dt: float = FRAME_DT / SIM_SUBSTEPS
        self.sim_time: float = 0.0
        self._merged: bool = False

        self._init_gas_background()
        self._add_droplets()

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
        """Fill the domain with low-density gas."""
        state: LbmState = self.domain.state
        n: int = int(self.model.nx)
        stride: int = n * n * n
        wp.launch(
            _init_gas,
            dim=(n, n, n),
            inputs=[state.f, state.density, RHO_GAS, 42, n, n, n, stride],
        )
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)

    def _add_droplets(self) -> None:
        """Place two droplets with opposing x-velocities."""
        state: LbmState = self.domain.state
        n: int = int(self.model.nx)
        stride: int = n * n * n
        # Left droplet: moves right (+x)
        wp.launch(
            _add_droplet_moving,
            dim=(n, n, n),
            inputs=[
                state.f, state.density,
                self.cx1, self.cy, self.cz, DROPLET_RADIUS,
                RHO_LIQUID, RHO_GAS, TRANSITION_WIDTH,
                APPROACH_VX, 0.0, 0.0, 43,
                n, n, n, stride,
            ],
        )
        wp.synchronize_device(self.model._device)
        # Right droplet: moves left (-x)
        wp.launch(
            _add_droplet_moving,
            dim=(n, n, n),
            inputs=[
                state.f, state.density,
                self.cx2, self.cy, self.cz, DROPLET_RADIUS,
                RHO_LIQUID, RHO_GAS, TRANSITION_WIDTH,
                -APPROACH_VX, 0.0, 0.0, 44,
                n, n, n, stride,
            ],
        )
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)
        rho_np: np.ndarray = state.density.numpy()
        print(f"  Liquid cells: {(rho_np > SSFR_THRESHOLD).sum()} (2 droplets)")

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
        if self.frame_count % 15 == 0:
            self._print_status()

    def _print_status(self) -> None:
        """Print diagnostic information."""
        rho_np: np.ndarray = self.domain.state.density.numpy()
        if np.any(~np.isfinite(rho_np)):
            print(f"[t={self.sim_time:.1f}s] *** DIVERGED ***", file=sys.stderr, flush=True)
            self.viewer._paused = True
            return
        liq: np.ndarray = rho_np > SSFR_THRESHOLD
        lc: int = int(liq.sum())
        # Detect number of disconnected liquid regions by checking x-profile
        liq_x: np.ndarray = liq.any(axis=(1, 2))  # any liquid in each x-slice
        # Count contiguous blocks in x
        blocks: int = 0
        in_block: bool = False
        for v in liq_x:
            if v and not in_block:
                blocks += 1
                in_block = True
            elif not v:
                in_block = False
        if blocks == 1 and not self._merged:
            self._merged = True
            print(f"\n*** COALESCENCE at t≈{self.sim_time:.1f}s ***\n", file=sys.stderr, flush=True)
        vm: float = float(np.sqrt(
            self.domain.state.velocity_x.numpy()[liq] ** 2
            + self.domain.state.velocity_y.numpy()[liq] ** 2
            + self.domain.state.velocity_z.numpy()[liq] ** 2
        ).max()) if liq.any() else 0.0
        print(
            f"[t={self.sim_time:.1f}s] liquid={lc} blocks={blocks} "
            f"merged={'YES' if self._merged else 'no'} "
            f"rho=[{rho_np.min():.3f},{rho_np.max():.3f}] v_max={vm:.4f} "
            f"sim={self._last_ms:.0f}ms",
            file=sys.stderr,
            flush=True,
        )

    def render(self) -> None:
        """Render via screen-space ray-marching."""
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
    """Run the TRT droplet coalescence visual demo."""
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer

    viewer, args = init_fluid_viewer()
    newton.examples.run(TrtDropletCoalescence(viewer), args)


if __name__ == "__main__":
    main()
