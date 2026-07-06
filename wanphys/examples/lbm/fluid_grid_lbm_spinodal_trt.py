# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Spinodal decomposition with TRT + Shan-Chen pseudopotential.

The domain is initialised with a uniform near-critical density and small
random noise.  Because this density lies inside the spinodal region of the
SC equation of state, the uniform state is thermodynamically unstable:
infinitesimal fluctuations grow spontaneously, phase-separating the fluid
into liquid (ρ ≈ 1.8) and gas (ρ ≈ 0.1) domains.

The early stage produces a characteristic interconnected labyrinth pattern.
Over time, surface tension drives coarsening: small domains shrink and
vanish while larger domains grow (Ostwald ripening).

This is the purest demonstration of the SC multiphase model — no gravity,
no walls, no initial structure.  The pattern emerges entirely from the
interplay of the SC interaction force and the TRT collision operator.

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
# Physics
# ---------------------------------------------------------------------------
# G = -5.0 → equilibrium densities: ρ_gas ≈ 0.1, ρ_liquid ≈ 1.8
# Initial density ~0.95 sits inside the spinodal → spontaneous decomposition.
# No gravity — the SC force alone drives the pattern formation.
# All boundaries periodic — the domain is a closed thermodynamic system.
TAU: float = 0.6                    # slightly higher viscosity for clean coarsening
LAMBDA_TRT: float = 0.03
G_SC: float = -5.0
PSI_TYPE: int = 1                   # PSI_EXP: ψ = 1 - exp(-ρ)
PSI_REF: float = 1.0
GRAVITY: float = 0.0                # no gravity — pure phase separation
OMEGA_REG: float = 0.5              # reg-TRT blend

# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
# Initial density controls the late-stage morphology (equilibrium: gas≈0.1, liquid≈1.8):
#   0.95 → ~50:50 bicontinuous → cylinder/slab (minimal surface in periodic box)
#   0.70 → ~35% liquid → connected lamellae
#   0.45 → ~20% liquid → isolated spherical droplets ← best visual
#   1.20 → ~35% gas → isolated bubbles
# NOTE: D3Q19 has cubic lattice anisotropy — interfaces will align slightly with
#   the grid axes after long coarsening. This is a known limitation of the SC model
#   on the D3Q19 lattice; it cannot be fully eliminated at this resolution.
INITIAL_DENSITY: float = 0.45       # low liquid fraction → isolated droplets
NOISE_AMPLITUDE: float = 0.10       # stronger noise to create more nuclei

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
SSFR_THRESHOLD: float = 0.7
RAY_MARCH_STEPS: int = 1600

# ---------------------------------------------------------------------------
# Time-stepping
# ---------------------------------------------------------------------------
FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 4


# ---------------------------------------------------------------------------
# Warp kernel
# ---------------------------------------------------------------------------


@wp.kernel
def _init_spinodal(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    rho0: float,
    noise_amp: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    """Uniform near-critical density with small random noise."""
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k

    # Integer hash for isotropic pseudo-random noise (no preferred direction)
    h = (i * 1664525 + j * 1013904223 + k * 22695477 + seed * 1103515245) & 0x7FFFFFFF
    n = (float(h) / 2147483648.0) - 1.0  # map to [-1, 1]
    rho = rho0 + noise_amp * n
    rho = wp.max(rho, 0.01)  # floor at gas density to avoid negative

    density[i, j, k] = rho

    # Equilibrium distributions at rest
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


class TrtSpinodalDecomposition:
    """Spinodal decomposition with TRT + Shan-Chen pseudopotential.

    The uniform near-critical initial state spontaneously separates into
    liquid and gas phases.  The coarsening proceeds from fine labyrinth
    patterns to larger, smoother domains over time.

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
            lambda_trt=LAMBDA_TRT,
            use_regularization=True,
            omega_reg=OMEGA_REG,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=GRAVITY,
            bc_periodic=(True, True, True),  # fully periodic — closed system
        )
        n: int = int(self.model.nx)

        print(f"Spinodal Decomposition (TRT): {n}³, tau={TAU}, G={G_SC}")
        print(f"  tau_plus={self.model.tau_plus:.3f}, tau_minus={self.model.tau_minus:.3f}")
        print(f"  omega_reg={OMEGA_REG}, use_regularization=True")
        print(f"  rho0={INITIAL_DENSITY}, noise={NOISE_AMPLITUDE}")
        print(f"  bc: fully periodic (X,Y,Z) — closed thermodynamic system")
        print(f"  gravity=0 — SC force drives all dynamics")

        self.domain: LbmDomain = LbmDomain(self.model)
        self.domain.create_state()
        self.sim_dt: float = FRAME_DT / SIM_SUBSTEPS
        self.sim_time: float = 0.0

        self._init_uniform_noise()

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

    def _init_uniform_noise(self) -> None:
        """Fill domain with near-critical uniform density + random noise."""
        state: LbmState = self.domain.state
        n: int = int(self.model.nx)
        stride: int = n * n * n
        wp.launch(
            _init_spinodal,
            dim=(n, n, n),
            inputs=[
                state.f, state.density,
                INITIAL_DENSITY, NOISE_AMPLITUDE,
                42, n, n, n, stride,
            ],
        )
        for a in ['f', 'density', 'velocity_x', 'velocity_y', 'velocity_z', 'solid_phi', 'solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)
        rho_np: np.ndarray = state.density.numpy()
        print(f"  Initial: rho=[{rho_np.min():.3f}, {rho_np.max():.3f}], mean={rho_np.mean():.4f}")

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
        liquid: int = int((rho_np > SSFR_THRESHOLD).sum())
        gas: int = int((rho_np < 0.3).sum())
        interface: int = int(rho_np.size) - liquid - gas
        vm: float = float(np.sqrt(
            self.domain.state.velocity_x.numpy() ** 2
            + self.domain.state.velocity_y.numpy() ** 2
            + self.domain.state.velocity_z.numpy() ** 2
        ).max())
        print(
            f"[t={self.sim_time:.1f}s] liquid={liquid} gas={gas} interface={interface} "
            f"rho=[{rho_np.min():.3f},{rho_np.max():.3f}] "
            f"v_max={vm:.4f} sim={self._last_ms:.0f}ms",
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
    """Run the TRT spinodal decomposition visual demo."""
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer

    viewer, args = init_fluid_viewer()
    newton.examples.run(TrtSpinodalDecomposition(viewer), args)


if __name__ == "__main__":
    main()
