# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""TRT (Two-Relaxation-Time) Shan-Chen dam-break.

TRT damps ghost modes at interfaces via independent even/odd relaxation,
enabling much stronger gravity than pure BGK while keeping the interface
intact.

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom
"""

from __future__ import annotations

import sys, time
import numpy as np
import warp as wp
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel, LbmState
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer

# ---------------------------------------------------------------------------
N: int = 128
DH: float = 0.02               # domain = 1.536m (20% smaller → fluid traverses faster)

TAU: float = 0.55               # ν = (0.55-0.5)/3 = 0.0167
LAMBDA_TRT: float = 0.03       # τ₊ = 0.03/0.05 + 0.5 = 1.1 (mild ghost damping)
                                # Λ scales with (τ−0.5): must reduce when τ → 0.5
                                # otherwise τ₊ explodes (was 4.25 with Λ=0.1875!)
G_SC: float = -5.0
SC_BOUNDARY_PSI: float = -1.0     # gas-like wall ψ: no mirror feedback → no droplets
PSI_TYPE: int = 1
PSI_REF: float = 1.0

GRAVITY: float = -0.005
OMEGA_REG: float = 0.5         # reg-TRT: odd part preserved, even part regularized

DAM_X_FRAC: float = 0.25
RHO_WATER: float = 1.8
RHO_AIR: float = 0.1

SSFR_THRESHOLD: float = 0.7
RAY_MARCH_STEPS: int = 1600     # 128³ diagonal: ~√(3)×128 ≈ 222 → ×7 samples/lu

FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 5    # 3 lattice steps/frame, ~24ms @ 128³ on RTX 4060


@wp.kernel
def _init(
    f: wp.array(dtype=float), density: wp.array3d(dtype=float),
    dam_x: int, rho_w: float, rho_a: float,
    seed: int, nx: int, ny: int, nz: int, stride: int,
) -> None:
    i, j, k = wp.tid(); idx = i * ny * nz + j * nz + k
    rho = rho_w if i < dam_x else rho_a
    n = wp.sin(float(i*127+j*311+k*541+seed)*0.001) * wp.cos(float(i*419+j*233+k*577+seed)*0.0013)
    rho = wp.max(rho + n * 0.005 * rho, 0.01)
    density[i, j, k] = rho
    f[0*stride+idx]=(1.0/3.0)*rho; f[1*stride+idx]=(1.0/18.0)*rho; f[2*stride+idx]=(1.0/18.0)*rho
    f[3*stride+idx]=(1.0/18.0)*rho; f[4*stride+idx]=(1.0/18.0)*rho; f[5*stride+idx]=(1.0/18.0)*rho
    f[6*stride+idx]=(1.0/18.0)*rho; f[7*stride+idx]=(1.0/36.0)*rho; f[8*stride+idx]=(1.0/36.0)*rho
    f[9*stride+idx]=(1.0/36.0)*rho; f[10*stride+idx]=(1.0/36.0)*rho; f[11*stride+idx]=(1.0/36.0)*rho
    f[12*stride+idx]=(1.0/36.0)*rho; f[13*stride+idx]=(1.0/36.0)*rho; f[14*stride+idx]=(1.0/36.0)*rho
    f[15*stride+idx]=(1.0/36.0)*rho; f[16*stride+idx]=(1.0/36.0)*rho; f[17*stride+idx]=(1.0/36.0)*rho
    f[18*stride+idx]=(1.0/36.0)*rho


class TrtDamBreak:
    def __init__(self, viewer: FluidViewerGL):
        self.viewer = viewer; viewer._paused = True

        self.model = LbmModel(
            fluid_grid_res=(N,N,N), fluid_grid_cell_size=DH,
            tau=TAU, G=G_SC, sc_boundary_psi=SC_BOUNDARY_PSI, psi_type=PSI_TYPE, psi_ref=PSI_REF,
            lambda_trt=LAMBDA_TRT, use_regularization=True, omega_reg=OMEGA_REG,
            gravity_x=0.0, gravity_y=0.0, gravity_z=GRAVITY,
        )
        n = int(self.model.nx); ws = n*DH
        print(f"Reg-TRT Dam-Break: {n}^3, tau={TAU}, lambda={LAMBDA_TRT}, omega_reg={OMEGA_REG}")
        print(f"  tau_plus={self.model.tau_plus:.3f}, tau_minus={self.model.tau_minus:.3f}")
        print(f"  omega_plus={self.model.omega_plus:.3f}, omega_minus={self.model.omega_minus:.3f}")
        print(f"  sc_boundary_psi={SC_BOUNDARY_PSI}, gz={GRAVITY}, dam at x<{DAM_X_FRAC*n:.0f}")

        self.domain = LbmDomain(self.model); self.domain.create_state()
        self.sim_dt = FRAME_DT / SIM_SUBSTEPS; self.sim_time = 0.0

        state = self.domain.state; stride = n*n*n; dam_x = int(n*DAM_X_FRAC)
        wp.launch(_init, dim=(n,n,n),
            inputs=[state.f, state.density, dam_x, RHO_WATER, RHO_AIR, 42, n,n,n,stride])
        for a in ['f','density','velocity_x','velocity_y','velocity_z','solid_phi','solid_body_id']:
            wp.copy(getattr(self.domain._state_out, a), getattr(state, a))
        wp.synchronize_device(self.model._device)
        print(f"  Water cells: {(state.density.numpy()>SSFR_THRESHOLD).sum()}")

        # Gentle gravity ramp to avoid shocking the interface
        target_gz = float(self.model.gravity_z)
        self.model.gravity_z = 0.0
        ramp = 60
        for s in range(ramp):
            self.model.gravity_z = target_gz * float(s + 1) / float(ramp)
            self.domain.step(self.sim_dt)
        self.model.gravity_z = target_gz
        wp.synchronize_device(self.model._device)

        self.ssfr = ScreenSpaceFluidRenderer(viewer=viewer, max_particles=1, particle_radius=0.01,
                                               device=self.model._device)
        viewer.register_post_render_callback(lambda v: self.ssfr.render(v))
        self.frame_count = 0; self._last_ms = 0.0
        print("Controls: [Space] unpause  [R] reset  [mouse] orbit")

    def step(self):
        t0 = time.perf_counter()
        for _ in range(SIM_SUBSTEPS): self.domain.step(self.sim_dt)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter()-t0)*1000; self.sim_time += FRAME_DT
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            r = self.domain.state.density.numpy(); w = r > SSFR_THRESHOLD
            if w.any():
                c = np.argwhere(w).mean(axis=0)
                vx = float(self.domain.state.velocity_x.numpy()[w].mean())
                vy = float(self.domain.state.velocity_y.numpy()[w].mean())
                vz = float(self.domain.state.velocity_z.numpy()[w].mean())
                print(f"[t={self.sim_time:.1f}s] water={w.sum()} COM=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) v=({vx:+.3f},{vy:+.3f},{vz:+.3f}) sim={self._last_ms:.0f}ms", file=sys.stderr, flush=True)
            # ---- boundary density diagnostics ----
            r_np = self.domain.state.density.numpy()
            x0_slab: np.ndarray = r_np[0, :, :]          # x=0 (left wall, adjacent to water)
            x1_slab: np.ndarray = r_np[-1, :, :]         # x=63 (right wall, opposite side)
            water_left: int = int((x0_slab > SSFR_THRESHOLD).sum())
            water_right: int = int((x1_slab > SSFR_THRESHOLD).sum())
            print(
                f"  boundary rho: x=0 [{x0_slab.min():.3f}, {x0_slab.max():.3f}] water_cells={water_left}"
                f"  |  x=-1 [{x1_slab.min():.3f}, {x1_slab.max():.3f}] water_cells={water_right}",
                file=sys.stderr, flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.ssfr.available:
            self.ssfr.set_density_field(density=self.domain.state.density,
                grid_origin=(0,0,0), cell_size=DH, threshold=SSFR_THRESHOLD, max_steps=RAY_MARCH_STEPS)
        self.viewer.end_frame()


def main():
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
    viewer, args = init_fluid_viewer()
    newton.examples.run(TrtDamBreak(viewer), args)

if __name__ == "__main__": main()
