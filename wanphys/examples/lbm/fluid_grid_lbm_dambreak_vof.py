# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""TRT FSLBM / VOF sharp free-surface dam-break (no Shan-Chen).

Uses ``phase_mode='vof_sharp'`` on the existing D3Q19 distribution LBM.
Paper note: BGK/TRT FSLBM dam-break can be fragile; keep mild gravity.

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom
"""

from __future__ import annotations

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

# Stability-first dam-break (paper Fig.8: BGK/TRT FSLBM is fragile).
TAU: float = 0.58
LAMBDA_TRT: float = 0.015
GRAVITY: float = -0.0012

DAM_X_FRAC: float = 0.25
FILL_Z_FRAC: float = 0.5
RHO_LIQUID: float = 1.0
VOF_RHO_GAS: float = 1.0
VOF_EPSILON: float = 1.0e-3
# PLIC surface tension (Eq. 12). Too large → pressure spikes / NaNs.
# 5e-4 was too weak to flatten lattice bumps after the flow settles.
VOF_GAMMA: float = 1.5e-3
VOF_KAPPA_SMOOTH: int = 2

SSFR_THRESHOLD: float = 0.2
RAY_MARCH_STEPS: int = 800

FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 8


class VofDamBreak:
    def __init__(self, viewer: FluidViewerGL):
        self.viewer = viewer
        viewer._paused = True

        self.model = LbmModel(
            fluid_grid_res=(N, N, N),
            fluid_grid_cell_size=DH,
            tau=TAU,
            G=0.0,
            phase_mode="vof_sharp",
            vof_rho_gas=VOF_RHO_GAS,
            vof_epsilon=VOF_EPSILON,
            vof_gamma=VOF_GAMMA,
            vof_kappa_smooth=VOF_KAPPA_SMOOTH,
            lambda_trt=LAMBDA_TRT,
            initial_density=RHO_LIQUID,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=GRAVITY,
        )
        n = int(self.model.nx)
        print(
            f"VOF Dam-Break: {n}^3, tau={TAU}, lambda={LAMBDA_TRT}, "
            f"gz={GRAVITY}, gamma={VOF_GAMMA}, kappa_smooth={VOF_KAPPA_SMOOTH}, "
            f"phase_mode=vof_sharp"
        )

        self.domain = LbmDomain(self.model)
        self.domain.create_state()
        self.sim_dt = FRAME_DT / SIM_SUBSTEPS
        self.sim_time = 0.0

        dam_x = int(n * DAM_X_FRAC)
        fill_z = int(n * FILL_Z_FRAC)
        state = self.domain.state
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
        self.domain.solver._vof_sharp.update_visual_field(state, n, n, n)
        wp.synchronize_device(self.model._device)

        phi = state.phi.numpy()
        ctype = state.cell_type.numpy()
        print(
            f"  liquid={int((ctype == 2).sum())}  interface={int((ctype == 1).sum())}  "
            f"gas={int((ctype == 0).sum())}  vol0={float(phi.sum()):.1f}"
        )
        self._vol0 = float(phi.sum())

        target_gz = float(self.model.gravity_z)
        self.model.gravity_z = 0.0
        ramp = 40
        for s in range(ramp):
            self.model.gravity_z = target_gz * float(s + 1) / float(ramp)
            self.domain.step(self.sim_dt)
        self.model.gravity_z = target_gz
        wp.synchronize_device(self.model._device)
        self._vol0 = float(self.domain.state.phi.numpy().sum())

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
        for _ in range(SIM_SUBSTEPS):
            self.domain.step(self.sim_dt)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            state = self.domain.state
            phi = state.phi.numpy()
            ctype = state.cell_type.numpy()
            rho = state.density.numpy()
            liquid = ctype == 2
            interface = ctype == 1
            wet = (ctype > 0) & np.isfinite(rho)
            rho_finite = rho[np.isfinite(rho)]
            vx = float(state.velocity_x.numpy()[wet].mean()) if wet.any() else 0.0
            vz = float(state.velocity_z.numpy()[wet].mean()) if wet.any() else 0.0
            rho_max = float(rho_finite.max()) if rho_finite.size else float("nan")
            kappa = None
            vof = self.domain.solver._vof_sharp
            if float(self.model.vof_gamma) > 0.0:
                kappa = vof._kappa.numpy()
            rough = collect_interface_roughness(phi, ctype, kappa=kappa)
            print(
                f"[t={self.sim_time:.1f}s] L={liquid.sum()} I={interface.sum()} "
                f"vol={float(np.nansum(phi)):.1f} (Δ={float(np.nansum(phi))-self._vol0:+.1f}) "
                f"rho_max={rho_max:.3f} "
                f"v=({vx:+.4f},{vz:+.4f}) "
                f"h_rms={rough.height_rms:.3f} h_p2p={rough.height_p2p:.2f} "
                f"κ_rms={rough.kappa_rms:.3f} "
                f"sim={self._last_ms:.0f}ms",
                file=sys.stderr,
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.ssfr.available:
            # Liquid=1, interface=φ — avoids hollow transparent rim on φ-only field.
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

    viewer, args = init_fluid_viewer()
    newton.examples.run(VofDamBreak(viewer), args)


if __name__ == "__main__":
    main()
