# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM rising bubble + dissolved gas (Phase-4 visual).

Single buoyant bubble with D3Q7 gas field; renders ``phi`` and prints
``max(c_value)`` / interface mean concentration.

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom

Run::

    python -m wanphys.examples.home_fslbm.fluid_grid_home_fslbm_rising_bubble_gas --viewer gl
"""

from __future__ import annotations

import sys
import time

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_fslbm import HomeFslbmDomain, HomeFslbmModel
from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer

N: int = 64
DH: float = 0.02
R: float = 8.0
CENTRE = (32.0, 32.0, 20.0)

OMEGA: float = 1.0
GRAVITY_Z: float = -1.0e-4
SURFACE_TENSION: float = C.SURFACE_TENSION
C0_GAS: float = 1.0e-3  # initial dissolved concentration

SSFR_THRESHOLD: float = 0.5
RAY_MARCH_STEPS: int = 1200
FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 2
CAMERA_PITCH: float = -18.0
CAMERA_YAW: float = -180.0


def _paint_bubble(state, nx: int, ny: int, nz: int) -> None:
    flag = np.full((nx, ny, nz), C.TYPE_F, dtype=np.uint8)
    phi = np.ones((nx, ny, nz), dtype=np.float32)
    mass = np.ones((nx, ny, nz), dtype=np.float32)
    cx0, cy0, cz0 = CENTRE
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                d = np.sqrt((i - cx0) ** 2 + (j - cy0) ** 2 + (k - cz0) ** 2)
                if d < R - 0.5:
                    flag[i, j, k] = C.TYPE_G
                    phi[i, j, k] = 0.0
                    mass[i, j, k] = 0.0
                elif d <= R + 0.5:
                    flag[i, j, k] = C.TYPE_I
                    phi[i, j, k] = 0.5
                    mass[i, j, k] = 0.5
    for a, v in ((flag, C.TYPE_S), (phi, 0.0), (mass, 0.0)):
        a[0, :, :] = v
        a[-1, :, :] = v
        a[:, 0, :] = v
        a[:, -1, :] = v
        a[:, :, 0] = v
        a[:, :, -1] = v
    device = state.flag.device
    wp.copy(state.flag, wp.array(flag, dtype=wp.uint8, device=device))
    wp.copy(state.phi, wp.array(phi, dtype=float, device=device))
    wp.copy(state.mass, wp.array(mass, dtype=float, device=device))


def _init_gas(state, nx: int, ny: int, nz: int) -> None:
    stride = nx * ny * nz
    g = np.zeros(7 * stride, dtype=np.float32)
    w = [0.25] + [0.125] * 6
    for di, wi in enumerate(w):
        g[di * stride : (di + 1) * stride] = wi * C0_GAS
    device = state.g_mom.device
    wp.copy(state.g_mom, wp.array(g, dtype=float, device=device))
    wp.copy(state.g_mom_post, state.g_mom)
    wp.copy(
        state.c_value,
        wp.array(np.full((nx, ny, nz), C0_GAS, dtype=np.float32), dtype=float, device=device),
    )


def _sync_double_buffer(domain: HomeFslbmDomain) -> None:
    src = domain.state
    dst = domain._state_out
    assert dst is not None
    for name in (
        "f_mom", "f_mom_post", "flag", "mass", "massex", "phi",
        "tag_matrix", "previous_tag", "previous_merge_tag",
        "bubble_volume", "bubble_init_volume", "bubble_rho",
        "g_mom", "g_mom_post", "c_value", "src", "delta_g",
        "disjoin_force", "label_matrix", "input_matrix", "merge_detector",
        "force_x", "force_y", "force_z",
    ):
        wp.copy(getattr(dst, name), getattr(src, name))
    dst.bubble_count = src.bubble_count
    dst.label_num = src.label_num


class HomeFslbmRisingBubbleGas:
    def __init__(self, viewer):
        self.viewer = viewer
        if hasattr(viewer, "_paused"):
            viewer._paused = True

        world_size = float(N) * DH
        self.model = HomeFslbmModel(
            fluid_grid_res=(N, N, N),
            fluid_grid_cell_size=DH,
            omega=OMEGA,
            gravity_z=GRAVITY_Z,
            surface_tension=SURFACE_TENSION,
            henry_constant=C.HENRY_CONSTANT,
        )
        print(
            f"HOME-FSLBM Rising Bubble + Gas: {N}^3, r={R}, gz={GRAVITY_Z}, "
            f"c0={C0_GAS}"
        )
        self.domain = HomeFslbmDomain(self.model)
        self.domain.create_state()
        self.sim_time = 0.0
        self.frame_count = 0
        self._last_ms = 0.0

        self.domain.solver.initialize_equilibrium(self.domain.state)
        _paint_bubble(self.domain.state, N, N, N)
        _init_gas(self.domain.state, N, N, N)
        self.domain.solver.init_bubbles(self.domain.state)
        _sync_double_buffer(self.domain)
        wp.synchronize_device(self.model._device)
        print(f"  bubble_count={self.domain.state.bubble_count}")

        self.ssfr: ScreenSpaceFluidRenderer | None = None
        if isinstance(viewer, FluidViewerGL):
            if hasattr(viewer, "set_camera"):
                viewer.set_camera(
                    pos=wp.vec3(world_size * 2.15, world_size * 0.5, world_size * 0.85),
                    pitch=CAMERA_PITCH,
                    yaw=CAMERA_YAW,
                )
            self.ssfr = ScreenSpaceFluidRenderer(
                viewer=viewer,
                max_particles=1,
                particle_radius=0.01,
                device=self.model._device,
            )
            viewer.register_post_render_callback(lambda v: self.ssfr.render(v))
        print("Controls: [Space] unpause  [R] reset  [mouse] orbit")

    def step(self):
        t0 = time.perf_counter()
        for _ in range(SIM_SUBSTEPS):
            self.domain.step(1.0)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT
        self.frame_count += 1

        if self.frame_count % 30 == 0:
            flag = self.domain.state.flag.numpy()
            c = self.domain.state.c_value.numpy()
            tag = self.domain.state.tag_matrix.numpy()
            iface = flag == C.TYPE_I
            c_iface = float(c[iface].mean()) if iface.any() else 0.0
            gas = tag > 0
            com = np.argwhere(gas).mean(axis=0) if gas.any() else np.zeros(3)
            print(
                f"[t={self.sim_time:.1f}s] bubbles={self.domain.state.bubble_count} "
                f"COM=({com[0]:.0f},{com[1]:.0f},{com[2]:.0f}) "
                f"max(c)={float(np.max(c)):.3e} mean_I(c)={c_iface:.3e} "
                f"sim={self._last_ms:.0f}ms",
                file=sys.stderr,
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.ssfr is not None and self.ssfr.available:
            self.ssfr.set_density_field(
                density=self.domain.state.phi,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=DH,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()


def main():
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer

    viewer, args = init_fluid_viewer()
    newton.examples.run(HomeFslbmRisingBubbleGas(viewer), args)


if __name__ == "__main__":
    main()
