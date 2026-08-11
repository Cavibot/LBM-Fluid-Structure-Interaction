# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM foam pair (Phase-4 visual gate).

Two close bubbles under disjoining pressure should not coalesce.
Renders free surface from ``phi``; stderr prints bubble_count, COM distance,
and sum(disjoin) probe.

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom

Run::

    python -m wanphys.examples.home_fslbm.fluid_grid_home_fslbm_foam_pair --viewer gl
"""

from __future__ import annotations

import sys
import time

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_fslbm import HomeFslbmDomain, HomeFslbmModel
from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
from wanphys._src.fluid.fluid_grid.home_fslbm import kernels_foam
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer

N: int = 64
DH: float = 0.02
R: float = 6.0
# Centres ~15 lu apart -> surface gap ~3
C0 = (24.0, 32.0, 32.0)
C1 = (39.0, 32.0, 32.0)

OMEGA: float = 1.0
GRAVITY_Z: float = 0.0
SURFACE_TENSION: float = C.SURFACE_TENSION
DISJOIN: float = C.DISJOINT_FACTOR

SSFR_THRESHOLD: float = 0.5
RAY_MARCH_STEPS: int = 1200
FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 2
CAMERA_PITCH: float = -18.0
CAMERA_YAW: float = -180.0


def _paint_two_bubbles(state, nx: int, ny: int, nz: int) -> None:
    flag = np.full((nx, ny, nz), C.TYPE_F, dtype=np.uint8)
    phi = np.ones((nx, ny, nz), dtype=np.float32)
    mass = np.ones((nx, ny, nz), dtype=np.float32)
    for cx0, cy0, cz0 in (C0, C1):
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


def _sync_double_buffer(domain: HomeFslbmDomain) -> None:
    src = domain.state
    dst = domain._state_out
    assert dst is not None
    for name in (
        "f_mom", "f_mom_post", "flag", "mass", "massex", "phi",
        "tag_matrix", "previous_tag", "previous_merge_tag",
        "bubble_volume", "bubble_init_volume", "bubble_rho",
        "g_mom", "g_mom_post", "disjoin_force", "label_matrix",
        "input_matrix", "merge_detector", "force_x", "force_y", "force_z",
    ):
        wp.copy(getattr(dst, name), getattr(src, name))
    dst.bubble_count = src.bubble_count
    dst.label_num = src.label_num


def _tag_coms(tag: np.ndarray) -> list[np.ndarray]:
    ids = sorted(int(x) for x in np.unique(tag) if x > 0)
    return [np.argwhere(tag == tid).mean(axis=0) for tid in ids]


class HomeFslbmFoamPair:
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
            disjoin_factor=DISJOIN,
        )
        print(
            f"HOME-FSLBM Foam Pair: {N}^3, r={R}, centres={C0}/{C1}, "
            f"disjoin={DISJOIN}"
        )
        self.domain = HomeFslbmDomain(self.model)
        self.domain.create_state()
        self.sim_time = 0.0
        self.frame_count = 0
        self._last_ms = 0.0

        self.domain.solver.initialize_equilibrium(self.domain.state)
        _paint_two_bubbles(self.domain.state, N, N, N)
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
        # Probe disjoin before solver reset clears it
        st = self.domain.state
        st.disjoin_force.zero_()
        sol = self.domain.solver
        wp.launch(
            kernels_foam.calculate_disjoint_kernel,
            dim=(N, N, N),
            inputs=[
                st.flag, st.phi, st.mass, st.massex, st.f_mom, st.tag_matrix,
                st.disjoin_force, sol._cx, sol._cy, sol._cz, sol._opposite,
                N, N, N, sol._stride,
            ],
        )
        wp.synchronize_device(self.model._device)
        sum_dj = float(st.disjoin_force.numpy().sum())

        for _ in range(SIM_SUBSTEPS):
            self.domain.step(1.0)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT
        self.frame_count += 1

        if self.frame_count % 30 == 0:
            tag = self.domain.state.tag_matrix.numpy()
            coms = _tag_coms(tag)
            dist = float(np.linalg.norm(coms[0] - coms[1])) if len(coms) >= 2 else -1.0
            print(
                f"[t={self.sim_time:.1f}s] bubbles={self.domain.state.bubble_count} "
                f"merge={self.domain.state.merge_flag} dist={dist:.2f} "
                f"sum(disjoin)~{sum_dj:.3f} sim={self._last_ms:.0f}ms",
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
    newton.examples.run(HomeFslbmFoamPair(viewer), args)


if __name__ == "__main__":
    main()
