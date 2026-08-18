# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM free-surface dam-break (Phase-2 visual smoke).

Water column collapses under gravity; free surface is rendered from ``phi``
via screen-space density ray-marching (threshold ~0.5).

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom
"""

from __future__ import annotations

import sys
import time

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_fslbm import HomeFslbmDomain, HomeFslbmModel
from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer

# ---------------------------------------------------------------------------
# Scene scale aligned with wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt:
# 128³ lattice, DH=0.02 → world ≈ 2.56 m; dam fills left 25% (full y/z).
N: int = 128
DH: float = 0.02
DAM_X_FRAC: float = 0.25
DAM_Z_FRAC: float = 0.50  # free surface above; avoid ceiling-sealed full-height column

OMEGA: float = 1.998
# Plan / dam_break_64 / falling_droplet golden use |g|=1e-3. The temporary
# 1e-4 workaround was only needed while gravity was incorrectly accumulated.
GRAVITY_Z: float = -1.0e-3
SURFACE_TENSION: float = 6.0 * 4e-3

INTERFACE_DELTA: float = 1.5

SSFR_THRESHOLD: float = 0.5
# 128³ diagonal: ~√3×128 ≈ 222 lu; match LBM TRT dambreak sample budget.
RAY_MARCH_STEPS: int = 1600

FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 4  # lattice steps per frame (HOME dt = 1)
GRAVITY_RAMP_STEPS: int = 60

# Camera: frame the ~2.56 m cube from +X, looking toward -X (Z-up).
CAMERA_PITCH: float = -18.0
CAMERA_YAW: float = -180.0


def _box_signed_distance(
    i: np.ndarray,
    j: np.ndarray,
    k: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
) -> np.ndarray:
    """Axis-aligned box SDF: negative inside, positive outside (cell centers)."""
    px = i.astype(np.float32)
    py = j.astype(np.float32)
    pz = k.astype(np.float32)

    dx = np.maximum(np.maximum(x0 - px, px - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - py, py - y1), 0.0)
    dz = np.maximum(np.maximum(z0 - pz, pz - z1), 0.0)
    outside = np.sqrt(dx * dx + dy * dy + dz * dz)

    inside_x = np.minimum(px - x0, x1 - px)
    inside_y = np.minimum(py - y0, y1 - py)
    inside_z = np.minimum(pz - z0, z1 - pz)
    inside = np.minimum(np.minimum(inside_x, inside_y), inside_z)

    in_box = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1) & (pz >= z0) & (pz <= z1)
    return np.where(in_box, -inside, outside).astype(np.float32)


def _setup_water_column(state, nx: int, ny: int, nz: int) -> None:
    """Fill left dam slab (LBM-style); rest of domain is gas.

    Matches ``fluid_grid_lbm_dambreak_trt`` in x/y: water for
    ``i < DAM_X_FRAC * nx``, full interior y. Height is ``DAM_Z_FRAC`` of
    the domain so a free surface remains above the column.
    """
    ii, jj, kk = np.meshgrid(
        np.arange(nx, dtype=np.int32),
        np.arange(ny, dtype=np.int32),
        np.arange(nz, dtype=np.int32),
        indexing="ij",
    )

    dam_x = max(2, int(float(nx) * DAM_X_FRAC))
    dam_z = max(2, int(float(nz) * DAM_Z_FRAC))
    # Soft box: i ∈ [1, dam_x), full y interior, z ∈ [1, dam_z).
    x0, x1 = 0.5, float(dam_x) - 0.5
    y0, y1 = 0.5, float(ny) - 1.5
    z0, z1 = 0.5, float(dam_z) - 0.5

    sdf = _box_signed_distance(ii, jj, kk, x0, x1, y0, y1, z0, z1)
    phi = np.clip((-sdf) / INTERFACE_DELTA + 0.5, 0.0, 1.0).astype(np.float32)

    flag = np.full((nx, ny, nz), C.CellFlag.TYPE_G, dtype=np.uint8)
    mass = np.zeros((nx, ny, nz), dtype=np.float32)

    fluid = phi >= 1.0 - 1e-6
    gas = phi <= 1e-6
    interface = ~(fluid | gas)

    flag[fluid] = C.CellFlag.TYPE_F
    mass[fluid] = 1.0
    flag[gas] = C.CellFlag.TYPE_G
    mass[gas] = 0.0
    flag[interface] = C.CellFlag.TYPE_I
    mass[interface] = phi[interface]

    device = state.flag.device
    wp.copy(state.phi, wp.array(phi, dtype=float, device=device))
    wp.copy(state.flag, wp.array(flag, dtype=wp.uint8, device=device))
    wp.copy(state.mass, wp.array(mass, dtype=float, device=device))


def _set_boundary_walls(state, nx: int, ny: int, nz: int) -> None:
    """Mark the six domain faces as TYPE_S (phi/mass cleared)."""
    flag = state.flag.numpy()
    phi = state.phi.numpy()
    mass = state.mass.numpy()

    for arr, solid_val in ((flag, C.CellFlag.TYPE_S), (phi, 0.0), (mass, 0.0)):
        arr[0, :, :] = solid_val
        arr[-1, :, :] = solid_val
        arr[:, 0, :] = solid_val
        arr[:, -1, :] = solid_val
        arr[:, :, 0] = solid_val
        arr[:, :, -1] = solid_val

    device = state.flag.device
    wp.copy(state.flag, wp.array(flag, dtype=wp.uint8, device=device))
    wp.copy(state.phi, wp.array(phi, dtype=float, device=device))
    wp.copy(state.mass, wp.array(mass, dtype=float, device=device))


def _init_rest_state(domain: HomeFslbmDomain) -> None:
    """Rest equilibrium + tag_matrix=-1 (ambient gas, no bubbles)."""
    domain.solver.initialize_equilibrium(domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))
    state = domain.state
    tag = np.full((N, N, N), -1, dtype=np.int32)
    wp.copy(state.tag_matrix, wp.array(tag, dtype=wp.int32, device=state.tag_matrix.device))
    state.massex.zero_()
    state.force_x.zero_()
    state.force_y.zero_()
    state.force_z.zero_()
    state.delta_phi.zero_()


def _setup_camera(viewer, world_size: float) -> None:
    """Place the GL camera so the dam-break cube fills the view."""
    if not hasattr(viewer, "set_camera"):
        return
    # From +X looking toward -X; y at domain mid, z elevated for a slight top-down tilt.
    viewer.set_camera(
        pos=wp.vec3(world_size * 2.15, world_size * 0.5, world_size * 0.85),
        pitch=CAMERA_PITCH,
        yaw=CAMERA_YAW,
    )


def _sync_double_buffer(domain: HomeFslbmDomain) -> None:
    """Copy active state into the back buffer after host-side init."""
    src = domain.state
    dst = domain._state_out
    assert dst is not None
    if src.shares_buffers_with(dst):
        return
    for name in (
        "f_mom",
        "f_mom_post",
        "flag",
        "mass",
        "massex",
        "phi",
        "force_x",
        "force_y",
        "force_z",
        "delta_phi",
        "tag_matrix",
    ):
        wp.copy(getattr(dst, name), getattr(src, name))


class HomeFslbmDamBreak:
    def __init__(self, viewer):
        self.viewer = viewer
        if hasattr(viewer, "_paused"):
            viewer._paused = True

        world_size = float(N) * DH
        dam_x = max(2, int(float(N) * DAM_X_FRAC))
        dam_z = max(2, int(float(N) * DAM_Z_FRAC))
        self.model = HomeFslbmModel(
            fluid_grid_res=(N, N, N),
            fluid_grid_cell_size=DH,
            omega=OMEGA,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=GRAVITY_Z,
            surface_tension=SURFACE_TENSION,
            atmosphere_open=False,
        )
        print(
            f"HOME-FSLBM Dam-Break: {N}^3, dh={DH}, world={world_size:.3f}m, "
            f"omega={OMEGA}, gz={GRAVITY_Z}, sigma={SURFACE_TENSION:.4g}, "
            f"dam at x<{dam_x}, z<{dam_z}"
        )

        self.domain = HomeFslbmDomain(self.model)
        self.domain.create_state()
        self.sim_time = 0.0
        self._lattice_steps = 0
        self.frame_count = 0
        self._last_ms = 0.0

        _setup_water_column(self.domain.state, N, N, N)
        _set_boundary_walls(self.domain.state, N, N, N)
        _init_rest_state(self.domain)
        _sync_double_buffer(self.domain)
        wp.synchronize_device(self.model._device)

        phi_np = self.domain.state.phi.numpy()
        water = int((phi_np > SSFR_THRESHOLD).sum())
        mass_sum = float(self.domain.state.mass.numpy().sum())
        print(f"  Water cells (phi>{SSFR_THRESHOLD}): {water}, sum(mass)={mass_sum:.1f}")

        target_gz = float(self.model.gravity_z)
        self.model.gravity_z = 0.0
        for s in range(GRAVITY_RAMP_STEPS):
            self.model.gravity_z = target_gz * float(s + 1) / float(GRAVITY_RAMP_STEPS)
            self.domain.step(1.0)
            self._lattice_steps += 1
        self.model.gravity_z = target_gz
        wp.synchronize_device(self.model._device)

        self.ssfr: ScreenSpaceFluidRenderer | None = None
        if isinstance(viewer, FluidViewerGL):
            _setup_camera(viewer, world_size)
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
            self._lattice_steps += 1
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += FRAME_DT
        self.frame_count += 1

        if self.frame_count % 30 == 0:
            phi = self.domain.state.phi.numpy()
            mass = self.domain.state.mass.numpy()
            w = phi > SSFR_THRESHOLD
            n_water = int(w.sum())
            mass_sum = float(mass.sum())
            if n_water > 0:
                c = np.argwhere(w).mean(axis=0)
                print(
                    f"[t={self.sim_time:.1f}s] water={n_water} "
                    f"COM=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) "
                    f"sum(mass)={mass_sum:.1f} sim={self._last_ms:.0f}ms",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[t={self.sim_time:.1f}s] water=0 sum(mass)={mass_sum:.1f} "
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
    newton.examples.run(HomeFslbmDamBreak(viewer), args)


if __name__ == "__main__":
    main()
