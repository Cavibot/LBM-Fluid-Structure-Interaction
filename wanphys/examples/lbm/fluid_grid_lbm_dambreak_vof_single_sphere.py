# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FREE VOF dam-break with one rigid sphere (ME core + optional empirical FSI).

Default: Eq.24 walls + reconstructed-link ME (pure research path).
``--me-drag`` opts into mild submerged linear drag. ``--empirical-fsi`` /
``--showcase-fsi`` enables the full showcase buoyancy/push/drag plugin.

Run:
    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_single_sphere \\
        --viewer gl --n 48

    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_single_sphere \\
        --viewer gl --n 48 --empirical-fsi
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Any

import newton
import newton.examples
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import (
    GridLbmRigidCoupling,
    lattice_gravity_to_world,
    recommended_me_force_scale,
)
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.generic import (
    make_home_vof_model,
)
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys.examples.lbm._home_vof_empirical_sphere_fsi import (
    EmpiricalSphereFsiConfig,
    EmpiricalSphereFsiPlugin,
    me_path_linear_drag_config,
)
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig

N: int = 48
DH: float = 0.02
TAU: float = 0.51
VOF_GAMMA: float = 1.5e-3
RHO_LIQUID: float = 1.0
DAM_X_FRAC: float = 0.25
FILL_Z_FRAC: float = 0.5
RIGID_GRAVITY_Z: float = -1.0
SPHERE_RADIUS: float = 0.08
SPHERE_DENSITY: float = 0.7
WALL_THICKNESS_CELLS: float = 2.0
ME_PATH_DRAG_XY: float = 0.75
ME_PATH_DRAG_Z: float = 10.0
SPHERE_FRICTION_MU: float = 0.18
SPHERE_CONTACT_KE: float = 250.0
SPHERE_CONTACT_KD: float = 900.0
SPHERE_CONTACT_RESTITUTION: float = 0.0
FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 12
GRAVITY_RAMP_STEPS: int = 40
SSFR_THRESHOLD: float = 0.35


class HomeVofDamBreakSingleSphere:
    def __init__(
        self,
        viewer: Any,
        *,
        n: int = N,
        feedback_force_scale: float | None = None,
        empirical_fsi: bool = False,
        enable_height_eq: bool = False,
        me_in_fused: bool = True,
        me_drag: bool = False,
        me_drag_xy: float = ME_PATH_DRAG_XY,
        me_drag_z: float = ME_PATH_DRAG_Z,
    ) -> None:
        self.viewer = viewer
        if isinstance(self.viewer, FluidViewerGL):
            self.viewer._paused = True
        self._n = int(n)
        self._empirical_fsi_enabled = bool(empirical_fsi)
        self._me_drag = bool(me_drag) and not self._empirical_fsi_enabled
        self._me_drag_xy = float(me_drag_xy)
        self._me_drag_z = float(me_drag_z)
        self._enable_height_eq = bool(enable_height_eq)
        self._height_eq_armed = False
        self._height_eq_arm_after_t = 8.0
        n_ref = 48
        gravity = -0.0020 * (float(n_ref) / float(self._n))
        self._substeps = max(SIM_SUBSTEPS, 12)
        self.sim_dt = FRAME_DT / float(self._substeps)
        self._lbm_gravity_z = float(gravity)
        self._rigid_gravity_z = lattice_gravity_to_world(gravity, DH, self.sim_dt)
        self._g_matched = self._rigid_gravity_z

        use_wall_eq = bool(self._empirical_fsi_enabled)
        self.model = make_home_vof_model(
            fluid_grid_res=(self._n, self._n, self._n),
            fluid_grid_cell_size=DH,
            tau=TAU,
            gravity_z=gravity,
            vof_gamma=VOF_GAMMA,
            initial_density=RHO_LIQUID,
            vof_orphan_max_cells=max(96, self._n),
            vof_home_wall_eq=use_wall_eq,
            vof_height_eq_rate=0.025,
            vof_height_eq_every=24,
            vof_height_eq_dh_cap=0.03,
        )
        self.domain = LbmDomain(self.model)
        self.domain.create_state()
        self.sim_time = 0.0
        self.frame_count = 0
        self._last_ms = 0.0
        self._sphere_volume = (4.0 / 3.0) * math.pi * SPHERE_RADIUS**3
        self._empirical: EmpiricalSphereFsiPlugin | None = None

        if feedback_force_scale is None:
            feedback_scale = recommended_me_force_scale(
                DH, self.sim_dt, rho_fluid=RHO_LIQUID, match_rigid_g=True
            )
        else:
            feedback_scale = float(feedback_force_scale)
        self._feedback_force_scale = feedback_scale
        self.model.vof_home_me_in_fused = bool(me_in_fused)

        self._init_fluid()
        self._init_rigid(feedback_force_scale=feedback_scale)
        self._ramp_gravity()

        self.ssfr: ScreenSpaceFluidRenderer | None = None
        if isinstance(self.viewer, FluidViewerGL):
            ssfr = ScreenSpaceFluidRenderer(
                viewer=self.viewer,
                max_particles=1,
                particle_radius=0.01,
                device=self.model._device,
            )
            self.ssfr = ssfr
            self.viewer.register_post_render_callback(lambda v: ssfr.render(v))

        print(
            f"HOME-VOF single sphere: {self._n}^3, gz={gravity:.5f}, "
            f"feedback=ME, force_scale={self._feedback_force_scale:.4g}, "
            f"me_in_fused={'on' if self.model.vof_home_me_in_fused else 'off'}, "
            f"me_drag={'on' if self._me_drag else 'off'}, "
            f"showcase_fsi={'on' if self._empirical_fsi_enabled else 'off'}, "
            f"wall_eq={use_wall_eq}, height_eq={self._enable_height_eq}"
        )

    def _init_fluid(self) -> None:
        dam_x = int(self._n * DAM_X_FRAC)
        fill_z = int(self._n * FILL_Z_FRAC)
        state = self.domain.state
        home = self.domain.solver._home_fp32
        assert home is not None
        home.seed_dam_break(
            state,
            dam_x=dam_x,
            fill_z=fill_z,
            rho_liquid=RHO_LIQUID,
            hydrostatic=False,
        )
        out = self.domain._state_out
        home.sync_to_state(out)
        for name in ("solid_phi", "solid_body_id", "vel_solid_u", "vel_solid_v", "vel_solid_w"):
            wp.copy(getattr(out, name), getattr(state, name))
        self.domain.solver._vof_sharp.update_visual_field(state, self._n, self._n, self._n)
        wp.synchronize_device(self.model._device)

    def _init_rigid(self, *, feedback_force_scale: float) -> None:
        n = self._n
        world = float(n) * DH
        wall_t = WALL_THICKNESS_CELLS * DH
        radius = SPHERE_RADIUS
        z_floor = radius + 1.5 * DH
        builder = RigidModelBuilder(gravity=self._rigid_gravity_z)
        wall_cfg = ShapeConfig(
            density=0.0,
            is_visible=False,
            is_solid=True,
            has_shape_collision=True,
            ke=SPHERE_CONTACT_KE,
            kd=SPHERE_CONTACT_KD,
            restitution=SPHERE_CONTACT_RESTITUTION,
        )

        def add_wall(label: str, center: tuple[float, float, float], he: tuple[float, float, float]) -> None:
            body = builder.add_body(position=center, label=label)
            builder.add_shape_box(body, hx=he[0], hy=he[1], hz=he[2], cfg=wall_cfg)

        add_wall("floor", (world * 0.5, world * 0.5, -wall_t * 0.5), (world * 0.5, world * 0.5, wall_t * 0.5))
        add_wall("ceil", (world * 0.5, world * 0.5, world + wall_t * 0.5), (world * 0.5, world * 0.5, wall_t * 0.5))
        add_wall("xmin", (-wall_t * 0.5, world * 0.5, world * 0.5), (wall_t * 0.5, world * 0.5, world * 0.5))
        add_wall("xmax", (world + wall_t * 0.5, world * 0.5, world * 0.5), (wall_t * 0.5, world * 0.5, world * 0.5))
        add_wall("ymin", (world * 0.5, -wall_t * 0.5, world * 0.5), (world * 0.5, wall_t * 0.5, world * 0.5))
        add_wall("ymax", (world * 0.5, world + wall_t * 0.5, world * 0.5), (world * 0.5, wall_t * 0.5, world * 0.5))

        cfg = ShapeConfig(
            density=SPHERE_DENSITY,
            is_visible=True,
            is_solid=True,
            mu=SPHERE_FRICTION_MU,
            ke=SPHERE_CONTACT_KE,
            kd=SPHERE_CONTACT_KD,
            restitution=SPHERE_CONTACT_RESTITUTION,
        )
        center = (world * 0.32, world * 0.5, z_floor)
        self.sphere_body_id = builder.add_body(position=center, label="sphere")
        builder.add_shape_sphere(self.sphere_body_id, radius=radius, cfg=cfg)

        self.rigid_domain = RigidDomain(builder.finalize(device=self.model._device))
        self.rigid_domain.create_state()
        if self.viewer is not None and hasattr(self.viewer, "set_model"):
            self.rigid_domain.model.setup_viewer(self.viewer)

        self.coupling = GridLbmRigidCoupling(self.domain, self.rigid_domain)
        self.coupling.add_body_sphere(body_idx=self.sphere_body_id, radius=radius)
        self.coupling.set_rigid_dynamics_enabled(False)
        self.coupling.set_two_way_feedback_enabled(True, force_scale=feedback_force_scale)
        self.coupling.set_feedback_mode("momentum_exchange")
        self.coupling.set_me_integration_mode("impulse")

        if self._empirical_fsi_enabled:
            cfg = EmpiricalSphereFsiConfig()
        elif self._me_drag:
            cfg = me_path_linear_drag_config(
                drag_xy=self._me_drag_xy,
                drag_z=self._me_drag_z,
            )
        else:
            cfg = None

        if cfg is not None:
            self._empirical = EmpiricalSphereFsiPlugin(
                device=str(self.model._device),
                body_ids=(self.sphere_body_id,),
                densities=(SPHERE_DENSITY,),
                radius=radius,
                volume=self._sphere_volume,
                rho_liquid=RHO_LIQUID,
                gravity_abs=abs(self._rigid_gravity_z),
                dh=DH,
                nx=self._n,
                ny=self._n,
                nz=self._n,
                config=cfg,
            )

    def _ramp_gravity(self) -> None:
        target_lbm = float(self.model.gravity_z)
        target_rigid = float(self._rigid_gravity_z)
        self.model.gravity_z = 0.0
        self.rigid_domain.model.set_gravity((0.0, 0.0, 0.0))
        for step_index in range(GRAVITY_RAMP_STEPS):
            alpha = float(step_index + 1) / float(GRAVITY_RAMP_STEPS)
            self.model.gravity_z = alpha * target_lbm
            self.rigid_domain.model.set_gravity((0.0, 0.0, alpha * target_rigid))
            self._step_coupled()
        self.model.gravity_z = target_lbm
        self.rigid_domain.model.set_gravity((0.0, 0.0, target_rigid))

    def _step_coupled(self) -> None:
        self.coupling.step(self.sim_dt)
        if self._empirical is not None:
            state = self.domain.state
            rigid = self.rigid_domain.state
            self._empirical.apply(
                phi=state.phi,
                cell=state.cell_type,
                solid=state.solid_phi,
                ux=state.velocity_x,
                uy=state.velocity_y,
                uz=state.velocity_z,
                body_q=rigid.body_q,
                body_qd=rigid.body_qd,
                body_f_apply=rigid.apply_body_forces,
                vel_scale=DH / max(self.sim_dt, 1.0e-12),
            )
        self.rigid_domain.step(self.sim_dt)

    def step(self) -> None:
        t0 = time.perf_counter()
        home = self.domain.solver._home_fp32
        if (
            self._enable_height_eq
            and home is not None
            and not self._height_eq_armed
            and self.sim_time >= self._height_eq_arm_after_t
        ):
            self.model.vof_height_eq = True
            self._height_eq_armed = True
            print(f"  [height-eq] armed at t={self.sim_time:.2f}", file=sys.stderr)
        for _ in range(self._substeps):
            self._step_coupled()
            self.sim_time += self.sim_dt
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.frame_count += 1
        if self.frame_count % 60 == 0:
            pos = np.asarray(
                self.rigid_domain.state.get_body_position(self.sphere_body_id),
                dtype=np.float64,
            )
            me_note = ""
            J = self.coupling.last_me_impulse
            if J is not None and J.shape[0] > self.sphere_body_id:
                j = J[self.sphere_body_id, 0:3]
                me_note = (
                    f" ME_J={np.linalg.norm(j):.3g}"
                    f" res={self.coupling.last_me_apply_rel:.2e}"
                )
            print(
                f"[t={self.sim_time:.1f}s] sphere=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                f"sim={self._last_ms:.0f}ms{me_note}",
                file=sys.stderr,
                flush=True,
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.rigid_domain.state.as_newton_state())
        state = self.domain.state
        if self.ssfr is not None:
            self.domain.solver._vof_sharp.update_visual_field(state, self._n, self._n, self._n)
            self.ssfr.set_density_field(state.density, threshold=SSFR_THRESHOLD)
        self.viewer.end_frame()


def main() -> None:
    parser = argparse.ArgumentParser(description="HOME-VOF dam-break single sphere")
    parser.add_argument("--viewer", default="gl")
    parser.add_argument("--n", type=int, default=N)
    parser.add_argument(
        "--feedback-force-scale",
        type=float,
        default=None,
        help="LBM→rigid ME scale (default: recommended_me_force_scale).",
    )
    parser.add_argument(
        "--empirical-fsi",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Showcase buoyancy/push/drag plugin (default: off). Alias: --showcase-fsi.",
    )
    parser.add_argument(
        "--showcase-fsi",
        action="store_true",
        help="Same as --empirical-fsi (showcase plugin + eq-wall).",
    )
    parser.add_argument(
        "--me-in-fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accumulate ME inside fused solid pulls (default: on).",
    )
    parser.add_argument(
        "--me-drag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in submerged linear drag on ME path (default: off).",
    )
    parser.add_argument("--me-drag-xy", type=float, default=ME_PATH_DRAG_XY)
    parser.add_argument("--me-drag-z", type=float, default=ME_PATH_DRAG_Z)
    parser.add_argument("--height-eq", action="store_true")
    viewer, args = init_fluid_viewer(parser)
    empirical = bool(args.empirical_fsi) or bool(args.showcase_fsi)
    example = HomeVofDamBreakSingleSphere(
        viewer,
        n=int(args.n),
        feedback_force_scale=(
            None
            if args.feedback_force_scale is None
            else float(args.feedback_force_scale)
        ),
        empirical_fsi=empirical,
        enable_height_eq=bool(args.height_eq),
        me_in_fused=bool(args.me_in_fused),
        me_drag=bool(args.me_drag),
        me_drag_xy=float(args.me_drag_xy),
        me_drag_z=float(args.me_drag_z),
    )
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
