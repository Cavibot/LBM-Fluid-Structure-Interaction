# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
# pyright: reportInvalidTypeForm=false

"""Fluid-grid robot two-way coupling example (ANYmal C walking through water).

The robot is built as a RigidDomain using the WanPhys MuJoCo solver factory.
The water is a FluidGridLiquidDomain using FluidGridLiquidSolver.
GridLiquidRigidCoupling advances both domains with two-way pressure/velocity coupling.

Run with:
    python -m wanphys.examples.fluid_grid_liquid_robot --viewer gl
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton._src.geometry.types import GeoType
from newton.viewer import ViewerGL
from newton.examples.robot.example_robot_anymal_c_walk import quat_rotate_inverse, lab_to_mujoco, mujoco_to_lab

from wanphys._src.fluid import FluidGridLiquidDomain, FluidGridLiquidModel, FluidGridLiquidSolver
from wanphys._src.fluid.fluid_grid.coupling import GridLiquidRigidCoupling
from wanphys.rigid import (
    RigidDomain,
    RigidModel,
    RigidModelBuilder,
    RigidState,
    ShapeConfig,
    create_mujoco_solver,
    register_mujoco_solver_attributes,
)


# -----------------------------------------------------------------------------
# User-editable config
# -----------------------------------------------------------------------------
GRID_RESOLUTION = (48, 32, 32)
CELL_SIZE = 0.05
PRESSURE_ITERATIONS = 30
EXTRAPOLATION_ITERATIONS = 2
PRESSURE_SOLVER = "jacobi"
USE_GRAPH = False

# Particle fill config
PARTICLE_SPACING_SCALE = 0.25
PARTICLE_RADIUS_SCALE = 0.45
FILL_RATIO_X = 0.5
FILL_RATIO_Y = 1.0
FILL_RATIO_Z = 0.3
FILL_OFFSET_X = 1.0
FILL_OFFSET_Y = 0.1
FILL_OFFSET_Z = 1.0

# Runtime config
FPS = 60
SUBSTEPS = 4
WARMUP_STEPS = 0
START_PAUSED = True
LOG_INTERVAL_FRAMES = 60

# Visualization mode
VISUALIZATION_MODE = "particles"
ENABLE_TRANSPARENT_GRID = True
USE_NARROW_BAND_SURFACE = False
USE_OUTER_SURFACE_ONLY = False
FLUID_RENDER_RES = (64, 64, 64)
FLUID_RENDER_ALPHA = 0.20
FLUID_RENDER_ROUGHNESS = 0.15
FLUID_RENDER_METALLIC = 0.00

@wp.kernel
def init_particles_grid(
    particles: wp.array(dtype=wp.vec3),
    fill_min: wp.vec3,
    spacing: float,
    nx: int,
    ny: int,
):
    tid = wp.tid()
    layer_size = nx * ny
    k = tid // layer_size
    rem = tid % layer_size
    j = rem // nx
    i = rem % nx

    px = fill_min[0] + (wp.float32(i) + 0.5) * spacing
    py = fill_min[1] + (wp.float32(j) + 0.5) * spacing
    pz = fill_min[2] + (wp.float32(k) + 0.5) * spacing

    particles[tid] = wp.vec3(px, py, pz)


def compute_rigid_obs(
    actions: torch.Tensor,
    state: RigidState,
    joint_pos_initial: torch.Tensor,
    device: torch.device | str,
    indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    joint_q = wp.to_torch(state.joint_q).to(device=device, dtype=torch.float32)
    joint_qd = wp.to_torch(state.joint_qd).to(device=device, dtype=torch.float32)
    root_quat_w = joint_q[3:7].unsqueeze(0)
    root_lin_vel_w = joint_qd[:3].unsqueeze(0)
    root_ang_vel_w = joint_qd[3:6].unsqueeze(0)
    joint_pos_current = joint_q[7:].unsqueeze(0)
    joint_vel_current = joint_qd[6:].unsqueeze(0)
    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    return torch.cat(
        [vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions],
        dim=1,
    )


class Example:
    def __init__(self, viewer: ViewerGL):
        self.viewer = viewer
        self.device = wp.get_device()

        self.fps = FPS
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_count = 0
        self.total_step_seconds = 0.0

        self.viewer._paused = bool(START_PAUSED)
        self.use_particle_visualization = VISUALIZATION_MODE == "particles"

        self._build_robot()
        self._build_coupling()

        if self.use_particle_visualization:
            self._build_particle_visualization()

        self.rigid_domain.model.setup_viewer(self.viewer)
        self.render_state = self.coupling.rigid_state

        # Adjust camera
        self.viewer.set_camera(pos=wp.vec3(1.6, -1.0, 1.0), pitch=-15.0, yaw=45.0)

        # Apply one step to see the right initial state in the viewer log
        self.step()

    def _build_robot(self):
        builder = RigidModelBuilder(up_axis=newton.Axis.Z)
        builder.set_default_joint_config(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.set_default_shape_config(ShapeConfig(ke=5.0e4, kd=5.0e2, kf=1.0e3, mu=0.75))
        register_mujoco_solver_attributes(builder)

        self.asset_path: Path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(self.asset_path / "urdf" / "anymal.urdf")
        builder.add_urdf(
            stage_path,
            xform=wp.transform(wp.vec3(1.6, 0.35, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        builder.add_ground_plane()

        initial_q = {
            "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
            "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
            "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
            "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
        }
        for name, value in initial_q.items():
            idx = next((i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}")), -1)
            if idx >= 0:
                builder.joint_q[idx + 6] = value

        for i in range(builder.joint_dof_count):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        rigid_model = builder.finalize()

        # Rigid domain with MuJoCo solver
        self.rigid_domain: RigidDomain = RigidDomain(
            rigid_model,
            solver=create_mujoco_solver(rigid_model, ls_iterations=50, njmax=50),
        )

        # Control & policy setup
        q0 = wp.to_torch(rigid_model.state().joint_q)
        self.torch_device = q0.device
        self.joint_pos_initial = q0[7:].unsqueeze(0).detach().clone()
        self.act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)

        policy_path = str(self.asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")
        self.policy = torch.jit.load(policy_path, map_location=self.torch_device)

        self.lab_to_mujoco_indices = torch.tensor(
            [lab_to_mujoco[i] for i in range(len(lab_to_mujoco))], device=self.torch_device
        )
        self.mujoco_to_lab_indices = torch.tensor(
            [mujoco_to_lab[i] for i in range(len(mujoco_to_lab))], device=self.torch_device
        )
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self._auto_forward = True

    def _build_coupling(self):
        # Fluid model & solver
        nx, ny, nz = GRID_RESOLUTION
        particle_spacing = CELL_SIZE * PARTICLE_SPACING_SCALE
        particle_radius = CELL_SIZE * PARTICLE_RADIUS_SCALE

        fill_nx = max(1, int(nx * CELL_SIZE * FILL_RATIO_X / particle_spacing))
        fill_ny = max(1, int(ny * CELL_SIZE * FILL_RATIO_Y / particle_spacing))
        fill_nz = max(1, int(nz * CELL_SIZE * FILL_RATIO_Z / particle_spacing))
        particle_count = fill_nx * fill_ny * fill_nz

        fluid_model = FluidGridLiquidModel(
            density=1500.0,
            fluid_grid_res=GRID_RESOLUTION,
            fluid_grid_cell_size=CELL_SIZE,
            pressure_iteration=PRESSURE_ITERATIONS,
            pressure_solver=PRESSURE_SOLVER,
            extrap_iterations=EXTRAPOLATION_ITERATIONS,
            particle_radius=particle_radius,
            particle_count=particle_count,
        )
        fluid_model.flip_pic_blend = 0.03

        fluid_solver = FluidGridLiquidSolver(fluid_model)
        fluid_solver.use_graph = bool(USE_GRAPH)

        fluid_domain = FluidGridLiquidDomain(fluid_model, fluid_solver)

        # Coupling
        self.coupling = GridLiquidRigidCoupling(
            fluid_domain=fluid_domain,
            rigid_domain=self.rigid_domain,
        )

        # Auto-register all collision shapes from the model for fluid coupling
        self._register_shapes_from_model()

        self.coupling.reset()

        # Initialise fluid particles
        fluid_state_in = self.coupling.fluid_state
        fill_min = wp.vec3(
            float(FILL_OFFSET_X),
            float(FILL_OFFSET_Y),
            float(FILL_OFFSET_Z),
        )
        wp.launch(
            kernel=init_particles_grid,
            dim=particle_count,
            inputs=[fluid_state_in.particle_q, fill_min, particle_spacing, fill_nx, fill_ny],
        )
        fluid_state_in.particle_v.zero_()

        fluid_state_out = self.coupling.fluid_state_out
        wp.copy(fluid_state_out.particle_q, fluid_state_in.particle_q)
        wp.copy(fluid_state_out.particle_v, fluid_state_in.particle_v)

    def _build_particle_visualization(self) -> None:
        state = self.coupling.fluid_state
        particle_count = int(state.particle_q.shape[0])
        particle_radius = float(CELL_SIZE) * float(PARTICLE_RADIUS_SCALE) * 0.8
        self.particle_radii = wp.zeros(particle_count, dtype=float, device=self.device)
        self.particle_radii.fill_(particle_radius)
        self.particle_colors = wp.zeros(particle_count, dtype=wp.vec3, device=self.device)
        self.particle_colors.fill_(wp.vec3(0.2, 0.5, 0.9))

    def apply_control(self):
        obs = compute_rigid_obs(
            self.act,
            self.rigid_domain.state,
            self.joint_pos_initial,
            self.torch_device,
            self.lab_to_mujoco_indices,
            self.gravity_vec,
            self.command,
        )
        with torch.no_grad():
            self.act = self.policy(obs)
            self.rearranged_act = torch.gather(self.act, 1, self.mujoco_to_lab_indices.unsqueeze(0))
            a = self.joint_pos_initial + 0.5 * self.rearranged_act
            a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            wp.copy(self.rigid_domain.control.joint_target_pos, a_wp)

    def step(self):
        t0 = time.perf_counter()

        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)

            if fwd or lat or rot:
                self._auto_forward = False

            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)

        if self._auto_forward:
            self.command[0, 0] = 1

        self.apply_control()

        for _ in range(self.sim_substeps):
            self.coupling.step(self.sim_dt)

        wp.synchronize()
        t1 = time.perf_counter()

        self.total_step_seconds += t1 - t0
        self.sim_time += self.frame_dt
        self.frame_count += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.rigid_domain.state.as_newton_state())

        if self.use_particle_visualization:
            state = self.coupling.fluid_state
            self.viewer.log_points(
                name="water_surface",
                points=state.particle_q,
                radii=self.particle_radii,
                colors=self.particle_colors,
            )
        self.viewer.end_frame()

    def _register_shapes_from_model(self) -> None:
        model = self.rigid_domain.model
        shape_type = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        shape_scale = model.shape_scale.numpy()

        for shape_idx in range(model.shape_count):
            gtype = int(shape_type[shape_idx])
            body_idx = int(shape_body[shape_idx])
            if body_idx < 0:
                continue

            sx, sy, sz = shape_scale[shape_idx]

            if gtype == GeoType.BOX:
                self.coupling.add_body_box(body_idx=body_idx, half_extents=(sx, sy, sz))
            elif gtype == GeoType.CAPSULE:
                self.coupling.add_body_capsule(body_idx=body_idx, radius=sx, half_height=sy)
            elif gtype == GeoType.SPHERE:
                self.coupling.add_body_sphere(body_idx=body_idx, radius=sx)
            if gtype == GeoType.MESH:
                mesh_id = int(model.shape_source_ptr.numpy()[shape_idx])
                if mesh_id != 0:
                    self.coupling.add_body_mesh(body_idx=body_idx, mesh_id=mesh_id, scale=sx)


if __name__ == "__main__":
    wp.init()
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer)
    newton.examples.run(example, args)
