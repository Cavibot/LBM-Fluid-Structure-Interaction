# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.tests.unittest_utils import find_nonfinite_members
from wanphys.sensors import Sensor
from wanphys.examples.utils import init_simulation_params
from wanphys.rigid import RigidDomain, RigidModel, RigidModelBuilder, ShapeConfig, create_mujoco_solver

def get_source_directory() -> str:
    return os.path.realpath(os.path.dirname(__file__))

def get_cfg_directory() -> str:
    return os.path.join(get_source_directory(), "cfg")

def get_cfg(filename: str) -> str:
    return os.path.join(get_cfg_directory(), filename)

@wp.kernel
def acc_to_color(
    alpha: float,
    imu_acc: wp.array(dtype=wp.vec3),
    buffer: wp.array(dtype=wp.vec3),
    color: wp.array(dtype=wp.vec3),
):
    """Kernel mapping an acceleration to a color, with exponential smoothing."""
    idx = wp.tid()
    if idx >= len(imu_acc):
        return

    stored = buffer[idx]

    limit = wp.vec3(80.0)
    acc = wp.max(wp.min(imu_acc[idx], limit), -limit)

    smoothed = (1.0 - alpha) * stored + alpha * acc
    buffer[idx] = smoothed

    c = wp.vec3(0.5) + 0.5 * (0.1 * wp.min(wp.abs(smoothed), wp.vec3(20.0)) - wp.vec3(0.5))
    color[idx] = wp.max(wp.min(c, wp.vec3(1.0)), wp.vec3(0.0))

class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        # setup simulation parameters first
        params = init_simulation_params(fps=200, substeps=10)
        self.fps = params.fps
        self.frame_dt = params.frame_dt
        self.sim_substeps = params.sim_substeps
        self.sim_dt = params.sim_dt
        self.sim_time = 0.0

        builder = RigidModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # pendulum
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("axis_cube.usda"))
        axis_cube_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/AxisCube/VisualCube"))

        self.visual_cubes = []
        self.visual_fillers = []
        self.imu_sites = []

        self.n_cubes = 3

        for cube_idx in range(self.n_cubes):
            scale = 0.2
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(0, 0.7 * (cube_idx - 1), 1),
                    wp.quat_from_axis_angle(
                        wp.vec3(cube_idx % 3 == 0, cube_idx % 3 == 1, cube_idx % 3 == 2), wp.pi / 2
                    ),
                )
            )

            visual_cube = builder.add_shape_mesh(
                body,
                scale=wp.vec3(scale),
                mesh=axis_cube_mesh,
                cfg=ShapeConfig(has_shape_collision=False, density=0.0, ke=1e3, kd=1e2),
            )

            scale_filler = scale * 0.98

            visual_filler = builder.add_shape_box(
                body,
                hx=scale_filler,
                hy=scale_filler,
                hz=scale_filler,
                cfg=ShapeConfig(has_shape_collision=False, density=0.0),
            )
            builder.add_shape_box(
                body, hx=scale, hy=scale, hz=scale, cfg=ShapeConfig(is_visible=False, density=200.0)
            )
            imu_site = builder.add_site(body, label=f"imu_site_{cube_idx}")

            self.visual_cubes.append(visual_cube)
            self.visual_fillers.append(visual_filler)
            self.imu_sites.append(imu_site)

        # finalize model
        model = builder.finalize()

        njmax = 100
        self.domain: RigidDomain = RigidDomain(model, solver=create_mujoco_solver(model, njmax=njmax))
        self.imu = Sensor(self.domain, get_cfg("imu_sensor.yaml"))

        self.state = self.domain.state

        self.model = model

        contact_capacity = self.domain.get_max_contact_count(default=njmax) or njmax
        self.contacts: newton.Contacts = newton.Contacts(contact_capacity, 0)

        self.buffer = wp.zeros(self.n_cubes, dtype=wp.vec3)
        self.colors = wp.zeros(self.n_cubes, dtype=wp.vec3)

        model.setup_viewer(self.viewer)
        
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.camera.pos = type(self.viewer.camera.pos)(3.0, 0.0, 2.0)
            self.viewer.camera.pitch = type(self.viewer.camera.pitch)(-20)


        self.viewer.update_shape_colors({cube: (0.1, 0.1, 0.1) for i, cube in enumerate(self.visual_fillers)})


    def step(self):
        for _ in range(self.sim_substeps):
            self.state.clear_forces()
            self.viewer.apply_forces(self.state.as_newton_state())
            self.domain.step(self.sim_dt, self.contacts)

            # read IMU acceleration
            self.imu.sensor.update()
            # average and compute color
            wp.launch(acc_to_color, dim=self.n_cubes, inputs=[0.025, self.imu.sensor.accelerometer, self.buffer, self.colors])
        
        self.domain.update_contacts(self.contacts)
        self.sim_time += self.frame_dt
        self.viewer.update_shape_colors({cube: self.colors.numpy()[i] for i, cube in enumerate(self.visual_cubes)})

    def test(self):
        pass

    def test_final(self):
        acc = self.imu.sensor.accelerometer.numpy()
        gravity_mag = float(np.linalg.norm(self.model.gravity.numpy()[0]))

        # Cubes settle with different faces up: cube 0  Y, cube 1 �? X, cube 2 �? Z
        expected_axes = [1, 0, 2]

        for i, expected_axis in enumerate(expected_axes):
            np.testing.assert_allclose(np.linalg.norm(acc[i]), gravity_mag, rtol=0.05)
            assert abs(acc[i][expected_axis]) > gravity_mag * 0.95

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state.as_newton_state())
        self.viewer.log_contacts(self.contacts, self.state.as_newton_state())
        self.viewer.end_frame()

if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create viewer and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
        


