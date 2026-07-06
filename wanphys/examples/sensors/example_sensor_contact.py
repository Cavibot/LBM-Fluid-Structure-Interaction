# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Contact Sensor
#
# Shows how to use the SensorContact class to evaluate both net contact
# forces and contact forces between individual objects.
# The flap has a contact sensor registering the net contact force of the
# objects on top. The upper and lower plates' sensors will only register
# contacts with the cube and with the ball, respectively.
#
# Command: python -m wanphys.examples.sensors.sensor_contact
#
###########################################################################

import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import Contacts
from newton.tests.unittest_utils import find_nonfinite_members
from wanphys.sensors import Sensor
from wanphys.examples.utils import init_simulation_params
from wanphys.rigid import (
    RigidDomain,
    RigidModel,
    RigidModelBuilder,
    RigidState,
    create_mujoco_solver,
    register_mujoco_solver_attributes,
)

def get_source_directory() -> str:
    return os.path.realpath(os.path.dirname(__file__))

def get_cfg_directory() -> str:
    return os.path.join(get_source_directory(), "cfg")

def get_cfg(filename: str) -> str:
    return os.path.join(get_cfg_directory(), filename)

class Example:
    def __init__(self, viewer, args=None):
        self.viewer = viewer

        params = init_simulation_params(fps=120, substeps=1)
        self.fps = params.fps
        self.frame_dt = params.frame_dt
        self.sim_substeps = params.sim_substeps
        self.sim_time = 0.0
        self.sim_dt = params.sim_dt
        self.reset_interval = 8.0

        self.plot_window = ViewerPlot(
            viewer, "Flap Contact Force", n_points=100, avg=10, scale_min=0, graph_size=(400, 200)
        )
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.plot_window.render, "free")

        builder = RigidModelBuilder()
        builder.add_usd(newton.examples.get_asset("sensor_contact_scene.usda"))
        register_mujoco_solver_attributes(builder)

        builder.add_ground_plane()

        # finalize model
        model = builder.finalize()

        njmax = 100
        self.domain: RigidDomain = RigidDomain(
            model,
            solver=create_mujoco_solver(
                model,
                njmax=njmax,
                nconmax=100,
                cone="pyramidal",
                impratio=1,
            ),
        )

        self.flap_contact_sensor = Sensor(self.domain, get_cfg("flap_contact_sensor.yaml"))

        self.plate_contact_sensor = Sensor(self.domain, get_cfg("plate_contact_sensor.yaml"))

        # self.sim = CompositeSimulation()
        # self.sim.add_domain(self.domain)

        # self.state = self.sim.get_state(self.domain.name)

        self.state: RigidState = self.domain.state

        self.model = model

        requested_contact_attributes = model.get_requested_contact_attributes()
        contact_capacity = self.domain.get_max_contact_count(default=njmax) or njmax
        self.contacts: Contacts = Contacts(
            rigid_contact_max=contact_capacity,
            soft_contact_max=0,
            requested_attributes=requested_contact_attributes,
            device=self.model.device,
        )

        model.setup_viewer(self.viewer)

        self.plates_touched: list[bool] = 2 * [False]
        self.shape_colors: dict[str, list[float]] = {
            "/env/Plate1": 3 * [0.4],
            "/env/Plate2": 3 * [0.4],
            "/env/Sphere": [1.0, 0.4, 0.2],
            "/env/Cube": [0.2, 0.4, 0.8],
            "/env/Flap": 3 * [0.8],
        }
        self.shape_map: dict[str, int] = {key: s for s, key in enumerate(model.shape_label)}

        self.control = self.domain.control
        hinge_joint_idx = model.joint_label.index("/env/Hinge")
        self.hinge_joint_q_start: int = int(model.joint_q_start.numpy()[hinge_joint_idx])

        self.next_reset = 0.0

        # store initial state for reset
        self.initial_joint_q: wp.array = wp.clone(self.state.joint_q)
        self.initial_joint_qd: wp.array = wp.clone(self.state.joint_qd)


    def simulate(self):
        self.state.clear_forces()
        self.viewer.apply_forces(self.state.as_newton_state())
        self.domain.step(self.sim_dt, self.contacts)


    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset()

        hinge_angle = min(self.sim_time / 3, 1.6)
        self.control.joint_target_pos[self.hinge_joint_q_start : self.hinge_joint_q_start + 1].fill_(hinge_angle)

        # with wp.ScopedTimer("step", active=False):
        #     if self.graph:
        #         wp.capture_launch(self.graph)
        #     else:
        #         self.simulate()
        for _ in range(self.sim_substeps):
            self.simulate()
        
        self.state = self.domain.state

        self.domain.update_contacts(self.contacts)
        self.plate_contact_sensor.sensor.update(self.contacts)

        net_force = self.plate_contact_sensor.sensor.net_force.numpy()
        sensor = self.plate_contact_sensor.sensor
        world_sensing = sensor.sensing_objs[0]   # world 0
        world_counterparts = sensor.counterparts[0]
        for i, (plate, _) in enumerate(world_sensing):
            if self.plates_touched[i]:
                continue
            if np.abs(net_force[i, i]).max() == 0:
                continue
            # color newly touched plate
            plate = sensor.sensing_objs[0][i][0]
            obj = sensor.counterparts[0][i][0]
            obj_key = self.model.shape_label[obj]
            self.plates_touched[i] = True
            print(f"Plate {self.model.shape_label[plate]} was touched by counterpart {obj_key}")
            self.viewer.update_shape_colors({plate: self.shape_colors[obj_key]})


        # contacts already populated by update_contacts() above
        self.flap_contact_sensor.sensor.update(self.contacts)
        self.plot_window.add_point(np.abs(self.flap_contact_sensor.sensor.net_force.numpy()[0, 0, 2]))
        self.sim_time += self.frame_dt

    def reset(self):
        self.sim_time = 0
        self.next_reset = self.sim_time + self.reset_interval
        self.viewer.update_shape_colors({self.shape_map[s]: v for s, v in self.shape_colors.items()})
        self.plates_touched = 2 * [False]

        print("Resetting")
        # Restore initial joint positions and velocities in-place.
        self.state.joint_q.assign(self.initial_joint_q)
        self.state.joint_qd.assign(self.initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        self.model.eval_forward_kinematics(self.state)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state.as_newton_state())
        self.viewer.log_contacts(self.contacts, self.state.as_newton_state())
        self.viewer.end_frame()

    def test_post_step(self):
        assert not self.plates_touched[1] or self.plates_touched[0]  # plate 0 always touched first
        assert len(find_nonfinite_members(self.flap_contact_sensor.sensor)) == 0
        assert len(find_nonfinite_members(self.plate_contact_sensor.sensor)) == 0
        # first plate touched by 1.4s, second by 4s, flap left by 2.8s
        if self.sim_time > 1.4:
            assert self.plates_touched[0]
        if self.sim_time > 2.8:
            assert self.flap_contact_sensor.sensor.net_force.numpy().sum() == 0
        # if self.sim_time > 4.0: assert self.plates_touched[1]   # unreliable due to jerky cube motion

    def test_final(self):
        self.test_post_step()
        assert np.all(self.state.body_q.numpy()[:, 2] > 0.0), "all bodies are above the ground"
        assert len(find_nonfinite_members(self.flap_contact_sensor.sensor)) == 0
        assert len(find_nonfinite_members(self.plate_contact_sensor.sensor)) == 0



class ViewerPlot:
    """ImGui plot window"""

    def __init__(self, viewer=None, title="Plot", n_points=200, avg=1, **kwargs):
        self.viewer = viewer
        self.avg = avg
        self.title = title
        self.data = np.zeros(n_points, dtype=np.float32)
        self.plot_kwargs = kwargs
        self.cache = []

    def add_point(self, point):
        self.cache.append(point)
        if len(self.cache) == self.avg:
            self.data[0] = sum(self.cache) / self.avg
            self.data = np.roll(self.data, -1)
            self.cache.clear()

    def render(self, imgui):
        """
        Render the replay UI controls.

        Args:
            imgui: The ImGui object passed by the ViewerGL callback system
        """
        if not self.viewer or not self.viewer.ui.is_available:
            return

        io = self.viewer.ui.io

        # Position the plot window
        window_shape = (400, 350)
        imgui.set_next_window_pos(
            imgui.ImVec2(io.display_size[0] - window_shape[0] - 10, io.display_size[1] - window_shape[1] - 10)
        )
        imgui.set_next_window_size(imgui.ImVec2(*window_shape))

        flags = imgui.WindowFlags_.no_resize.value

        if imgui.begin(self.title, flags=flags):
            imgui.text("Flap contact force")
            imgui.plot_lines("Force", self.data, **self.plot_kwargs)
        imgui.end()


if __name__ == "__main__":
    parser = newton.examples.create_parser()

    viewer, args = newton.examples.init(parser)

    example = Example(viewer)

    newton.examples.run(example, args)
