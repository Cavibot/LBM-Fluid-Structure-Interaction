# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WCSPH fluid scene rendered through the tiled camera sensor wrapper.

This example keeps the existing YAML-based sensor flow:

    Sensor(domain, "cfg/tiled_camera.yaml", extra_params={...})

The rigid scene is rendered by the Newton viewer, while the sensor panel shows
the same single-world fluid scene through the tiled camera sensor.

This example requires CUDA and will not run on CPU.

Run with:
    uv run python -m wanphys.examples.sensors.example_sensor_tiled_camera_fluid
"""

from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.viewer import ViewerGL

from wanphys._src.core import CompositeSimulation
from wanphys._src.fluid import ParticleFluidDomain
from wanphys._src.fluid.fluid_particle.builder import ParticleFluidBuilder
from wanphys.collision import CollisionPipeline
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig
from wanphys.sensors import Sensor

try:
    from newton.geometry import ParticleFlags
except ImportError:
    from newton._src.geometry import ParticleFlags


def get_source_directory() -> str:
    return os.path.realpath(os.path.dirname(__file__))


def get_cfg_directory() -> str:
    return os.path.join(get_source_directory(), "cfg")


def get_cfg(filename: str) -> str:
    return os.path.join(get_cfg_directory(), filename)

@dataclass
class SensorStateBridge:
    body_q: wp.array


@wp.kernel
def apply_twoway_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=int),
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_shape: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    margin: float,
    max_push_frac: float,
    vel_damp: float,
    particle_mass: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    sim_dt: float,
):
    """Two-way soft contact: push particle out and accumulate reaction on rigid body."""
    tid = wp.tid()
    count = soft_contact_count[0]
    if tid >= count:
        return

    p = soft_contact_particle[tid]
    s = soft_contact_shape[tid]
    if p < 0 or s < 0:
        return
    if (particle_flags[p] & ParticleFlags.ACTIVE) == 0:
        return

    n = soft_contact_normal[tid]
    n_len = wp.length(n)
    if n_len < 1.0e-6:
        return
    n = n / n_len

    body_pos_local = soft_contact_body_pos[tid]
    body_vel_local = soft_contact_body_vel[tid]

    rigid = shape_body[s]
    X_wb = wp.transform_identity()
    if rigid >= 0:
        X_wb = body_q[rigid]

    surf_w = wp.transform_point(X_wb, body_pos_local)

    com_w = wp.vec3(0.0)
    lever = wp.vec3(0.0)
    v_body_linear = wp.vec3(0.0)
    v_body_angular = wp.vec3(0.0)
    if rigid >= 0:
        com_w = wp.transform_point(X_wb, body_com[rigid])
        lever = surf_w - com_w
        body_sv = body_qd[rigid]
        v_body_linear = wp.spatial_top(body_sv)
        v_body_angular = wp.spatial_bottom(body_sv)
    body_vel_w = v_body_linear + wp.transform_vector(X_wb, body_vel_local) + wp.cross(v_body_angular, lever)

    x = particle_q[p]
    v = particle_qd[p]
    v_old = v
    r = particle_radius[p]

    d = wp.dot(x - surf_w, n)
    pen = (margin + r) - d
    if pen <= 0.0:
        return

    max_push = wp.max(0.0, max_push_frac) * r
    push = wp.min(pen, max_push)
    x = x + n * push

    rel_v = v - body_vel_w
    vn = wp.dot(rel_v, n)
    if vn < 0.0:
        rel_v = rel_v - vn * n
    rel_v = rel_v * (1.0 - vel_damp)
    v = rel_v + body_vel_w

    particle_q[p] = x
    particle_qd[p] = v

    if rigid >= 0 and sim_dt > 0.0:
        force = ((v_old - v) * particle_mass[p]) / sim_dt
        torque = wp.cross(lever, force)
        wp.atomic_add(body_f, rigid, wp.spatial_vector(force, torque))


class WCSPHRigidTwoWayCoupling(CompositeSimulation):
    """Two-way WCSPH-rigid coupling used by this sensor example."""

    def __init__(
        self,
        rigid: RigidDomain,
        wcsph: ParticleFluidDomain,
        contact_iters: int = 3,
        contact_margin: float = 0.002,
        contact_max_push_frac: float = 0.30,
        contact_vel_damp: float = 0.015,
    ):
        super().__init__()
        self.rigid = rigid
        self.wcsph = wcsph
        self.contact_iters = contact_iters
        self.contact_margin = contact_margin
        self.contact_max_push_frac = contact_max_push_frac
        self.contact_vel_damp = contact_vel_damp
        self.rigid.create_state()
        self.wcsph.create_state()

    def step(self, dt: float) -> None:
        fluid_model = self.wcsph.model
        rigid_state = self.rigid.state

        self.wcsph.step(dt)
        fluid_state = self.wcsph.state

        for _ in range(max(self.contact_iters, 0)):
            contacts = CollisionPipeline.collide_rigid_fluid(self.rigid, self.wcsph)
            if contacts is None:
                break

            wp.launch(
                apply_twoway_contacts,
                dim=contacts.soft_contact_max,
                inputs=[
                    fluid_state.particle_q,
                    fluid_state.particle_qd,
                    fluid_model.particle_radius,
                    fluid_model.particle_flags,
                    rigid_state.body_q,
                    rigid_state.body_qd,
                    self.rigid.model.shape_body,
                    contacts.soft_contact_count,
                    contacts.soft_contact_particle,
                    contacts.soft_contact_shape,
                    contacts.soft_contact_body_pos,
                    contacts.soft_contact_body_vel,
                    contacts.soft_contact_normal,
                    float(self.contact_margin),
                    float(self.contact_max_push_frac),
                    float(self.contact_vel_damp),
                    fluid_model.particle_mass,
                    rigid_state.body_f,
                    self.rigid.model.body_com,
                    dt,
                ],
                device=fluid_model._device,
            )

        rigid_contacts = CollisionPipeline.collide_rigid(self.rigid)
        self.rigid.step(dt, contacts=rigid_contacts)
        rigid_state.clear_forces()
        self._time += dt

    def reset(self) -> None:
        super().reset()
        self.rigid.create_state()
        self.wcsph.create_state()


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = self._require_cuda_device()
        self.device_name = str(self.device)
        self.scene_scale = 0.70
        self.viewer_base_camera_pos = self._scale_xyz((-0.4, -7.6, 2.5))
        self.viewer_base_camera_pitch = -12.0
        self.viewer_base_camera_yaw = 90.0

        self.fps = int(args.fps)
        self.frame_dt = 1.0 / float(self.fps)
        self.substeps = max(1, int(args.substeps))
        self.sim_dt = self.frame_dt / float(self.substeps)
        self.sim_time = 0.0
        self.fluid_origin = tuple(float(v) for v in args.origin)
        self.fluid_dims = tuple(int(v) for v in args.block_dims)
        self.fluid_spacing = float(args.spacing)
        self.waterline_z = self.fluid_origin[2] + self.fluid_spacing * (self.fluid_dims[2] - 0.5)

        self.image_output = 0
        self.texture_id = 0
        self.pixel_buffer = 0
        self.texture_buffer = None
        self.ui_padding = 10
        self.ui_side_panel_width = 280
        self.worlds_per_row = 2
        self.worlds_per_col = 2
        self.world_count_total = 1
        self.camera_count = 4
        self.camera_angle_offsets = (
            (-30.0, -4.0),
            (-10.0, 8.0),
            (12.0, -8.0),
            (32.0, 4.0),
        )

        if isinstance(self.viewer, ViewerGL):
            self.viewer.register_ui_callback(self.display, "free")

        self.rigid_domain = self._build_rigid_domain()
        self.fluid_domain = self._build_fluid_domain()
        self.sim = WCSPHRigidTwoWayCoupling(
            rigid=self.rigid_domain,
            wcsph=self.fluid_domain,
            contact_iters=int(args.contact_iters),
            contact_margin=float(args.contact_margin),
            contact_max_push_frac=float(args.contact_max_push_frac),
            contact_vel_damp=float(args.contact_vel_damp),
        )

        self.fluid_colors = wp.full(
            self.fluid_domain.model.particle_count,
            value=wp.vec3(0.10, 0.45, 0.95),
            dtype=wp.vec3,
            device=self.device,
        )

        self.rigid_domain.model.setup_viewer(self.viewer)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = False
        if hasattr(self.viewer, "show_collision"):
            self.viewer.show_collision = False
        if hasattr(self.viewer, "show_contacts"):
            self.viewer.show_contacts = False
        if hasattr(self.viewer, "show_visual"):
            self.viewer.show_visual = True

        if hasattr(self.viewer, "camera"):
            self.viewer.camera.pos = wp.vec3(*self.viewer_base_camera_pos)
            self.viewer.camera.pitch = self.viewer_base_camera_pitch
            self.viewer.camera.yaw = self.viewer_base_camera_yaw
        if hasattr(self.viewer, "camera_control_enable"):
            self.viewer.camera_control_enable = True

        self.sensor_render_width = int(args.sensor_width)
        self.sensor_render_height = int(args.sensor_height)
        if isinstance(self.viewer, ViewerGL):
            display_width = self.viewer.ui.io.display_size[0] - self.ui_side_panel_width - self.ui_padding * 4
            display_height = self.viewer.ui.io.display_size[1] - self.ui_padding * 2
            if display_width > 0 and display_height > 0:
                self.sensor_render_width = max(128, int(display_width // self.worlds_per_row))
                self.sensor_render_height = max(96, int(display_height // self.worlds_per_col))

        self.sensor_body_q = wp.empty_like(self.rigid_domain.state.body_q)
        self.sensor_state = SensorStateBridge(body_q=self.sensor_body_q)
        self.tiled_camera_sensor = Sensor(
            self.rigid_domain,
            get_cfg("tiled_camera.yaml"),
            extra_params={"particle_domain": self.fluid_domain},
        )

        fov = float(args.sensor_fov)
        if isinstance(self.viewer, ViewerGL):
            fov = float(self.viewer.camera.fov)

        self.camera_rays = self.tiled_camera_sensor.sensor.compute_pinhole_camera_rays(
            self.sensor_render_width,
            self.sensor_render_height,
            [math.radians(fov)] * self.camera_count,
        )
        self.color_image = self.tiled_camera_sensor.sensor.create_color_image_output(
            self.sensor_render_width,
            self.sensor_render_height,
            self.camera_count,
        )
        self.depth_image = self.tiled_camera_sensor.sensor.create_depth_image_output(
            self.sensor_render_width,
            self.sensor_render_height,
            self.camera_count,
        )
        self.normal_image = self.tiled_camera_sensor.sensor.create_normal_image_output(
            self.sensor_render_width,
            self.sensor_render_height,
            self.camera_count,
        )
        self.albedo_image = self.tiled_camera_sensor.sensor.create_albedo_image_output(
            self.sensor_render_width,
            self.sensor_render_height,
            self.camera_count,
        )
        self.depth_range = wp.array([0.2, float(args.depth_far)], dtype=wp.float32, device=self.device)

        if isinstance(self.viewer, ViewerGL):
            self.create_texture()

    def _build_rigid_domain(self) -> RigidDomain:
        builder = RigidModelBuilder()
        builder.add_ground_plane(cfg=ShapeConfig(mu=0.5))
        bound = float(self.args.bound)
        self._add_tank(
            builder,
            lower=(-bound, -bound, 0.0),
            upper=(bound, bound, bound * 2.0),
            thickness=0.12,
        )

        box_body = builder.add_body(
            xform=wp.transform((-0.5, 0.0, 2.0), wp.quat_identity()),
            label="testBox",
        )
        builder.add_shape_box(
            body=box_body,
            hx=0.2,
            hy=0.2,
            hz=0.1,
            cfg=ShapeConfig(density=float(self.args.box_density), mu=0.5),
        )
        self.float_box_body = box_body

        builder.add_shape_cylinder(
            body=-1,
            radius=0.18 * self.scene_scale,
            half_height=0.45 * self.scene_scale,
            xform=wp.transform(p=wp.vec3(*self._scale_xyz((1.10, 0.55, 0.45))), q=wp.quat_identity()),
            cfg=ShapeConfig(mu=0.1),
        )

        return RigidDomain(builder.finalize(device=self.device_name))

    def _add_tank(
        self,
        builder: RigidModelBuilder,
        *,
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
        thickness: float,
        friction: float = 0.05,
    ) -> None:
        hidden_wall_cfg = ShapeConfig(mu=friction, is_visible=False)
        min_x, min_y, min_z = lower
        max_x, max_y, max_z = upper
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        center_z = 0.5 * (min_z + max_z)
        half_x = 0.5 * (max_x - min_x)
        half_y = 0.5 * (max_y - min_y)
        half_z = 0.5 * (max_z - min_z)
        overlap = thickness * 2.0

        boxes = [
            ((center_x, center_y, min_z - thickness / 2), half_x + overlap, half_y + overlap, thickness / 2, hidden_wall_cfg),
            ((min_x - thickness / 2, center_y, center_z), thickness / 2, half_y + overlap, half_z + overlap, hidden_wall_cfg),
            ((max_x + thickness / 2, center_y, center_z), thickness / 2, half_y + overlap, half_z + overlap, hidden_wall_cfg),
            ((center_x, max_y + thickness / 2, center_z), half_x + overlap, thickness / 2, half_z + overlap, hidden_wall_cfg),
            ((center_x, min_y - thickness / 2, center_z), half_x + overlap, thickness / 2, half_z + overlap, hidden_wall_cfg),
            ((center_x, center_y, max_z + thickness / 2), half_x + overlap, half_y + overlap, thickness / 2, hidden_wall_cfg),
        ]

        for position, hx, hy, hz, cfg in boxes:
            builder.add_shape_box(
                body=-1,
                hx=hx,
                hy=hy,
                hz=hz,
                xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
                cfg=cfg,
            )

    def _build_fluid_domain(self) -> ParticleFluidDomain:
        builder = ParticleFluidBuilder()
        self._emit_fluid_block(
            builder,
            spacing=float(self.args.spacing),
            origin=tuple(float(v) for v in self.args.origin),
            dims=tuple(int(v) for v in self.args.block_dims),
            rest_density=float(self.args.rest_density),
            velocity=tuple(float(v) for v in self.args.initial_velocity),
        )
        fluid_data = builder.finalize(device=self.device)

        parameters = {
            "solver_type": "wcsph",
            "h": float(self.args.h),
            "rest_density": float(self.args.rest_density),
            "viscosity": float(self.args.viscosity),
            "device": self.device,
            "use_graph": True,
            "particle_count": fluid_data.particle_count,
            "particle_mass": wp.clone(fluid_data.particle_mass),
            "particle_flags": wp.clone(fluid_data.particle_flags),
            "particle_radius": wp.clone(fluid_data.particle_radius),
        }
        domain = ParticleFluidDomain(fluid_data.particle_q, fluid_data.particle_qd, parameters)
        domain.model.contact_margin = float(self.args.contact_margin)
        domain.model.contact_max_push_frac = float(self.args.contact_max_push_frac)
        domain.model.contact_vel_damp = float(self.args.contact_vel_damp)
        return domain

    def _require_cuda_device(self):
        device = wp.get_device()
        if getattr(device, "is_cuda", False):
            return device
        if not wp.is_cuda_available():
            raise RuntimeError("example_sensor_tiled_camera_fluid requires CUDA and does not support CPU execution.")
        wp.set_device("cuda:0")
        device = wp.get_device()
        if not getattr(device, "is_cuda", False):
            raise RuntimeError("Failed to switch to a CUDA device for example_sensor_tiled_camera_fluid.")
        return device

    def _scale_xyz(self, xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(float(v) * self.scene_scale for v in xyz)

    def _emit_fluid_block(
        self,
        builder: ParticleFluidBuilder,
        *,
        spacing: float,
        origin: tuple[float, float, float],
        dims: tuple[int, int, int],
        rest_density: float,
        velocity: tuple[float, float, float],
    ) -> None:
        nx, ny, nz = dims
        particle_mass = rest_density * (spacing ** 3)
        v0 = wp.vec3(*velocity)
        start = np.array(origin, dtype=np.float32)

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    p = start + spacing * np.array([ix, iy, iz], dtype=np.float32)
                    builder.add_particle(
                        pos=wp.vec3(float(p[0]), float(p[1]), float(p[2])),
                        vel=v0,
                        mass=float(particle_mass),
                        radius=0.5 * spacing,
                    )

    def get_camera_transforms(self) -> wp.array(dtype=wp.transformf):
        if isinstance(self.viewer, ViewerGL):
            transforms = [self._viewer_camera_transform(yaw_offset, pitch_offset) for yaw_offset, pitch_offset in self.camera_angle_offsets]
            return wp.array(
                [[transform] * self.world_count_total for transform in transforms],
                dtype=wp.transformf,
            )
        transform = wp.transformf(wp.vec3f(*self.viewer_base_camera_pos), wp.quatf(0.5, 0.5, 0.5, 0.5))
        return wp.array(
            [[transform] * self.world_count_total for _ in range(self.camera_count)],
            dtype=wp.transformf,
        )

    def _viewer_camera_transform(self, yaw_offset: float, pitch_offset: float) -> wp.transformf:
        from newton._src.viewer.camera import Camera as ViewerCamera  # noqa: PLC0415

        camera = self.viewer.camera
        tmp_camera = ViewerCamera(
            fov=camera.fov,
            near=camera.near,
            far=camera.far,
            width=camera.width,
            height=camera.height,
            pos=(float(camera.pos.x), float(camera.pos.y), float(camera.pos.z)),
            up_axis=camera.up_axis,
        )
        tmp_camera.pitch = max(-89.0, min(89.0, camera.pitch + pitch_offset))
        tmp_camera.yaw = camera.yaw + yaw_offset
        rotation = wp.quat_from_matrix(wp.mat33f(tmp_camera.get_view_matrix().reshape(4, 4)[:3, :3]))
        return wp.transformf(camera.pos, rotation)

    def step(self):
        for _ in range(self.substeps):
            self.sim.step(self.sim_dt)
        if hasattr(self.viewer, "apply_forces"):
            self.viewer.apply_forces(self.rigid_domain.state.as_newton_state())
        self.sim_time = self.sim.time

    def render(self):
        # Match the pure tiled-camera example: update sensor texture before
        # entering the ViewerGL frame so camera input is not held mid-frame.
        wp.synchronize()
        self.render_sensors()
        self.viewer.begin_frame(self.sim_time)
        # ViewerGL reads rigid state back on the CPU path during logging.
        # Sync here to avoid racing outstanding CUDA work.
        wp.synchronize()
        self.viewer.log_state(self.sim.rigid.state.as_newton_state())
        self.viewer.log_points(
            "wcsph_fluid",
            points=self.sim.wcsph.state.particle_q,
            radii=self.sim.wcsph.model.particle_radius,
            colors=self.fluid_colors,
            hidden=False,
        )
        self.viewer.end_frame()

    def render_sensors(self):
        fluid_state = self.sim.wcsph.state
        sensor_impl = self.tiled_camera_sensor.sensor

        # The fluid solver swaps state buffers, so refresh the particle pointer
        # before every tiled-camera render.
        sensor_impl.render_context.particles_position = fluid_state.particle_q
        sensor_impl.render_context.particles_radius = self.sim.wcsph.model.particle_radius
        sensor_impl.render_context.particles_world_index = self.sim.wcsph.model.particle_world_ids
        wp.copy(self.sensor_body_q, self.sim.rigid.state.body_q)

        sensor_impl.update(
            self.get_camera_transforms(),
            self.camera_rays,
            color_image=self.color_image,
            depth_image=self.depth_image,
            normal_image=self.normal_image,
            albedo_image=self.albedo_image,
        )
        self.update_texture()

    def create_texture(self):
        from pyglet import gl  # noqa: PLC0415

        width = self.sensor_render_width * self.worlds_per_row
        height = self.sensor_render_height * self.worlds_per_col

        texture_id = gl.GLuint()
        gl.glGenTextures(1, texture_id)
        self.texture_id = texture_id.value

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        pixel_buffer = gl.GLuint()
        gl.glGenBuffers(1, pixel_buffer)
        self.pixel_buffer = pixel_buffer.value
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self.texture_buffer = wp.RegisteredGLBuffer(self.pixel_buffer)

    def update_texture(self):
        if not self.texture_id or self.texture_buffer is None:
            return

        from pyglet import gl  # noqa: PLC0415

        texture_buffer = self.texture_buffer.map(
            dtype=wp.uint8,
            shape=(
                self.worlds_per_col * self.sensor_render_height,
                self.worlds_per_row * self.sensor_render_width,
                4,
            ),
        )

        if self.image_output == 0:
            self.tiled_camera_sensor.sensor.flatten_color_image_to_rgba(
                self.color_image,
                texture_buffer,
                self.worlds_per_row,
            )
        elif self.image_output == 1:
            self.tiled_camera_sensor.sensor.flatten_color_image_to_rgba(
                self.albedo_image,
                texture_buffer,
                self.worlds_per_row,
            )
        elif self.image_output == 2:
            self.tiled_camera_sensor.sensor.flatten_depth_image_to_rgba(
                self.depth_image,
                texture_buffer,
                self.worlds_per_row,
                self.depth_range,
            )
        else:
            self.tiled_camera_sensor.sensor.flatten_normal_image_to_rgba(
                self.normal_image,
                texture_buffer,
                self.worlds_per_row,
            )
        self.texture_buffer.unmap()

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.sensor_render_width * self.worlds_per_row,
            self.sensor_render_height * self.worlds_per_col,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def gui(self, ui):
        ui.text("Sensor Output")
        ui.separator()
        ui.text("View follows the main viewer camera")
        if ui.radio_button("Show Color Output", self.image_output == 0):
            self.image_output = 0
        if ui.radio_button("Show Albedo Output", self.image_output == 1):
            self.image_output = 1
        if ui.radio_button("Show Depth Output", self.image_output == 2):
            self.image_output = 2
        if ui.radio_button("Show Normal Output", self.image_output == 3):
            self.image_output = 3

    def display(self, imgui):
        width = self.viewer.ui.io.display_size[0] - self.ui_side_panel_width - self.ui_padding * 4
        height = self.viewer.ui.io.display_size[1] - self.ui_padding * 2

        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(self.viewer.ui.io.display_size)

        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_mouse_inputs.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
            | imgui.WindowFlags_.no_scrollbar.value
        )

        if imgui.begin("Sensors", flags=flags):
            pos_x = self.ui_side_panel_width + self.ui_padding * 2
            pos_y = self.ui_padding

            if self.texture_id > 0:
                imgui.set_cursor_pos(imgui.ImVec2(pos_x, pos_y))
                imgui.image(imgui.ImTextureRef(self.texture_id), imgui.ImVec2(width, height))

            line_color = imgui.get_color_u32(imgui.Col_.window_bg)
            draw_list = imgui.get_window_draw_list()
            for x in range(1, self.worlds_per_row):
                draw_list.add_line(
                    imgui.ImVec2(pos_x + x * (width / self.worlds_per_row), pos_y),
                    imgui.ImVec2(pos_x + x * (width / self.worlds_per_row), pos_y + height),
                    line_color,
                    2.0,
                )
            for y in range(1, self.worlds_per_col):
                draw_list.add_line(
                    imgui.ImVec2(pos_x, pos_y + y * (height / self.worlds_per_col)),
                    imgui.ImVec2(pos_x + width, pos_y + y * (height / self.worlds_per_col)),
                    line_color,
                    2.0,
                )

        imgui.end()

    def test_final(self):
        self.render_sensors()

        color_image = self.color_image.numpy()
        assert color_image.shape == (
            self.rigid_domain.model.world_count,
            self.camera_count,
            self.sensor_render_height,
            self.sensor_render_width,
        )
        assert color_image.min() < color_image.max()

        depth_image = self.depth_image.numpy()
        assert depth_image.shape == (
            self.rigid_domain.model.world_count,
            self.camera_count,
            self.sensor_render_height,
            self.sensor_render_width,
        )
        assert np.any(depth_image > 0.0)


def build_parser():
    parser = newton.examples.create_parser()

    parser.add_argument("--fps", type=int, default=250)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--contact-iters", type=int, default=3)

    parser.add_argument("--spacing", type=float, default=0.02)
    parser.add_argument("--block-dims", type=int, nargs=3, default=(70, 70, 70), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--origin", type=float, nargs=3, default=(-1.5, -0.6, 0.2), metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--initial-velocity",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "VZ"),
    )

    parser.add_argument("--h", type=float, default=0.04)
    parser.add_argument("--rest-density", type=float, default=1000.0)
    parser.add_argument("--viscosity", type=float, default=0.15)
    parser.add_argument("--bound", type=float, default=2.1)
    parser.add_argument("--restitution", type=float, default=0.30)
    parser.add_argument("--damping", type=float, default=0.02)
    parser.add_argument("--contact-margin", type=float, default=0.002)
    parser.add_argument("--contact-max-push-frac", type=float, default=0.30)
    parser.add_argument("--contact-vel-damp", type=float, default=0.015)
    parser.add_argument("--box-density", type=float, default=100.0)

    parser.add_argument("--sensor-width", type=int, default=320)
    parser.add_argument("--sensor-height", type=int, default=200)
    parser.add_argument("--sensor-fov", type=float, default=50.0)
    parser.add_argument("--depth-far", type=float, default=6.0)
    return parser


def main():
    wp.init()
    parser = build_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
