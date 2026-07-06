from __future__ import annotations

import math

import warp as wp
import numpy as np

import newton
import newton.examples
from newton.viewer import ViewerGL

from wanphys._src.fluid import FluidGridLiquidModel, FluidGridLiquidDomain, FluidGridLiquidSolver
from wanphys._src.fluid.fluid_grid.coupling import GridLiquidRigidCoupling
from wanphys.rigid import RigidDomain, RigidModel, RigidModelBuilder, ShapeConfig, create_xpbd_solver

from pxr import Usd, UsdGeom
import newton.usd

HCP_ROW_SPACING = 0.8660254037844386
HCP_LAYER_SPACING = 0.816496580927726
INITIAL_PARTICLE_JITTER = 0.18
INITIAL_FILL_MARGIN_CELLS = 0.0
INITIAL_PARTICLE_SEED = 17

@wp.kernel
def init_particles(
    particles: wp.array(dtype=wp.vec3),
    offset: wp.vec3,
    spacing_x: float,
    spacing_y: float,
    spacing_z: float,
    jitter: float,
    boundary_min: wp.vec3,
    boundary_max: wp.vec3,
    dim_x: int,
    dim_y: int,
    random_seed: int,
):
    tid = wp.tid()

    layer_size = dim_x * dim_y
    k = tid // layer_size
    rem = tid % layer_size
    j = rem // dim_x
    i = rem % dim_x

    # Alternate each row/layer to avoid imprinting a Cartesian lattice onto the level set rebuild.
    row_shift = 0.5 * float((j + k) & 1)

    px = offset[0] + (float(i) + row_shift) * spacing_x
    py = offset[1] + float(j) * spacing_y
    pz = offset[2] + float(k) * spacing_z

    state = wp.rand_init(tid + random_seed * 131071)
    jx = (wp.randf(state) - 0.5) * 2.0 * jitter
    jy = (wp.randf(state) - 0.5) * 2.0 * jitter
    jz = (wp.randf(state) - 0.5) * 2.0 * jitter

    pad_x = jitter + 0.25 * spacing_x
    pad_y = jitter + 0.25 * spacing_y
    pad_z = jitter + 0.25 * spacing_z

    px = wp.clamp(px + jx, boundary_min[0] + pad_x, boundary_max[0] - pad_x)
    py = wp.clamp(py + jy, boundary_min[1] + pad_y, boundary_max[1] - pad_y)
    pz = wp.clamp(pz + jz, boundary_min[2] + pad_z, boundary_max[2] - pad_z)

    particles[tid] = wp.vec3(px, py, pz)

class GridLiquidCouplingExample:
    """Grid liquid-rigid two-way coupling example.

    A dam-break pours water over buoyant rigid bodies inside the tank.
    The box, capsule, and sphere are all pushed by the fluid pressure.
    """

    def __init__(self, viewer: ViewerGL):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.box_released = False

        self.n_grid = 64
        self.grid_size = 3.2
        self.dh = self.grid_size / self.n_grid

        self.dam_width = self.grid_size * 0.4
        self.dam_height = self.grid_size * 0.8

        self.viewer = viewer
        self.viewer._paused = True

        print(f"Initializing MAC Grid Water Simulation: {self.n_grid}^3 grid (two-way coupling)...")
        print(f"Dam dimensions: {self.dam_width}m x {self.dam_height}m")

        # ----------------------------------------------------------------
        # Rigid bodies: buoyant box + capsule + sphere
        # ----------------------------------------------------------------
        self.box_half_extent = self.dh * 5
        self.box_mass = 20.0
        self.capsule_radius = self.dh * 3.0
        self.capsule_half_height = self.dh * 5.0
        self.capsule_mass = 16.0
        self.sphere_radius = self.dh * 4.0
        self.box_initial_tilt = math.radians(18.0)
        box_start_x = self.grid_size * 0.5
        box_start_y = self.grid_size * 0.5
        box_start_z = self.grid_size * 1.3
        capsule_start_x = self.grid_size * 0.68
        capsule_start_y = self.grid_size * 0.5
        capsule_start_z = self.grid_size * 1.5
        sphere_start_x = 0.5 * (box_start_x + capsule_start_x)
        sphere_start_y = self.grid_size * 0.5
        sphere_start_z = self.grid_size * 1.7

        builder = RigidModelBuilder()
        box_rot_x = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), self.box_initial_tilt)
        box_rot_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * self.box_initial_tilt)
        box_initial_rot = wp.mul(box_rot_z, box_rot_x)
        box_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(box_start_x, box_start_y, box_start_z),
                box_initial_rot
            )
        )
        capsule_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.25 * math.pi)
        capsule_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(capsule_start_x, capsule_start_y, capsule_start_z),
                capsule_rot
            )
        )
        sphere_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(sphere_start_x, sphere_start_y, sphere_start_z),
                wp.quat_identity()
            )
        )
        density_val = self.box_mass / ((2.0 * self.box_half_extent) ** 3)
        cfg = ShapeConfig(density=density_val)
        builder.add_shape_box(
            body=box_body,
            hx=self.box_half_extent,
            hy=self.box_half_extent,
            hz=self.box_half_extent,
            cfg=cfg,
        )
        capsule_volume = math.pi * self.capsule_radius * self.capsule_radius * (2.0 * self.capsule_half_height)
        capsule_volume += (4.0 / 3.0) * math.pi * (self.capsule_radius ** 3)
        capsule_cfg = ShapeConfig(density=self.capsule_mass / capsule_volume)
        builder.add_shape_capsule(
            body=capsule_body,
            radius=self.capsule_radius,
            half_height=self.capsule_half_height,
            cfg=capsule_cfg,
        )
        capsule_density = self.capsule_mass / capsule_volume
        sphere_density = 0.5 * (density_val + capsule_density)
        sphere_cfg = ShapeConfig(density=sphere_density)
        builder.add_shape_sphere(
            body=sphere_body,
            radius=self.sphere_radius,
            cfg=sphere_cfg,
        )
        # Add boundary walls so the rigid bodies don't fall through or escape the tank
        wall_cfg = ShapeConfig(is_visible=False)
        xmax = self.dh * self.n_grid
        ymax = self.dh * self.n_grid
        margin = 0.0
        builder.add_shape_plane((0.0, 0.0, 1.0, -margin), width=20.0, length=20.0, cfg=wall_cfg)  # Floor z=0
        builder.add_shape_plane((1.0, 0.0, 0.0, -margin), width=0.0, length=0.0, cfg=wall_cfg)    # Left x=0
        builder.add_shape_plane((-1.0, 0.0, 0.0, xmax+margin), width=0.0, length=0.0, cfg=wall_cfg) # Right
        builder.add_shape_plane((0.0, 1.0, 0.0, -margin), width=0.0, length=0.0, cfg=wall_cfg)    # Back y=0
        builder.add_shape_plane((0.0, -1.0, 0.0, ymax+margin), width=0.0, length=0.0, cfg=wall_cfg) # Front

        # usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        # newton_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        # bunny_pos = wp.vec3(self.grid_size * 0.68, self.grid_size * 0.5, self.grid_size * 1.1)
        # bunny_rot = wp.quat(0.5, 0.5, 0.5, 0.5)
        # body_mesh = builder.add_body(xform=wp.transform(p=bunny_pos, q=bunny_rot), label="mesh")

        # builder.add_shape_mesh(body_mesh, mesh=newton_mesh)

        # vertices_np = np.ascontiguousarray(newton_mesh.vertices, dtype=np.float32)
        # indices_np = np.ascontiguousarray(newton_mesh.indices, dtype=np.int32)
        # vertices_wp = wp.array(vertices_np, dtype=wp.vec3, device=wp.get_device())
        # indices_wp = wp.array(indices_np, dtype=wp.int32, device=wp.get_device())
        # wp_bunny_mesh = wp.Mesh(points=vertices_wp, indices=indices_wp)
        # self.wp_bunny_mesh = wp_bunny_mesh

        rigid_model = builder.finalize(device="cuda")
        self.rigid_domain: RigidDomain = RigidDomain(
            rigid_model,
            solver=create_xpbd_solver(rigid_model, iterations=20),
        )

        self.render_model = rigid_model
        self.render_model.setup_viewer(self.viewer)
        self.render_state = self.render_model.state()

        # ----------------------------------------------------------------
        # Fluid
        # ----------------------------------------------------------------
        particle_spacing = self.dh * 0.5
        row_spacing = particle_spacing * HCP_ROW_SPACING
        layer_spacing = particle_spacing * HCP_LAYER_SPACING
        fill_margin = self.dh * INITIAL_FILL_MARGIN_CELLS
        fill_depth = self.grid_size * 1.0

        fill_size_x = max(self.dam_width - 2.0 * fill_margin, particle_spacing)
        fill_size_y = max(fill_depth - 2.0 * fill_margin, row_spacing)
        fill_size_z = max(self.dam_height - 2.0 * fill_margin, layer_spacing)

        nx = max(1, int((fill_size_x - 0.5 * particle_spacing) / particle_spacing))
        ny = max(1, int(fill_size_y / row_spacing))
        nz = max(1, int(fill_size_z / layer_spacing))
        total_particles = nx * ny * nz

        self.model = FluidGridLiquidModel(
            fluid_grid_res=(self.n_grid, self.n_grid, self.n_grid),
            fluid_grid_cell_size=self.dh,
            pressure_iteration=20,
            extrap_iterations=2,
            particle_radius=self.dh * 1.2,
            particle_count=total_particles
        )
        self.model.flip_pic_blend = 0.03
        self.solver = FluidGridLiquidSolver(self.model)
        self.grid_liquid_domain = FluidGridLiquidDomain(self.model, self.solver)

        # ----------------------------------------------------------------
        # Two-way coupled simulation
        # ----------------------------------------------------------------
        self.sim = GridLiquidRigidCoupling(
            fluid_domain=self.grid_liquid_domain,
            rigid_domain=self.rigid_domain,
        )
        self.sim.add_body_box(
            body_idx=box_body,
            half_extents=(self.box_half_extent, self.box_half_extent, self.box_half_extent),
        )
        self.sim.add_body_sphere(
            body_idx=sphere_body,
            radius=self.sphere_radius,
        )
        self.sim.add_body_capsule(
            body_idx=capsule_body,
            radius=self.capsule_radius,
            half_height=self.capsule_half_height,
        )
        
        # self.sim.add_body_mesh(
        #     body_idx=body_mesh,
        #     mesh_id=wp_bunny_mesh.id
        # )
        self.sim.reset()
        self.sim.set_rigid_dynamics_enabled(False)

        # ----------------------------------------------------------------
        # Initialise fluid particles
        # ----------------------------------------------------------------
        self.particle_radii = wp.zeros(total_particles, dtype=float, device=self.model._device)
        self.particle_radii.fill_(self.model.particle_radius * 0.2)

        sim_state_in = self.sim.fluid_state
        fill_min = wp.vec3(fill_margin, fill_margin, fill_margin)
        fill_max = wp.vec3(self.dam_width - fill_margin, fill_depth - fill_margin, self.dam_height - fill_margin)
        wp.launch(
            kernel=init_particles,
            dim=total_particles,
            inputs=[
                sim_state_in.particle_q,
                fill_min,
                particle_spacing,
                row_spacing,
                layer_spacing,
                particle_spacing * INITIAL_PARTICLE_JITTER,
                fill_min,
                fill_max,
                nx,
                ny,
                INITIAL_PARTICLE_SEED,
            ]
        )
        sim_state_out = self.sim.fluid_state_out
        wp.copy(sim_state_out.particle_q, sim_state_in.particle_q)
        wp.copy(sim_state_out.particle_v, sim_state_in.particle_v)

        # Rendering
        self.particle_colors = wp.zeros(total_particles, dtype=wp.vec3, device=self.model._device)
        self.particle_colors.fill_(wp.vec3(0.2, 0.5, 0.9))  # Water particles

        self.frame_count = 0

    def step(self):
        for _ in range(self.sim_substeps):
            self.sim.step(self.sim_dt)

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # Render rigid bodies
        rigid_state_in = self.sim.rigid_state
        self.viewer.log_state(rigid_state_in.as_newton_state())

        # Render fluid particles
        current_sim_state = self.sim.fluid_state
        self.viewer.log_points(
            name="water_surface",
            points=current_sim_state.particle_q,
            radii=self.particle_radii,
            colors=self.particle_colors,
        )

        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Rigid Body Control")
        ui.separator()
        ui.text("The fluid runs immediately.")
        ui.text("The box, capsule, and sphere stay locked until released.")

        if not self.box_released:
            if ui.button("Release Bodies"):
                self.box_released = True
                self.sim.set_rigid_dynamics_enabled(True)
        else:
            ui.text("Bodies released")

if __name__ == "__main__":
    wp.init()
    viewer, args = newton.examples.init()
    example = GridLiquidCouplingExample(viewer)
    newton.examples.run(example, args)
