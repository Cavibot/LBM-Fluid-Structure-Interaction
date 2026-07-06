from __future__ import annotations

import os
import time
import numpy as np
import warp as wp

import newton
import newton.examples
from newton.viewer import ViewerGL
 
from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer
from wanphys._src.fluid.fluid_viewer.viewer_gl import ScreenSpaceFluidRenderer
from wanphys._src.fluid import FluidGridLiquidModel, FluidGridLiquidState, FluidGridLiquidSolver, FluidGridLiquidDomain
from wanphys._src.core import CompositeSimulation
from wanphys.rigid import RigidModelBuilder

from pxr import Usd, UsdGeom
import newton.usd

HCP_ROW_SPACING = 0.8660254037844386
HCP_LAYER_SPACING = 0.816496580927726
INITIAL_PARTICLE_JITTER = 0.18
INITIAL_FILL_MARGIN_CELLS = 0.0
INITIAL_PARTICLE_SEED = 17
SCALE = 1

ENABLE_SSFR = False
@wp.kernel
def init_solid_box(
    solid_phi: wp.array3d(dtype=float),
    dh: float,
    center: wp.vec3,
    half_extents: wp.vec3
):
    i, j, k = wp.tid()
    p = wp.vec3(float(i) * dh, float(j) * dh, float(k) * dh)

    d = wp.vec3(
        wp.abs(p[0] - center[0]) - half_extents[0],
        wp.abs(p[1] - center[1]) - half_extents[1],
        wp.abs(p[2] - center[2]) - half_extents[2]
    )

    outside_dist = wp.length(wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0)))
    inside_dist = wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)
    
    # Negative inside solid
    solid_phi[i, j, k] = outside_dist + inside_dist


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

class Example:
    def __init__(self, viewer: ViewerGL):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.n_grid = 64 * SCALE
        self.grid_size = 3.2
        self.dh = self.grid_size / self.n_grid

        self.dam_width = self.grid_size * 0.4
        self.dam_height = self.grid_size * 0.4

        self.viewer = viewer
        self.viewer._paused = True

        print(f"Initializing MAC Grid Water Simulation: {self.n_grid}^3 grid...")
        print(f"Dam dimensions: {self.dam_width}m x {self.dam_height}m")

        # Static scene for rendering
        builder = RigidModelBuilder()
        self.box_pos = wp.vec3(self.dh * 100 * SCALE, self.dh * 20 * SCALE, self.dh * 20)
        self.box_half_extent = wp.vec3(self.dh * 5 * SCALE, self.dh * 5 * SCALE, self.dh * 20)
        self.box_pos1 = wp.vec3(self.dh * 100 * SCALE, self.dh * 40 * SCALE, self.dh * 20)
        self.box_half_extent1 = wp.vec3(self.dh * 5 * SCALE, self.dh * 5 * SCALE, self.dh * 20)
        body_box = builder.add_body(xform=wp.transform(p=self.box_pos, q=wp.quat_identity()), label="box")
        body_box1 = builder.add_body(xform=wp.transform(p=self.box_pos1, q=wp.quat_identity()), label="box1")

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        demo_mesh = newton.usd.get_mesh(usd_stage.GetPrimAtPath("/root/bunny"))
        bunny_pos = wp.vec3(self.dh * 64 * SCALE, self.dh * 32 * SCALE, self.dh * -1)
        bunny_rot = wp.quat(0.5, 0.5, 0.5, 0.5) 
        bunny_scale = 1.0
        body_mesh = builder.add_body(xform=wp.transform(p=bunny_pos, q=bunny_rot), label="mesh")

        builder.add_shape_mesh(body_mesh, mesh=demo_mesh)
        builder.add_shape_box(body_box, hx=self.box_half_extent.x, hy=self.box_half_extent.y, hz=self.box_half_extent.z)
        builder.add_shape_box(body_box1, hx=self.box_half_extent1.x, hy=self.box_half_extent1.y, hz=self.box_half_extent1.z)
        self.render_model = builder.finalize()
        
        self.render_model.setup_viewer(self.viewer)
        self.render_state = self.render_model.state()  
       
        # Setup model, state, solver for simulation
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

        print(f"Adding staggered particle pack: {nx} x {ny} x {nz} (Total: {total_particles})")
        print(
            "Initial layout spacing: "
            f"dx={particle_spacing:.4f}, dy={row_spacing:.4f}, dz={layer_spacing:.4f}, "
            f"jitter={particle_spacing * INITIAL_PARTICLE_JITTER:.4f}"
        )

        self.model = FluidGridLiquidModel(
            fluid_grid_res=(self.n_grid * 2, self.n_grid, self.n_grid),
            fluid_grid_cell_size=self.dh,
            pressure_iteration=30,
            extrap_iterations=5,
            particle_radius=self.dh * 1.2,
            particle_count=total_particles
        )
        self.model.pressure_solver = "jacobi"
        self.model.flip_pic_blend = 0.03 
        self.model.sort_particles_by_cell = True
        self.model.sort_particles_every_n_steps = 30

        self.solver = FluidGridLiquidSolver(self.model)
        self.grid_liquid_domain = FluidGridLiquidDomain(self.model, self.solver)
        
        self.particle_radii = wp.zeros(total_particles, dtype=float, device=self.model._device)
        self.particle_radii.fill_(self.model.particle_radius * 0.3)

        # States are now owned by the domain's internal double buffer.
        self.state_in = self.grid_liquid_domain.create_state()

        # Bake solid sdf
        self.solver.bake_box(self.state_in, self.box_pos, self.box_half_extent)
        self.solver.bake_box(self.state_in, self.box_pos1, self.box_half_extent1)
        self.solver.bake_mesh(self.state_in, demo_mesh, pos=bunny_pos, rot=bunny_rot, scale=bunny_scale)

        # Init fluid particles into the CompositeSimulation's active state
        sim_state_in = self.state_in
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

        # Rendering and output
        self.particle_colors = wp.zeros(total_particles, dtype=wp.vec3, device=self.model._device)
        self.particle_colors.fill_(wp.vec3(0.2, 0.5, 0.9))  # Water particles

        self.output_dir = "ply_output"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"PLY Export Enabled. Output directory: {os.path.abspath(self.output_dir)}")
        
        self.output_dir = "ply_output"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"PLY Export Enabled. Output directory: {os.path.abspath(self.output_dir)}")
        self.export_box_ply("collider_box0.ply", self.box_pos, self.box_half_extent)
        self.export_box_ply("collider_box1.ply", self.box_pos1, self.box_half_extent1)
        usd_points = demo_mesh._vertices
        usd_indices = demo_mesh._indices
        
        #self.export_mesh_ply("collider_bunny.ply", usd_points, usd_indices, bunny_pos, bunny_rot, bunny_scale)
        # ----------------------------

        self.ssfr = None
        self.use_ssfr = bool(ENABLE_SSFR and isinstance(self.viewer, ViewerGL))
        if self.use_ssfr:
            self.ssfr = ScreenSpaceFluidRenderer(
                viewer=self.viewer,
                max_particles=total_particles,
                particle_radius=self.model.particle_radius,
                device=self.model._device,
            )
            self.viewer.register_post_render_callback(lambda current_viewer: self.ssfr.render(current_viewer))

        self.frame_count = 0
        self._frame_wall_start: float | None = None
        self._last_sim_ms = 0.0
        self._last_status_len = 0

        self.total_simulation_time = 0.0

    def step(self):
        self._frame_wall_start = time.perf_counter()
        for _ in range(self.sim_substeps):
            self.grid_liquid_domain.step(self.sim_dt)
        wp.synchronize_device(self.model._device)
        self._last_sim_ms = (time.perf_counter() - self._frame_wall_start) * 1000.0

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        # Render solid
        self.viewer.log_state(self.render_state.as_newton_state())
        # Render fluid
        current_sim_state = self.state_in
        # Render fluid
        if self.ssfr is not None and self.ssfr.available:
            self.ssfr.set_particles(current_sim_state.particle_q)
            self.viewer.log_points(name="water_surface", points=None)
        else:
            self.viewer.log_points(
                name="water_surface",
                points=current_sim_state.particle_q,
                radii=self.particle_radii,
                colors=self.particle_colors,
            )

        self.viewer.end_frame()
        wp.synchronize_device(self.model._device)

        frame_total_ms = 0.0
        if self._frame_wall_start is not None:
            frame_total_ms = (time.perf_counter() - self._frame_wall_start) * 1000.0

        status_line = (
            f"frame={self.frame_count:05d} "
            f"sim_ms={self._last_sim_ms:.3f} "
            f"frame_ms={frame_total_ms:.3f}"
        )
        padding = max(0, self._last_status_len - len(status_line))
        print(f"\r{status_line}{' ' * padding}", end="", flush=True)
        self._last_status_len = len(status_line)
        self.frame_count += 1
        self.total_simulation_time += self._last_sim_ms

        # if not self.viewer._paused:
        #     # --- Export particle sequence ---
        #     if self.frame_count <= 180:
        #         self.save_particles_ply(self.frame_count)
        #         if self.frame_count % 10 == 0:
        #             print(f"Exported frame {self.frame_count} - Time: {self.sim_time:.3f}s")
        #     # ----------------------
        #     self.frame_count += 1


    def export_box_ply(self, filename, pos, half_extent):
        """Export a box mesh to PLY."""
        filepath = os.path.join(self.output_dir, filename)
        c, h = pos, half_extent
        
        vertices = [
            [c[0]-h[0], c[1]-h[1], c[2]-h[2]], [c[0]+h[0], c[1]-h[1], c[2]-h[2]],
            [c[0]+h[0], c[1]+h[1], c[2]-h[2]], [c[0]-h[0], c[1]+h[1], c[2]-h[2]],
            [c[0]-h[0], c[1]-h[1], c[2]+h[2]], [c[0]+h[0], c[1]-h[1], c[2]+h[2]],
            [c[0]+h[0], c[1]+h[1], c[2]+h[2]], [c[0]-h[0], c[1]+h[1], c[2]+h[2]],
        ]
        faces = [
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5],
        ]
        with open(filepath, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\nend_header\n")
            for v in vertices: f.write(f"{v[0]} {v[1]} {v[2]}\n")
            for face in faces: f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        print(f"Static box exported to {filepath}")

    def export_mesh_ply(self, filename, points, indices, pos, rot, scale):
        """Export a transformed mesh to PLY."""
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert Warp quaternion (x, y, z, w) to a rotation matrix
        x, y, z, w = rot[0], rot[1], rot[2], rot[3]
        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        # Apply scale, rotation, and translation
        pos_arr = np.array([pos[0], pos[1], pos[2]])
        transformed_points = points @ R.T * scale + pos_arr
        
        # Ensure indices is one-dimensional
        indices = indices.flatten()
        num_faces = len(indices) // 3

        with open(filepath, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(transformed_points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {num_faces}\n")
            f.write("property list uchar int vertex_indices\nend_header\n")
            for v in transformed_points: 
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            for i in range(num_faces):
                f.write(f"3 {indices[i*3]} {indices[i*3+1]} {indices[i*3+2]}\n")
        print(f"Static mesh exported to {filepath}")

    def save_particles_ply(self, frame_idx):
        """Export fluid particles to PLY."""
        filename = os.path.join(self.output_dir, f"particles_{frame_idx:05d}.ply")
        
        # Read particle data from the composite simulation active state
        current_sim_state = self.state_in
        points = current_sim_state.particle_q.numpy()
        colors = self.particle_colors.numpy()
        
        n_points = len(points)
        colors_u8 = (colors * 255).astype(np.uint8)
        
        vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        vertex_data = np.empty(n_points, dtype=vertex_dtype)
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['red'] = colors_u8[:, 0]
        vertex_data['green'] = colors_u8[:, 1]
        vertex_data['blue'] = colors_u8[:, 2]

        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n_points}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        with open(filename, "wb") as f:
            f.write(header.encode('ascii'))
            f.write(vertex_data.tobytes())

if __name__ == "__main__":
    wp.init()
    viewer, args = init_fluid_viewer()
    example = Example(viewer)
    newton.examples.run(example, args)
    print(f"\nAverage simulation time per frame: {example.total_simulation_time / example.frame_count:.3f} ms")
