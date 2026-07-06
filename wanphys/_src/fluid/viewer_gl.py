# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""OpenGL viewer extension for grid-based Eulerian fluid simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .state import FluidState
    from .model import FluidModel

try:
    from newton.viewer import ViewerGL
except ImportError:
    ViewerGL = None


class FluidViewerGL:
    """Real-time OpenGL viewer for Eulerian fluid simulation.
    
    This class extends Newton's ViewerGL to visualize grid-based fluids using:
    - Marker particles for velocity field visualization
    - Optional velocity vector arrows
    - Domain boundary box
    
    The viewer integrates seamlessly with Newton's existing viewer system,
    using the same log_points(), log_lines(), and UI callback mechanisms.
    
    Args:
        viewer: Newton ViewerGL instance to extend.
        model: FluidModel configuration.
        num_particles: Number of marker particles to create.
        show_vectors: Whether to show velocity vector arrows.
        vector_spacing: Grid spacing for velocity arrows (in cells).
    """
    
    def __init__(
        self,
        viewer: ViewerGL,
        model: FluidModel,
        num_particles: int = 5000,
        show_vectors: bool = False,
        vector_spacing: int = 4,
    ):
        if ViewerGL is None:
            raise ImportError(
                "Newton's ViewerGL is required for FluidViewerGL. "
                "Make sure newton package is properly installed."
            )
        
        if not isinstance(viewer, ViewerGL):
            raise TypeError("viewer must be an instance of newton.viewer.ViewerGL")
        
        self.viewer = viewer
        self.model = model
        self.num_particles = num_particles
        self.show_vectors = show_vectors
        self.vector_spacing = vector_spacing
        
        # Visualization settings
        self.show_particles = True
        self.show_boundary = True
        self.particle_size = model.cell_size * 0.25
        self.vector_scale = 0.1
        
        # Initialize marker particles
        self._init_particles()
        
        # Initialize boundary visualization
        self._init_boundary()
        
        # Register UI callback for fluid-specific controls
        self.viewer.register_ui_callback(self._render_fluid_ui, position="side")
        
        # Statistics
        self.max_velocity = 0.0
        self.avg_velocity = 0.0
        self.max_pressure = 0.0
        self.max_divergence = 0.0
    
    def _init_particles(self):
        """Initialize marker particles randomly in the fluid domain."""
        nx, ny, nz = self.model.nx, self.model.ny, self.model.nz
        ox, oy, oz = self.model.origin
        dx = self.model.cell_size
        
        # Create random particle positions in grid space
        np.random.seed(42)
        
        particle_i = np.random.uniform(0.5, nx - 0.5, self.num_particles)
        particle_j = np.random.uniform(0.5, ny - 0.5, self.num_particles)
        particle_k = np.random.uniform(0.5, nz - 0.5, self.num_particles)
        
        # Convert to world space
        particle_x = ox + particle_i * dx
        particle_y = oy + particle_j * dx
        particle_z = oz + particle_k * dx
        
        self.particle_positions = np.stack([particle_x, particle_y, particle_z], axis=1).astype(np.float32)
        
        # Warp arrays for GPU-based advection
        self.particle_pos_wp = wp.array(self.particle_positions, dtype=wp.vec3, device=self.model._device)
        self.particle_vel_wp = wp.zeros(self.num_particles, dtype=wp.vec3, device=self.model._device)
    
    def _init_boundary(self):
        """Create boundary box visualization."""
        nx, ny, nz = self.model.nx, self.model.ny, self.model.nz
        ox, oy, oz = self.model.origin
        dx = self.model.cell_size
        
        # Domain corners
        min_corner = np.array([ox, oy, oz], dtype=np.float32)
        max_corner = np.array([ox + nx * dx, oy + ny * dx, oz + nz * dx], dtype=np.float32)
        
        # Create 12 edges of the box
        edges = []
        # Bottom face
        edges.extend([
            (min_corner, [max_corner[0], min_corner[1], min_corner[2]]),
            ([max_corner[0], min_corner[1], min_corner[2]], [max_corner[0], min_corner[1], max_corner[2]]),
            ([max_corner[0], min_corner[1], max_corner[2]], [min_corner[0], min_corner[1], max_corner[2]]),
            ([min_corner[0], min_corner[1], max_corner[2]], min_corner),
        ])
        # Top face
        edges.extend([
            ([min_corner[0], max_corner[1], min_corner[2]], [max_corner[0], max_corner[1], min_corner[2]]),
            ([max_corner[0], max_corner[1], min_corner[2]], max_corner),
            (max_corner, [min_corner[0], max_corner[1], max_corner[2]]),
            ([min_corner[0], max_corner[1], max_corner[2]], [min_corner[0], max_corner[1], min_corner[2]]),
        ])
        # Vertical edges
        edges.extend([
            (min_corner, [min_corner[0], max_corner[1], min_corner[2]]),
            ([max_corner[0], min_corner[1], min_corner[2]], [max_corner[0], max_corner[1], min_corner[2]]),
            ([max_corner[0], min_corner[1], max_corner[2]], max_corner),
            ([min_corner[0], min_corner[1], max_corner[2]], [min_corner[0], max_corner[1], max_corner[2]]),
        ])
        
        self.boundary_starts = wp.array([np.array(e[0], dtype=np.float32) for e in edges], dtype=wp.vec3, device=self.model._device)
        self.boundary_ends = wp.array([np.array(e[1], dtype=np.float32) for e in edges], dtype=wp.vec3, device=self.model._device)
    
    def update(self, state: FluidState, dt: float):
        """Update particle positions by advecting through velocity field.
        
        Args:
            state: Current fluid state.
            dt: Timestep for visualization update (typically 1/60 for 60 FPS).
        """
        # Launch kernel to advect particles
        wp.launch(
            _advect_particles,
            dim=self.num_particles,
            inputs=[
                self.particle_pos_wp,
                self.particle_vel_wp,
                state.velocity_x,
                state.velocity_y,
                state.velocity_z,
                self.model.origin,
                self.model.cell_size,
                self.model.nx,
                self.model.ny,
                self.model.nz,
                dt,
            ],
            device=self.model._device,
        )
        
        # Update statistics
        particle_velocities = self.particle_vel_wp.numpy()
        vel_magnitude = np.linalg.norm(particle_velocities, axis=1)
        self.max_velocity = float(np.max(vel_magnitude)) if len(vel_magnitude) > 0 else 0.0
        self.avg_velocity = float(np.mean(vel_magnitude)) if len(vel_magnitude) > 0 else 0.0
        
        # Pressure statistics
        pressure_np = state.pressure.numpy()
        self.max_pressure = float(np.max(np.abs(pressure_np)))
        
        # Divergence statistics
        div_np = state.divergence.numpy()
        self.max_divergence = float(np.max(np.abs(div_np)))
    
    def render(self, state: FluidState, time: float, dt: float = 1.0 / 60.0):
        """Render the fluid state to the Newton ViewerGL.
        
        Args:
            state: Current fluid state.
            time: Current simulation time.
            dt: Time delta for particle advection (default: 1/60s for smooth 60 FPS).
        """
        # Update particles
        self.update(state, dt)
        
        # Render particles if enabled
        if self.show_particles:
            self._render_particles()
        
        # Render boundary box if enabled
        if self.show_boundary:
            self._render_boundary()
        
        # Render velocity vectors if enabled
        if self.show_vectors:
            self._render_vectors(state)
    
    def _render_particles(self):
        """Render marker particles as colored spheres."""
        # Get velocity magnitude for coloring
        particle_velocities = self.particle_vel_wp.numpy()
        vel_magnitude = np.linalg.norm(particle_velocities, axis=1)
        
        # Normalize to [0, 1] for coloring
        max_vel = np.max(vel_magnitude) if np.max(vel_magnitude) > 1e-6 else 1.0
        vel_normalized = np.clip(vel_magnitude / max_vel, 0.0, 1.0)
        
        # Create colors (blue = slow, cyan = medium, red = fast)
        colors = np.zeros((self.num_particles, 3), dtype=np.float32)
        colors[:, 0] = vel_normalized  # Red channel
        colors[:, 1] = 0.3 * (1.0 - vel_normalized)  # Green channel
        colors[:, 2] = 1.0 - vel_normalized  # Blue channel
        
        # Convert to warp arrays
        colors_wp = wp.array(colors, dtype=wp.vec3, device=self.model._device)
        radii_wp = wp.full(self.num_particles, self.particle_size, dtype=wp.float32, device=self.model._device)
        
        # Log to viewer
        self.viewer.log_points(
            "fluid/particles",
            points=self.particle_pos_wp,
            radii=radii_wp,
            colors=colors_wp,
            hidden=False,
        )
    
    def _render_boundary(self):
        """Render domain boundary box."""
        boundary_color = wp.vec3(0.5, 0.5, 0.5)
        num_edges = len(self.boundary_starts)
        colors = wp.full(num_edges, boundary_color, dtype=wp.vec3, device=self.model._device)
        
        self.viewer.log_lines(
            "fluid/boundary",
            starts=self.boundary_starts,
            ends=self.boundary_ends,
            colors=colors,
            width=0.005,
            hidden=False,
        )
    
    def _render_vectors(self, state: FluidState):
        """Render velocity vector field as arrows."""
        nx, ny, nz = self.model.nx, self.model.ny, self.model.nz
        ox, oy, oz = self.model.origin
        dx = self.model.cell_size
        spacing = self.vector_spacing
        
        # Sample velocity field at regular intervals
        positions = []
        vectors = []
        
        for i in range(0, nx, spacing):
            for j in range(0, ny, spacing):
                for k in range(0, nz, spacing):
                    # Sample velocity at cell center (interpolate from staggered grid)
                    vx = 0.5 * (state.velocity_x.numpy()[i, j, k] + state.velocity_x.numpy()[i + 1, j, k]) if i < nx else 0.0
                    vy = 0.5 * (state.velocity_y.numpy()[i, j, k] + state.velocity_y.numpy()[i, j + 1, k]) if j < ny else 0.0
                    vz = 0.5 * (state.velocity_z.numpy()[i, j, k] + state.velocity_z.numpy()[i, j, k + 1]) if k < nz else 0.0
                    
                    pos = np.array([
                        ox + (i + 0.5) * dx,
                        oy + (j + 0.5) * dx,
                        oz + (k + 0.5) * dx
                    ], dtype=np.float32)
                    
                    vel = np.array([vx, vy, vz], dtype=np.float32)
                    vel_mag = np.linalg.norm(vel)
                    
                    if vel_mag > 1e-6:  # Only show non-zero velocities
                        positions.append(pos)
                        vectors.append(pos + vel * self.vector_scale)
        
        if len(positions) > 0:
            starts = wp.array(positions, dtype=wp.vec3, device=self.model._device)
            ends = wp.array(vectors, dtype=wp.vec3, device=self.model._device)
            colors = wp.full(len(positions), wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3, device=self.model._device)
            
            self.viewer.log_lines(
                "fluid/vectors",
                starts=starts,
                ends=ends,
                colors=colors,
                width=0.01,
                hidden=False,
            )
        else:
            # Clear vectors if none to show
            self.viewer.log_lines("fluid/vectors", None, None, None)
    
    def _render_fluid_ui(self, imgui):
        """Render fluid-specific UI controls in the side panel.
        
        Args:
            imgui: ImGui module passed by the viewer.
        """
        
        if imgui.collapsing_header("Fluid Visualization"):
            # Particle controls
            changed, self.show_particles = imgui.checkbox("Show Particles", self.show_particles)
            
            if self.show_particles:
                changed, new_size = imgui.slider_float(
                    "Particle Size", self.particle_size, 0.001, self.model.cell_size, "%.3f"
                )
                if changed:
                    self.particle_size = new_size
            
            # Boundary controls
            changed, self.show_boundary = imgui.checkbox("Show Boundary", self.show_boundary)
            
            # Vector field controls
            changed, self.show_vectors = imgui.checkbox("Show Velocity Vectors", self.show_vectors)
            
            if self.show_vectors:
                changed, new_spacing = imgui.slider_int("Vector Spacing", self.vector_spacing, 1, 8)
                if changed:
                    self.vector_spacing = new_spacing
                
                changed, new_scale = imgui.slider_float(
                    "Vector Scale", self.vector_scale, 0.01, 1.0, "%.2f"
                )
                if changed:
                    self.vector_scale = new_scale
            
            imgui.separator()
            
            # Statistics
            imgui.text(f"Max Velocity: {self.max_velocity:.3f} m/s")
            imgui.text(f"Avg Velocity: {self.avg_velocity:.3f} m/s")
            imgui.text(f"Max Pressure: {self.max_pressure:.2f} Pa")
            imgui.text(f"Max Divergence: {self.max_divergence:.6f}")
            
            imgui.separator()
            imgui.text(f"Particles: {self.num_particles}")
            imgui.text(f"Grid: {self.model.nx}x{self.model.ny}x{self.model.nz}")
            imgui.text(f"Cell Size: {self.model.cell_size:.3f} m")


# ============================================================================
# Warp Kernels
# ============================================================================

@wp.func
def _sample_velocity(
    pos: wp.vec3,
    vel_x: wp.array3d(dtype=wp.float32),
    vel_y: wp.array3d(dtype=wp.float32),
    vel_z: wp.array3d(dtype=wp.float32),
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
) -> wp.vec3:
    """Sample velocity at a particle position using trilinear interpolation."""
    
    # Convert world position to grid coordinates
    grid_pos = (pos - origin) / dx
    
    # Sample u (x-velocity) - staggered at (i, j+0.5, k+0.5)
    ix = grid_pos[0]
    iy = grid_pos[1] - 0.5
    iz = grid_pos[2] - 0.5
    
    ix = wp.clamp(ix, 0.0, float(nx))
    iy = wp.clamp(iy, 0.0, float(ny - 1))
    iz = wp.clamp(iz, 0.0, float(nz - 1))
    
    i0 = wp.clamp(int(wp.floor(ix)), 0, nx - 1)
    j0 = wp.clamp(int(wp.floor(iy)), 0, ny - 2)
    k0 = wp.clamp(int(wp.floor(iz)), 0, nz - 2)
    
    i1 = wp.min(i0 + 1, nx)
    j1 = j0 + 1
    k1 = k0 + 1
    
    fx = ix - float(i0)
    fy = iy - float(j0)
    fz = iz - float(k0)
    
    # Trilinear interpolation for u
    u000 = vel_x[i0, j0, k0]
    u100 = vel_x[i1, j0, k0]
    u010 = vel_x[i0, j1, k0]
    u110 = vel_x[i1, j1, k0]
    u001 = vel_x[i0, j0, k1]
    u101 = vel_x[i1, j0, k1]
    u011 = vel_x[i0, j1, k1]
    u111 = vel_x[i1, j1, k1]
    
    u00 = u000 * (1.0 - fx) + u100 * fx
    u01 = u001 * (1.0 - fx) + u101 * fx
    u10 = u010 * (1.0 - fx) + u110 * fx
    u11 = u011 * (1.0 - fx) + u111 * fx
    
    u0 = u00 * (1.0 - fy) + u10 * fy
    u1 = u01 * (1.0 - fy) + u11 * fy
    
    u = u0 * (1.0 - fz) + u1 * fz
    
    # Sample v (y-velocity) - staggered at (i+0.5, j, k+0.5)
    ix = grid_pos[0] - 0.5
    iy = grid_pos[1]
    iz = grid_pos[2] - 0.5
    
    ix = wp.clamp(ix, 0.0, float(nx - 1))
    iy = wp.clamp(iy, 0.0, float(ny))
    iz = wp.clamp(iz, 0.0, float(nz - 1))
    
    i0 = wp.clamp(int(wp.floor(ix)), 0, nx - 2)
    j0 = wp.clamp(int(wp.floor(iy)), 0, ny - 1)
    k0 = wp.clamp(int(wp.floor(iz)), 0, nz - 2)
    
    i1 = i0 + 1
    j1 = wp.min(j0 + 1, ny)
    k1 = k0 + 1
    
    fx = ix - float(i0)
    fy = iy - float(j0)
    fz = iz - float(k0)
    
    v000 = vel_y[i0, j0, k0]
    v100 = vel_y[i1, j0, k0]
    v010 = vel_y[i0, j1, k0]
    v110 = vel_y[i1, j1, k0]
    v001 = vel_y[i0, j0, k1]
    v101 = vel_y[i1, j0, k1]
    v011 = vel_y[i0, j1, k1]
    v111 = vel_y[i1, j1, k1]
    
    v00 = v000 * (1.0 - fx) + v100 * fx
    v01 = v001 * (1.0 - fx) + v101 * fx
    v10 = v010 * (1.0 - fx) + v110 * fx
    v11 = v011 * (1.0 - fx) + v111 * fx
    
    v0 = v00 * (1.0 - fy) + v10 * fy
    v1 = v01 * (1.0 - fy) + v11 * fy
    
    v = v0 * (1.0 - fz) + v1 * fz
    
    # Sample w (z-velocity) - staggered at (i+0.5, j+0.5, k)
    ix = grid_pos[0] - 0.5
    iy = grid_pos[1] - 0.5
    iz = grid_pos[2]
    
    ix = wp.clamp(ix, 0.0, float(nx - 1))
    iy = wp.clamp(iy, 0.0, float(ny - 1))
    iz = wp.clamp(iz, 0.0, float(nz))
    
    i0 = wp.clamp(int(wp.floor(ix)), 0, nx - 2)
    j0 = wp.clamp(int(wp.floor(iy)), 0, ny - 2)
    k0 = wp.clamp(int(wp.floor(iz)), 0, nz - 1)
    
    i1 = i0 + 1
    j1 = j0 + 1
    k1 = wp.min(k0 + 1, nz)
    
    fx = ix - float(i0)
    fy = iy - float(j0)
    fz = iz - float(k0)
    
    w000 = vel_z[i0, j0, k0]
    w100 = vel_z[i1, j0, k0]
    w010 = vel_z[i0, j1, k0]
    w110 = vel_z[i1, j1, k0]
    w001 = vel_z[i0, j0, k1]
    w101 = vel_z[i1, j0, k1]
    w011 = vel_z[i0, j1, k1]
    w111 = vel_z[i1, j1, k1]
    
    w00 = w000 * (1.0 - fx) + w100 * fx
    w01 = w001 * (1.0 - fx) + w101 * fx
    w10 = w010 * (1.0 - fx) + w110 * fx
    w11 = w011 * (1.0 - fx) + w111 * fx
    
    w0 = w00 * (1.0 - fy) + w10 * fy
    w1 = w01 * (1.0 - fy) + w11 * fy
    
    w = w0 * (1.0 - fz) + w1 * fz
    
    return wp.vec3(u, v, w)


@wp.kernel
def _advect_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    vel_x: wp.array3d(dtype=wp.float32),
    vel_y: wp.array3d(dtype=wp.float32),
    vel_z: wp.array3d(dtype=wp.float32),
    origin: wp.vec3,
    dx: float,
    nx: int,
    ny: int,
    nz: int,
    dt: float,
):
    """Advect particles through the velocity field."""
    i = wp.tid()
    
    pos = positions[i]
    
    # Sample velocity at particle position
    vel = _sample_velocity(pos, vel_x, vel_y, vel_z, origin, dx, nx, ny, nz)
    
    # Update position (Euler integration)
    pos_new = pos + vel * dt
    
    # Domain bounds with small margin
    margin = dx * 0.1
    domain_min = origin + wp.vec3(margin, margin, margin)
    domain_max = origin + wp.vec3(float(nx) * dx - margin, float(ny) * dx - margin, float(nz) * dx - margin)
    
    # Recycle particles that hit bottom back to near their horizontal position at top
    # Z is the vertical axis in Newton's coordinate system
    if pos_new[2] < domain_min[2]:
        # Keep horizontal position, reset to top
        pos_new = wp.vec3(
            wp.clamp(pos[0], domain_min[0], domain_max[0]),
            wp.clamp(pos[1], domain_min[1], domain_max[1]),
            domain_max[2] - margin,  # Reset to top (Z axis)
        )
    elif pos_new[2] > domain_max[2]:
        # Hit top, send to bottom (shouldn't happen with gravity down)
        pos_new = wp.vec3(
            wp.clamp(pos[0], domain_min[0], domain_max[0]),
            wp.clamp(pos[1], domain_min[1], domain_max[1]),
            domain_min[2] + margin,
        )
    else:
        # Clamp horizontal positions (X, Y)
        pos_new = wp.vec3(
            wp.clamp(pos_new[0], domain_min[0], domain_max[0]),
            wp.clamp(pos_new[1], domain_min[1], domain_max[1]),
            pos_new[2],  # Z is free to move
        )
    
    positions[i] = pos_new
    velocities[i] = vel
