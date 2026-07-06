# Fluid Visualization with Newton's OpenGL Viewer

This document explains how to visualize grid-based Eulerian fluid simulations using Newton's real-time OpenGL viewer.

## Overview

The `FluidViewerGL` class extends Newton's ViewerGL to provide real-time visualization of grid-based fluids using:

- **Marker Particles**: Tracer particles advected through the velocity field, colored by speed
- **Velocity Vectors**: Optional arrow visualization of the velocity field
- **Domain Boundary**: Wireframe box showing the fluid domain bounds
- **Interactive UI**: ImGui-based controls for visualization options and statistics

## Quick Start

```python
from newton.viewer import ViewerGL
from wanphys.fluid import FluidModel, FluidDomain, FluidViewerGL
from wanphys.core import CompositeSimulation

# Create fluid model
model = FluidModel(
    resolution=(32, 32, 32),
    cell_size=0.1,
    origin=(-1.6, 0.0, -1.6),
    density=1000.0,
    gravity=(0.0, -9.81, 0.0),
    pressure_iterations=50,
)

# Create simulation
fluid = FluidDomain(model)
sim = CompositeSimulation()
sim.add_domain(fluid)

# Create OpenGL viewer
viewer = ViewerGL(width=1920, height=1080)

# Create fluid visualization extension
fluid_viz = FluidViewerGL(
    viewer,
    model,
    num_particles=5000,
    show_vectors=False,
)

# Simulation loop
dt = 1.0 / 60.0
while not viewer.renderer.has_exit():
    sim.step(dt)
    
    viewer.begin_frame(sim.time)
    fluid_viz.render(sim.get_state("fluid"), sim.time, dt)
    viewer.end_frame()
```

## Features

### Marker Particles

Particles are advected through the velocity field using trilinear interpolation from the staggered MAC grid. Particle colors indicate velocity magnitude:
- **Blue**: Slow/stationary
- **Cyan**: Medium speed  
- **Red**: Fast

### Velocity Vectors

When enabled, arrow visualizations show the velocity field at regular grid intervals. Useful for understanding flow patterns but can clutter the view with high resolution grids.

### Interactive Controls

The viewer adds a "Fluid Visualization" panel with:
- Toggle particles, boundary, and vectors on/off
- Adjust particle size
- Control vector spacing and scale
- View real-time statistics (max velocity, pressure, divergence)

### Camera Controls

- **Mouse Left-Drag**: Rotate camera
- **Mouse Right-Drag**: Pan camera
- **Mouse Scroll**: Zoom in/out
- **WASD**: Move camera horizontally
- **Q/E**: Move camera up/down
- **ESC**: Exit viewer

## Integration with Newton

`FluidViewerGL` integrates seamlessly with Newton's viewer architecture:

- Uses `ViewerGL.log_points()` for particle rendering
- Uses `ViewerGL.log_lines()` for boundary and vectors
- Registers UI callbacks for fluid-specific controls
- Compatible with all Newton viewer features (camera, picking, etc.)

## Performance Tips

- Reduce `num_particles` if frame rate drops (default: 5000)
- Disable velocity vectors for better performance
- Use larger `vector_spacing` if vectors are enabled
- The advection kernel runs on GPU for best performance

## Example

See `wanphys/examples/fluid_tank_gl.py` for a complete working example.

Run with:
```bash
uv run python -m wanphys.examples.fluid_tank_gl
```

## Architecture

The integration works as follows:

1. `FluidViewerGL` wraps an existing `ViewerGL` instance
2. Marker particles are stored as Warp arrays on GPU
3. Each render call:
   - Advects particles through velocity field (GPU kernel)
   - Converts particle data to colors based on velocity
   - Logs to viewer using standard APIs
4. UI callbacks are registered with viewer for controls

This design allows fluid visualization to coexist with other Newton model visualizations in the same viewer.
