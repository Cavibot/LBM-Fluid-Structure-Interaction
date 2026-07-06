# WanPhys Onboarding Guide

Welcome to WanPhys! This guide will help you get started with the framework, whether you're running examples, writing simulations, or contributing to development.

## Table of Contents

1. [What is WanPhys?](#what-is-wanphys)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Running Examples](#running-examples)
5. [Writing Your First Simulation](#writing-your-first-simulation)
6. [Visualizing Simulations](#visualizing-simulations)
7. [Development Workflow](#development-workflow)
8. [Getting Help](#getting-help)

---

## What is WanPhys?

WanPhys is a collaborative GPU-accelerated physics simulation framework built on top of Newton. It supports multiple physics domains:

- **Rigid Bodies** - Collision detection, joints, articulations
- **Cloth** - FEM-based soft body simulation
- **Fluid** - Eulerian and Lagrangian fluid dynamics
- **Coupling** - Multi-domain interactions (coming soon)

### Key Features

- **GPU Acceleration** - Powered by NVIDIA Warp for CUDA performance
- **Modular Design** - Clean separation between Model, State, and Solver
- **Isolation Layer** - Decoupled from Newton backend for future migration
- **Composable** - Mix multiple physics domains in one simulation
- **Team-Friendly** - Organized structure for parallel development

### Strategic Goal

WanPhys is gradually replacing Newton's internal implementations with custom GPU-optimized code while maintaining a stable API through isolation layers.

---

## Quick Start

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **CUDA-capable GPU** (optional but recommended)
- **Git**
- **UV package manager** (faster than pip)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd WanPhys
   ```

2. **Install UV (if not already installed):**
   ```bash
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies:**
   ```bash
   uv sync --extra dev --extra examples
   ```

4. **Verify installation:**
   ```bash
   uv run python -m wanphys.examples.rigid_falling_bodies --viewer null --num-frames 10
   ```

   If you see physics simulation output without errors, you're ready!

---

## Project Structure

```
WanPhys/
├── newton/                   # Newton physics engine (upstream dependency)
│   ├── _src/                # Newton implementation
│   ├── examples/            # Newton viewer framework
│   └── tests/               # Newton tests
│
├── wanphys/                 # WanPhys extensions
│   ├── core/                # Core abstractions (Domain, State, Solver)
│   │   ├── composite.py    # CompositeSimulation ABC
│   │   ├── domain.py       # Domain protocol
│   │
│   ├── rigid/               # Rigid body domain
│   │   ├── model.py        # RigidModel (WanPhys model facade)
│   │   ├── state.py        # RigidState (convenience API)
│   │   ├── solver.py       # RigidSolver (isolation layer)
│   │   └── domain.py       # RigidDomain
│   │
│   ├── cloth/               # Cloth simulation domain
│   │   ├── model.py        # ClothModel
│   │   ├── solver.py       # ClothSolver (VBD)
│   │   └── domain.py       # ClothDomain
│   │
│   ├── fluid/               # Fluid simulation domain
│   │   ├── model.py        # FluidModel
│   │   ├── solver.py       # FluidSolver (Eulerian)
│   │   └── domain.py       # FluidDomain
│   │
│   └── examples/            # Example simulations
│       ├── utils.py        # Boilerplate reduction utilities
│       ├── rigid_*.py      # Rigid body examples
│       ├── cloth_*.py      # Cloth examples
│       └── fluid_*.py      # Fluid examples
│
├── CLAUDE.md                # AI assistant instructions
├── DESIGN.md                # Architecture deep dive
├── ONBOARDING.md            # This file
└── pyproject.toml           # Project configuration
```

### Key Concepts

#### Model-State-Solver Pattern

WanPhys uses a three-component abstraction:

1. **Model** - Static scene configuration (geometry, materials, constraints)
2. **State** - Dynamic simulation data (positions, velocities, forces)
3. **Solver** - Physics integration algorithm (XPBD, VBD, MPM, etc.)

#### Domain System

Each physics domain (rigid, cloth, fluid) implements the `Domain` protocol:

```python
class Domain:
    @property
    def name(self) -> str: ...

    def create_state(self) -> State: ...

    def step(self, state_in: State, state_out: State, dt: float) -> None: ...
```

#### Composite Simulation

For coupled multi-domain simulations, subclass `CompositeSimulation` and implement
your own `step()` logic with explicit step ordering.

For single-domain simulations, use an explicit step loop:

```python
from wanphys.collision import CollisionPipeline

rigid_domain.create_state()

# Explicit step loop
rigid_domain.state.clear_forces()
rigid_domain.pre_step(dt)
contacts = CollisionPipeline.collide_rigid(rigid_domain)
rigid_domain.step(dt, contacts=contacts)
rigid_domain.post_step(dt)
```

---

## Running Examples

### Available Examples

WanPhys includes examples demonstrating each domain:

| Example | Domain | Description |
|---------|--------|-------------|
| `rigid_falling_bodies.py` | Rigid | Multiple shapes falling with collisions |
| `rigid_pendulum.py` | Rigid | Simple pendulum with energy tracking |
| `cloth_flag.py` | Cloth | Flag simulation with wind |
| `fluid_pbf_dam_break.py` | Fluid | PBF dam break simulation |
| `fluid_dfsph_dam_break.py` | Fluid | DFSPH dam break with rigid coupling |
| `fluid_wcsph_dam_break.py` | Fluid | WCSPH dam break with rigid coupling |
| `fluid_grid_basic.py` | Fluid | Grid-based fluid simulation |
| `fluid_pbf_emitter_corals.py` | Fluid | PBF fluid with particle emitter |
| `point_cloud_demo.py` | Utility | Point cloud visualization |
| `broad_phase_benchmark.py` | Collision | Broad phase algorithm benchmark |
| `rigid_fluid_gated_benchmark.py` | Coupling | Rigid-fluid coupling benchmark |
| `example_sensor_contact.py` | Sensor | Contact sensor demonstration |
| `example_sensor_imu.py` | Sensor | IMU sensor demonstration |
| `example_sensor_tiled_camera.py` | Sensor | Tiled camera ray-tracing sensor |

### Running an Example

```bash
# Interactive OpenGL viewer (default)
uv run python -m wanphys.examples.rigid_falling_bodies

# Headless mode (no visualization, faster)
uv run python -m wanphys.examples.rigid_falling_bodies --viewer null --num-frames 100

# Export to USD (for Blender, Omniverse)
uv run python -m wanphys.examples.rigid_pendulum --viewer usd --output-path pendulum.usd --num-frames 300

# Rerun.io visualization
uv run python -m wanphys.examples.cloth_flag --viewer rerun
```

### Common Options

- `--viewer {gl,usd,rerun,null}` - Viewer type
- `--device {cuda:0,cpu}` - Compute device
- `--headless` - Run OpenGL without window
- `--num-frames N` - Number of simulation frames
- `--output-path PATH` - Output file for USD

---

## Writing Your First Simulation

Let's create a simple rigid body simulation from scratch, using WanPhys's isolation layer.

### Step 1: Import Dependencies

```python
from wanphys.collision import CollisionPipeline
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig, create_xpbd_solver
from wanphys.examples.utils import init_warp, init_simulation_params
```

### Step 2: Initialize

```python
def main():
    # Initialize Warp
    device = init_warp()

    # Set simulation parameters
    params = init_simulation_params(fps=60, substeps=4)
```

### Step 3: Build the Scene

```python
    # Create model using WanPhys's rigid builder
    builder = RigidModelBuilder(up_axis=1)  # Y-up

    # Add ground plane
    builder.add_ground_plane()

    # Add a falling sphere
    cfg = ShapeConfig(density=1000.0, mu=0.5, restitution=0.3)
    body = builder.add_body(position=(0.0, 5.0, 0.0))
    builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg)

    # Finalize WanPhys rigid model
    model = builder.finalize(device=device)
```

### Step 4: Create Domain and Initialize State

Create the domain from the WanPhys model and solver:

```python
    solver = create_xpbd_solver(model, iterations=20)
    rigid_domain = RigidDomain(model, solver=solver)
    rigid_domain.create_state()
```

### Step 5: Run Simulation Loop

```python
    # Simulate for 3 seconds
    time = 0.0
    for step in range(180):
        rigid_domain.state.clear_forces()
        contacts = CollisionPipeline.collide_rigid(rigid_domain)
        rigid_domain.step(params.sim_dt, contacts=contacts)
        time += params.sim_dt

        if step % 30 == 0:  # Print every 0.5 seconds
            state = rigid_domain.state
            # Use RigidState's convenience methods
            pos = state.get_body_position(0)
            vel = state.get_body_linear_velocity(0)
            print(f"t={time:.2f}s: pos={pos}, vel={vel}")

if __name__ == "__main__":
    main()
```

### Complete Example

Save as `my_first_sim.py`:

```python
from wanphys.collision import CollisionPipeline
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig, create_xpbd_solver
from wanphys.examples.utils import init_warp, init_simulation_params

def main():
    # Initialize
    device = init_warp()
    params = init_simulation_params(fps=60, substeps=4)

    # Build scene
    builder = RigidModelBuilder(up_axis=1)
    builder.add_ground_plane()

    cfg = ShapeConfig(density=1000.0, mu=0.5, restitution=0.3)
    body = builder.add_body(position=(0.0, 5.0, 0.0))
    builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg)

    model = builder.finalize(device=device)
    solver = create_xpbd_solver(model, iterations=20)
    rigid_domain = RigidDomain(model, solver=solver)
    rigid_domain.create_state()

    # Simulate
    time = 0.0
    for step in range(180):
        rigid_domain.state.clear_forces()
        contacts = CollisionPipeline.collide_rigid(rigid_domain)
        rigid_domain.step(params.sim_dt, contacts=contacts)
        time += params.sim_dt
        if step % 30 == 0:
            state = rigid_domain.state
            pos = state.get_body_position(0)
            print(f"t={time:.2f}s: position={pos}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
uv run python my_first_sim.py
```

**Note on Collision Handling:**
Collision detection is explicit — call `CollisionPipeline.collide_rigid(domain)` before `domain.step()`. The pipeline:

1. **Collision Detection** - Finds which bodies are touching (via Newton's collision kernel)
2. **Returns Contacts** - Pass the contacts object to `domain.step(dt, contacts=contacts)`
3. **Collision Response** - XPBD solver receives contacts and computes forces to prevent penetration
4. **State Integration** - Updates positions and velocities

The key insight: **Detection** and **Response** are separate steps, and you control the order explicitly.

---

## Development Workflow

### Collaboration Rules

**CRITICAL:** This is a multi-team project. Follow these rules:

1. **Never modify Newton base classes** - Use isolation layers instead
2. **Work only in your team's directory** - e.g., `wanphys/rigid/`, `wanphys/fluid/`
3. **Use composition over inheritance** - Wrap Newton classes, don't inherit
4. **Coordinate cross-team changes** - Create dedicated coupling branches

**For detailed Chinese team collaboration guidelines, see [协作指南.md](协作指南.md)** - covers architecture strategy, multi-team development workflow, and integration approach.

### Branch Model

```
main              # Production
  ├── dev         # Development (branch from here)
  │   ├── wanphys/rigid/*      # Team branches
  │   ├── wanphys/fluid/*
  │   └── wanphys/cloth/*
  └── feature/coupling_rigid_fluid  # Cross-team branches
```

### Code Style

- **PEP 8** compliant
- **120 character** line length
- **Google-style docstrings**
- **Type hints** encouraged

Example:
```python
def create_rigid_domain(
    model: newton.Model,
    **kwargs,
) -> RigidDomain:
    """Create WanPhys rigid domain.

    Args:
        model: Physics model configuration.
        **kwargs: Additional solver parameters.

    Returns:
        Initialized RigidDomain.
    """
    ...
```

---

## Visualizing Simulations

Newton provides multiple viewer backends for visualization. WanPhys examples integrate seamlessly with these viewers.

### Viewer Backends

| Viewer | Description | Use Case |
|--------|-------------|----------|
| **gl** | OpenGL viewer (interactive) | Development, debugging, demos |
| **usd** | USD file export | Blender, Omniverse, offline rendering |
| **rerun** | Rerun.io streaming | Remote visualization, analysis |
| **null** | No visualization | Benchmarking, headless servers |

### Adding Viewer to Your Simulation

Let's extend the previous example to add visualization:

#### Step 1: Use Newton's Example Framework

```python
import warp as wp
import newton.examples  # Newton's viewer framework

from wanphys.collision import CollisionPipeline
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig, create_xpbd_solver
from wanphys.examples.utils import init_simulation_params
```

#### Step 2: Create Example Class

```python
class MySimulation:
    def __init__(self, viewer, args=None):
        self.viewer = viewer

        # Simulation parameters
        params = init_simulation_params(fps=60, substeps=4)
        self.sim_dt = params.sim_dt
        self.sim_time = 0.0

        # Build scene
        builder = RigidModelBuilder(up_axis=1)
        builder.add_ground_plane()

        cfg = ShapeConfig(density=1000.0, mu=0.5, restitution=0.3)
        body = builder.add_body(position=(0.0, 5.0, 0.0))
        builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg)

        # Create WanPhys domain
        model = builder.finalize()
        solver = create_xpbd_solver(model, iterations=20)
        self.rigid_domain = RigidDomain(model, solver=solver)
        self.rigid_domain.create_state()
        self.contacts = None

        # Setup viewer
        model.setup_viewer(self.viewer)
        self.viewer.set_camera(
            pos=wp.vec3(0.0, 3.0, 8.0),
            pitch=-20.0,
            yaw=-180.0,
        )

    def step(self):
        """Advance simulation by one frame."""
        self.rigid_domain.state.clear_forces()
        self.viewer.apply_forces(self.rigid_domain.state.as_newton_state())  # viewer bridge
        contacts = CollisionPipeline.collide_rigid(self.rigid_domain)
        self.rigid_domain.step(self.sim_dt, contacts=contacts)
        self.contacts = contacts

        self.sim_time += self.sim_dt

    def render(self):
        """Render current state to viewer."""
        state = self.rigid_domain.state
        newton_state = state.as_newton_state()  # viewer bridge
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(newton_state)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, newton_state)
        self.viewer.end_frame()
```

#### Step 3: Main Entry Point

```python
if __name__ == "__main__":
    # Initialize viewer using Newton's framework
    viewer, args = newton.examples.init()

    # Create and run simulation
    example = MySimulation(viewer, args)
    newton.examples.run(example, args)
```

### Adding Mouse Interaction (Force Picking)

To enable interactive mouse forces, call `viewer.apply_forces()` inline in `step()` after `clear_forces()` but before `domain.step()`:

```python
def step(self):
    self.rigid_domain.state.clear_forces()
    self.viewer.apply_forces(self.rigid_domain.state.as_newton_state())  # viewer bridge
    contacts = CollisionPipeline.collide_rigid(self.rigid_domain)
    self.rigid_domain.step(self.sim_dt, contacts=contacts)
```

**How it works:**
1. OpenGL viewer detects mouse clicks and drags
2. Viewer calculates force direction and magnitude
3. `apply_forces()` applies spring forces to the clicked body
4. Forces are integrated by the solver in the same timestep

**Usage (OpenGL viewer):**
- **Click + drag** a body to apply forces
- Forces feel like a spring pulling the body toward the mouse
- Release to let the body go

**Note:** This only works with interactive viewers (OpenGL). USD/Rerun/Null viewers ignore the call.

See `wanphys/examples/rigid_falling_bodies.py` and `wanphys/examples/rigid_pendulum.py` for complete examples.

### Running with Different Viewers

```bash
# Interactive OpenGL (default)
uv run python my_simulation.py

# Headless mode (no window)
uv run python my_simulation.py --viewer gl --headless

# Export to USD for Blender/Omniverse
uv run python my_simulation.py --viewer usd --output-path output.usd --num-frames 300

# Stream to Rerun.io
uv run python my_simulation.py --viewer rerun

# Benchmarking (no visualization)
uv run python my_simulation.py --viewer null --num-frames 1000
```

### Camera Control

Position the camera to frame your scene:

```python
# Method 1: Position + Angles
setup_viewer(
    viewer, model, state,
    camera_pos=(x, y, z),     # Camera position
    camera_pitch=angle,       # Up/down (degrees)
    camera_yaw=angle,         # Left/right (degrees)
)

# Method 2: Position + Target
setup_viewer(
    viewer, model, state,
    camera_pos=(5, 5, 5),     # Camera position
    camera_target=(0, 0, 0),  # Look at origin
)
```

**Tips:**
- For falling objects: Place camera above and in front
- For pendulums: Side view with centered target
- For cloth: Front view looking at the fabric

### Interactive Controls (OpenGL)

When using `--viewer gl`:
- **Mouse drag** - Rotate camera
- **Mouse wheel** - Zoom in/out
- **Right click + drag** - Pan camera
- **Space** - Pause/resume
- **Click body + drag** - Apply forces

### Custom UI Panels (OpenGL)

Add interactive controls to your viewer:

```python
class MySimulation:
    def __init__(self, viewer, args=None):
        # ... setup code ...

        # Register UI callback (OpenGL only)
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self.gui, position="side")

    def gui(self, imgui):
        """Custom UI panel."""
        imgui.text("Simulation Controls")
        imgui.separator()

        # Display current state
        imgui.text(f"Time: {self.sim_time:.2f}s")

        pos = self.state.get_body_position(0)
        imgui.text(f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        # Interactive sliders
        changed, self.gravity = imgui.slider_float(
            "Gravity", self.gravity, -20.0, 0.0
        )
```

See `wanphys/examples/rigid_pendulum.py` and `wanphys/examples/cloth_flag.py` for full examples with custom UI.

---

## Getting Help

### Documentation

- **`CLAUDE.md`** - Project instructions and commands
- **`DESIGN.md`** - Architecture and migration strategy
- **`协作指南.md`** - Team collaboration guidelines (Chinese) - multi-team workflow and integration
- **`wanphys/examples/README.md`** - Examples guide
- **`wanphys/examples/BOILERPLATE_REDUCTION.md`** - Utility functions reference

### Common Issues

#### "CUDA device not found"
- Check GPU drivers: `nvidia-smi`
- Fall back to CPU: `--device cpu`

#### "Module not found: newton"
- Install dependencies: `uv sync --extra dev --extra examples`

#### "Pre-commit hook failed"
- Review auto-fixes: `git diff`
- Stage fixes: `git add .`
- Commit again

#### Example won't run
- Check you're using new names: `rigid_falling_bodies` not `falling_bodies_viewer`
- Use `--viewer null` for headless testing

### Getting Support

1. **Check documentation** - Most questions are answered in CLAUDE.md or DESIGN.md
2. **Look at examples** - See how similar problems are solved
3. **Ask your team** - Coordinate within your domain team
4. **Create an issue** - For bugs or feature requests

---

## Next Steps

Now that you're set up:

1. **Run all examples** - Get familiar with each domain
2. **Read DESIGN.md** - Understand the architecture
3. **Try writing a simple simulation** - Use the template above
4. **Add visualization** - Integrate Newton's viewer for interactive development
5. **Explore your domain** - Dive into `wanphys/rigid/`, `wanphys/cloth/`, or `wanphys/fluid/`
6. **Contribute** - Follow collaboration rules and submit PRs

Welcome to WanPhys! 🚀

---

**Last Updated:** 2025-01-31
**WanPhys Version:** 0.1.0 (Development)
