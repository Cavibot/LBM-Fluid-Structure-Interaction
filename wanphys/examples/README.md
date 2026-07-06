# WanPhys Examples

Example demonstrations of WanPhys domains and features.

## Running Examples

### Direct Execution

Run examples directly as Python modules:

```bash
# Rigid body pendulum (interactive OpenGL)
python -m wanphys.examples.rigid_pendulum

# Rigid body falling bodies
python -m wanphys.examples.rigid_falling_bodies

# Rigid body primitive shape stack
python -m wanphys.examples.rigid_basic_shapes

# Rigid bunnies falling into a static box
python -m wanphys.examples.rigid_bunny_in_box

# Cloth flag simulation
python -m wanphys.examples.cloth_flag

# PBF dam break
python -m wanphys.examples.fluid_pbf_dam_break

# DFSPH dam break with rigid coupling
python -m wanphys.examples.fluid_dfsph_dam_break

# WCSPH dam break with rigid coupling
python -m wanphys.examples.fluid_wcsph_dam_break

# Grid-based fluid
python -m wanphys.examples.fluid_grid_basic

# PBF fluid with particle emitter
python -m wanphys.examples.fluid_pbf_emitter_corals

# Point cloud visualization
python -m wanphys.examples.point_cloud_demo

# Sensors
python -m wanphys.examples.sensors.example_sensor_contact
python -m wanphys.examples.sensors.example_sensor_imu
python -m wanphys.examples.sensors.example_sensor_tiled_camera

# LBM (Lattice Boltzmann Method) examples live under wanphys.examples.lbm
python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak --viewer gl
python -m wanphys.examples.lbm.fluid_grid_lbm_cavity
python -m wanphys.examples.lbm.fluid_grid_lbm_twophase
```

### Viewer Options

Examples using Newton's viewer support multiple backends:

```bash
# OpenGL viewer (default, interactive)
python -m wanphys.examples.rigid_pendulum --viewer gl

# Headless mode (no window)
python -m wanphys.examples.rigid_pendulum --viewer gl --headless

# USD export (for Omniverse, Blender, etc.)
python -m wanphys.examples.rigid_pendulum --viewer usd --output-path pendulum.usd --num-frames 300

# Rerun.io visualization
python -m wanphys.examples.rigid_pendulum --viewer rerun

# Null viewer (benchmarking, no visualization)
python -m wanphys.examples.rigid_pendulum --viewer null --num-frames 100
```

### Command-Line Options

Common options (from `newton.examples.create_parser()`):

- `--device DEVICE` - Warp device (e.g., `cuda:0`, `cpu`)
- `--viewer {gl,usd,rerun,null}` - Viewer type
- `--headless` - Run OpenGL viewer without window
- `--num-frames N` - Number of frames for USD/null viewers
- `--output-path PATH` - Output file for USD viewer

### Smoke Testing

Run all examples in headless mode with output validation:

```bash
uv run python scripts/smoke_test_examples.py
uv run python scripts/smoke_test_examples.py --pattern lbm     # filter by name
uv run python scripts/smoke_test_examples.py --verbose          # show all output
```

### LBM Launcher

A PowerShell menu launcher for the LBM examples is provided:

```bash
pwsh scripts/run_lbm.ps1          # show menu
pwsh scripts/run_lbm.ps1 3        # run example #3 directly
pwsh scripts/run_lbm.ps1 list     # list examples
```

## Example Categories

### Rigid Body

- **`rigid_pendulum.py`** - Interactive pendulum with energy tracking and viewer
- **`rigid_falling_bodies.py`** - Multiple bodies falling with collisions
- **`rigid_basic_shapes.py`** - Primitive rigid shapes falling under XPBD or semi-implicit solvers
- **`rigid_bunny_in_box.py`** - Bunny-shaped rigid bodies falling into a static box

### Fluid (Particle-based)

- **`fluid_pbf_dam_break.py`** - PBF dam break simulation
- **`fluid_dfsph_dam_break.py`** - DFSPH dam break with rigid-fluid coupling
- **`fluid_wcsph_dam_break.py`** - WCSPH dam break with rigid-fluid coupling
- **`fluid_pbf_emitter_corals.py`** - PBF fluid with dynamic particle emitter

### Fluid (Grid-based)

- **`fluid_grid_basic.py`** - Grid-based fluid simulation

### Fluid (Lattice Boltzmann)

All LBM examples live in `wanphys/examples/lbm/` and run as `wanphys.examples.lbm.<name>`:

- **`fluid_grid_lbm.py`** - Baseline LBM: droplet oscillation in a closed box (bounce-back walls, no gravity)
- **`fluid_grid_lbm_cavity.py`** - Lid-driven cavity recirculation (classic CFD validation)
- **`fluid_grid_lbm_cylinder.py`** - Cylinder cross-flow with Kármán vortex street
- **`fluid_grid_lbm_container.py`** - Rigid container one-way FSI (wall rasterization check)
- **`fluid_grid_lbm_dambreak.py`** - Single-phase dam break with passive marker advection
- **`fluid_grid_lbm_dambreak_reg.py`** - Regularized BGK dam break (strong gravity stability)
- **`fluid_grid_lbm_dambreak_trt.py`** - TRT dam break (Two-Relaxation-Time, high accuracy)
- **`fluid_grid_lbm_dambreak_sc.py`** - Shan-Chen two-phase dam break (surface tension)
- **`fluid_grid_lbm_dambreak_sc_rigid.py`** - Shan-Chen dam break with fixed rigid sphere
- **`fluid_grid_lbm_twophase.py`** - SC two-phase droplet equilibrium (sphericity test)
- **`fluid_grid_lbm_twophase_gravity.py`** - SC droplet fall and splash
- **`fluid_grid_lbm_fsi.py`** - Prescribed-motion sphere one-way FSI
- **`fluid_grid_lbm_oneway_moving_rigid_visual.py`** - Oscillating sphere one-way coupling (density/speed render)
- **`fluid_grid_lbm_oneway_moving_rigid_visual_trt.py`** - Same scene with selectable BGK/TRT collision
- **`fluid_grid_lbm_twoway_fsi.py`** - Two-way FSI smoke test (headless, returns rigid forces)
- **`fluid_grid_lbm_twoway_fsi_visual.py`** - Two-way FSI with tracer/field render and force arrows
- **`fluid_grid_lbm_debug.py`** - Static 16³ render sanity check (zero sim steps)
- **`fluid_grid_lbm_validate.py`** - Non-visual validation suite (mass conservation, stability, Zou-He BCs)
- **`_lbm_twoway_scene.py`** - Shared two-way scene infrastructure (not runnable directly)

### Cloth

- **`cloth_flag.py`** - Flag simulation with wind forces

### Sensors

- **`example_sensor_contact.py`** - Contact force sensor
- **`example_sensor_imu.py`** - Inertial measurement unit sensor
- **`example_sensor_tiled_camera.py`** - Ray-tracing tiled camera sensor

### Benchmarks

- **`broad_phase_benchmark.py`** - Broad phase collision detection algorithm comparison
- **`rigid_fluid_gated_benchmark.py`** - Rigid-fluid coupling performance benchmark

### Utilities (not directly runnable)

- **`fluid_particle_emitter.py`** - `PlaneEmitter` and `ParticlePoolAllocator` classes, used by emitter examples
- **`utils.py`** - Common helpers (`init_warp`, `setup_viewer`, `SimulationParams`)

## Example Structure

### Viewer-Integrated Examples

Examples with interactive visualization follow this pattern:

```python
import newton.examples
from wanphys.collision import CollisionPipeline
from wanphys.rigid import RigidDomain, RigidModelBuilder, create_xpbd_solver

class Example:
    def __init__(self, viewer, args=None):
        builder = RigidModelBuilder()
        build_scene(builder)
        model = builder.finalize()
        solver = create_xpbd_solver(model)
        self.rigid = RigidDomain(model, solver=solver)
        self.rigid.create_state()
        self.viewer = viewer
        model.setup_viewer(self.viewer)

    def step(self):
        self.rigid.state.clear_forces()
        contacts = CollisionPipeline.collide_rigid(self.rigid)
        self.rigid.step(self.sim_dt, contacts=contacts)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.rigid.state.as_newton_state())
        self.viewer.end_frame()

if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
```

## Contributing Examples

When adding new examples:

1. **Follow naming convention**: `category_description.py` (e.g., `rigid_double_pendulum.py`)
2. **Add docstring** with description and usage
3. **Use WanPhys API** (not direct Newton calls)
4. **Include `if __name__ == "__main__"`** so the smoke test can discover it
5. **Support `--viewer null`** for headless testing
6. **Update this README** under appropriate category

## Troubleshooting

### Viewer won't open

- Check OpenGL support: `python -m wanphys.examples.rigid_pendulum --viewer gl`
- Use headless mode: `--headless`
- Try null viewer: `--viewer null`

### CUDA errors

- Switch to CPU: `--device cpu`
- Check CUDA installation

### Import errors

```bash
uv sync --extra dev --extra examples
```
