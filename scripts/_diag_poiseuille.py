"""Minimal Poiseuille diagnostic — dump velocity field."""
import sys, types as _t
def _mk(n, p=None):
    m = _t.ModuleType(n)
    if p: m.__path__ = [p]
    sys.modules[n] = m
_mk("wanphys", "wanphys"); _mk("wanphys._src", "wanphys/_src")
_mk("wanphys._src.fluid", "wanphys/_src/fluid")
_mk("wanphys._src.fluid.fluid_grid", "wanphys/_src/fluid/fluid_grid")
for nm in ["wanphys._src.fluid.fluid_grid.lbm","wanphys._src.fluid.fluid_grid.liquid",
           "wanphys._src.fluid.fluid_grid.coupling","wanphys._src.collision",
           "wanphys._src.collision.rigid","wanphys._src.geometry",
           "wanphys.collision","wanphys.geometry","wanphys.utils"]:
    _mk(nm)
sys.modules["wanphys.collision"].CollisionPipeline = None
sys.modules["wanphys.geometry"].CollisionTriMeshStyle3D = None
sys.modules["wanphys.geometry"].CollisionTriMeshVBD = None
sys.modules["wanphys.utils"].load_mesh_file = None
sys.modules["wanphys.utils"].load_point_cloud = None
for a in ["FluidGridLiquidDomain","FluidGridLiquidModel","FluidGridLiquidSolver","FluidGridLiquidState"]:
    setattr(sys.modules["wanphys._src.fluid.fluid_grid.liquid"], a, None)
for a in ["LbmDomain","LbmModel","LbmSolver","LbmState"]:
    setattr(sys.modules["wanphys._src.fluid.fluid_grid.lbm"], a, None)

import warp as wp; wp.init()
import numpy as np
from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFSLbmModel
from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFSLbmSolver
from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFSLbmState

# Tiny Poiseuille: 8x16x4 for quick debug
Nx, Ny, Nz = 8, 16, 4
gx = 1e-4  # stronger force to see effect faster
tau = 0.55
model = HomeFSLbmModel(
    fluid_grid_res=(Nx, Ny, Nz), fluid_grid_cell_size=1.0, tau=tau,
    bc_types=("periodic","periodic","bounce_back","bounce_back","periodic","periodic"),
    gravity_x=gx,
)
solver = HomeFSLbmSolver(model)
sa = HomeFSLbmState(model); sb = HomeFSLbmState(model)
solver.initialize_state(sa, rho0=1.0)
tn = solver._total_num

# Print initial flag layout at y profile
fn_init = sa.f_mom.numpy()
fl_init = sa.flag.numpy()
print("Initial flag at x=4, z=2:")
for iy in range(Ny):
    f = fl_init[4, iy, 2]
    print(f"  y={iy:2d}: flag={f:#010b} TYPE_S={bool(f&8)} TYPE_F={bool(f&1)}")

# Run a few steps, dump velocity profile
for step in range(200):
    solver.step(sa, sb, 1.0)
    sa, sb = sb, sa
    sa.f_mom, sa.f_mom_post = sa.f_mom_post, sa.f_mom

fn = sa.f_mom.numpy()
fl = sa.flag.numpy()
print("\nVelocity profile u_x(y) at x=4, z=2, step=200:")
print(f"  {'y':>4s}  {'flag':>10s}  {'ux':>12s}  {'uy':>12s}")
for iy in range(Ny):
    idx = 2*Nx*Ny + iy*Nx + 4  # z=2, y=iy, x=4
    ux = float(fn[tn + idx])
    uy = float(fn[2*tn + idx])
    f = fl[4, iy, 2]
    print(f"  {iy:4d}  {f:#010b}  {ux:12.6e}  {uy:12.6e}")

# Check: is u_x higher near walls or center?
ux_profile = [float(fn[tn + 2*Nx*Ny + iy*Nx + 4]) for iy in range(1, Ny-1)]
print(f"\nux at fluid cells: min={min(ux_profile):.6e} max={max(ux_profile):.6e}")
print(f"ux[1]={ux_profile[0]:.6e} ux[mid]={ux_profile[Ny//2-1]:.6e} ux[{Ny-2}]={ux_profile[-1]:.6e}")
