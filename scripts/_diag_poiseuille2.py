"""Inline Poiseuille diagnostic — dump velocity field at test params."""
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

Nx, Ny, Nz = 16, 64, 4
gx = 1e-5; tau = 0.55
model = HomeFSLbmModel(
    fluid_grid_res=(Nx, Ny, Nz), fluid_grid_cell_size=1.0, tau=tau,
    bc_types=("periodic","periodic","bounce_back","bounce_back","periodic","periodic"),
    gravity_x=gx,
)
solver = HomeFSLbmSolver(model)
sa = HomeFSLbmState(model); sb = HomeFSLbmState(model)
solver.initialize_state(sa, rho0=1.0)
tn = solver._total_num

# Run just 2000 steps (shorter) and dump profile at multiple times
for step in range(2000):
    solver.step(sa, sb, 1.0)
    sa, sb = sb, sa
    sa.f_mom, sa.f_mom_post = sa.f_mom_post, sa.f_mom

fn = sa.f_mom.numpy()
fl = sa.flag.numpy()

print(f"Velocity profile u_x(y) at x=8, z=2, step=2000:")
print(f"  {'y':>4s}  {'flag':>10s}  {'ux':>12s}  {'rho':>12s}")
for iy in [0, 1, 2, 4, 8, 16, 24, 31, 32, 40, 48, 56, 60, 62, 63]:
    idx = 2*Nx*Ny + iy*Nx + 8
    ux = float(fn[tn + idx])
    rho = float(fn[idx])
    f = fl[8, iy, 2]
    print(f"  {iy:4d}  {f:#010b}  {ux:12.6e}  {rho:12.6f}")

# Also check u_y to verify no cross-flow
uy_max = max(abs(float(fn[2*tn + 2*Nx*Ny + iy*Nx + 8])) for iy in range(Ny))
print(f"\nmax |u_y| across channel: {uy_max:.3e}")

# Check nx=0 and nx=Nx-1 (should be periodic, same velocity)
ux_x0 = float(fn[tn + 2*Nx*Ny + 32*Nx + 0])
ux_x1 = float(fn[tn + 2*Nx*Ny + 32*Nx + Nx-1])
print(f"ux at center y=32, x=0 vs x=Nx-1: {ux_x0:.6e} vs {ux_x1:.6e}")
