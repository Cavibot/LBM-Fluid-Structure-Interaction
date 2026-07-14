"""Diagnostic: isolate if collision relaxation causes gravity decay."""
import sys
import wanphys._src.fluid.fluid_grid.home_fslbm.tests._safe_import
import warp as wp; wp.init()
import numpy as np
from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFSLbmModel
from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFSLbmSolver
from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFSLbmState

for tau_val, label in [(1.0, "omega=1.0 (no relax)"), (0.55, "omega=1.82")]:
    print(f"\n=== {label} ===")
    gy = -1e-4
    model = HomeFSLbmModel(
        fluid_grid_res=(8,8,8), fluid_grid_cell_size=1.0, tau=tau_val,
        bc_types=('periodic',)*6, gravity_y=gy,
    )
    solver = HomeFSLbmSolver(model)
    sa = HomeFSLbmState(model); sb = HomeFSLbmState(model)
    solver.initialize_state(sa, rho0=1.0)
    tn = solver._total_num

    for step in range(10):
        solver.step(sa, sb, 1.0)
        sa, sb = sb, sa
        sa.f_mom, sa.f_mom_post = sa.f_mom_post, sa.f_mom
        if (step+1) % 5 == 0:
            fn = sa.f_mom.numpy()
            uy = float(np.mean([fn[2*tn+idx] for idx in range(tn)]))
            du = uy / (abs(gy)*(step+1))
            print(f"  step {step+1}: uy={uy:.6e}  ratio={du:.4f}")
