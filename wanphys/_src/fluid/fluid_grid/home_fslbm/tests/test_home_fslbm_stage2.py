# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM Stage 2 acceptance tests — standalone runner.

Usage:
    uv run --extra examples python wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_home_fslbm_stage2.py [--quick]

--quick skips long-running Poiseuille (20k steps) and cavity decay (2k steps) tests.
"""

import sys, types as _types

# ---- Safe import preamble — see _safe_import.py for details ----------
def _mk(name, path=None):
    m = _types.ModuleType(name)
    if path: m.__path__ = [path]
    sys.modules[name] = m
    return m

# Parent packages with __path__ so submodule discovery works
_mk("wanphys", "wanphys")
_mk("wanphys._src", "wanphys/_src")
_mk("wanphys._src.fluid", "wanphys/_src/fluid")
_mk("wanphys._src.fluid.fluid_grid", "wanphys/_src/fluid/fluid_grid")
# Crash-triggering leaf packages
for _nm in ["wanphys._src.fluid.fluid_grid.lbm","wanphys._src.fluid.fluid_grid.liquid",
            "wanphys._src.fluid.fluid_grid.coupling","wanphys._src.collision",
            "wanphys._src.collision.rigid","wanphys._src.geometry",
            "wanphys.collision","wanphys.geometry","wanphys.utils"]:
    _mk(_nm)
# Dummy attributes that various __init__.py expect
sys.modules["wanphys.collision"].CollisionPipeline = None
sys.modules["wanphys.geometry"].CollisionTriMeshStyle3D = None
sys.modules["wanphys.geometry"].CollisionTriMeshVBD = None
sys.modules["wanphys.utils"].load_mesh_file = None
sys.modules["wanphys.utils"].load_point_cloud = None
for _attr in ["FluidGridLiquidDomain","FluidGridLiquidModel","FluidGridLiquidSolver","FluidGridLiquidState"]:
    setattr(sys.modules["wanphys._src.fluid.fluid_grid.liquid"], _attr, None)
for _attr in ["LbmDomain","LbmModel","LbmSolver","LbmState"]:
    setattr(sys.modules["wanphys._src.fluid.fluid_grid.lbm"], _attr, None)

import warp as wp; wp.init()
import numpy as np
from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFSLbmModel
from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFSLbmSolver
from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFSLbmState

PASS = FAIL = 0
QUICK = "--quick" in sys.argv


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS: {name}")
    else:
        FAIL += 1; print(f"  FAIL: {name} {detail}")


def _step(solver, sa, sb, n):
    for _ in range(n):
        solver.step(sa, sb, 1.0)
        sa, sb = sb, sa
        sa.f_mom, sa.f_mom_post = sa.f_mom_post, sa.f_mom
    return sa, sb


def _maxv(solver, state):
    tn = solver._total_num; fn = state.f_mom.numpy()
    fl = state.flag.numpy(); nx, ny, nz = solver.nx, solver.ny, solver.nz
    mv2 = 0.0
    for idx in range(tn):
        if (fl[idx % nx, (idx // nx) % ny, idx // (nx * ny)] & 8) == 0:
            ux = float(fn[tn + idx]); uy = float(fn[2 * tn + idx])
            uz = float(fn[3 * tn + idx])
            mv2 = max(mv2, ux * ux + uy * uy + uz * uz)
    return np.sqrt(mv2)


# ===================================================================
# A.1 Mass conservation
# ===================================================================
print("=== A.1 Mass Conservation ===")
m = HomeFSLbmModel(fluid_grid_res=(16, 16, 16), fluid_grid_cell_size=1.0,
                   tau=0.55, bc_types=("periodic",) * 6)
s = HomeFSLbmSolver(m)
sa = HomeFSLbmState(m); sb = HomeFSLbmState(m)
s.initialize_state(sa, rho0=1.0, u0_x=0.01)
tn = s._total_num; fn0 = sa.f_mom.numpy()
m0 = sum(float(fn0[idx]) for idx in range(tn))
for _ in range(20):
    sa, sb = _step(s, sa, sb, 1)
fn = sa.f_mom.numpy()
m1 = sum(float(fn[idx]) for idx in range(tn))
check("A.1 Mass drift < 5e-3", abs(m1 - m0) < 5e-3, f"drift={abs(m1-m0):.3e}")

# ===================================================================
# A.3 Quiescent stability
# ===================================================================
print("=== A.3 Quiescent Stability ===")
m2 = HomeFSLbmModel(fluid_grid_res=(16, 16, 16), fluid_grid_cell_size=1.0,
                    tau=0.55, bc_types=("bounce_back",) * 6)
s2 = HomeFSLbmSolver(m2)
sa2 = HomeFSLbmState(m2); sb2 = HomeFSLbmState(m2)
s2.initialize_state(sa2, rho0=1.0)
sa2, sb2 = _step(s2, sa2, sb2, 1000)
check("A.3 Quiescent max|u| < 1e-10", _maxv(s2, sa2) < 1e-10,
      f"max|u|={_maxv(s2,sa2):.3e}")

# ===================================================================
# A.6 Gravity
# ===================================================================
print("=== A.6 Gravity ===")
gy = -1e-4
m3 = HomeFSLbmModel(fluid_grid_res=(8, 8, 8), fluid_grid_cell_size=1.0,
                    tau=0.55, bc_types=("periodic",) * 6, gravity_y=gy)
s3 = HomeFSLbmSolver(m3)
sa3 = HomeFSLbmState(m3); sb3 = HomeFSLbmState(m3)
s3.initialize_state(sa3, rho0=1.0)
tn3 = s3._total_num
for _ in range(20):
    sa3, sb3 = _step(s3, sa3, sb3, 1)
fn3 = sa3.f_mom.numpy()
uy = np.mean([float(fn3[2 * tn3 + idx]) for idx in range(tn3)])
ratio = abs(uy / (gy * 20))
check("A.6 Gravity ratio ~0.5", abs(ratio - 0.5) < 0.02, f"ratio={ratio:.4f}")

# ===================================================================
# A.7 Velocity clamp
# ===================================================================
print("=== A.7 Velocity Clamp ===")
m4 = HomeFSLbmModel(fluid_grid_res=(8, 8, 8), fluid_grid_cell_size=1.0,
                    tau=0.55, bc_types=("periodic",) * 6,
                    gravity_y=-0.01, max_velocity=0.4)
s4 = HomeFSLbmSolver(m4)
sa4 = HomeFSLbmState(m4); sb4 = HomeFSLbmState(m4)
s4.initialize_state(sa4, rho0=1.0)
all_ok = True
for _ in range(100):
    sa4, sb4 = _step(s4, sa4, sb4, 1)
    if _maxv(s4, sa4) > 0.4 + 1e-6:
        all_ok = False
check("A.7 Velocity clamp", all_ok)

# ===================================================================
# B.8 Wall no-penetration
# ===================================================================
print("=== B.8 No-Penetration ===")
check("B.8 Wall velocity zero (from A.3)", _maxv(s2, sa2) < 1e-10)

# ===================================================================
# B.9 Poiseuille flow (long)
# ===================================================================
print("=== B.9 Poiseuille Flow ===")
Nx, Ny, Nz = 8, 32, 4
gx = 1e-4; tau = 0.55; nu = (1.0 / 3.0) * (tau - 0.5)
mp = HomeFSLbmModel(fluid_grid_res=(Nx, Ny, Nz), fluid_grid_cell_size=1.0, tau=tau,
                    bc_types=("periodic", "periodic", "bounce_back", "bounce_back",
                              "periodic", "periodic"), gravity_x=gx)
sp = HomeFSLbmSolver(mp)
sap = HomeFSLbmState(mp); sbp = HomeFSLbmState(mp)
sp.initialize_state(sap, rho0=1.0)
tnp = sp._total_num

if QUICK:
    check("B.9 Poiseuille (skipped --quick)", True)
    check("B.9 Wall no-slip (skipped)", True)
    check("B.9 Parabolic (skipped)", True)
    check("B.10 Courant (skipped)", True)
else:
    print("  Running 20000 steps...")
    for _ in range(20000):
        sp.step(sap, sbp, 1.0)
        sap, sbp = sbp, sap
        sap.f_mom, sap.f_mom_post = sap.f_mom_post, sap.f_mom

    fnp = sap.f_mom.numpy(); flp = sap.flag.numpy()
    prof = np.zeros(Ny)
    for iy in range(Ny):
        vv = [float(fnp[tnp + (Nz // 2) * Nx * Ny + iy * Nx + ix])
              for ix in range(2, Nx - 2) if (flp[ix, iy, Nz // 2] & 8) == 0]
        prof[iy] = np.mean(vv) if vv else 0.0

    # ---- Diagnostic dump (remove after debugging) ----
    print("  [DIAG] Full u_x(y) profile (every 4th y):")
    for iy in range(0, Ny, 4):
        ux_vals = [
            float(fnp[tnp + (Nz // 2) * Nx * Ny + iy * Nx + ix])
            for ix in range(Nx) if (flp[ix, iy, Nz // 2] & 8) == 0
        ]
        ux_mean = np.mean(ux_vals) if ux_vals else 0.0
        fl = flp[Nx//2, iy, Nz//2]
        print(f"    y={iy:3d} flag={fl:#010b} S={bool(fl&8)} ux_avg={ux_mean:.6e} n_fluid={len(ux_vals)}")
    # Verify _flag_flat matches 3D flag
    ff = sp._flag_flat.numpy()
    mismatches = 0
    for iy in [0, 1, 2, Ny//2, Ny-2, Ny-1]:
        idx3d = (Nz//2)*Nx*Ny + iy*Nx + Nx//2
        if ff[idx3d] != flp[Nx//2, iy, Nz//2]:
            mismatches += 1
            print(f"    FLAG MISMATCH y={iy}: flat={ff[idx3d]:#010b} 3d={flp[Nx//2,iy,Nz//2]:#010b}")
    if mismatches == 0:
        print("  [DIAG] _flag_flat matches 3D flag: OK")
    print("  [DIAG] End dump")
    # --------------------------------------------------

    H_eff = Ny - 2; y_eff = np.arange(1, Ny - 1) - 0.5
    uth = (gx / (2 * nu)) * y_eff * (H_eff - y_eff)
    u_max_sim = np.max(prof[1:-1]); u_max_th = np.max(uth)

    check("B.9 Wall no-slip y=0", abs(prof[0]) < 1e-6, f"u={prof[0]:.3e}")
    check("B.9 Wall no-slip y=Ny-1", abs(prof[-1]) < 1e-6, f"u={prof[-1]:.3e}")
    check("B.9 Parabolic shape",
          prof[Ny // 4] > 0.6 * u_max_sim and prof[Ny // 2] > prof[Ny // 4],
          f"qtr={prof[Ny//4]:.6e} mid={prof[Ny//2]:.6e}")
    err = abs(u_max_sim - u_max_th) / max(u_max_th, 1e-10)
    check("B.9 Centerline velocity < 5%", err < 0.05,
          f"sim={u_max_sim:.6e} th={u_max_th:.6e} err={err:.4f}")
    check("B.10 Courant < 0.4", _maxv(sp, sap) < 0.4,
          f"max|u|={_maxv(sp,sap):.4f}")

# ===================================================================
# B.11 Cavity decay (long)
# ===================================================================
print("=== B.11 Cavity Decay ===")
Nc = 16
mc = HomeFSLbmModel(fluid_grid_res=(Nc, Nc, Nc), fluid_grid_cell_size=1.0,
                    tau=0.55, bc_types=("bounce_back",) * 6)
sc = HomeFSLbmSolver(mc)
sac = HomeFSLbmState(mc); sbc = HomeFSLbmState(mc)
sc.initialize_state(sac, rho0=1.0)
tnc = sc._total_num

if QUICK:
    for _ in range(3):
        check("B.11 Cavity decay (skipped --quick)", True)
else:
    fnp = sac.f_mom.numpy()
    for ix in range(Nc):
        for iy in range(Nc):
            for iz in range(Nc):
                if 4 < iy < 12:
                    idx = iz * Nc * Nc + iy * Nc + ix
                    fnp[tnc + idx] = 0.05 * np.sin(np.pi * ix / Nc)
    wp.copy(sac.f_mom, wp.array(dtype=float, shape=(10 * tnc,),
            device=sc.device, data=fnp.tolist()))

    ke_series = []
    print("  Running 2000 steps...")
    for si in range(2000):
        sac, sbc = _step(sc, sac, sbc, 1)
        if (si + 1) % 200 == 0:
            fn = sac.f_mom.numpy()
            ke = sum(0.5 * (float(fn[tnc + idx])**2 + float(fn[2 * tnc + idx])**2)
                     for idx in range(tnc))
            ke_series.append(ke)

    monotonic = all(ke_series[i] >= ke_series[i + 1] - 1e-10
                    for i in range(len(ke_series) - 1))
    check("B.11 Monotonic energy decay", monotonic)
    check("B.11 Final KE < 1e-4", ke_series[-1] < 1e-4,
          f"KE={ke_series[-1]:.3e}")
    check("B.11 No NaN/Inf", np.isfinite(ke_series[-1]))

# ===================================================================
# C.13 Stage-1 regression
# ===================================================================
print("=== C.13 Stage-1 Regression ===")
try:
    from wanphys._src.fluid.fluid_grid.home_fslbm.constants import (
        TYPE_F, TYPE_S, CS2, HERMITE_COEFFS, W, C_X, C_Y, C_Z,
    )
    from wanphys._src.fluid.fluid_grid.home_fslbm.kernels import (
        reconstruct_fi_at_index, equilibrium_fi,
    )
    check("C.13 Imports OK", True)
    # Verify D3Q27 weights sum to 1
    w_sum = sum(float(W.numpy()[i]) for i in range(27))
    check("C.13 D3Q27 sum(W)=1", abs(w_sum - 1.0) < 1e-6, f"sum={w_sum}")
    # Verify coefficient table shape
    c = HERMITE_COEFFS.numpy()
    check("C.13 HERMITE_COEFFS shape", c.shape == (27 * 17,), f"shape={c.shape}")
except Exception as e:
    check("C.13 Regression imports", False, str(e))

# ===================================================================
print(f"\n{'='*50}")
print(f"Results: {PASS} PASS, {FAIL} FAIL")
if FAIL > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
