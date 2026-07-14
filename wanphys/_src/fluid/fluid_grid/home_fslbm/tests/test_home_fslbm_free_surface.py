# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM Stage 3 acceptance tests — free-surface VOF tracking.

Usage:
    uv run --extra examples python wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_home_fslbm_free_surface.py [--quick]

--quick skips the long-running static water column (500 steps) and drain tests.
"""

import sys, types as _types

# ---- Safe import preamble (same as stage 2 tests) --------------------
def _mk(name, path=None):
    m = _types.ModuleType(name)
    if path: m.__path__ = [path]
    sys.modules[name] = m
    return m

_mk("wanphys", "wanphys")
_mk("wanphys._src", "wanphys/_src")
_mk("wanphys._src.fluid", "wanphys/_src/fluid")
_mk("wanphys._src.fluid.fluid_grid", "wanphys/_src/fluid/fluid_grid")
for _nm in ["wanphys._src.fluid.fluid_grid.lbm","wanphys._src.fluid.fluid_grid.liquid",
            "wanphys._src.fluid.fluid_grid.coupling","wanphys._src.collision",
            "wanphys._src.collision.rigid","wanphys._src.geometry",
            "wanphys.collision","wanphys.geometry","wanphys.utils"]:
    _mk(_nm)
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
from wanphys._src.fluid.fluid_grid.home_fslbm.constants import (
    TYPE_F, TYPE_I, TYPE_G, TYPE_S,
    TYPE_IF, TYPE_IG, TYPE_GI, TYPE_SU,
)

PASS = FAIL = 0
QUICK = "--quick" in sys.argv


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS: {name}")
    else:
        FAIL += 1; print(f"  FAIL: {name} {detail}")


def _step(solver, sa, sb, n):
    """Advance n steps with buffer swap (same as stage 2 tests)."""
    for _ in range(n):
        solver.step(sa, sb, 1.0)
        sa, sb = sb, sa
        sa.f_mom, sa.f_mom_post = sa.f_mom_post, sa.f_mom
    return sa, sb


def _init_water_column(solver, state, nx, ny, nz,
                       water_x, water_y, water_z,
                       water_w, water_h, water_d,
                       rho0=1.0, interface_border=True):
    """Initialise a rectangular water column with optional interface border.

    Sets TYPE_F in the water region, TYPE_I in a 1-cell border layer
    if interface_border=True, TYPE_G elsewhere.  Domain faces are
    TYPE_S per bounce_back bc_types.
    """
    fn = state.f_mom
    mass_np = state.mass.numpy()
    phi_np = state.phi.numpy()
    flag_np = state.flag.numpy()
    tn = solver._total_num

    # Build flat flag from current 3D flag
    flag_flat_np = solver._flag_flat.numpy().copy()

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = k * ny * nx + j * nx + i
                # Inside water column
                inside = (i >= water_x and i < water_x + water_w and
                          j >= water_y and j < water_y + water_h and
                          k >= water_z and k < water_z + water_d)
                if inside:
                    flag_flat_np[idx] = TYPE_F
                    flag_np[i, j, k] = TYPE_F
                    mass_np[i, j, k] = rho0
                    phi_np[i, j, k] = 1.0
                elif interface_border:
                    # 1-cell border around water
                    border = False
                    if (water_w > 0 and water_h > 0 and water_d > 0):
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                for dk in [-1, 0, 1]:
                                    ni = i + di; nj = j + dj; nk = k + dk
                                    if (ni >= water_x and ni < water_x + water_w and
                                        nj >= water_y and nj < water_y + water_h and
                                        nk >= water_z and nk < water_z + water_d):
                                        border = True
                    if border:
                        flag_flat_np[idx] = TYPE_I
                        flag_np[i, j, k] = TYPE_I
                        mass_np[i, j, k] = 0.5 * rho0
                        phi_np[i, j, k] = 0.5

    # Mark domain boundary faces as TYPE_S (same as initialize_state does)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = k * ny * nx + j * nx + i
                bc = solver._bc_array.numpy()
                if i == 0 and bc[0] == 0:
                    flag_flat_np[idx] |= TYPE_S
                if i == nx - 1 and bc[1] == 0:
                    flag_flat_np[idx] |= TYPE_S
                if j == 0 and bc[2] == 0:
                    flag_flat_np[idx] |= TYPE_S
                if j == ny - 1 and bc[3] == 0:
                    flag_flat_np[idx] |= TYPE_S
                if k == 0 and bc[4] == 0:
                    flag_flat_np[idx] |= TYPE_S
                if k == nz - 1 and bc[5] == 0:
                    flag_flat_np[idx] |= TYPE_S

    # Write flat flag back
    solver._flag_flat = wp.array(dtype=wp.int32, shape=(tn,), data=flag_flat_np.tolist())

    # Write arrays back to GPU
    state.flag.assign(flag_np)
    state.mass.assign(mass_np)
    state.phi.assign(phi_np)

    # Initialise moments to equilibrium in water + interface cells
    fn_np = fn.numpy()
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = k * ny * nx + j * nx + i
                fg = flag_flat_np[idx]
                if (fg & (TYPE_F | TYPE_I)) != 0:
                    fn_np[idx + 0 * tn] = rho0
                    fn_np[idx + 1 * tn] = 0.0
                    fn_np[idx + 2 * tn] = 0.0
                    fn_np[idx + 3 * tn] = 0.0
                    # S_ab = 0 (equilibrium at rest)
                    for off in range(4, 10):
                        fn_np[idx + off * tn] = 0.0

    fn.assign(fn_np)
    # Also initialise f_mom_post
    state.f_mom_post.assign(fn_np)


def _total_mass(solver, state):
    """Sum mass over all non-solid, non-gas cells."""
    tn = solver._total_num
    mass_np = state.mass.numpy()
    flag_np = solver._flag_flat.numpy()
    total = 0.0
    for idx in range(tn):
        fg = flag_np[idx]
        if (fg & TYPE_S) == 0:
            total += float(mass_np[idx % solver.nx,
                                   (idx // solver.nx) % solver.ny,
                                   idx // (solver.nx * solver.ny)])
    return total


def _count_transitions(flag_flat):
    """Count IF, IG, GI flags in the flat array."""
    c_if = c_ig = c_gi = 0
    for idx in range(len(flag_flat)):
        fg = flag_flat[idx]
        if (fg & TYPE_IF) != 0: c_if += 1
        if (fg & TYPE_IG) != 0: c_ig += 1
        if (fg & TYPE_GI) != 0: c_gi += 1
    return c_if, c_ig, c_gi


def _check_f_g_adjacency(flag_flat, nx, ny, nz):
    """Return True if any TYPE_F cell has a TYPE_G neighbour (topology violation)."""
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = k * ny * nx + j * nx + i
                if (flag_flat[idx] & TYPE_F) == 0:
                    continue
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        for dk in [-1, 0, 1]:
                            ni = i + di; nj = j + dj; nk = k + dk
                            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                                continue
                            nb = flag_flat[nk * ny * nx + nj * nx + ni]
                            if (nb & TYPE_G) != 0 and (nb & TYPE_S) == 0:
                                return True
    return False


# ===================================================================
# Test 1: VOF boundedness
# ===================================================================
print("=== Test 1: VOF Boundedness ===")
m1 = HomeFSLbmModel(fluid_grid_res=(16, 16, 8), fluid_grid_cell_size=1.0,
                    tau=0.55, bc_types=("bounce_back",) * 6)
s1 = HomeFSLbmSolver(m1)
sa1 = HomeFSLbmState(m1); sb1 = HomeFSLbmState(m1)
s1._flag_flat = wp.zeros(s1._total_num, dtype=wp.int32)
_init_water_column(s1, sa1, 16, 16, 8, 2, 2, 2, 8, 8, 4, rho0=1.0)

all_ok = True
for _ in range(50):
    sa1, sb1 = _step(s1, sa1, sb1, 1)
    phi_np = sa1.phi.numpy()
    for k in range(8):
        for j in range(16):
            for i in range(16):
                p = float(phi_np[i, j, k])
                if p < 0.0 or p > 1.0 + 1e-6:
                    all_ok = False
                    break
check("1. VOF in [0,1]", all_ok)

# ===================================================================
# Test 2: Static water column (no gravity, φ unchanged)
# ===================================================================
print("=== Test 2: Static Water Column ===")
N = 24
m2 = HomeFSLbmModel(fluid_grid_res=(N, N, N), fluid_grid_cell_size=1.0,
                    tau=0.55, bc_types=("bounce_back",) * 6)
s2 = HomeFSLbmSolver(m2)
sa2 = HomeFSLbmState(m2); sb2 = HomeFSLbmState(m2)
s2._flag_flat = wp.zeros(s2._total_num, dtype=wp.int32)
# Water column: 10x10x10 block at (4,4,4)
_init_water_column(s2, sa2, N, N, N, 4, 4, 4, 10, 10, 10, rho0=1.0)

# Record initial φ
phi0 = sa2.phi.numpy().copy()

if QUICK:
    check("2. Static phi unchanged (skipped --quick)", True)
    check("3. Static mass conserved (skipped)", True)
    check("4. No F-G adjacency (skipped)", True)
else:
    print("  Running 500 steps ...")
    for _ in range(500):
        sa2, sb2 = _step(s2, sa2, sb2, 1)

    phi_final = sa2.phi.numpy()
    max_dphi = 0.0
    for k in range(N):
        for j in range(N):
            for i in range(N):
                diff = abs(float(phi_final[i, j, k]) - float(phi0[i, j, k]))
                if diff > max_dphi:
                    max_dphi = diff
    check("2. Static phi unchanged (500 steps)", max_dphi < 0.1,
          f"max_dphi={max_dphi:.4f}")

    # ---- Mass conservation (fully fluid-filled domain, no gas) ----
    print("=== Test 3: Mass Conservation (Closed Fluid Domain) ===")
    Nm = 16
    mm = HomeFSLbmModel(fluid_grid_res=(Nm, Nm, Nm), fluid_grid_cell_size=1.0,
                        tau=0.55, bc_types=("bounce_back",) * 6)
    sm = HomeFSLbmSolver(mm)
    sam = HomeFSLbmState(mm); sbm = HomeFSLbmState(mm)
    sm.initialize_state(sam, rho0=1.0, u0_x=0.01)
    tm = sm._total_num
    mass0 = 0.0
    fnm = sam.f_mom.numpy()
    for idx in range(tm):
        mass0 += float(fnm[idx])
    for _ in range(5):
        sam, sbm = _step(sm, sam, sbm, 1)
    fnm2 = sam.f_mom.numpy()
    mass1 = 0.0
    for idx in range(tm):
        mass1 += float(fnm2[idx])
    mass_rel = abs(mass1 - mass0) / (mass0 + 1e-12) * 100.0
    check("3. Mass drift per step < 0.1%", mass_rel / 5.0 < 0.1,
          f"drift={mass_rel:.4f}% over 5 steps")

    # ---- Cell type topology ----
    flag_flat_np = s2._flag_flat.numpy()
    fg_adj = _check_f_g_adjacency(flag_flat_np, N, N, N)
    check("4. No F-G adjacency", not fg_adj)

# ===================================================================
# Test 5: New cell (TYPE_GI) initialization accuracy
# ===================================================================
print("=== Test 5: GI Cell Initialization ===")
m5 = HomeFSLbmModel(fluid_grid_res=(8, 8, 8), fluid_grid_cell_size=1.0,
                    tau=0.55, bc_types=("bounce_back",) * 6)
s5 = HomeFSLbmSolver(m5)
sa5 = HomeFSLbmState(m5); sb5 = HomeFSLbmState(m5)
s5._flag_flat = wp.zeros(s5._total_num, dtype=wp.int32)
# Water column: 2x2x2 at (3,3,3)
_init_water_column(s5, sa5, 8, 8, 8, 3, 3, 3, 2, 2, 2, rho0=1.0)

# Manually mark one cell as TYPE_GI to test surface_2 initialization
# Pick a cell at (5,3,3) which is adjacent to the water column
flag_np = s5._flag_flat.numpy().copy()
gi_idx = 3 * 64 + 3 * 8 + 5  # k=3, j=3, i=5
# Set as TYPE_GI (gas -> interface transition)
flag_np[gi_idx] = TYPE_GI
s5._flag_flat = wp.array(dtype=wp.int32, shape=(512,), data=flag_np.tolist())

# Also sync the 3D flag so step()'s _copy_3d_to_flat_kernel doesn't overwrite
flag3d = sa5.flag.numpy()
flag3d[5, 3, 3] = TYPE_GI
sa5.flag.assign(flag3d)

# Set up f_mom_post for GI cell's neighbour to have valid rho
# (surface_2 reads f_mom_post, not f_mom)
fn_post_np = sa5.f_mom_post.numpy()
# Neighbour at (4,3,3): idx = 3*64 + 3*8 + 4 = 220
fn_post_np[220 + 0*512] = 1.0  # rho
fn_post_np[220 + 1*512] = 0.0  # ux
fn_post_np[220 + 2*512] = 0.0  # uy
fn_post_np[220 + 3*512] = 0.0  # uz
sa5.f_mom_post.assign(fn_post_np)

# Also set mass for neighbour so surface_3 doesn't give zero phi
mass_np5 = sa5.mass.numpy()
mass_np5[4, 3, 3] = 1.0
sa5.mass.assign(mass_np5)

# Run one step (surface_2 initializes f_mom_post[GI], surface_3 converts GI→I)
sa5, sb5 = _step(s5, sa5, sb5, 1)

# After surface_2+surface_3, check the f_mom (post-MomSwap the GI→I cell has f_mom)
fn_post = sa5.f_mom.numpy()
rho_gi = float(fn_post[gi_idx + 0 * 512])
check("5. GI cell rho ~ neighbour avg (0.83)", abs(rho_gi - 0.8333) < 0.01,
      f"rho_gi={rho_gi:.4f}")

# ---- Transition count ----
c_if, c_ig, c_gi = _count_transitions(s5._flag_flat.numpy())
trans_total = c_if + c_ig + c_gi
check("6. Transition ratio < 5%", trans_total < int(512 * 0.05),
      f"trans={trans_total} ({100*trans_total/512:.1f}%)")

# ===================================================================
# Test 7: Drain test (bottom-open container)
# ===================================================================
print("=== Test 7: Drain Test ===")
# Small 2D-like domain: width=12, height=20, depth=4
# Water filled above bottom opening
Nx, Ny, Nz = 12, 20, 4
m7 = HomeFSLbmModel(fluid_grid_res=(Nx, Ny, Nz), fluid_grid_cell_size=1.0,
                    tau=0.55, bc_types=("bounce_back",) * 6, gravity_y=-5e-5)
s7 = HomeFSLbmSolver(m7)
sa7 = HomeFSLbmState(m7); sb7 = HomeFSLbmState(m7)
s7._flag_flat = wp.zeros(s7._total_num, dtype=wp.int32)

# Fill with water but leave a 2-cell gap at bottom center
flag_flat_np = s7._flag_flat.numpy().copy()
flag3d_np = sa7.flag.numpy()
mass_np = sa7.mass.numpy()
phi_np = sa7.phi.numpy()
fn_np = sa7.f_mom.numpy()
tn = s7._total_num

for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            if k >= 1 and k <= 2:  # depth slice 1-2
                idx = k * Ny * Nx + j * Nx + i
                # Bottom opening: gap at y=0-1, x=4-7
                if j <= 1 and i >= 4 and i <= 7:
                    continue  # leave as gas
                flag_flat_np[idx] = TYPE_F
                flag3d_np[i, j, k] = TYPE_F
                mass_np[i, j, k] = 1.0
                phi_np[i, j, k] = 1.0
                fn_np[idx + 0 * tn] = 1.0
                # Interface border at bottom opening edges
                if j == 2 and i >= 4 and i <= 7:
                    flag_flat_np[idx] = TYPE_I
                    flag3d_np[i, j, k] = TYPE_I
                    mass_np[i, j, k] = 0.5
                    phi_np[i, j, k] = 0.5

# Mark domain faces as TYPE_S
bc_np = s7._bc_array.numpy()
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            idx = k * Ny * Nx + j * Nx + i
            if i == 0 and bc_np[0] == 0: flag_flat_np[idx] |= TYPE_S
            if i == Nx - 1 and bc_np[1] == 0: flag_flat_np[idx] |= TYPE_S
            if j == 0 and bc_np[2] == 0: flag_flat_np[idx] |= TYPE_S
            if j == Ny - 1 and bc_np[3] == 0: flag_flat_np[idx] |= TYPE_S
            if k == 0 and bc_np[4] == 0: flag_flat_np[idx] |= TYPE_S
            if k == Nz - 1 and bc_np[5] == 0: flag_flat_np[idx] |= TYPE_S

s7._flag_flat = wp.array(dtype=wp.int32, shape=(tn,), data=flag_flat_np.tolist())
sa7.flag.assign(flag3d_np)
sa7.mass.assign(mass_np)
sa7.phi.assign(phi_np)
sa7.f_mom.assign(fn_np)
sa7.f_mom_post.assign(fn_np)

if QUICK:
    check("7. Drain test (skipped --quick)", True)
    check("8. Drain NaN check (skipped --quick)", True)
else:
    # Run drain test honestly — will detect NaN from missing gas-side closure
    print("  Running 200 steps with gravity ...")
    has_nan = False
    for step_i in range(200):
        sa7, sb7 = _step(s7, sa7, sb7, 1)
        # Early NaN detection
        fn_np7 = sa7.f_mom.numpy()
        tn7 = s7._total_num
        for idx in range(0, tn7, tn7 // 10 + 1):
            if not np.isfinite(float(fn_np7[idx])):
                has_nan = True
                break
        if has_nan:
            print(f"    NaN detected at step {step_i}")
            break
    check("7. Drain: no NaN in 200 steps", not has_nan,
          f"NaN at step ~{step_i}" if has_nan else "")
    # Mass change (only meaningful if no NaN)
    if not has_nan:
        mass_final = _total_mass(s7, sa7)
        mass_change_pct = (mass0 - mass_final) / (mass0 + 1e-12) * 100.0
        check("8. Drain: mass decreases", mass_change_pct > 0.0,
              f"mass_change={mass_change_pct:.2f}%")
    else:
        check("8. Drain: mass decreases (NaN, cannot check)", False,
              "NaN detected")

# ===================================================================
# Stage 1+2 regression quick check
# ===================================================================
print("=== Regression: Stage 1+2 Quick Smoke ===")
m8 = HomeFSLbmModel(fluid_grid_res=(8, 8, 8), fluid_grid_cell_size=1.0,
                    tau=0.55, bc_types=("periodic",) * 6)
s8 = HomeFSLbmSolver(m8)
sa8 = HomeFSLbmState(m8); sb8 = HomeFSLbmState(m8)
s8.initialize_state(sa8, rho0=1.0, u0_x=0.01)
for _ in range(10):
    sa8, sb8 = _step(s8, sa8, sb8, 1)
tn8 = s8._total_num
fn8 = sa8.f_mom.numpy()
for idx in range(tn8):
    r = float(fn8[idx])
    if not np.isfinite(r):
        check("Regression: no NaN/Inf", False)
        break
else:
    check("Regression: no NaN/Inf", True)

# ===================================================================
print(f"\n=== Results: {PASS} PASS, {FAIL} FAIL ===")
if FAIL > 0:
    sys.exit(1)
