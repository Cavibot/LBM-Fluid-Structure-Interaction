#!/usr/bin/env python3
"""Reproduce bubble_coupling_volume_delta_phi evolution (golden parity check)."""
from __future__ import annotations

import numpy as np
import warp as wp

wp.init()

# Direct imports (avoid wanphys/__init__.py heavy dependency chain when possible)
from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel
from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver
from wanphys._src.fluid.fluid_grid.home_fslbm.tests.test_regression_bubble import (
    SCENES,
    build_fluid_bubble_scene,
    load_bubble_golden,
    reorder_ref_scalar_to_warp,
    _grid_shape,
)


def main() -> None:
    scene = "bubble_coupling_volume_delta_phi"
    meta = SCENES[scene]
    golden = load_bubble_golden(scene)
    nx, ny, nz = _grid_shape(golden, meta["N"])
    steps = int(golden.get("steps", meta.get("default_steps", 1)))

    model = HomeFslbmModel(
        fluid_grid_res=(nx, ny, nz),
        fluid_grid_cell_size=1.0,
        max_bubbles=256,
        gravity_z=0.0,
        omega=1.0 / (3.0 * 1e-4 + 0.5),
    )
    solver = HomeFslbmSolver(model)
    domain = HomeFslbmDomain(model, solver=solver)
    domain.create_state()
    state = domain.state

    flag, phi = build_fluid_bubble_scene(nx, meta["builder"], meta["builder_kwargs"])
    wp.copy(state.flag, wp.array(flag, dtype=wp.uint8))
    wp.copy(state.phi, wp.array(phi, dtype=float))
    solver.initialize_equilibrium(state)
    solver.init_bubbles(state)

    g_dphi = reorder_ref_scalar_to_warp(golden["delta_phi"], nx, ny, nz)
    wp.copy(state.delta_phi, wp.array(g_dphi.astype(np.float32), dtype=float))

    print(f"Running {scene} for {steps} steps...")
    for _ in range(steps):
        domain.step(1.0)

    wp.synchronize()
    state = domain.state
    vol = float(state.bubble_volume.numpy()[0])
    rho = float(state.bubble_rho.numpy()[0])
    golden_vol = float(np.asarray(golden["bubble_volume"]).ravel()[0])
    print(f"final bubble_volume={vol} (golden={golden_vol}) bubble_rho={rho}")


if __name__ == "__main__":
    main()
