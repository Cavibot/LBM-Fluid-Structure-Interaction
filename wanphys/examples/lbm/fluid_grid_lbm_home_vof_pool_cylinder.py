# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless HOME-FREE VOF: still pool + static cylinder obstacle (P1 smoke).

Demonstrates generic IC (``seed_pool``), open-top BC, cylinder→``solid_phi``,
and ``LbmFeedbackMode.NONE`` (no fluid→rigid force).

Run:
    uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_home_vof_pool_cylinder \\
        --n 32 --steps 40
"""

from __future__ import annotations

import argparse
import time

import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling, LbmFeedbackMode
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    CELL_INTERFACE,
    CELL_LIQUID,
    HomeDomainBC,
    apply_home_domain_bc_to_model,
    make_home_vof_model,
)
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig


def main() -> None:
    p = argparse.ArgumentParser(description="HOME-VOF pool + static cylinder")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--fill-z-frac", type=float, default=0.45)
    args = p.parse_args()

    n = int(args.n)
    dh = 0.02
    fill_z = max(2, int(n * float(args.fill_z_frac)))
    model = make_home_vof_model(
        fluid_grid_res=(n, n, n),
        fluid_grid_cell_size=dh,
        tau=0.7,
        gravity_z=-0.0004,
        vof_gamma=0.0,
    )
    apply_home_domain_bc_to_model(model, HomeDomainBC.open_top())
    domain = LbmDomain(model)
    domain.create_state()
    home = domain.solver._home_fp32
    assert home is not None
    home.refresh_domain_bc()
    home.seed_pool(domain.state, fill_z=fill_z)
    home.sync_to_state(domain._state_out)

    world = n * dh
    radius = 0.12
    half_h = 0.2
    builder = RigidModelBuilder(gravity=0.0)
    cfg = ShapeConfig(density=0.0, is_visible=False, is_solid=True, has_shape_collision=False)
    # Static cylinder standing in the pool (local Z up).
    body = builder.add_body(
        position=(world * 0.5, world * 0.5, half_h + dh),
        label="static_cyl",
    )
    builder.add_shape_capsule(body, radius=radius, half_height=half_h, cfg=cfg)
    rigid = RigidDomain(builder.finalize(device=model._device))
    rigid.create_state()

    coupling = GridLbmRigidCoupling(domain, rigid)
    coupling.add_body_cylinder(body, radius=radius, half_height=half_h)
    coupling.set_rigid_dynamics_enabled(False)
    coupling.set_two_way_feedback_enabled(False)
    coupling.set_feedback_mode(LbmFeedbackMode.NONE)

    ctype0 = domain.state.cell_type.numpy()
    print(
        f"pool+cylinder: {n}^3 fill_z={fill_z} L={int((ctype0 == CELL_LIQUID).sum())} "
        f"I={int((ctype0 == CELL_INTERFACE).sum())} feedback={coupling.feedback_mode}"
    )

    t0 = time.perf_counter()
    for step in range(int(args.steps)):
        coupling.step(1.0)
        if step % 10 == 0 or step == args.steps - 1:
            wp.synchronize_device(model._device)
            solid = int((domain.state.solid_phi.numpy() < 0.0).sum())
            phi = float(domain.state.phi.numpy().sum())
            print(f"step={step:3d}  φ_sum={phi:.1f}  solid_cells={solid}")
    print(f"done in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
