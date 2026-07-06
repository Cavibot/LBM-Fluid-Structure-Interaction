"""Diagnostic: check effective body force from apply_guo_force path."""
from __future__ import annotations
import numpy as np
import warp as wp
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel


def main() -> None:
    wp.init()
    wp.set_device("cpu")

    g = -0.001
    tau = 0.55
    omega = 1.0 / tau
    nx = ny = nz = 16

    model = LbmModel(fluid_grid_res=(nx, ny, nz), tau=tau,
                     gravity_x=0.0, gravity_y=0.0, gravity_z=g)
    domain = LbmDomain(model)
    domain.create_state()
    domain.solver.initialize_equilibrium(domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))

    # single step (interior cells unaffected by walls in 1 step)
    domain.step(dt=1.0)

    uz = domain.state.velocity_z.numpy()
    # interior 6x6x6
    interior = uz[5:-5, 5:-5, 5:-5]
    print(f"tau={tau} omega={omega:.4f} g={g}")
    print(f"(1-omega/2)        = {1-omega/2:.4f}")
    print(f"(1-omega/2)(1-omega)= {(1-omega/2)*(1-omega):.4f}")
    print(f"interior uz mean = {interior.mean():.6e}  (expected correct ~ g = {g})")
    print(f"interior uz max  = {interior.max():.6e}  min = {interior.min():.6e}")

    # Run 20 steps, watch sign of interior uz
    for _ in range(19):
        domain.step(dt=1.0)
    uz20 = domain.state.velocity_z.numpy()
    interior20 = uz20[5:-5, 5:-5, 5:-5]
    print(f"after 20 steps: interior uz mean = {interior20.mean():.6e} (sign of g = {g})")


if __name__ == "__main__":
    main()
