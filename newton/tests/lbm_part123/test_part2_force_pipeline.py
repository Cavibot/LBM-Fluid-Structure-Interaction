"""Part 2 acceptance tests for unified force density and injection."""

from __future__ import annotations

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm import FullFLbmState, HomeLbmState, LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.constants import W


class TestPart2ForcePipeline(unittest.TestCase):
    def test_gravity_is_force_density_and_half_step_velocity_for_every_operator(self) -> None:
        gravity = 2.0e-5
        paths = (
            ("fullf", "srt"),
            ("fullf", "trt"),
            ("fullf", "raw_mrt"),
            ("fullf", "nocm_mrt"),
            ("home", "srt"),
            ("home", "trt"),
            ("home", "nocm_mrt"),
        )
        for encoding, collision in paths:
            with self.subTest(encoding=encoding, collision=collision):
                model = LbmModel(
                    fluid_grid_res=(2, 2, 2),
                    device="cpu",
                    encoding=encoding,
                    collision=collision,
                    gravity_x=gravity,
                    tau=0.8,
                    bc_periodic=(True, True, True),
                )
                domain = LbmDomain(model)
                state = domain.create_state()
                domain.solver.initialize_equilibrium(state, rho0=1.0)
                domain.step(1.0)

                np.testing.assert_allclose(
                    domain.state.force_x.numpy(), gravity, atol=2.0e-8, rtol=2.0e-6
                )
                np.testing.assert_allclose(
                    domain.state.velocity_x.numpy(),
                    0.5 * gravity,
                    atol=2.0e-8,
                    rtol=2.0e-6,
                )

                if isinstance(domain.state, HomeLbmState):
                    total_post_momentum = float(
                        np.sum(domain.state.rho_u_x.numpy(), dtype=np.float64)
                    )
                else:
                    f = domain.state.f_post.numpy().reshape(19, -1)
                    total_post_momentum = float(
                        np.sum(
                            f[1] - f[2] + f[7] - f[8] + f[9] - f[10]
                            + f[11] - f[12] + f[13] - f[14],
                            dtype=np.float64,
                        )
                    )
                # Full MRT paths reconstruct O(1) populations through a
                # float32 inverse moment transform before their O(1e-4)
                # momentum is differenced.  Allow the corresponding
                # cancellation floor while still resolving the injected F.
                np.testing.assert_allclose(
                    total_post_momentum,
                    8.0 * gravity,
                    rtol=1.0e-3,
                    atol=2.0e-7,
                )

    def test_explicit_none_rejects_nonzero_legacy_force_fields(self) -> None:
        with self.assertRaises(ValueError):
            LbmModel(
                fluid_grid_res=(2, 2, 2),
                device="cpu",
                force_model="none",
                gravity_y=-1.0e-5,
            )

    def test_shan_chen_force_computation_feeds_every_declared_collision_space(self) -> None:
        paths = (
            ("fullf", "srt"),
            ("fullf", "trt"),
            ("fullf", "raw_mrt"),
            ("fullf", "nocm_mrt"),
            ("home", "srt"),
            ("home", "trt"),
            ("home", "nocm_mrt"),
        )
        density = np.ones((4, 4, 4), dtype=np.float32)
        density[2, 2, 2] = 1.1
        for encoding, collision in paths:
            with self.subTest(encoding=encoding, collision=collision):
                domain = LbmDomain(
                    LbmModel(
                        fluid_grid_res=(4, 4, 4),
                        device="cpu",
                        encoding=encoding,
                        collision=collision,
                        G=-0.2,
                        sc_force_stride=1,
                        tau=0.8,
                        bc_periodic=(True, True, True),
                    )
                )
                state = domain.create_state()
                if isinstance(state, FullFLbmState):
                    populations = np.stack(
                        [float(weight) * density for weight in W], axis=0
                    ).astype(np.float32)
                    state.f_post.assign(populations.reshape(-1))
                else:
                    assert isinstance(state, HomeLbmState)
                    state.rho.assign(density)
                    for field in state.kinetic_fields[1:]:
                        field.zero_()
                state.density.assign(density)
                domain.step(1.0)
                force = np.stack(
                    (
                        domain.state.force_x.numpy(),
                        domain.state.force_y.numpy(),
                        domain.state.force_z.numpy(),
                    ),
                    axis=0,
                )
                self.assertTrue(np.all(np.isfinite(force)))
                self.assertGreater(float(np.max(np.abs(force))), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
