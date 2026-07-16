"""Execution-matrix tests for encoding providers and collision collectors."""

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel


class TestLbmStreamingMatrix(unittest.TestCase):
    def test_all_six_concrete_combinations_keep_uniform_equilibrium(self) -> None:
        for encoding in ("fullf", "home"):
            for collision in ("srt", "trt", "home_nocm"):
                with self.subTest(encoding=encoding, collision=collision):
                    model = LbmModel(
                        fluid_grid_res=(3, 3, 3),
                        device="cpu",
                        encoding=encoding,
                        collision=collision,
                        lambda_trt=0.001 if collision == "trt" else 0.0,
                        bc_periodic=(True, True, True),
                    )
                    domain = LbmDomain(model)
                    state = domain.create_state()
                    domain.solver.initialize_equilibrium(state, 1.0, (0.02, -0.01, 0.005))
                    domain.step(1.0)
                    rho = domain.state.density.numpy()
                    ux = domain.state.velocity_x.numpy()
                    self.assertTrue(np.all(np.isfinite(rho)))
                    np.testing.assert_allclose(rho, 1.0, atol=2.0e-6)
                    np.testing.assert_allclose(ux, 0.02, atol=2.0e-6)

    def test_static_halfway_bounce_back_preserves_total_mass(self) -> None:
        model = LbmModel(
            fluid_grid_res=(4, 4, 4), device="cpu", collision="srt"
        )
        domain = LbmDomain(model)
        state = domain.create_state()
        domain.solver.initialize_equilibrium(state, 1.0, (0.03, 0.0, 0.0))
        mass0 = float(np.sum(state.f.numpy(), dtype=np.float64))
        domain.step(1.0)
        mass1 = float(np.sum(domain.state.f.numpy(), dtype=np.float64))
        self.assertAlmostEqual(mass0, mass1, places=5)


if __name__ == "__main__":
    unittest.main()
