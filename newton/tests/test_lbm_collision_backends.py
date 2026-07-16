"""EPC collision regression tests."""

import unittest

import numpy as np

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel


class TestLbmCollisionBackends(unittest.TestCase):
    def _one_step(self, collision: str, lambda_trt: float) -> np.ndarray:
        model = LbmModel(
            fluid_grid_res=(3, 3, 3), device="cpu", collision=collision,
            lambda_trt=lambda_trt, bc_periodic=(True, True, True), tau=0.8,
        )
        domain = LbmDomain(model)
        state = domain.create_state()
        domain.solver.initialize_equilibrium(state, 1.0, (0.01, 0.02, -0.01))
        domain.step(1.0)
        return domain.state.f.numpy()

    def test_srt_equilibrium_is_fixed_point(self) -> None:
        post = self._one_step("srt", 0.0)
        self.assertTrue(np.all(np.isfinite(post)))
        self.assertAlmostEqual(float(np.sum(post, dtype=np.float64)), 27.0, places=5)

    def test_trt_reduces_to_srt_when_rates_are_equal(self) -> None:
        srt = self._one_step("srt", 0.0)
        # lambda=(tau-0.5)^2 gives tau_plus=tau_minus=tau.
        trt = self._one_step("trt", (0.8 - 0.5) ** 2)
        np.testing.assert_allclose(trt, srt, atol=2.0e-7, rtol=2.0e-7)


if __name__ == "__main__":
    unittest.main()
