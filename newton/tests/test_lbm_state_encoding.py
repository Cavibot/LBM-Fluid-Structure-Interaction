"""State/model contract tests for the FullF/HOME architecture."""

import unittest
from types import SimpleNamespace

from wanphys._src.fluid.fluid_grid.coupling.grid_lbm_rigid_coupling import GridLbmRigidCoupling
from wanphys._src.fluid.fluid_grid.lbm import (
    FullFLbmState,
    HomeLbmState,
    LbmDomain,
    LbmModel,
    LbmState,
)


class TestLbmStateEncoding(unittest.TestCase):
    def test_default_state_is_fullf_with_compatibility_alias(self) -> None:
        domain = LbmDomain(LbmModel(fluid_grid_res=(2, 3, 4), device="cpu"))
        state = domain.create_state()
        self.assertIsInstance(state, FullFLbmState)
        self.assertIsInstance(state, LbmState)
        self.assertIs(state.f, state.f_post)
        self.assertEqual(state.f_post.size, 19 * 2 * 3 * 4)

    def test_home_state_has_ten_persistent_fields_and_no_fullf(self) -> None:
        domain = LbmDomain(LbmModel(fluid_grid_res=(2, 3, 4), device="cpu", encoding="home"))
        state = domain.create_state()
        self.assertIsInstance(state, HomeLbmState)
        self.assertEqual(len(state.kinetic_fields), 10)
        self.assertFalse(hasattr(state, "f"))
        self.assertFalse(hasattr(state, "f_post"))

    def test_domain_double_buffers_use_same_concrete_type(self) -> None:
        domain = LbmDomain(LbmModel(fluid_grid_res=(2, 2, 2), device="cpu", encoding="home"))
        domain.create_state()
        self.assertIsInstance(domain._state_in, HomeLbmState)
        self.assertIsInstance(domain._state_out, HomeLbmState)

    def test_collision_resolution_and_reserved_backends(self) -> None:
        self.assertEqual(
            LbmModel(fluid_grid_res=(2, 2, 2), device="cpu").resolved_collision,
            "srt",
        )
        self.assertEqual(
            LbmModel(fluid_grid_res=(2, 2, 2), device="cpu", lambda_trt=0.001).resolved_collision,
            "trt",
        )
        for name in ("raw_mrt", "nocm_mrt"):
            with self.assertRaises(NotImplementedError):
                LbmModel(fluid_grid_res=(2, 2, 2), device="cpu", collision=name)

    def test_invalid_combinations_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            LbmModel(fluid_grid_res=(2, 2, 2), device="cpu", encoding="unknown")
        with self.assertRaises(ValueError):
            LbmModel(fluid_grid_res=(2, 2, 2), device="cpu", collision="unknown")
        with self.assertRaises(NotImplementedError):
            LbmModel(fluid_grid_res=(2, 2, 2), device="cpu", encoding="home", G=-1.0)
        with self.assertRaises(NotImplementedError):
            LbmModel(
                fluid_grid_res=(2, 2, 2),
                device="cpu",
                encoding="fullf",
                collision="home_nocm",
                G=-1.0,
            )
        with self.assertRaises(ValueError):
            LbmModel(
                fluid_grid_res=(2, 2, 2),
                device="cpu",
                collision="srt",
                use_regularization=True,
            )

    def test_home_rigid_coupling_fails_at_construction(self) -> None:
        model = LbmModel(fluid_grid_res=(2, 2, 2), device="cpu", encoding="home")
        fake_fluid_domain = SimpleNamespace(model=model)
        with self.assertRaises(NotImplementedError):
            GridLbmRigidCoupling(fake_fluid_domain, SimpleNamespace())

    def test_home_nocm_does_not_allocate_population_scratch(self) -> None:
        domain = LbmDomain(
            LbmModel(
                fluid_grid_res=(2, 2, 2),
                device="cpu",
                encoding="home",
                collision="home_nocm",
            )
        )
        self.assertIsNone(domain.solver._f_star)
        self.assertIsNone(domain.solver._f_post)

    def test_runtime_boundary_update_rejects_unmigrated_types(self) -> None:
        domain = LbmDomain(LbmModel(fluid_grid_res=(2, 2, 2), device="cpu"))
        with self.assertRaises(NotImplementedError):
            domain.solver.set_boundary_condition(0, 1)


if __name__ == "__main__":
    unittest.main()
