"""Host-side tests for Part 0 naming, capability and boundary contracts."""

from __future__ import annotations

import unittest

from wanphys._src.fluid.fluid_grid.lbm.boundaries import resolve_boundary_faces
from wanphys._src.fluid.fluid_grid.lbm.constants import (
    BC_BOUNCE_BACK,
    BC_OUTFLOW,
    BC_VELOCITY_INLET,
)
from wanphys._src.fluid.fluid_grid.lbm.contracts import (
    BoundaryModel,
    CapabilityStatus,
    Collision,
    CollisionSpace,
    Encoding,
    ForceModel,
    InterfaceModel,
    boundary_capability_status,
    capability_status,
    collision_contract,
    normalize_collision,
    normalize_interface_model,
    resolve_force_model,
)
from wanphys._src.fluid.fluid_grid.lbm.model import LbmModel, normalize_boundary_type


class TestPart0Contracts(unittest.TestCase):
    def test_deprecated_home_nocm_alias_resolves_to_canonical_collision(self) -> None:
        collision, deprecated = normalize_collision("home_nocm")
        self.assertIs(collision, Collision.NOCM_MRT)
        self.assertTrue(deprecated)

    def test_collision_contract_keeps_encoding_and_collision_separate(self) -> None:
        fullf = collision_contract(Encoding.FULLF, Collision.NOCM_MRT)
        home = collision_contract(Encoding.HOME, Collision.NOCM_MRT)
        self.assertIs(fullf.space, CollisionSpace.NOCM_MOMENT)
        self.assertIs(home.space, CollisionSpace.HOME_ENCODED_MOMENT)
        self.assertNotEqual(fullf.native_output, home.native_output)

    def test_home_raw_mrt_is_an_explicit_fail_fast_combination(self) -> None:
        self.assertIs(
            capability_status(Encoding.HOME, Collision.RAW_MRT, ForceModel.NONE),
            CapabilityStatus.FAIL_FAST,
        )
        with self.assertRaises(NotImplementedError):
            collision_contract(Encoding.HOME, Collision.RAW_MRT)

    def test_force_model_is_derived_from_interface_and_gravity(self) -> None:
        self.assertIs(
            resolve_force_model(InterfaceModel.OFF, (0.0, -1.0e-5, 0.0)),
            ForceModel.GRAVITY,
        )
        self.assertIs(
            resolve_force_model("shan_chen", (0.0, 0.0, 0.0)),
            ForceModel.SHAN_CHEN,
        )
        self.assertIs(
            resolve_force_model("shan_chen", (0.0, -1.0e-5, 0.0)),
            ForceModel.GRAVITY_SHAN_CHEN,
        )
        self.assertIs(normalize_interface_model("off"), InterfaceModel.OFF)
        with self.assertRaisesRegex(ValueError, "Unknown LBM interface_model"):
            normalize_interface_model("legacy")

    def test_open_face_edges_and_corners_fall_back_as_complete_cells(self) -> None:
        resolution = resolve_boundary_faces(
            (
                BC_VELOCITY_INLET,
                BC_OUTFLOW,
                BC_BOUNCE_BACK,
                BC_BOUNCE_BACK,
                BC_BOUNCE_BACK,
                BC_BOUNCE_BACK,
            ),
            (8, 6, 5),
        )
        self.assertTrue(resolution.has_open_boundaries)
        self.assertIsNotNone(resolution.conflict)
        assert resolution.conflict is not None
        # Each x-face has 2*ny + 2*(nz-2) edge/corner cells.
        self.assertEqual(resolution.conflict.cell_count, 2 * (2 * 6 + 2 * (5 - 2)))
        self.assertIn("static bounce-back", resolution.conflict.warning_message)

    def test_string_boundary_axis_normalizes_without_leaking_into_collision(self) -> None:
        self.assertEqual(normalize_boundary_type("zou_he")[0], BC_VELOCITY_INLET)
        self.assertEqual(normalize_boundary_type("convective")[0], BC_OUTFLOW)
        with self.assertRaises(ValueError):
            normalize_boundary_type("surface")
        model = LbmModel(
            fluid_grid_res=(4, 4, 4),
            device="cpu",
            boundary_models=("zou_he", "convective", "bounce_back", "bounce_back", "bounce_back", "bounce_back"),
        )
        self.assertEqual(model.bc_types[:2], (BC_VELOCITY_INLET, BC_OUTFLOW))
        self.assertEqual(model.boundary_models[:2], ("zou_he", "convective"))

    def test_surface_is_not_a_six_face_boundary(self) -> None:
        self.assertIs(
            boundary_capability_status(BoundaryModel.SURFACE),
            CapabilityStatus.PLANNED,
        )
        with self.assertRaises(ValueError):
            LbmModel(
                fluid_grid_res=(4, 4, 4),
                device="cpu",
                bc_types=("surface", "bounce_back", "bounce_back", "bounce_back", "bounce_back", "bounce_back"),
            )


if __name__ == "__main__":
    unittest.main()
