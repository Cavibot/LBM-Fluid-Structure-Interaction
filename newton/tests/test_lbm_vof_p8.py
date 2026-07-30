# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""P8 authoritative VOF dam-break visualization and CLI acceptance tests."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import DebugVofView, VofCellType
from wanphys._src.fluid.fluid_grid.lbm.vof.initialization import (
    prepare_initial_vof,
)
from wanphys.examples.lbm import fluid_grid_lbm_vof_dambreak as dambreak

_SMALL_SHAPE = (8, 3, 6)


def _config(**overrides: object) -> dambreak.AuthoritativeVofDamBreakConfig:
    values: dict[str, object] = {
        "grid_res": _SMALL_SHAPE,
        "gravity_z": -2.0e-5,
    }
    values.update(overrides)
    return dambreak.AuthoritativeVofDamBreakConfig(**values)


class TestVofP8DamBreakState(unittest.TestCase):
    def test_initializer_builds_legal_d3q19_interface_layer(self) -> None:
        phi = dambreak.build_authoritative_dam_break_phi(_SMALL_SHAPE)
        canonical, cell_type = prepare_initial_vof(
            phi,
            shape=_SMALL_SHAPE,
            periodic=(False, False, False),
        )

        np.testing.assert_array_equal(canonical, phi)
        self.assertGreater(
            np.count_nonzero(cell_type == int(VofCellType.GAS)),
            0,
        )
        self.assertGreater(
            np.count_nonzero(cell_type == int(VofCellType.INTERFACE)),
            0,
        )
        self.assertGreater(
            np.count_nonzero(cell_type == int(VofCellType.LIQUID)),
            0,
        )

    def test_scene_uses_authoritative_vof_without_solids_or_sc_mock(self) -> None:
        scene = dambreak.AuthoritativeVofDamBreakScene(_config())
        state = scene.domain.state

        self.assertEqual(scene.model.interface_model, "vof")
        self.assertIsNotNone(state.vof)
        self.assertIsNone(state.debug_mock_sc_to_vof)
        self.assertTrue(np.all(state.solid_phi.numpy() >= 0.0))

    def test_authoritative_adapter_is_zero_copy_and_rejects_stale_geometry(
        self,
    ) -> None:
        scene = dambreak.AuthoritativeVofDamBreakScene(_config())
        vof = scene.domain.state.vof
        assert vof is not None
        phi_before = vof.phi.numpy().copy()
        normal_before = vof.normal.numpy().copy()

        view = DebugVofView.from_authoritative_vof(vof)

        self.assertIs(view.phi, vof.phi)
        self.assertIs(view.cell_type, vof.cell_type)
        self.assertIs(view.normal, vof.normal)
        self.assertEqual(view.epoch, vof.epoch)
        self.assertEqual(view.source, "authoritative_vof")
        np.testing.assert_array_equal(vof.phi.numpy(), phi_before)
        np.testing.assert_array_equal(vof.normal.numpy(), normal_before)

        vof.geometry_epoch -= 1
        with self.assertRaisesRegex(ValueError, "current geometry"):
            DebugVofView.from_authoritative_vof(vof)

    def test_headless_fullf_smoke_is_bounded_and_valid(self) -> None:
        ledger = dambreak.run_headless(_config(encoding="fullf"), num_steps=3)

        self.assertEqual(len(ledger), 3)
        self.assertEqual([entry.epoch for entry in ledger], [1, 2, 3])
        self.assertTrue(
            all(entry.geometry_epoch == entry.epoch for entry in ledger)
        )
        self.assertLessEqual(ledger[-1].relative_mass_error, 5.0e-6)
        self.assertEqual(
            ledger[-1].invalid_liquid_gas_adjacency_count,
            0,
        )

    def test_headless_home_smoke_is_bounded_and_valid(self) -> None:
        ledger = dambreak.run_headless(_config(encoding="home"), num_steps=3)

        self.assertEqual(len(ledger), 3)
        self.assertEqual(ledger[-1].epoch, 3)
        self.assertEqual(ledger[-1].non_finite_count, 0)
        self.assertEqual(
            ledger[-1].nonpositive_active_density_count,
            0,
        )

    def test_csv_rows_and_final_mass_match_ledger(self) -> None:
        ledger = dambreak.run_headless(_config(), num_steps=2)
        with tempfile.TemporaryDirectory() as directory:
            path = dambreak.write_diagnostics_csv(
                Path(directory) / "vof-dambreak.csv",
                ledger,
            )
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), len(ledger))
        self.assertEqual(int(rows[-1]["step"]), 2)
        self.assertEqual(int(rows[-1]["epoch"]), ledger[-1].epoch)
        self.assertAlmostEqual(
            float(rows[-1]["total_mass"]),
            ledger[-1].total_mass,
        )

    def test_parser_defaults_and_critical_overrides(self) -> None:
        parser = dambreak.create_parser()
        defaults = parser.parse_args([])
        explicit = parser.parse_args(
            [
                "--viewer",
                "null",
                "--num-frames",
                "7",
                "--encoding",
                "home",
                "--grid-res",
                "9",
                "4",
                "7",
                "--surface-tension",
                "0.01",
                "--show-normals",
            ]
        )

        self.assertEqual(tuple(defaults.grid_res), (48, 12, 32))
        self.assertEqual(defaults.encoding, "fullf")
        self.assertEqual(defaults.runtime_profile, "auto")
        self.assertIsNone(defaults.steps_per_frame)
        self.assertEqual(defaults.validation_interval, 60)
        self.assertEqual(explicit.viewer, "null")
        self.assertEqual(explicit.num_frames, 7)
        self.assertEqual(explicit.encoding, "home")
        self.assertEqual(tuple(explicit.grid_res), (9, 4, 7))
        self.assertAlmostEqual(explicit.surface_tension, 0.01)
        self.assertTrue(explicit.show_normals)

        headless = dambreak._config_from_args(defaults, interactive=False)
        interactive = dambreak._config_from_args(defaults, interactive=True)
        self.assertEqual(headless.runtime_profile, "strict")
        self.assertEqual(headless.steps_per_frame, 1)
        self.assertEqual(interactive.runtime_profile, "device")
        self.assertEqual(interactive.steps_per_frame, 4)

    def test_cuda_request_has_availability_guard(self) -> None:
        config = _config(device="cuda:0")
        if wp.is_cuda_available():
            scene = dambreak.AuthoritativeVofDamBreakScene(config)
            self.assertEqual(str(scene.model._device), "cuda:0")
        else:
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                dambreak.AuthoritativeVofDamBreakScene(config)


class TestVofP8VisualSource(unittest.TestCase):
    def test_render_reads_authoritative_phi_and_geometry(self) -> None:
        class FakeViewer:
            def __init__(self) -> None:
                self.begin_count = 0
                self.end_count = 0
                self.post_render = None

            def register_post_render_callback(self, callback: object) -> None:
                self.post_render = callback

            def begin_frame(self, time: float) -> None:
                del time
                self.begin_count += 1

            def end_frame(self) -> None:
                self.end_count += 1

        class FakeScreenSpaceFluidRenderer:
            def __init__(self, **kwargs: object) -> None:
                self.available = True
                self.constructor = kwargs
                self.density = None

            def set_density_field(self, **kwargs: object) -> None:
                self.density = kwargs["density"]

            def render(self, viewer: object) -> None:
                del viewer

        class FakeInterfaceVisualizer:
            def __init__(self, *args: object) -> None:
                self.constructor = args
                self.view = None
                self.solid_phi = None

            def render(
                self,
                viewer: object,
                view: DebugVofView,
                solid_phi: object,
                *,
                show_normals: bool,
            ) -> int:
                del viewer, show_normals
                self.view = view
                self.solid_phi = solid_phi
                return 1

        viewer = FakeViewer()
        module = "wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak"
        with (
            patch(
                f"{module}.ScreenSpaceFluidRenderer",
                FakeScreenSpaceFluidRenderer,
            ),
            patch(
                f"{module}.VofInterfaceVisualizer",
                FakeInterfaceVisualizer,
            ),
        ):
            example = dambreak.AuthoritativeVofDamBreakVisualExample(
                viewer,
                _config(),
            )
            example.render()

        state = example.scene.domain.state
        assert state.vof is not None
        self.assertIs(example.ssfr.density, state.vof.phi)
        self.assertIs(example.interface_visualizer.view.phi, state.vof.phi)
        self.assertEqual(
            example.interface_visualizer.view.source,
            "authoritative_vof",
        )
        self.assertIs(example.interface_visualizer.solid_phi, state.solid_phi)
        self.assertEqual(viewer.begin_count, 1)
        self.assertEqual(viewer.end_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
