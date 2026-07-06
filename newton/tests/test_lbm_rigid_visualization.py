# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused smoke tests for WanPhys LBM rigid visualization examples."""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import types
import unittest
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

import newton.viewer

VISUAL_MODULE_NAME: str = "wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi_visual"


def _import_visual_module() -> types.ModuleType:
    try:
        return importlib.import_module(VISUAL_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name == VISUAL_MODULE_NAME:
            raise unittest.SkipTest(f"{VISUAL_MODULE_NAME} is not available yet") from exc
        raise


def _call_with_supported_kwargs(entry: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    signature: inspect.Signature = inspect.signature(entry)
    accepts_var_kwargs: bool = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return entry(**kwargs)

    supported_kwargs: dict[str, Any] = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return entry(**supported_kwargs)


def _run_smoke_helper(module: types.ModuleType) -> Any:
    smoke_kwargs: dict[str, Any] = {
        "steps": 2,
        "num_frames": 2,
        "viewer": "null",
        "viewer_mode": "null",
        "device": None,
        "force_scale": 1.0,
        "advance_rigid": False,
        "headless": True,
        "quiet": True,
    }
    for helper_name in ("run_smoke", "run_null_smoke", "smoke_test"):
        helper: Any = getattr(module, helper_name, None)
        if callable(helper):
            return _call_with_supported_kwargs(helper, smoke_kwargs)

    example_type: Any = None
    for class_name in ("TwowayFsiVisualExample", "LbmTwoWayFsiVisualExample"):
        example_type = getattr(module, class_name, None)
        if example_type is not None:
            break
    if example_type is None:
        raise AssertionError(
            f"{VISUAL_MODULE_NAME} should expose run_smoke(...) or a two-way visual example class for null smoke testing"
        )

    config_type: Any = getattr(module, "LbmTwoWaySceneConfig", None)
    config: Any = _call_with_supported_kwargs(config_type, smoke_kwargs) if callable(config_type) else None
    viewer: Any = newton.viewer.ViewerNull(num_frames=2)
    args: argparse.Namespace = argparse.Namespace(
        viewer="null",
        num_frames=2,
        output_path=None,
        device=None,
        force_scale=1.0,
        advance_rigid=False,
        headless=True,
        quiet=True,
    )
    example: Any = _call_with_supported_kwargs(example_type, {"viewer": viewer, "args": args, "config": config})
    for _step_index in range(2):
        example.step()

    return _read_example_telemetry(example)


def _read_example_telemetry(example: Any) -> Any:
    for accessor_name in ("smoke_metrics", "get_smoke_metrics", "collect_smoke_metrics"):
        accessor: Any = getattr(example, accessor_name, None)
        if callable(accessor):
            return accessor()

    telemetry: Any = getattr(example, "telemetry", None)
    if callable(telemetry):
        return telemetry()
    if telemetry is not None:
        return telemetry

    metrics: Any = getattr(example, "metrics", None)
    if metrics is not None:
        return metrics

    last_telemetry: Any = getattr(example, "last_telemetry", None)
    if last_telemetry is not None:
        return last_telemetry

    raise AssertionError(
        "TwowayFsiVisualExample should expose smoke_metrics(), get_smoke_metrics(), "
        "collect_smoke_metrics(), telemetry, metrics, or last_telemetry"
    )


def _read_metric(telemetry: Any, name: str, tuple_index: int) -> float:
    if isinstance(telemetry, Mapping):
        value: Any = telemetry[name]
        return float(value)

    if hasattr(telemetry, name):
        return float(getattr(telemetry, name))

    if isinstance(telemetry, tuple | list):
        return float(telemetry[tuple_index])

    raise AssertionError(
        f"smoke telemetry should provide {name!r} as a mapping key, object attribute, or tuple/list slot"
    )


def _make_dambreak_mem_metrics(metrics_module: types.ModuleType, **overrides: Any) -> Any:
    baseline: dict[str, Any] = {
        "frame": 12,
        "sim_time": 0.2,
        "body_position": np.asarray([0.5, 0.5, 0.7], dtype=np.float64),
        "body_velocity": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        "analytic_force": np.asarray([0.0, 0.0, -1.0e-4, 0.0, 0.0, 0.0], dtype=np.float64),
        "mem_force": np.asarray([1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "density_all_finite": True,
        "density_min": 0.1,
        "density_max": 1.0,
        "flow_speed_max": 0.04,
        "water_contact_fraction": 0.2,
        "initial_water_contact_fraction": 0.0,
        "displacement_norm": 0.01,
        "coupling_mode": "hybrid",
        "sphere_density": 0.35,
        "min_body_z": 0.68,
        "horizontal_displacement": 0.02,
    }
    baseline.update(overrides)
    return metrics_module.DambreakMemMetrics(**baseline)


class _FakeDensityFieldRenderer:
    """Capture the field selected by the visual example without creating OpenGL."""

    def __init__(self) -> None:
        self.available: bool = True
        self.density: Any | None = None
        self.threshold: float | None = None

    def set_density_field(
        self,
        *,
        density: Any,
        grid_origin: tuple[float, float, float],
        cell_size: float,
        threshold: float,
        max_steps: int,
    ) -> None:
        self.density = density
        self.threshold = float(threshold)


class TestLbmRigidVisualization(unittest.TestCase):
    def test_dambreak_mem_validation_reports_nonfinite_density(self) -> None:
        """M1 validation should expose the first invalid field in test mode."""
        metrics_module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._metrics")
        metrics: Any = _make_dambreak_mem_metrics(metrics_module, density_max=float("nan"))

        with self.assertRaisesRegex(ValueError, "density range is not finite"):
            metrics_module.validate_m1_metrics(metrics, require_contact=False, require_mem=False)

    def test_dambreak_mem_validation_reports_partially_nonfinite_density_field(self) -> None:
        """M1 validation should reject nonfinite density fields even when finite min/max exist."""
        metrics_module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._metrics")
        metrics: Any = _make_dambreak_mem_metrics(metrics_module, density_all_finite=False)

        with self.assertRaisesRegex(ValueError, "density field contains nonfinite values"):
            metrics_module.validate_m1_metrics(metrics, require_contact=False, require_mem=False)

    def test_dambreak_mem_validation_reports_nonfinite_acceptance_scalars(self) -> None:
        """M1 validation should reject nonfinite scalar acceptance metrics."""
        metrics_module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._metrics")
        cases: tuple[tuple[str, str], ...] = (
            ("water_contact_fraction", "water_contact_fraction is not finite"),
            ("initial_water_contact_fraction", "initial_water_contact_fraction is not finite"),
            ("displacement_norm", "displacement_norm is not finite"),
        )

        for field_name, message in cases:
            with self.subTest(field_name=field_name):
                metrics: Any = _make_dambreak_mem_metrics(metrics_module, **{field_name: float("nan")})
                with self.assertRaisesRegex(ValueError, message):
                    metrics_module.validate_m1_metrics(metrics, require_contact=True, require_mem=False)

    def test_dambreak_mem_m1_parser_defaults_to_hybrid(self) -> None:
        """M1 should default to the stable hybrid coupling mode."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )
        parser: argparse.ArgumentParser = module.create_parser()
        args: argparse.Namespace = parser.parse_args([])

        self.assertEqual(str(args.coupling_mode), "hybrid")
        self.assertEqual(int(args.grid_res[0]), 32)
        self.assertGreater(float(args.force_scale), 0.0)

    def test_dambreak_mem_m1_parser_accepts_density_controls(self) -> None:
        """M1 CLI should expose shared density preset and explicit density controls."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )
        parser: argparse.ArgumentParser = module.create_parser()
        args: argparse.Namespace = parser.parse_args(["--density-preset", "heavy", "--sphere-density", "2.25"])

        self.assertEqual(str(args.density_preset), "heavy")
        self.assertAlmostEqual(float(args.sphere_density), 2.25)

    def test_dambreak_mem_m1_visual_example_constructs_with_null_viewer(self) -> None:
        """The M1 visual example should construct without opening GL in tests."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )
        scene_module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._scene")
        config: Any = scene_module.DambreakMemM1Config(grid_res=(24, 18, 20), tracer_count=0)

        example: Any = module.LbmDambreakMemM1VisualExample(
            viewer=None,
            config=config,
            print_every=1000,
        )

        self.assertIsNotNone(example.scene)
        self.assertEqual(str(example.metrics.coupling_mode), "hybrid")

    def test_dambreak_mem_m1_visual_example_test_final_rejects_zero_step(self) -> None:
        """The M1 visual example should not let test mode pass without stepping."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )
        scene_module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._scene")
        config: Any = scene_module.DambreakMemM1Config(grid_res=(24, 18, 20), tracer_count=0)
        example: Any = module.LbmDambreakMemM1VisualExample(
            viewer=None,
            config=config,
            print_every=1000,
        )

        with self.assertRaisesRegex(ValueError, "M1 visual did not advance any frames"):
            example.test_final()

    def test_dambreak_mem_m1_visual_example_renders_with_viewer_null(self) -> None:
        """The M1 visual example should render with ViewerNull without GL callbacks."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )
        scene_module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._scene")
        viewer: Any = newton.viewer.ViewerNull(num_frames=1)
        config: Any = scene_module.DambreakMemM1Config(grid_res=(24, 18, 20), tracer_count=0)
        example: Any = module.LbmDambreakMemM1VisualExample(
            viewer=viewer,
            config=config,
            print_every=1000,
        )

        example.step()
        example.render()
        example.test_final()

    def test_dambreak_mem_m1_visual_example_unpauses_test_viewer(self) -> None:
        """The M1 visual example test entrypoint should not stay paused before run()."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )
        args: argparse.Namespace = argparse.Namespace(test=True)
        viewer: Any = types.SimpleNamespace(_paused=True)

        module._unpause_viewer_for_test(viewer, args)

        self.assertFalse(bool(viewer._paused))

    def test_dambreak_mem_m1_visual_example_main_routes_headless_test_to_smoke(self) -> None:
        """The M1 visual example should avoid the unbounded GL run loop in headless test mode."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )
        args: argparse.Namespace = argparse.Namespace(
            device=None,
            grid_res=(24, 18, 20),
            cell_size=0.04,
            tau=1.0,
            force_scale=0.005,
            coupling_mode="hybrid",
            tracer_count=0,
            analytic_force_scale=100.0,
            vertical_drag_scale=0.2,
            flow_drag_scale=8.0,
            print_every=1,
            num_frames=1,
            test=True,
            headless=True,
        )
        viewer: Any = types.SimpleNamespace(_paused=True)
        smoke_calls: list[dict[str, Any]] = []
        run_calls: list[tuple[Any, argparse.Namespace]] = []
        original_init: Any = module.init_fluid_viewer
        original_run_smoke: Any = module.run_smoke
        original_run: Any = module.newton.examples.run

        def fake_init_fluid_viewer(parser: argparse.ArgumentParser) -> tuple[Any, argparse.Namespace]:
            return viewer, args

        def fake_run_smoke(**kwargs: Any) -> Any:
            smoke_calls.append(dict(kwargs))
            return None

        def fake_run(example: Any, run_args: argparse.Namespace) -> None:
            run_calls.append((example, run_args))

        try:
            module.init_fluid_viewer = fake_init_fluid_viewer
            module.run_smoke = fake_run_smoke
            module.newton.examples.run = fake_run

            module.main()
        finally:
            module.init_fluid_viewer = original_init
            module.run_smoke = original_run_smoke
            module.newton.examples.run = original_run

        self.assertEqual(len(smoke_calls), 1)
        self.assertEqual(smoke_calls[0]["num_frames"], 1)
        self.assertEqual(smoke_calls[0]["grid_res"], (24, 18, 20))
        self.assertFalse(run_calls)

    def test_dambreak_mem_m1_smoke_reports_finite_hybrid_metrics(self) -> None:
        """M1 should produce finite metrics and retained MEM diagnostics in hybrid mode."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )

        metrics: Any = module.run_smoke(
            num_frames=80,
            grid_res=(24, 18, 20),
            tracer_count=0,
            coupling_mode="hybrid",
            force_scale=0.005,
            dt=1.0 / 1920.0,
            print_every=80,
        )

        self.assertEqual(str(metrics.coupling_mode), "hybrid")
        self.assertTrue(np.all(np.isfinite(metrics.body_position)))
        self.assertTrue(np.all(np.isfinite(metrics.body_velocity)))
        self.assertTrue(np.all(np.isfinite(metrics.analytic_force)))
        self.assertTrue(np.all(np.isfinite(metrics.mem_force)))
        self.assertTrue(np.isfinite(float(metrics.density_min)))
        self.assertTrue(np.isfinite(float(metrics.density_max)))

    def test_dambreak_mem_m1_metrics_expose_density_and_trajectory(self) -> None:
        """M1 metrics should expose resolved density and base trajectory diagnostics."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )

        metrics: Any = module.run_smoke(
            num_frames=12,
            grid_res=(24, 18, 20),
            tracer_count=0,
            coupling_mode="hybrid",
            force_scale=0.005,
            density_preset="heavy",
            dt=1.0 / 1920.0,
            print_every=0,
        )

        self.assertAlmostEqual(float(metrics.sphere_density), 3.0)
        self.assertTrue(np.isfinite(float(metrics.min_body_z)))
        self.assertTrue(np.isfinite(float(metrics.horizontal_displacement)))
        self.assertLessEqual(float(metrics.min_body_z), float(metrics.body_position[2]))
        self.assertGreaterEqual(float(metrics.horizontal_displacement), 0.0)

    def test_dambreak_mem_m2_parser_defaults_to_light_hybrid(self) -> None:
        """M2 should expose a separate density-preset visual entrypoint."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m2_density_presets_visual"
        )
        parser: argparse.ArgumentParser = module.create_parser()
        args: argparse.Namespace = parser.parse_args([])

        self.assertEqual(str(args.density_preset), "light")
        self.assertEqual(str(args.coupling_mode), "hybrid")
        self.assertAlmostEqual(float(args.force_scale), 0.005)

    def test_dambreak_mem_m2_main_uses_m2_parser_defaults(self) -> None:
        """M2 main should run with M2 parser defaults, not M1's parser directly."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m2_density_presets_visual"
        )
        m1_module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )
        captured_kwargs: list[dict[str, Any]] = []
        original_create_parser: Any = m1_module.create_parser
        original_init_fluid_viewer: Any = m1_module.init_fluid_viewer
        original_run_smoke: Any = m1_module.run_smoke

        def fake_m1_create_parser() -> argparse.ArgumentParser:
            parser: argparse.ArgumentParser = original_create_parser()
            parser.set_defaults(density_preset="heavy")
            return parser

        def fake_init_fluid_viewer(parser: argparse.ArgumentParser) -> tuple[Any, argparse.Namespace]:
            args: argparse.Namespace = parser.parse_args([])
            args.test = True
            args.headless = True
            args.num_frames = 1
            return types.SimpleNamespace(close=lambda: None), args

        def fake_run_smoke(**kwargs: Any) -> None:
            captured_kwargs.append(dict(kwargs))

        try:
            m1_module.create_parser = fake_m1_create_parser
            m1_module.init_fluid_viewer = fake_init_fluid_viewer
            m1_module.run_smoke = fake_run_smoke

            module.main()
        finally:
            m1_module.create_parser = original_create_parser
            m1_module.init_fluid_viewer = original_init_fluid_viewer
            m1_module.run_smoke = original_run_smoke

        self.assertEqual(len(captured_kwargs), 1)
        self.assertEqual(str(captured_kwargs[0]["density_preset"]), "light")
        self.assertEqual(str(captured_kwargs[0]["coupling_mode"]), "hybrid")

    def test_dambreak_mem_m2_single_preset_smoke_reports_finite_metrics(self) -> None:
        """M2 should run one selected density preset with finite diagnostics."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m2_density_presets_visual"
        )

        metrics: Any = module.run_smoke(
            num_frames=60,
            grid_res=(24, 18, 20),
            density_preset="heavy",
            tracer_count=0,
            force_scale=0.005,
            dt=1.0 / 120.0,
            print_every=0,
        )

        self.assertAlmostEqual(float(metrics.sphere_density), 3.0, places=9)
        self.assertTrue(np.all(np.isfinite(metrics.body_position)))
        self.assertTrue(np.all(np.isfinite(metrics.body_velocity)))
        self.assertTrue(np.all(np.isfinite(metrics.analytic_force)))
        self.assertTrue(np.all(np.isfinite(metrics.mem_force)))
        self.assertTrue(bool(metrics.density_all_finite))

    def test_dambreak_mem_m2_light_heavy_comparison_separates_trajectories(self) -> None:
        """M2 should prove light and heavy presets produce different single-sphere trajectories."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m2_density_presets_visual"
        )

        comparison: Any = module.run_density_comparison(
            num_frames=180,
            grid_res=(24, 18, 20),
            tracer_count=0,
            force_scale=0.005,
            dt=1.0 / 120.0,
            print_every=0,
        )

        light: Any = comparison["light"]
        heavy: Any = comparison["heavy"]
        self.assertGreater(float(light.body_position[2]), float(heavy.body_position[2]))
        self.assertGreater(float(light.min_body_z), float(heavy.min_body_z))
        self.assertGreater(float(light.horizontal_displacement), 1.0e-5)
        self.assertTrue(bool(light.density_all_finite))
        self.assertTrue(bool(heavy.density_all_finite))
        self.assertGreater(float(light.max_water_contact_fraction), float(light.initial_water_contact_fraction))
        self.assertGreater(float(heavy.max_water_contact_fraction), float(heavy.initial_water_contact_fraction))
        self.assertGreater(float(np.linalg.norm(light.mem_force[:3])), 0.0)
        self.assertGreater(float(np.linalg.norm(heavy.mem_force[:3])), 0.0)

    def test_dambreak_mem_m1_rejects_invalid_explicit_sphere_density(self) -> None:
        """M1 should reject explicit nonfinite or nonpositive sphere densities before scene build."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )
        invalid_densities: tuple[float, ...] = (float("nan"), float("inf"), 0.0, -1.0)

        for sphere_density in invalid_densities:
            with self.subTest(sphere_density=sphere_density):
                with self.assertRaisesRegex(ValueError, "sphere_density must be a finite positive value"):
                    module.run_smoke(
                        num_frames=0,
                        grid_res=(24, 18, 20),
                        tracer_count=0,
                        coupling_mode="hybrid",
                        force_scale=0.005,
                        sphere_density=sphere_density,
                        dt=1.0 / 1920.0,
                        print_every=0,
                    )

    def test_dambreak_mem_m1_metrics_expose_force_audit_values(self) -> None:
        """M1.1 metrics should explain fluid gravity versus scaled rigid gravity."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )

        metrics: Any = module.run_smoke(
            num_frames=1,
            grid_res=(24, 18, 20),
            tracer_count=0,
            coupling_mode="hybrid",
            force_scale=0.005,
            dt=1.0 / 1920.0,
            print_every=0,
        )

        self.assertTrue(np.isfinite(float(metrics.fluid_gravity_z)))
        self.assertTrue(np.isfinite(float(metrics.unscaled_gravity_accel_z)))
        self.assertTrue(np.isfinite(float(metrics.scaled_gravity_accel_z)))
        self.assertAlmostEqual(float(metrics.unscaled_gravity_accel_z), float(metrics.fluid_gravity_z), places=9)
        self.assertLess(float(metrics.scaled_gravity_accel_z), float(metrics.unscaled_gravity_accel_z))

    def test_dambreak_mem_m1_format_includes_force_audit_fields(self) -> None:
        """M1.1 logs should explain fluid and rigid gravity scales."""
        metrics_module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._metrics")
        render_module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._render")
        metrics: Any = metrics_module.DambreakMemMetrics(
            frame=1,
            sim_time=0.01,
            body_position=np.asarray([0.4, 0.4, 0.7], dtype=np.float64),
            body_velocity=np.asarray([0.0, 0.0, -0.01], dtype=np.float64),
            analytic_force=np.asarray([0.0, 0.0, -1.0e-4, 0.0, 0.0, 0.0], dtype=np.float64),
            mem_force=np.asarray([1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            density_all_finite=True,
            density_min=0.1,
            density_max=1.9,
            flow_speed_max=0.02,
            water_contact_fraction=0.0,
            initial_water_contact_fraction=0.0,
            displacement_norm=0.01,
            coupling_mode="hybrid",
            fluid_gravity_z=-0.0005,
            unscaled_gravity_accel_z=-0.0005,
            scaled_gravity_accel_z=-0.05,
            buoyancy_accel_z=0.0,
            vertical_drag_accel_z=0.0,
            flow_drag_accel_x=0.0,
            flow_drag_accel_y=0.0,
            sphere_density=0.35,
            min_body_z=0.69,
            horizontal_displacement=0.02,
        )

        text: str = render_module.format_m1_metrics(metrics)

        self.assertIn("fluid_g=", text)
        self.assertIn("rigid_g=", text)
        self.assertIn("buoy_accel=", text)
        self.assertIn("initial_contact=", text)
        self.assertIn("max_contact=", text)
        self.assertIn("sphere_density=", text)
        self.assertIn("min_z=", text)
        self.assertIn("hdisp=", text)

    def test_dambreak_mem_m1_default_starts_dry(self) -> None:
        """M1 should start the sphere outside the initial dam and pool water."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )

        metrics: Any = module.run_smoke(
            num_frames=0,
            grid_res=(24, 18, 20),
            tracer_count=0,
            coupling_mode="hybrid",
            force_scale=0.005,
            dt=1.0 / 1920.0,
        )

        self.assertEqual(float(metrics.initial_water_contact_fraction), 0.0)
        self.assertEqual(float(metrics.water_contact_fraction), 0.0)
        self.assertEqual(float(metrics.max_water_contact_fraction), 0.0)

    def test_dambreak_mem_m1_dry_start_falls_before_water_contact(self) -> None:
        """A dry-start M1 sphere should initially move downward, not receive an early water kick upward."""
        scene_module: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._scene")
        config: Any = scene_module.DambreakMemM1Config(
            grid_res=(24, 18, 20),
            tracer_count=0,
            coupling_mode="hybrid",
            force_scale=0.005,
        )
        scene: Any = scene_module.build_m1_scene(config)
        initial_z: float = float(scene.base.initial_body_position[2])
        metrics: Any = scene_module.collect_m1_metrics(scene)

        for _frame_index in range(12):
            metrics = scene_module.step_m1_scene(scene, dt=1.0 / 120.0)

        self.assertEqual(float(metrics.max_water_contact_fraction), 0.0)
        self.assertLess(float(metrics.body_position[2]), initial_z)
        self.assertLess(float(metrics.body_velocity[2]), 0.0)

    def test_dambreak_mem_m1_rejects_invalid_coupling_mode(self) -> None:
        """M1 should fail fast for unsupported coupling modes."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )

        with self.assertRaisesRegex(ValueError, "unsupported coupling_mode"):
            module.run_smoke(
                num_frames=0,
                grid_res=(24, 18, 20),
                tracer_count=0,
                coupling_mode="invalid",
                force_scale=0.005,
                dt=1.0 / 1920.0,
            )

    def test_dambreak_mem_m1_analytic_mode_reports_only_analytic_force(self) -> None:
        """Analytic mode should keep MEM feedback disabled while retaining analytic force diagnostics."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )

        metrics: Any = module.run_smoke(
            num_frames=80,
            grid_res=(24, 18, 20),
            tracer_count=0,
            coupling_mode="analytic",
            force_scale=0.005,
            dt=1.0 / 1920.0,
            print_every=0,
        )

        self.assertEqual(str(metrics.coupling_mode), "analytic")
        self.assertGreater(float(np.linalg.norm(metrics.analytic_force[:3])), 0.0)
        self.assertEqual(float(np.linalg.norm(metrics.mem_force[:3])), 0.0)

    def test_dambreak_mem_m1_mem_mode_reports_only_mem_force(self) -> None:
        """MEM mode should suppress analytic force while exposing retained MEM feedback."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )

        metrics: Any = module.run_smoke(
            num_frames=80,
            grid_res=(24, 18, 20),
            tracer_count=0,
            coupling_mode="mem",
            force_scale=0.005,
            dt=1.0 / 1920.0,
            print_every=0,
        )

        self.assertEqual(str(metrics.coupling_mode), "mem")
        self.assertEqual(float(np.linalg.norm(metrics.analytic_force[:3])), 0.0)
        self.assertGreater(float(np.linalg.norm(metrics.mem_force[:3])), 0.0)

    def test_dambreak_mem_m1_contact_and_mem_become_observable(self) -> None:
        """M1 should make contact with water and report nonzero MEM feedback."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )

        metrics: Any = module.run_smoke(
            num_frames=180,
            grid_res=(24, 18, 20),
            tracer_count=0,
            coupling_mode="hybrid",
            force_scale=0.005,
            dt=1.0 / 120.0,
            print_every=180,
            require_contact=True,
            require_mem=True,
        )

        self.assertEqual(float(metrics.initial_water_contact_fraction), 0.0)
        self.assertGreater(float(metrics.max_water_contact_fraction), float(metrics.initial_water_contact_fraction))
        self.assertGreater(float(np.linalg.norm(metrics.mem_force[:3])), 0.0)
        self.assertGreater(float(metrics.displacement_norm), 0.0)

    def test_dambreak_mem_m1_documented_hybrid_command_remains_finite(self) -> None:
        """The documented M1 hybrid scene should survive a long headless run."""
        module: types.ModuleType = importlib.import_module(
            "wanphys.examples.lbm.dambreak_mem.m1_hybrid_single_sphere_visual"
        )

        metrics: Any = module.run_smoke(
            num_frames=900,
            grid_res=(24, 18, 20),
            tracer_count=0,
            coupling_mode="hybrid",
            force_scale=0.005,
            dt=1.0 / 1920.0,
            print_every=300,
            require_contact=True,
            require_mem=True,
        )

        self.assertTrue(np.all(np.isfinite(metrics.body_position)))
        self.assertTrue(np.all(np.isfinite(metrics.body_velocity)))
        self.assertTrue(np.all(np.isfinite(metrics.analytic_force)))
        self.assertTrue(np.all(np.isfinite(metrics.mem_force)))
        self.assertTrue(bool(metrics.density_all_finite))
        self.assertTrue(np.isfinite(float(metrics.max_water_contact_fraction)))
        self.assertGreater(float(metrics.max_water_contact_fraction), float(metrics.initial_water_contact_fraction))
        self.assertGreater(float(np.linalg.norm(metrics.mem_force[:3])), 0.0)
        self.assertGreater(float(metrics.displacement_norm), 0.0)

    def test_dambreak_mem_force_breakdown_light_sphere_buoyancy_exceeds_weight(self) -> None:
        """A light contacted sphere should receive upward analytic stabilization."""
        forces: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._forces")

        breakdown: Any = forces.compute_hybrid_force_breakdown(
            sphere_radius=0.16,
            sphere_density=0.35,
            rho_water=1.8,
            gravity_z=-0.001,
            water_contact_fraction=0.8,
            body_velocity=(0.0, 0.0, -0.02),
            local_flow_velocity=(0.04, 0.0, 0.0),
            analytic_force_scale=100.0,
            vertical_drag_scale=0.2,
            flow_drag_scale=8.0,
        )

        self.assertGreater(float(breakdown.force[2]), 0.0)
        self.assertGreater(float(breakdown.buoyancy_z), abs(float(breakdown.gravity_z)))
        self.assertGreater(float(breakdown.force[0]), 0.0)

    def test_dambreak_mem_force_audit_reports_scaled_gravity_acceleration(self) -> None:
        """M1.1 should make dry rigid gravity scale explicit."""
        forces: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._forces")

        breakdown: Any = forces.compute_hybrid_force_breakdown(
            sphere_radius=0.16,
            sphere_density=3.0,
            rho_water=1.8,
            gravity_z=-0.0005,
            water_contact_fraction=0.0,
            body_velocity=(0.0, 0.0, 0.0),
            local_flow_velocity=(0.0, 0.0, 0.0),
            analytic_force_scale=100.0,
            vertical_drag_scale=0.2,
            flow_drag_scale=8.0,
        )

        self.assertAlmostEqual(float(breakdown.unscaled_gravity_accel_z), -0.0005, places=9)
        self.assertAlmostEqual(float(breakdown.scaled_gravity_accel_z), -0.05, places=9)
        self.assertEqual(float(breakdown.buoyancy_accel_z), 0.0)
        self.assertEqual(float(breakdown.flow_drag_accel_x), 0.0)

    def test_dambreak_mem_force_breakdown_dry_sphere_keeps_gravity(self) -> None:
        """A dry sphere should still fall under analytic gravity."""
        forces: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._forces")

        breakdown: Any = forces.compute_hybrid_force_breakdown(
            sphere_radius=0.16,
            sphere_density=3.0,
            rho_water=1.8,
            gravity_z=-0.001,
            water_contact_fraction=0.0,
            body_velocity=(0.0, 0.0, 0.0),
            local_flow_velocity=(0.04, 0.0, 0.0),
            analytic_force_scale=100.0,
            vertical_drag_scale=0.2,
            flow_drag_scale=8.0,
        )

        self.assertLess(float(breakdown.force[2]), 0.0)
        self.assertEqual(float(breakdown.buoyancy_z), 0.0)
        self.assertEqual(float(breakdown.force[0]), 0.0)

    def test_dambreak_mem_force_breakdown_dry_sphere_keeps_residual_vertical_damping(self) -> None:
        """A dry moving sphere should keep the small demo-stabilizing vertical damping floor."""
        forces: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._forces")

        breakdown: Any = forces.compute_hybrid_force_breakdown(
            sphere_radius=0.16,
            sphere_density=3.0,
            rho_water=1.8,
            gravity_z=-0.001,
            water_contact_fraction=0.0,
            body_velocity=(0.0, 0.0, -0.5),
            local_flow_velocity=(0.04, 0.0, 0.0),
            analytic_force_scale=100.0,
            vertical_drag_scale=0.2,
            flow_drag_scale=8.0,
        )

        self.assertAlmostEqual(float(breakdown.vertical_drag_z), 0.002)
        self.assertEqual(float(breakdown.buoyancy_z), 0.0)
        self.assertEqual(float(breakdown.force[0]), 0.0)

    def test_dambreak_mem_force_breakdown_exposes_gravity_force_alias(self) -> None:
        """The breakdown should expose a force-named gravity term while preserving the legacy alias."""
        forces: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._forces")

        breakdown: Any = forces.compute_hybrid_force_breakdown(
            sphere_radius=0.16,
            sphere_density=3.0,
            rho_water=1.8,
            gravity_z=-0.001,
            water_contact_fraction=0.0,
            body_velocity=(0.0, 0.0, 0.0),
            local_flow_velocity=(0.04, 0.0, 0.0),
            analytic_force_scale=100.0,
            vertical_drag_scale=0.2,
            flow_drag_scale=8.0,
        )

        self.assertEqual(float(breakdown.gravity_force_z), float(breakdown.gravity_z))
        self.assertLess(float(breakdown.gravity_force_z), 0.0)

    def test_dambreak_mem_force_breakdown_vertical_formula_is_explicit(self) -> None:
        """The returned force should apply scale only to weight and buoyancy terms."""
        forces: types.ModuleType = importlib.import_module("wanphys.examples.lbm.dambreak_mem._forces")

        analytic_force_scale: float = 50.0
        breakdown: Any = forces.compute_hybrid_force_breakdown(
            sphere_radius=0.16,
            sphere_density=0.8,
            rho_water=1.8,
            gravity_z=-0.001,
            water_contact_fraction=0.5,
            body_velocity=(0.0, 0.0, -0.25),
            local_flow_velocity=(0.04, 0.0, 0.0),
            analytic_force_scale=analytic_force_scale,
            vertical_drag_scale=0.2,
            flow_drag_scale=8.0,
        )
        expected_force_z: float = (
            (float(breakdown.gravity_force_z) + float(breakdown.buoyancy_z)) * analytic_force_scale
            + float(breakdown.vertical_drag_z)
        )

        self.assertAlmostEqual(float(breakdown.force[2]), expected_force_z)

    def test_twoway_visual_null_smoke_reports_finite_telemetry(self) -> None:
        """The visual example should be smoke-testable without OpenGL."""
        module: types.ModuleType = _import_visual_module()

        telemetry: Any = _run_smoke_helper(module)
        metrics: dict[str, float] = {
            "force_norm": _read_metric(telemetry, "force_norm", 0),
            "torque_norm": _read_metric(telemetry, "torque_norm", 1),
            "speed_max": _read_metric(telemetry, "speed_max", 2),
            "flow_speed_max": _read_metric(telemetry, "flow_speed_max", 3),
            "body_speed": _read_metric(telemetry, "body_speed", 4),
            "displacement_norm": _read_metric(telemetry, "displacement_norm", 5),
        }

        for metric_name, metric_value in metrics.items():
            self.assertTrue(math.isfinite(metric_value), msg=f"{metric_name} should be finite")
            self.assertGreaterEqual(metric_value, 0.0, msg=f"{metric_name} should be a norm or max speed")

    def test_twoway_visual_display_field_removes_uniform_inflow(self) -> None:
        """The display scalar should not render the whole uniform-flow domain as liquid."""
        scene_helpers: types.ModuleType = importlib.import_module("wanphys.examples.lbm._lbm_twoway_scene")
        config: Any = scene_helpers.LbmTwoWaySceneConfig()
        scene: Any = scene_helpers.build_twoway_scene(config)

        scene_helpers.step_twoway_scene(scene, dt=1.0 / 120.0, substeps=2)
        field: Any = scene.speed_field.numpy()
        far_corner_value: float = float(field[0, 0, 0])
        display_max: float = float(field.max())

        self.assertLess(far_corner_value, 1.0e-6)
        self.assertGreater(display_max, 0.0)

    def test_twoway_visual_field_mode_parser_defaults_to_density(self) -> None:
        """Two-way field rendering should default to the one-way density look."""
        module: types.ModuleType = _import_visual_module()
        parser: argparse.ArgumentParser = module.create_parser()
        field_actions: list[argparse.Action] = [
            action for action in parser._actions if action.dest == "field_mode"
        ]

        self.assertEqual(len(field_actions), 1)
        self.assertEqual(field_actions[0].default, "density")
        self.assertEqual(tuple(field_actions[0].choices), ("density", "speed", "perturbation"))

    def test_twoway_visual_run_smoke_accepts_all_field_modes(self) -> None:
        """Headless smoke should accept density, speed, and perturbation field modes."""
        module: types.ModuleType = _import_visual_module()

        for field_mode in ("density", "speed", "perturbation"):
            with self.subTest(field_mode=field_mode):
                metrics: Any = module.run_smoke(num_frames=1, tracer_count=0, field_mode=field_mode)
                self.assertTrue(math.isfinite(float(metrics.speed_max)))

    def test_twoway_visual_render_field_mode_selects_scalar_source(self) -> None:
        """The GL field path should select density, full speed, or perturbation fields explicitly."""
        module: types.ModuleType = _import_visual_module()
        config: Any = module.LbmTwoWaySceneConfig(tracer_count=0)

        for field_mode, expected_attr in (
            ("density", "density"),
            ("speed", "flow_speed_field"),
            ("perturbation", "speed_field"),
        ):
            with self.subTest(field_mode=field_mode):
                viewer: Any = newton.viewer.ViewerNull(num_frames=1)
                example: Any = module.LbmTwoWayFsiVisualExample(
                    viewer,
                    config=config,
                    fluid_render_mode="field",
                    field_mode=field_mode,
                    print_every=999,
                )
                renderer: _FakeDensityFieldRenderer = _FakeDensityFieldRenderer()
                example.ssfr = renderer
                example.render()

                if expected_attr == "density":
                    expected_field: Any = example.scene.fluid_domain.state.density
                else:
                    expected_field = getattr(example.scene, expected_attr)
                self.assertIs(renderer.density, expected_field)
                self.assertIsNotNone(renderer.threshold)

    def test_twoway_visual_perturbation_threshold_tracks_strong_motion_field(self) -> None:
        """Perturbation field rendering should not turn broad weak motion into a full-volume blob."""
        module: types.ModuleType = _import_visual_module()
        viewer: Any = newton.viewer.ViewerNull(num_frames=1)
        config: Any = module.LbmTwoWaySceneConfig(
            tracer_count=0,
            advance_rigid=True,
            force_scale=100.0,
            flow_velocity=(0.04, 0.0, 0.0),
        )
        example: Any = module.LbmTwoWayFsiVisualExample(
            viewer,
            config=config,
            fluid_render_mode="field",
            field_mode="perturbation",
            print_every=999,
        )
        for _frame_index in range(20):
            example.step()

        renderer: _FakeDensityFieldRenderer = _FakeDensityFieldRenderer()
        example.ssfr = renderer
        example.render()

        self.assertIs(renderer.density, example.scene.speed_field)
        self.assertIsNotNone(renderer.threshold)
        self.assertGreaterEqual(float(renderer.threshold), float(config.flow_velocity[0]) * 0.5)

    def test_twoway_visual_keeps_driven_flow_alive(self) -> None:
        """The visual acceptance scene should keep a sustained cross-flow."""
        scene_helpers: types.ModuleType = importlib.import_module("wanphys.examples.lbm._lbm_twoway_scene")
        config: Any = scene_helpers.LbmTwoWaySceneConfig()
        scene: Any = scene_helpers.build_twoway_scene(config)

        for _frame_index in range(120):
            scene_helpers.step_twoway_scene(scene, dt=1.0 / 120.0, substeps=2)

        state: Any = scene.fluid_domain.state
        ux: np.ndarray = np.asarray(state.velocity_x.numpy(), dtype=np.float64)
        uy: np.ndarray = np.asarray(state.velocity_y.numpy(), dtype=np.float64)
        uz: np.ndarray = np.asarray(state.velocity_z.numpy(), dtype=np.float64)
        solid_phi: np.ndarray = np.asarray(state.solid_phi.numpy(), dtype=np.float64)
        fluid_mask: np.ndarray = solid_phi >= 0.0
        velocity_mag: np.ndarray = np.sqrt(ux * ux + uy * uy + uz * uz)

        self.assertGreater(float(np.max(velocity_mag[fluid_mask])), float(config.flow_velocity[0]) * 0.5)

    def test_twoway_visual_motion_smoke_reports_retained_mem_feedback(self) -> None:
        """A real two-way motion run should report nonzero retained MEM feedback."""
        module: types.ModuleType = _import_visual_module()

        metrics: Any = module.run_smoke(
            num_frames=24,
            tracer_count=0,
            advance_rigid=True,
            force_scale=100.0,
            flow_speed=0.04,
            grid_res=(16, 12, 12),
            field_mode="perturbation",
        )

        self.assertTrue(math.isfinite(float(metrics.mem_force_norm)))
        self.assertGreater(float(metrics.mem_force_norm), 0.0)
        self.assertGreater(float(metrics.body_displacement[0]), 0.0)
        self.assertGreater(float(metrics.displacement_norm), 0.0)

    def test_twoway_visual_advance_rigid_keeps_force_and_motion_finite(self) -> None:
        """The documented real-MEM visual settings should remain finite."""
        module: types.ModuleType = _import_visual_module()

        metrics: Any = module.run_smoke(
            num_frames=120,
            tracer_count=0,
            advance_rigid=True,
            force_scale=100.0,
            flow_speed=0.04,
            grid_res=(16, 12, 12),
            field_mode="perturbation",
        )

        self.assertTrue(np.all(np.isfinite(metrics.body_position)))
        self.assertTrue(np.all(np.isfinite(metrics.body_velocity)))
        self.assertTrue(math.isfinite(float(metrics.flow_speed_max)))
        self.assertGreater(float(metrics.mem_force_norm), 0.0)
        self.assertGreater(float(metrics.displacement_norm), 0.0)

    def test_twoway_visual_documented_high_force_scale_remains_finite_long_enough(self) -> None:
        """The documented GL real-MEM command should not become nonfinite after startup."""
        scene_helpers: types.ModuleType = importlib.import_module("wanphys.examples.lbm._lbm_twoway_scene")
        config: Any = scene_helpers.LbmTwoWaySceneConfig(
            grid_res=(16, 12, 12),
            tracer_count=0,
            advance_rigid=True,
            force_scale=100.0,
            flow_velocity=(0.04, 0.0, 0.0),
        )
        scene: Any = scene_helpers.build_twoway_scene(config)
        metrics: Any | None = None

        for _frame_index in range(300):
            metrics = scene_helpers.step_twoway_scene(scene, dt=1.0 / 120.0, substeps=2)
            if _frame_index % 20 != 0:
                continue

            state: Any = scene.fluid_domain.state
            rho: np.ndarray = np.asarray(state.density.numpy(), dtype=np.float64)
            ux: np.ndarray = np.asarray(state.velocity_x.numpy(), dtype=np.float64)
            uy: np.ndarray = np.asarray(state.velocity_y.numpy(), dtype=np.float64)
            uz: np.ndarray = np.asarray(state.velocity_z.numpy(), dtype=np.float64)
            diagnostic: str = (
                f"frame={_frame_index} pos={metrics.body_position.tolist()} "
                f"vel={metrics.body_velocity.tolist()} flow_max={metrics.flow_speed_max}"
            )
            self.assertTrue(np.all(np.isfinite(rho)), msg=diagnostic)
            self.assertTrue(np.all(np.isfinite(ux)), msg=diagnostic)
            self.assertTrue(np.all(np.isfinite(uy)), msg=diagnostic)
            self.assertTrue(np.all(np.isfinite(uz)), msg=diagnostic)
            self.assertTrue(np.all(np.isfinite(metrics.body_position)), msg=diagnostic)
            self.assertTrue(np.all(np.isfinite(metrics.body_velocity)), msg=diagnostic)

        self.assertIsNotNone(metrics)
        self.assertGreater(float(metrics.displacement_norm), 0.0)

    def test_twoway_visual_tracers_advect_with_flow(self) -> None:
        """Tracer points should move when the LBM scene is stepped."""
        scene_helpers: types.ModuleType = importlib.import_module("wanphys.examples.lbm._lbm_twoway_scene")
        config: Any = scene_helpers.LbmTwoWaySceneConfig(tracer_count=64)
        scene: Any = scene_helpers.build_twoway_scene(config)

        self.assertIsNotNone(scene.tracer_positions)
        initial_positions: np.ndarray = np.asarray(scene.tracer_positions.numpy(), dtype=np.float64)
        for _frame_index in range(10):
            scene_helpers.step_twoway_scene(scene, dt=1.0 / 120.0, substeps=2)
        current_positions: np.ndarray = np.asarray(scene.tracer_positions.numpy(), dtype=np.float64)
        displacement: np.ndarray = np.linalg.norm(current_positions - initial_positions, axis=1)

        self.assertGreater(float(np.max(displacement)), 0.01)

    def test_twoway_visual_tracers_stay_clear_of_walls_and_body(self) -> None:
        """Tracer points should stay away from walls and solid cells."""
        scene_helpers: types.ModuleType = importlib.import_module("wanphys.examples.lbm._lbm_twoway_scene")
        config: Any = scene_helpers.LbmTwoWaySceneConfig(tracer_count=64)
        scene: Any = scene_helpers.build_twoway_scene(config)
        self.assertIsNotNone(scene.tracer_positions)

        initial_positions: np.ndarray = np.asarray(scene.tracer_positions.numpy(), dtype=np.float64)
        dh: float = float(config.dh)
        nx: int = int(config.grid_res[0])
        ny: int = int(config.grid_res[1])
        nz: int = int(config.grid_res[2])

        self.assertGreater(float(initial_positions[:, 1].min()), 2.0 * dh)
        self.assertLess(float(initial_positions[:, 1].max()), float(ny) * dh - 2.0 * dh)
        self.assertGreater(float(initial_positions[:, 2].min()), 2.0 * dh)
        self.assertLess(float(initial_positions[:, 2].max()), float(nz) * dh - 2.0 * dh)

        for _frame_index in range(180):
            scene_helpers.step_twoway_scene(scene, dt=1.0 / 120.0, substeps=2)

        state: Any = scene.fluid_domain.state
        positions: np.ndarray = np.asarray(scene.tracer_positions.numpy(), dtype=np.float64)
        solid_phi: np.ndarray = np.asarray(state.solid_phi.numpy(), dtype=np.float64)
        cell_idx: np.ndarray = np.clip(np.floor(positions / dh).astype(np.int32), 0, np.array([nx - 1, ny - 1, nz - 1]))
        phi_at: np.ndarray = solid_phi[cell_idx[:, 0], cell_idx[:, 1], cell_idx[:, 2]]

        self.assertGreater(float(phi_at.min()), 0.5 * dh)


if __name__ == "__main__":
    unittest.main(verbosity=2)

