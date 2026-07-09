# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Run an LBM example and measure average frame-rate and GPU memory.

Monkey-patches ``newton.examples.run()`` to collect per-frame timing and VRAM
samples, then reports aggregate statistics at exit.

**Viewer choice matters for render cost:**

- ``--viewer gl`` (default): real OpenGL context (headless invisible window).
  Captures **full rendering cost** including SSFR ray-marching and buffer swaps.
- ``--viewer null``: sim-only. SSFR post-render callbacks are no-ops, so
  ``render()`` cost is near-zero.  Useful for isolating simulation performance.

Usage::

    # Dam-break, real rendering cost (GL headless)
    uv run python scripts/bench/bench_lbm_run.py \\
        -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt -f 300

    # Droplet fall, GL viewer with visible window
    uv run python scripts/bench/bench_lbm_run.py \\
        -m wanphys.examples.lbm.fluid_grid_lbm_droplet_fall_trt \\
        -f 200 --viewer gl --no-headless

    # Pool-drop (rigid coupling), sim-only
    uv run python scripts/bench/bench_lbm_run.py \\
        -m wanphys.examples.lbm.fluid_grid_lbm_pool_drop \\
        -f 400 --viewer null

    # Two-sphere dam-break, save per-frame CSV
    uv run python scripts/bench/bench_lbm_run.py \\
        -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_two_spheres \\
        -f 200 --csv-per-frame frames.csv

    # Scan multiple examples, append to aggregate CSV
    for mod in \\
        fluid_grid_lbm_dambreak_trt \\
        fluid_grid_lbm_droplet_fall_trt \\
        fluid_grid_lbm_spinodal_trt \\
    ; do
        uv run python scripts/bench/bench_lbm_run.py \\
            -m wanphys.examples.lbm.$mod -f 300 --csv results.csv
    done
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# GPU helpers — must work before Warp is initialised by the example
# ---------------------------------------------------------------------------


def _get_nvsmi_memory() -> tuple[int, int]:
    """Return ``(used_mib, total_mib)`` from nvidia-smi, or ``(-1, -1)``."""
    try:
        raw: str = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        parts: list[str] = raw.split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return -1, -1


def _get_gpu_name() -> str:
    """Return GPU product name, or ``"unknown"``."""
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        ).strip()
    except Exception:
        return "unknown"


def _get_warp_mempool(device: str = "cuda:0") -> tuple[int, int]:
    """Return ``(current_mib, peak_mib)`` from Warp mempool."""
    import warp as wp
    cur: int = int(wp.get_mempool_used_mem_current(device)) // (1024 * 1024)
    peak: int = int(wp.get_mempool_used_mem_high(device)) // (1024 * 1024)
    return cur, peak


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class _BenchCollector:
    """Gathers per-frame timing and memory samples."""

    def __init__(self) -> None:
        self.total_ms: list[float] = []
        self.step_ms: list[float] = []
        self.render_ms: list[float] = []
        self.nvsmi_samples: list[tuple[int, int]] = []
        self.warp_samples: list[tuple[int, int]] = []
        self._mem_interval: int = 5
        self._frame_idx: int = 0

    def record_frame(self, total_ms: float, step_ms: float, render_ms: float) -> None:
        self.total_ms.append(total_ms)
        self.step_ms.append(step_ms)
        self.render_ms.append(render_ms)
        self._frame_idx += 1
        if self._frame_idx % self._mem_interval == 0:
            self.nvsmi_samples.append(_get_nvsmi_memory())
            try:
                self.warp_samples.append(_get_warp_mempool())
            except Exception:
                pass

    def summary(self) -> dict[str, Any]:
        nvsmi_used: list[int] = [s[0] for s in self.nvsmi_samples if s[0] >= 0]
        nvsmi_total: list[int] = [s[1] for s in self.nvsmi_samples if s[1] >= 0]
        warp_cur: list[int] = [s[0] for s in self.warp_samples]
        warp_peak: list[int] = [s[1] for s in self.warp_samples]

        avg_total: float = statistics.mean(self.total_ms) if self.total_ms else 0.0
        return {
            "frames": len(self.total_ms),
            "avg_total_ms": avg_total,
            "avg_total_fps": 1000.0 / avg_total if avg_total > 0 else 0.0,
            "min_total_ms": min(self.total_ms) if self.total_ms else 0.0,
            "max_total_ms": max(self.total_ms) if self.total_ms else 0.0,
            "p50_total_ms": float(np.percentile(self.total_ms, 50)) if self.total_ms else 0.0,
            "p99_total_ms": float(np.percentile(self.total_ms, 99)) if self.total_ms else 0.0,
            "stdev_total_ms": statistics.stdev(self.total_ms) if len(self.total_ms) > 1 else 0.0,
            "avg_step_ms": statistics.mean(self.step_ms) if self.step_ms else 0.0,
            "avg_step_fps": (
                1000.0 / statistics.mean(self.step_ms)
                if self.step_ms and statistics.mean(self.step_ms) > 0 else 0.0
            ),
            "min_step_ms": min(self.step_ms) if self.step_ms else 0.0,
            "max_step_ms": max(self.step_ms) if self.step_ms else 0.0,
            "avg_render_ms": statistics.mean(self.render_ms) if self.render_ms else 0.0,
            "min_render_ms": min(self.render_ms) if self.render_ms else 0.0,
            "max_render_ms": max(self.render_ms) if self.render_ms else 0.0,
            "avg_nvsmi_mib": statistics.mean(nvsmi_used) if nvsmi_used else -1,
            "peak_nvsmi_mib": max(nvsmi_used) if nvsmi_used else -1,
            "gpu_total_mib": statistics.mean(nvsmi_total) if nvsmi_total else -1,
            "avg_warp_mib": statistics.mean(warp_cur) if warp_cur else -1,
            "peak_warp_mib": max(warp_peak) if warp_peak else -1,
            "max_warp_mib": max(warp_cur) if warp_cur else -1,
        }


def _format_mib(value: float) -> str:
    if value < 0:
        return "N/A"
    if value >= 1024:
        return f"{value / 1024:.2f} GiB"
    return f"{value:.0f} MiB"


# ---------------------------------------------------------------------------
# Report & CSV output
# ---------------------------------------------------------------------------

def _print_report(example_name: str, collector: _BenchCollector,
                  viewer_type: str, warmup: int, baseline_nvsmi: int,
                  substeps: int | None = None) -> None:
    s: dict[str, Any] = collector.summary()

    print(f"\n{'=' * 64}", file=sys.stderr)
    print(f"  LBM Example Benchmark  —  {example_name}", file=sys.stderr)
    print(f"{'=' * 64}", file=sys.stderr)
    print(f"  GPU           : {_get_gpu_name()}", file=sys.stderr)
    print(f"  Viewer        : {viewer_type}", file=sys.stderr)
    if substeps is not None:
        print(f"  Substeps      : {substeps}", file=sys.stderr)
    print(f"  Warm-up frames: {warmup} (excluded from stats)", file=sys.stderr)
    print(f"  Measured frames: {s['frames']}", file=sys.stderr)
    print(f"{'─' * 64}", file=sys.stderr)
    print(f"  Total Frame (step + render)", file=sys.stderr)
    print(f"    Average       : {s['avg_total_ms']:.2f} ms  →  {s['avg_total_fps']:.1f} FPS", file=sys.stderr)
    print(f"    Min / Max     : {s['min_total_ms']:.2f} / {s['max_total_ms']:.2f} ms", file=sys.stderr)
    print(f"    P50 / P99     : {s['p50_total_ms']:.2f} / {s['p99_total_ms']:.2f} ms", file=sys.stderr)
    print(f"    StdDev        : {s['stdev_total_ms']:.2f} ms", file=sys.stderr)
    print(f"{'─' * 64}", file=sys.stderr)
    print(f"  Step only (simulation)", file=sys.stderr)
    print(f"    Average       : {s['avg_step_ms']:.2f} ms  →  {s['avg_step_fps']:.1f} FPS", file=sys.stderr)
    print(f"    Min / Max     : {s['min_step_ms']:.2f} / {s['max_step_ms']:.2f} ms", file=sys.stderr)
    print(f"{'─' * 64}", file=sys.stderr)
    print(f"  Render only (SSFR + OpenGL)", file=sys.stderr)
    print(f"    Average       : {s['avg_render_ms']:.2f} ms", file=sys.stderr)
    print(f"    Min / Max     : {s['min_render_ms']:.2f} / {s['max_render_ms']:.2f} ms", file=sys.stderr)
    print(f"{'─' * 64}", file=sys.stderr)
    print(f"  GPU Memory (nvidia-smi, total process)", file=sys.stderr)
    if baseline_nvsmi > 0:
        print(f"    Baseline      : {_format_mib(float(baseline_nvsmi))}", file=sys.stderr)
    print(f"    Average       : {_format_mib(float(s['avg_nvsmi_mib']))}", file=sys.stderr)
    print(f"    Peak          : {_format_mib(float(s['peak_nvsmi_mib']))}", file=sys.stderr)
    if s["gpu_total_mib"] > 0:
        print(f"    GPU Total     : {_format_mib(float(s['gpu_total_mib']))}", file=sys.stderr)
    print(f"{'─' * 64}", file=sys.stderr)
    print(f"  GPU Memory (Warp mempool)", file=sys.stderr)
    print(f"    Average       : {_format_mib(float(s['avg_warp_mib']))}", file=sys.stderr)
    print(f"    Peak          : {_format_mib(float(s['peak_warp_mib']))}", file=sys.stderr)
    print(f"{'=' * 64}\n", file=sys.stderr)


def _save_per_frame_csv(path: str, collector: _BenchCollector) -> None:
    with open(path, "w", newline="") as f:
        writer: Any = csv.writer(f)
        writer.writerow(["frame", "total_ms", "fps", "step_ms", "render_ms",
                          "nvsmi_mib", "warp_cur_mib", "warp_peak_mib"])
        for i in range(len(collector.total_ms)):
            mem_idx: int = i // collector._mem_interval
            nvsmi_used: int = (
                collector.nvsmi_samples[mem_idx][0]
                if mem_idx < len(collector.nvsmi_samples) else -1
            )
            warp_cur: int = -1
            warp_peak: int = -1
            if mem_idx < len(collector.warp_samples):
                warp_cur, warp_peak = collector.warp_samples[mem_idx]
            t_ms: float = collector.total_ms[i]
            writer.writerow([
                i + 1,
                f"{t_ms:.3f}",
                f"{1000.0 / t_ms:.1f}" if t_ms > 0 else "0",
                f"{collector.step_ms[i]:.3f}",
                f"{collector.render_ms[i]:.3f}",
                nvsmi_used, warp_cur, warp_peak,
            ])


def _save_aggregate_csv(path: str, example_name: str, viewer: str,
                        warmup: int, baseline_nvsmi: int,
                        collector: _BenchCollector,
                        substeps: int | None = None,
                        grid_res: int | None = None) -> None:
    s: dict[str, Any] = collector.summary()
    file_exists: bool = Path(path).exists()
    with open(path, "a", newline="") as f:
        writer: Any = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "example", "viewer", "substeps", "grid_res", "cells", "frames", "warmup",
                "avg_total_ms", "avg_total_fps", "p50_total_ms", "p99_total_ms",
                "avg_step_ms", "avg_render_ms",
                "baseline_nvsmi_mib", "avg_nvsmi_mib", "peak_nvsmi_mib",
                "avg_warp_mib", "peak_warp_mib",
                "gpu_name",
            ])
        n: int = grid_res if grid_res else 0
        writer.writerow([
            example_name, viewer, substeps if substeps is not None else "",
            grid_res if grid_res is not None else "", n * n * n,
            s["frames"], warmup,
            f"{s['avg_total_ms']:.3f}", f"{s['avg_total_fps']:.1f}",
            f"{s['p50_total_ms']:.3f}", f"{s['p99_total_ms']:.3f}",
            f"{s['avg_step_ms']:.3f}", f"{s['avg_render_ms']:.3f}",
            baseline_nvsmi, s["avg_nvsmi_mib"], s["peak_nvsmi_mib"],
            s["avg_warp_mib"], s["peak_warp_mib"],
            _get_gpu_name(),
        ])


# ---------------------------------------------------------------------------
# Monkey-patches (applied *before* the example module is imported)
# ---------------------------------------------------------------------------

def _patch_viewer_null() -> None:
    """Add ``FluidViewerGL``-specific methods to ``ViewerNull`` as no-ops.

    Some LBM examples unconditionally call
    ``viewer.register_post_render_callback()`` in ``__init__``.
    """
    from newton._src.viewer.viewer_null import ViewerNull

    for _method in ("register_post_render_callback", "register_key_press",
                    "register_key_release"):
        if not hasattr(ViewerNull, _method):
            def _noop(self: Any, callback: Any, _name: str = _method) -> None:
                pass
            setattr(ViewerNull, _method, _noop)


def _patch_viewer_gl_auto_exit(max_frames: int) -> None:
    """Make ``ViewerGL`` (and subclasses) exit after *max_frames* frames.

    Patches ``end_frame`` to count frames and ``is_running`` to return
    ``False`` once the limit is reached.  This allows the GL viewer to
    run deterministically without user interaction.
    """
    from newton._src.viewer.viewer_gl import ViewerGL

    _original_end_frame = ViewerGL.end_frame

    def _patched_end_frame(self: Any) -> None:
        _original_end_frame(self)
        self._bench_frame_count: int = getattr(self, "_bench_frame_count", 0) + 1

    _original_is_running = ViewerGL.is_running

    def _patched_is_running(self: Any) -> bool:
        if hasattr(self, "_bench_frame_count") and self._bench_frame_count >= max_frames:
            return False
        return _original_is_running(self)

    ViewerGL.end_frame = _patched_end_frame  # type: ignore[attr-defined]
    ViewerGL.is_running = _patched_is_running  # type: ignore[attr-defined]


def _make_patched_run(warmup_frames: int, collector: _BenchCollector) -> Any:
    """Return a drop-in replacement for ``newton.examples.run()``."""

    def _patched_run(example: Any, args: Any) -> None:
        # Register GUI callback if the example provides one
        if hasattr(example, "gui") and hasattr(example.viewer, "register_ui_callback"):
            example.viewer.register_ui_callback(lambda ui: example.gui(ui), position="side")

        # Auto-unpause: many LBM examples start paused to let the user
        # adjust the camera.  Force unpause so the benchmark runs.
        if hasattr(example.viewer, "_paused"):
            example.viewer._paused = False

        frame_idx: int = 0

        while example.viewer.is_running():
            # ---- step ----
            t0: float = time.perf_counter()
            paused: bool = (
                example.viewer.is_paused()
                if hasattr(example.viewer, "is_paused") else False
            )
            if not paused:
                example.step()
            t1: float = time.perf_counter()

            # ---- render ----
            example.render()
            t2: float = time.perf_counter()

            # Collect (skip warm-up and paused frames)
            if frame_idx >= warmup_frames and not paused:
                collector.record_frame(
                    total_ms=(t2 - t0) * 1000.0,
                    step_ms=(t1 - t0) * 1000.0,
                    render_ms=(t2 - t1) * 1000.0,
                )

            if not paused:
                frame_idx += 1

        example.viewer.close()

    return _patched_run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run an LBM example and measure average FPS + VRAM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--module", type=str, required=True,
        help="Python module path, e.g. wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt",
    )
    parser.add_argument(
        "-f", "--num-frames", type=int, default=300,
        help="Total frames to run (including warm-up).",
    )
    parser.add_argument(
        "-w", "--warmup-frames", type=int, default=30,
        help="Initial frames excluded from statistics.",
    )
    parser.add_argument(
        "--viewer", type=str, default="gl",
        choices=["gl", "null"],
        help="'gl' = real OpenGL + SSFR rendering (headless by default). "
             "'null' = sim-only, no rendering overhead.",
    )
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True,
        help="GL viewer: invisible window (default). --no-headless for visible.",
    )
    parser.add_argument(
        "--csv-per-frame", type=str, default=None,
        help="Write per-frame timing + memory to CSV.",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Append aggregate summary row to CSV.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override Warp device.",
    )
    parser.add_argument(
        "--substeps", type=int, default=None,
        help="Override SIM_SUBSTEPS in the example module (e.g. --substeps 2).",
    )
    parser.add_argument(
        "--grid-res", type=int, default=None,
        help="Override N (grid resolution) in the example module (e.g. --grid-res 64).",
    )
    parser.add_argument(
        "--quiet", action=argparse.BooleanOptionalAction, default=True,
        help="Suppress Warp compilation messages.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args: argparse.Namespace = create_parser().parse_args()

    if args.warmup_frames >= args.num_frames:
        print("ERROR: --warmup-frames must be < --num-frames", file=sys.stderr)
        sys.exit(1)

    # ---- Apply monkey-patches BEFORE anything from the example is imported ----
    _patch_viewer_null()
    if args.viewer == "gl":
        _patch_viewer_gl_auto_exit(args.num_frames)

    import newton.examples
    collector: _BenchCollector = _BenchCollector()
    newton.examples.run = _make_patched_run(args.warmup_frames, collector)  # type: ignore[attr-defined]

    baseline_nvsmi: int = _get_nvsmi_memory()[0]
    example_short: str = args.module.rsplit(".", 1)[-1]

    # Build sys.argv for the target module.  The module's main() parses these.
    saved_argv: list[str] = sys.argv.copy()
    sys.argv = [
        args.module,
        "--viewer", args.viewer,
        "--num-frames", str(args.num_frames),
    ]
    if args.device:
        sys.argv.extend(["--device", args.device])
    if args.quiet:
        sys.argv.append("--quiet")
    if args.viewer == "gl" and args.headless:
        sys.argv.append("--headless")

    substeps_str: str = f"  substeps={args.substeps}" if args.substeps else ""
    print(
        f"Benchmarking {example_short}  "
        f"(viewer={args.viewer}  frames={args.num_frames}  warmup={args.warmup_frames}{substeps_str})",
        file=sys.stderr,
    )

    # Import the example module (without executing its __main__ guard) so we
    # can override SIM_SUBSTEPS before calling main().
    import importlib
    try:
        mod: Any = importlib.import_module(args.module)
    except Exception as exc:
        print(f"ERROR: Failed to import {args.module}: {exc}", file=sys.stderr)
        sys.exit(1)

    overrides: list[str] = []
    if args.substeps is not None:
        if hasattr(mod, "SIM_SUBSTEPS"):
            mod.SIM_SUBSTEPS = args.substeps
            overrides.append(f"SIM_SUBSTEPS={args.substeps}")
        else:
            print(f"  ⚠ Module has no SIM_SUBSTEPS constant; --substeps ignored", file=sys.stderr)
    if args.grid_res is not None:
        if hasattr(mod, "N"):
            mod.N = args.grid_res
            overrides.append(f"N={args.grid_res}")
        else:
            print(f"  ⚠ Module has no N constant; --grid-res ignored", file=sys.stderr)
    if overrides:
        print(f"  → Overrode: {', '.join(overrides)}", file=sys.stderr)

    try:
        mod.main()
    finally:
        sys.argv = saved_argv

    # ---- Report ----
    if len(collector.total_ms) == 0:
        print("ERROR: No frames collected. Try reducing --warmup-frames.", file=sys.stderr)
        sys.exit(1)

    _print_report(example_short, collector, args.viewer, args.warmup_frames, baseline_nvsmi,
                  substeps=args.substeps)

    if args.csv_per_frame is not None:
        _save_per_frame_csv(args.csv_per_frame, collector)
        print(f"  → Per-frame data saved to {args.csv_per_frame}", file=sys.stderr)

    if args.csv is not None:
        _save_aggregate_csv(args.csv, example_short, args.viewer,
                            args.warmup_frames, baseline_nvsmi, collector,
                            substeps=args.substeps, grid_res=args.grid_res)
        print(f"  → Aggregate row appended to {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
