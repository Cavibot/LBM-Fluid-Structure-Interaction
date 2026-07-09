# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless LBM performance benchmark.

Measures average frame time (→ FPS) and GPU memory usage for a pure-LBM
dam-break scenario across a configurable number of frames.

Usage::

    # Quick test: 200 frames at default 128³
    uv run python scripts/bench/bench_lbm.py

    # 256³ grid, 500 frames
    uv run python scripts/bench/bench_lbm.py -n 256 -f 500

    # 64³, 3 substeps, 100 warm-up + 300 measured frames
    uv run python scripts/bench/bench_lbm.py -n 64 -s 3 -w 100 -f 300

    # CSV output for later analysis
    uv run python scripts/bench/bench_lbm.py -n 128 -f 300 --csv results.csv

    # Scan multiple resolutions
    uv run python scripts/bench/bench_lbm.py --scan 64,96,128,160,192 -f 200
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
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel, LbmState

# ---------------------------------------------------------------------------
# Default physics parameters (matching fluid_grid_lbm_dambreak_trt.py)
# ---------------------------------------------------------------------------
DEFAULT_N: int = 128
DEFAULT_DH: float = 0.02
DEFAULT_TAU: float = 0.55
DEFAULT_LAMBDA_TRT: float = 0.03
DEFAULT_G_SC: float = -5.0
DEFAULT_PSI_TYPE: int = 1
DEFAULT_PSI_REF: float = 1.0
DEFAULT_GRAVITY: float = -0.005
DEFAULT_OMEGA_REG: float = 0.5
DEFAULT_SC_BOUNDARY_PSI: float = -1.0

DEFAULT_WARMUP_FRAMES: int = 60
DEFAULT_BENCH_FRAMES: int = 300
DEFAULT_SUBSTEPS: int = 3
DEFAULT_FRAME_DT: float = 1.0 / 60.0

RHO_WATER: float = 1.8
RHO_AIR: float = 0.1
DAM_X_FRAC: float = 0.25


# ---------------------------------------------------------------------------
# D3Q19 equilibrium initialisation kernel
# ---------------------------------------------------------------------------
@wp.kernel
def _init_dambreak(
    f: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    dam_x: int,
    rho_w: float,
    rho_a: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    rho = rho_w if i < dam_x else rho_a
    noise = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001) * wp.cos(
        float(i * 419 + j * 233 + k * 577 + seed) * 0.0013
    )
    rho = wp.max(rho + noise * 0.005 * rho, 0.01)
    density[i, j, k] = rho
    f[0 * stride + idx] = (1.0 / 3.0) * rho
    f[1 * stride + idx] = (1.0 / 18.0) * rho
    f[2 * stride + idx] = (1.0 / 18.0) * rho
    f[3 * stride + idx] = (1.0 / 18.0) * rho
    f[4 * stride + idx] = (1.0 / 18.0) * rho
    f[5 * stride + idx] = (1.0 / 18.0) * rho
    f[6 * stride + idx] = (1.0 / 18.0) * rho
    f[7 * stride + idx] = (1.0 / 36.0) * rho
    f[8 * stride + idx] = (1.0 / 36.0) * rho
    f[9 * stride + idx] = (1.0 / 36.0) * rho
    f[10 * stride + idx] = (1.0 / 36.0) * rho
    f[11 * stride + idx] = (1.0 / 36.0) * rho
    f[12 * stride + idx] = (1.0 / 36.0) * rho
    f[13 * stride + idx] = (1.0 / 36.0) * rho
    f[14 * stride + idx] = (1.0 / 36.0) * rho
    f[15 * stride + idx] = (1.0 / 36.0) * rho
    f[16 * stride + idx] = (1.0 / 36.0) * rho
    f[17 * stride + idx] = (1.0 / 36.0) * rho
    f[18 * stride + idx] = (1.0 / 36.0) * rho


# ---------------------------------------------------------------------------
# GPU memory query helpers
# ---------------------------------------------------------------------------
def _get_warp_mempool_used(device: str = "cuda:0") -> int:
    """Return current Warp mempool usage in MiB."""
    return int(wp.get_mempool_used_mem_current(device)) // (1024 * 1024)


def _get_warp_mempool_peak(device: str = "cuda:0") -> int:
    """Return peak Warp mempool usage in MiB."""
    return int(wp.get_mempool_used_mem_high(device)) // (1024 * 1024)


def _get_nvsmi_memory() -> tuple[int, int]:
    """Return (used_mib, total_mib) from nvidia-smi, or (-1, -1) on failure."""
    try:
        raw: str = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        parts: list[str] = raw.split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return -1, -1


def _get_gpu_name() -> str:
    """Return the GPU product name, or 'unknown'."""
    try:
        raw: str = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        return raw
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
class LbmBenchmark:
    """Headless pure-LBM dam-break benchmark."""

    def __init__(
        self,
        grid_res: int = DEFAULT_N,
        dh: float = DEFAULT_DH,
        tau: float = DEFAULT_TAU,
        lambda_trt: float = DEFAULT_LAMBDA_TRT,
        g_sc: float = DEFAULT_G_SC,
        psi_type: int = DEFAULT_PSI_TYPE,
        psi_ref: float = DEFAULT_PSI_REF,
        gravity: float = DEFAULT_GRAVITY,
        omega_reg: float = DEFAULT_OMEGA_REG,
        sc_boundary_psi: float = DEFAULT_SC_BOUNDARY_PSI,
        substeps: int = DEFAULT_SUBSTEPS,
        frame_dt: float = DEFAULT_FRAME_DT,
        device: str | None = None,
    ) -> None:
        self.grid_res: int = grid_res
        self.substeps: int = substeps
        self.frame_dt: float = frame_dt
        self.sim_dt: float = frame_dt / float(substeps)

        # ---- Build model --------------------------------------------------
        self.model: LbmModel = LbmModel(
            fluid_grid_res=(grid_res, grid_res, grid_res),
            fluid_grid_cell_size=dh,
            tau=tau,
            G=g_sc,
            sc_boundary_psi=sc_boundary_psi,
            psi_type=psi_type,
            psi_ref=psi_ref,
            lambda_trt=lambda_trt,
            use_regularization=True,
            omega_reg=omega_reg,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=gravity,
        )
        if device is not None:
            self.model._device = wp.get_device(device)

        # ---- Create domain & state ----------------------------------------
        self.domain: LbmDomain = LbmDomain(self.model)
        self.domain.create_state()

        # ---- Initialise fluid ---------------------------------------------
        n: int = grid_res
        stride: int = n * n * n
        dam_x: int = int(float(n) * DAM_X_FRAC)
        state: LbmState = self.domain.state
        wp.launch(
            _init_dambreak,
            dim=(n, n, n),
            inputs=[state.f, state.density, dam_x, RHO_WATER, RHO_AIR, 42, n, n, n, stride],
            device=self.model._device,
        )
        state_out: LbmState = self.domain._state_out
        for field_name in ("f", "density", "velocity_x", "velocity_y", "velocity_z",
                           "solid_phi", "solid_body_id"):
            wp.copy(getattr(state_out, field_name), getattr(state, field_name))
        wp.synchronize_device(self.model._device)

        # ---- Gravity ramp (gentle) ----------------------------------------
        target_gz: float = float(self.model.gravity_z)
        self.model.gravity_z = 0.0
        ramp_steps: int = max(30, min(60, grid_res // 4))
        for _step in range(ramp_steps):
            self.model.gravity_z = target_gz * float(_step + 1) / float(ramp_steps)
            for _sub in range(self.substeps):
                self.domain.step(self.sim_dt)
        self.model.gravity_z = target_gz
        wp.synchronize_device(self.model._device)

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance one frame (substeps × domain.step)."""
        for _ in range(self.substeps):
            self.domain.step(self.sim_dt)

    # ------------------------------------------------------------------
    def run_benchmark(
        self,
        warmup_frames: int = DEFAULT_WARMUP_FRAMES,
        bench_frames: int = DEFAULT_BENCH_FRAMES,
        sample_interval: int = 1,
    ) -> dict[str, Any]:
        """Run the benchmark and return a dictionary of results.

        Parameters
        ----------
        warmup_frames:
            Number of frames to run before starting measurements (JIT warm-up).
        bench_frames:
            Number of frames to measure.
        sample_interval:
            Sample memory every N frames (1 = every frame).

        Returns
        -------
        dict with keys:
            - frame_times_ms: list of per-frame wall-clock times (ms)
            - mem_samples_mib: list of (warp_used, warp_peak, nvsmi_used) tuples
            - warmup_frames, bench_frames, grid_res, substeps
        """
        device_str: str = str(self.model._device)
        frame_times_ms: list[float] = []
        mem_samples: list[tuple[int, int, int]] = []  # (warp_used, warp_peak, nvsmi_used)

        # ---- Warm-up phase ------------------------------------------------
        for i in range(warmup_frames):
            self.step()
        wp.synchronize_device(self.model._device)

        # ---- Benchmark phase ----------------------------------------------
        for i in range(bench_frames):
            t0: float = time.perf_counter()
            self.step()
            wp.synchronize_device(self.model._device)
            elapsed_ms: float = (time.perf_counter() - t0) * 1000.0
            frame_times_ms.append(elapsed_ms)

            if i % sample_interval == 0:
                mem_samples.append((
                    _get_warp_mempool_used(device_str),
                    _get_warp_mempool_peak(device_str),
                    _get_nvsmi_memory()[0],
                ))

        # ---- Compute statistics -------------------------------------------
        avg_ms: float = statistics.mean(frame_times_ms)
        min_ms: float = min(frame_times_ms)
        max_ms: float = max(frame_times_ms)
        p50_ms: float = float(np.percentile(frame_times_ms, 50))
        p99_ms: float = float(np.percentile(frame_times_ms, 99))
        stdev_ms: float = statistics.stdev(frame_times_ms) if len(frame_times_ms) > 1 else 0.0

        warp_used_vals: list[int] = [m[0] for m in mem_samples]
        warp_peak_vals: list[int] = [m[1] for m in mem_samples]
        nvsmi_vals: list[int] = [m[2] for m in mem_samples]

        return {
            "grid_res": self.grid_res,
            "substeps": self.substeps,
            "warmup_frames": warmup_frames,
            "bench_frames": bench_frames,
            "frame_times_ms": frame_times_ms,
            "mem_samples": mem_samples,
            # Frame time statistics
            "avg_frame_ms": avg_ms,
            "avg_fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
            "min_frame_ms": min_ms,
            "max_frame_ms": max_ms,
            "p50_frame_ms": p50_ms,
            "p99_frame_ms": p99_ms,
            "stdev_frame_ms": stdev_ms,
            # Memory statistics (Warp mempool)
            "avg_warp_mem_mib": statistics.mean(warp_used_vals) if warp_used_vals else -1,
            "peak_warp_mem_mib": max(warp_peak_vals) if warp_peak_vals else -1,
            "max_warp_mem_mib": max(warp_used_vals) if warp_used_vals else -1,
            # Memory statistics (nvidia-smi, total GPU usage)
            "avg_nvsmi_mem_mib": statistics.mean(nvsmi_vals) if nvsmi_vals else -1,
            "peak_nvsmi_mem_mib": max(nvsmi_vals) if nvsmi_vals else -1,
        }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def _format_mib(value: float) -> str:
    if value < 0:
        return "N/A"
    if value >= 1024:
        return f"{value / 1024:.2f} GiB"
    return f"{value:.0f} MiB"


def print_report(results: dict[str, Any]) -> None:
    """Pretty-print benchmark results to stderr."""
    n: int = results["grid_res"]
    total_cells: int = n * n * n

    print(f"\n{'=' * 68}", file=sys.stderr)
    print(f"  LBM Benchmark Results", file=sys.stderr)
    print(f"{'=' * 68}", file=sys.stderr)
    print(f"  GPU          : {_get_gpu_name()}", file=sys.stderr)
    print(f"  Grid         : {n}³ = {total_cells:,} cells", file=sys.stderr)
    print(f"  Substeps     : {results['substeps']}", file=sys.stderr)
    print(f"  Warm-up      : {results['warmup_frames']} frames", file=sys.stderr)
    print(f"  Measured     : {results['bench_frames']} frames", file=sys.stderr)
    print(f"{'─' * 68}", file=sys.stderr)
    print(f"  Frame Time", file=sys.stderr)
    print(f"  ──────────", file=sys.stderr)
    print(f"    Average     : {results['avg_frame_ms']:.2f} ms  →  {results['avg_fps']:.1f} FPS", file=sys.stderr)
    print(f"    Min / Max   : {results['min_frame_ms']:.2f} / {results['max_frame_ms']:.2f} ms", file=sys.stderr)
    print(f"    P50 / P99   : {results['p50_frame_ms']:.2f} / {results['p99_frame_ms']:.2f} ms", file=sys.stderr)
    print(f"    StdDev      : {results['stdev_frame_ms']:.2f} ms", file=sys.stderr)
    print(f"{'─' * 68}", file=sys.stderr)
    print(f"  GPU Memory (Warp mempool)", file=sys.stderr)
    print(f"  ─────────────────────────", file=sys.stderr)
    print(f"    Average     : {_format_mib(results['avg_warp_mem_mib'])}", file=sys.stderr)
    print(f"    Peak        : {_format_mib(results['peak_warp_mem_mib'])}", file=sys.stderr)
    print(f"{'─' * 68}", file=sys.stderr)
    print(f"  GPU Memory (nvidia-smi, total process)", file=sys.stderr)
    print(f"  ────────────────────────────────────────", file=sys.stderr)
    print(f"    Average     : {_format_mib(results['avg_nvsmi_mem_mib'])}", file=sys.stderr)
    print(f"    Peak        : {_format_mib(results['peak_nvsmi_mem_mib'])}", file=sys.stderr)

    gpu_total_mib: int = _get_nvsmi_memory()[1]
    if gpu_total_mib > 0:
        print(f"    GPU Total   : {_format_mib(float(gpu_total_mib))}", file=sys.stderr)
    print(f"{'=' * 68}\n", file=sys.stderr)


def save_csv(results: dict[str, Any], path: str) -> None:
    """Append one row of aggregate results to a CSV file."""
    file_exists: bool = Path(path).exists()
    with open(path, "a", newline="") as f:
        writer: Any = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "grid_res", "cells", "substeps", "warmup_frames", "bench_frames",
                "avg_frame_ms", "avg_fps", "min_frame_ms", "max_frame_ms",
                "p50_frame_ms", "p99_frame_ms", "stdev_frame_ms",
                "avg_warp_mem_mib", "peak_warp_mem_mib",
                "avg_nvsmi_mem_mib", "peak_nvsmi_mem_mib",
                "gpu_name",
            ])
        writer.writerow([
            results["grid_res"],
            results["grid_res"] ** 3,
            results["substeps"],
            results["warmup_frames"],
            results["bench_frames"],
            f"{results['avg_frame_ms']:.3f}",
            f"{results['avg_fps']:.1f}",
            f"{results['min_frame_ms']:.3f}",
            f"{results['max_frame_ms']:.3f}",
            f"{results['p50_frame_ms']:.3f}",
            f"{results['p99_frame_ms']:.3f}",
            f"{results['stdev_frame_ms']:.3f}",
            results["avg_warp_mem_mib"],
            results["peak_warp_mem_mib"],
            results["avg_nvsmi_mem_mib"],
            results["peak_nvsmi_mem_mib"],
            _get_gpu_name(),
        ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def create_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Headless LBM performance benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--grid-res", type=int, default=DEFAULT_N,
        help="Grid resolution (cubic: n³ cells).",
    )
    parser.add_argument(
        "--scan", type=str, default=None,
        help="Comma-separated list of resolutions to scan, e.g. '64,96,128,160,192'.",
    )
    parser.add_argument(
        "-s", "--substeps", type=int, default=DEFAULT_SUBSTEPS,
        help="Simulation substeps per frame.",
    )
    parser.add_argument(
        "-w", "--warmup-frames", type=int, default=DEFAULT_WARMUP_FRAMES,
        help="Number of warm-up frames (JIT compilation).",
    )
    parser.add_argument(
        "-f", "--bench-frames", type=int, default=DEFAULT_BENCH_FRAMES,
        help="Number of measured frames.",
    )
    parser.add_argument(
        "--dh", type=float, default=DEFAULT_DH,
        help="Grid cell size.",
    )
    parser.add_argument(
        "--tau", type=float, default=DEFAULT_TAU,
        help="BGK relaxation time.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Warp device string (default: auto-detect).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Append aggregate results to a CSV file.",
    )
    parser.add_argument(
        "--sample-mem-interval", type=int, default=1,
        help="Sample GPU memory every N frames.",
    )
    parser.add_argument(
        "--csv-per-frame", type=str, default=None,
        help="Write per-frame timing data to a CSV file (for histogram/plot).",
    )
    return parser


def main() -> None:
    args: argparse.Namespace = create_parser().parse_args()

    resolutions: list[int]
    if args.scan is not None:
        resolutions = [int(x.strip()) for x in args.scan.split(",")]
    else:
        resolutions = [args.grid_res]

    if len(resolutions) > 1:
        print(f"Scanning {len(resolutions)} resolutions: {resolutions}", file=sys.stderr)

    all_results: list[dict[str, Any]] = []
    for n in resolutions:
        print(f"\n--- Benchmarking {n}³ grid ({n * n * n:,} cells) ---", file=sys.stderr)
        bench: LbmBenchmark = LbmBenchmark(
            grid_res=n,
            dh=args.dh,
            tau=args.tau,
            substeps=args.substeps,
            device=args.device,
        )
        results: dict[str, Any] = bench.run_benchmark(
            warmup_frames=args.warmup_frames,
            bench_frames=args.bench_frames,
            sample_interval=args.sample_mem_interval,
        )
        print_report(results)

        if args.csv is not None:
            save_csv(results, args.csv)
            print(f"  → Appended aggregate row to {args.csv}", file=sys.stderr)

        if args.csv_per_frame is not None:
            per_frame_path: str = args.csv_per_frame
            if len(resolutions) > 1:
                stem: str = str(Path(per_frame_path).with_suffix(""))
                ext: str = str(Path(per_frame_path).suffix)
                per_frame_path = f"{stem}_n{n}{ext}"
            with open(per_frame_path, "w", newline="") as f:
                writer: Any = csv.writer(f)
                writer.writerow(["frame", "time_ms", "fps", "warp_mem_mib", "warp_peak_mib", "nvsmi_mem_mib"])
                for i, t_ms in enumerate(results["frame_times_ms"]):
                    mem_idx: int = min(i // args.sample_mem_interval, len(results["mem_samples"]) - 1)
                    warp_used, warp_peak, nvsmi_used = results["mem_samples"][mem_idx]
                    writer.writerow([
                        i + 1,
                        f"{t_ms:.3f}",
                        f"{1000.0 / t_ms:.1f}" if t_ms > 0 else "0",
                        warp_used,
                        warp_peak,
                        nvsmi_used,
                    ])
            print(f"  → Wrote per-frame data to {per_frame_path}", file=sys.stderr)

        all_results.append(results)

    # ---- Scan summary ----------------------------------------------------
    if len(all_results) > 1:
        print(f"\n{'=' * 68}", file=sys.stderr)
        print(f"  Scan Summary", file=sys.stderr)
        print(f"{'=' * 68}", file=sys.stderr)
        print(f"  {'N':>6s}  {'Cells':>10s}  {'Avg ms':>8s}  {'Avg FPS':>8s}  "
              f"{'P99 ms':>8s}  {'Warp Mem':>9s}  {'nvsmi Mem':>9s}", file=sys.stderr)
        print(f"  {'─' * 6}  {'─' * 10}  {'─' * 8}  {'─' * 8}  "
              f"{'─' * 8}  {'─' * 9}  {'─' * 9}", file=sys.stderr)
        for r in all_results:
            n_cells: int = r["grid_res"] ** 3
            print(
                f"  {r['grid_res']:>6d}  {n_cells:>10,d}  "
                f"{r['avg_frame_ms']:>7.2f}  {r['avg_fps']:>7.1f}  "
                f"{r['p99_frame_ms']:>7.2f}  "
                f"{_format_mib(float(r['peak_warp_mem_mib'])):>9s}  "
                f"{_format_mib(float(r['peak_nvsmi_mem_mib'])):>9s}",
                file=sys.stderr,
            )
        print(f"{'=' * 68}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
