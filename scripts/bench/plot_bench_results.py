# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot LBM benchmark results from aggregate CSV files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Help Windows and Linux hosts render Chinese labels when a matching font exists.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR: Path = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR: Path = SCRIPT_DIR / "bench_results"
DEFAULT_EXAMPLE_CSV: Path = DEFAULT_OUTPUT_DIR / "substeps2" / "aggregate.csv"
DEFAULT_SCALE_CSV: Path = DEFAULT_OUTPUT_DIR / "dambreak_scale" / "aggregate.csv"

EXAMPLE_LABELS: dict[str, str] = {
    "fluid_grid_lbm_dambreak_trt": "溃坝 TRT",
    "fluid_grid_lbm_droplet_fall_trt": "液滴下落 TRT",
    "fluid_grid_lbm_dambreak_two_spheres": "双球耦合",
    "fluid_grid_lbm_pool_drop": "刚体入水",
    "fluid_grid_lbm_droplet_coalescence_trt": "液滴合并 TRT",
    "fluid_grid_lbm_spinodal_trt": "Spinodal TRT",
}

BAR_COLORS: list[str] = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
LINE_COLOR: str = "#4C72B0"


@dataclass(frozen=True)
class ExampleResults:
    labels: list[str]
    fps: list[float]
    warp_mib: list[float]
    nvsmi_peak_mib: list[float]


@dataclass(frozen=True)
class ScaleResults:
    resolutions: list[int]
    fps: list[float]
    warp_mib: list[float]
    nvsmi_peak_mib: list[float]


def _fmt_mem(val_mib: float) -> str:
    if val_mib >= 1024:
        return f"{val_mib / 1024:.1f} GiB"
    return f"{val_mib:.0f} MiB"


def _require_value(row: dict[str, str], names: tuple[str, ...], source: Path) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value != "":
            return value
    raise ValueError(f"{source} is missing one of these columns: {', '.join(names)}")


def _float_value(row: dict[str, str], names: tuple[str, ...], source: Path) -> float:
    return float(_require_value(row, names, source))


def _example_label(name: str) -> str:
    return EXAMPLE_LABELS.get(name, name.replace("fluid_grid_lbm_", "").replace("_", " "))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_example_results(path: Path) -> ExampleResults:
    """Read aggregate rows from ``bench_lbm_run.py`` for example comparison."""
    rows = _read_csv_rows(path)
    labels: list[str] = []
    fps: list[float] = []
    warp_mib: list[float] = []
    nvsmi_peak_mib: list[float] = []

    for row in rows:
        example = _require_value(row, ("example",), path)
        labels.append(_example_label(example))
        fps.append(_float_value(row, ("avg_total_fps", "avg_fps"), path))
        warp_mib.append(_float_value(row, ("peak_warp_mib", "peak_warp_mem_mib"), path))
        nvsmi_peak_mib.append(_float_value(row, ("peak_nvsmi_mib", "peak_nvsmi_mem_mib"), path))

    if not labels:
        raise ValueError(f"No benchmark rows found in {path}")
    return ExampleResults(labels, fps, warp_mib, nvsmi_peak_mib)


def load_scale_results(path: Path) -> ScaleResults:
    """Read aggregate rows from ``bench_lbm.py`` or ``bench_lbm_run.py``."""
    rows = _read_csv_rows(path)
    parsed: list[tuple[int, float, float, float]] = []
    for row in rows:
        grid_res = int(_float_value(row, ("grid_res",), path))
        parsed.append((
            grid_res,
            _float_value(row, ("avg_fps", "avg_total_fps"), path),
            _float_value(row, ("peak_warp_mem_mib", "peak_warp_mib"), path),
            _float_value(row, ("peak_nvsmi_mem_mib", "peak_nvsmi_mib"), path),
        ))

    if not parsed:
        raise ValueError(f"No benchmark rows found in {path}")

    parsed.sort(key=lambda item: item[0])
    return ScaleResults(
        resolutions=[item[0] for item in parsed],
        fps=[item[1] for item in parsed],
        warp_mib=[item[2] for item in parsed],
        nvsmi_peak_mib=[item[3] for item in parsed],
    )


def plot_example_comparison(results: ExampleResults, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle("LBM 示例性能对比", fontsize=13, fontweight="bold", y=1.02)

    x: np.ndarray = np.arange(len(results.labels))
    colors = [BAR_COLORS[i % len(BAR_COLORS)] for i in range(len(results.labels))]

    ax: plt.Axes = axes[0]
    bars = ax.bar(x, results.fps, color=colors, edgecolor="white", linewidth=0.8, width=0.55)
    ax.set_title("平均帧率 (FPS)", fontsize=12, fontweight="bold")
    ax.set_ylabel("FPS")
    ax.set_xticks(x)
    ax.set_xticklabels(results.labels, fontsize=10)
    ax.set_ylim(0, max(results.fps) * 1.2)
    for bar, val in zip(bars, results.fps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, results.warp_mib, color=colors, edgecolor="white", linewidth=0.8, width=0.55)
    ax.set_title("Warp 显存占用", fontsize=12, fontweight="bold")
    ax.set_ylabel("显存 (MiB)")
    ax.set_xticks(x)
    ax.set_xticklabels(results.labels, fontsize=10)
    ax.set_ylim(0, max(results.warp_mib) * 1.3)
    for bar, val in zip(bars, results.warp_mib):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                _fmt_mem(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    bars = ax.bar(x, results.nvsmi_peak_mib, color=colors, edgecolor="white", linewidth=0.8, width=0.55)
    ax.set_title("进程 GPU 显存峰值 (nvidia-smi)", fontsize=12, fontweight="bold")
    ax.set_ylabel("显存 (MiB)")
    ax.set_xticks(x)
    ax.set_xticklabels(results.labels, fontsize=10)
    ax.set_ylim(0, max(results.nvsmi_peak_mib) * 1.15)
    for bar, val in zip(bars, results.nvsmi_peak_mib):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                _fmt_mem(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "example_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_resolution_scaling(results: ScaleResults, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    fig.suptitle("溃坝 TRT 分辨率缩放性能", fontsize=13, fontweight="bold", y=1.02)

    x_labels: list[str] = [f"{r}³" for r in results.resolutions]

    ax: plt.Axes = axes[0]
    ax.plot(results.resolutions, results.fps, marker="o", color=LINE_COLOR,
            linewidth=2.2, markersize=9, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor=LINE_COLOR)
    ax.set_title("平均帧率 (FPS)", fontsize=12, fontweight="bold")
    ax.set_xlabel("网格分辨率")
    ax.set_ylabel("FPS")
    ax.set_xticks(results.resolutions)
    ax.set_xticklabels(x_labels, fontsize=10)
    for r, v in zip(results.resolutions, results.fps):
        offset = 14 if v > 100 else 12
        ax.annotate(f"{v:.1f}", (r, v), textcoords="offset points",
                    xytext=(0, offset), ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results.resolutions, results.warp_mib, marker="s", color="#55A868",
            linewidth=2.2, markersize=9, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor="#55A868")
    ax.set_title("Warp 显存占用", fontsize=12, fontweight="bold")
    ax.set_xlabel("网格分辨率")
    ax.set_ylabel("显存 (MiB)")
    ax.set_xticks(results.resolutions)
    ax.set_xticklabels(x_labels, fontsize=10)
    for r, v in zip(results.resolutions, results.warp_mib):
        ax.annotate(_fmt_mem(v), (r, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(results.resolutions, results.nvsmi_peak_mib, marker="^", color="#C44E52",
            linewidth=2.2, markersize=9, markerfacecolor="white",
            markeredgewidth=2, markeredgecolor="#C44E52")
    ax.set_title("进程 GPU 显存峰值 (nvidia-smi)", fontsize=12, fontweight="bold")
    ax.set_xlabel("网格分辨率")
    ax.set_ylabel("显存 (MiB)")
    ax.set_xticks(results.resolutions)
    ax.set_xticklabels(x_labels, fontsize=10)
    for r, v in zip(results.resolutions, results.nvsmi_peak_mib):
        ax.annotate(_fmt_mem(v), (r, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolution_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot LBM benchmark figures from aggregate CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--example-csv", type=Path, default=DEFAULT_EXAMPLE_CSV)
    parser.add_argument("--scale-csv", type=Path, default=DEFAULT_SCALE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    example_results = load_example_results(args.example_csv)
    scale_results = load_scale_results(args.scale_csv)
    example_path = plot_example_comparison(example_results, args.output_dir)
    scale_path = plot_resolution_scaling(scale_results, args.output_dir)
    print(f"  -> Saved {example_path}")
    print(f"  -> Saved {scale_path}")
    print("Done.")


if __name__ == "__main__":
    main()
