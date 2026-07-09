from __future__ import annotations

import csv
from pathlib import Path

from scripts.bench import plot_bench_results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_example_results_reads_bench_lbm_run_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "aggregate.csv"
    _write_csv(
        csv_path,
        [
            {
                "example": "fluid_grid_lbm_dambreak_trt",
                "avg_total_fps": 65.6,
                "peak_warp_mib": 600,
                "peak_nvsmi_mib": 2920,
            },
            {
                "example": "fluid_grid_lbm_droplet_fall_trt",
                "avg_total_fps": 79.8,
                "peak_warp_mib": 601,
                "peak_nvsmi_mib": 2872,
            },
        ],
    )

    results = plot_bench_results.load_example_results(csv_path)

    assert results.labels == ["溃坝 TRT", "液滴下落 TRT"]
    assert results.fps == [65.6, 79.8]
    assert results.warp_mib == [600.0, 601.0]
    assert results.nvsmi_peak_mib == [2920.0, 2872.0]


def test_load_scale_results_reads_bench_lbm_csv_sorted_by_resolution(tmp_path: Path) -> None:
    csv_path = tmp_path / "scale.csv"
    _write_csv(
        csv_path,
        [
            {
                "grid_res": 128,
                "avg_fps": 65.5,
                "peak_warp_mem_mib": 600,
                "peak_nvsmi_mem_mib": 2877,
            },
            {
                "grid_res": 64,
                "avg_fps": 324.8,
                "peak_warp_mem_mib": 75,
                "peak_nvsmi_mem_mib": 2351,
            },
        ],
    )

    results = plot_bench_results.load_scale_results(csv_path)

    assert results.resolutions == [64, 128]
    assert results.fps == [324.8, 65.5]
    assert results.warp_mib == [75.0, 600.0]
    assert results.nvsmi_peak_mib == [2351.0, 2877.0]
