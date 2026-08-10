# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the isolated PLIC reversible-vortex resolution/Courant audit."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicDomain,
    PlicGeometricTransport,
)


@dataclass(frozen=True)
class PlicVortexAuditConfig:
    resolutions: tuple[int, ...] = (32, 48, 64, 96)
    courant_limits: tuple[float, ...] = (0.175, 0.35)
    normalized_half_period: float = 0.546875
    circle_center: tuple[float, float] = (0.36, 0.59)
    circle_radius: float = 0.16
    initial_samples_per_axis: int = 12
    endpoint_tolerance: float = 4.0e-7
    device: str = "cuda:0"


def _sample_circle(config: PlicVortexAuditConfig, resolution: int) -> np.ndarray:
    shape = (resolution + 2, resolution + 2, 3)
    fill = np.zeros(shape, dtype=np.float32)
    center = 1.0 + resolution * np.asarray(config.circle_center)
    radius = resolution * config.circle_radius
    samples = config.initial_samples_per_axis
    offsets = (np.arange(samples, dtype=np.float64) + 0.5) / samples - 0.5
    for i in range(1, resolution + 1):
        for j in range(1, resolution + 1):
            inside = 0
            for dx in offsets:
                for dy in offsets:
                    inside += int(
                        np.linalg.norm(np.asarray((i + dx, j + dy)) - center)
                        <= radius
                    )
            fill[i, j, 1] = inside / (samples * samples)
    return fill


def _vortex_courant(
    shape: tuple[int, int, int], maximum_courant: float
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    nx, ny, nz = shape
    stream = np.zeros((nx, ny), dtype=np.float64)
    for i in range(1, nx):
        x = (i - 1) / (nx - 2)
        for j in range(1, ny):
            y = (j - 1) / (ny - 2)
            stream[i, j] = np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2
    cx = np.zeros((nx + 1, ny, nz), dtype=np.float64)
    cy = np.zeros((nx, ny + 1, nz), dtype=np.float64)
    cz = np.zeros((nx, ny, nz + 1), dtype=np.float64)
    for i in range(1, nx):
        for j in range(1, ny - 1):
            cx[i, j, 1] = stream[i, j + 1] - stream[i, j]
    for i in range(1, nx - 1):
        for j in range(1, ny):
            cy[i, j, 1] = -(stream[i + 1, j] - stream[i, j])
    scale = maximum_courant / max(
        float(np.max(np.abs(cx))), float(np.max(np.abs(cy)))
    )
    fields = tuple(
        field.astype(np.float32) * scale for field in (cx, cy, cz)
    )
    divergence = (
        fields[0][2:nx, 1 : ny - 1, 1]
        - fields[0][1 : nx - 1, 1 : ny - 1, 1]
        + fields[1][1 : nx - 1, 2:ny, 1]
        - fields[1][1 : nx - 1, 1 : ny - 1, 1]
    )
    return fields, float(np.max(np.abs(divergence), initial=0.0))


def _shape_anisotropy(fill: np.ndarray) -> float:
    mass = np.asarray(fill[:, :, 1], dtype=np.float64)
    total = float(np.sum(mass))
    if total <= 0.0:
        raise ValueError("shape anisotropy requires positive liquid volume")
    x, y = np.indices(mass.shape, dtype=np.float64)
    center = np.asarray(
        (np.sum(x * mass) / total, np.sum(y * mass) / total)
    )
    dx = x - center[0]
    dy = y - center[1]
    covariance = np.asarray(
        [
            [np.sum(mass * dx * dx), np.sum(mass * dx * dy)],
            [np.sum(mass * dx * dy), np.sum(mass * dy * dy)],
        ]
    ) / total
    eigenvalues = np.linalg.eigvalsh(covariance)
    return math.sqrt(float(eigenvalues[-1] / max(eigenvalues[0], 1.0e-30)))


def _render_frame(
    path: Path,
    fill: np.ndarray,
    *,
    step: int,
    total_steps: int,
    maximum_courant: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5.4, 5.0), dpi=180)
    image = axis.imshow(
        fill[:, :, 1].T,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        interpolation="nearest",
    )
    axis.contour(
        fill[:, :, 1].T, levels=[0.5], colors="#075985", linewidths=1.0
    )
    phase = "forward" if step <= total_steps // 2 else "reverse"
    axis.set_title(
        f"Isolated PLIC vortex, {phase} {step}/{total_steps}, C={maximum_courant:.3f}"
    )
    axis.set_xlabel("x lattice cell")
    axis.set_ylabel("y lattice cell")
    fig.colorbar(image, ax=axis, label="fill fraction")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_overview(path: Path, snapshots: dict[int, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 10.0), dpi=180)
    for axis, (step, fill) in zip(
        axes.ravel(), snapshots.items(), strict=True
    ):
        axis.imshow(
            fill[:, :, 1].T,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            interpolation="nearest",
        )
        axis.contour(
            fill[:, :, 1].T,
            levels=[0.5],
            colors="#075985",
            linewidths=0.8,
        )
        axis.set_title(f"step {step}")
        axis.set_xlim(0, fill.shape[0] - 1)
        axis.set_ylim(0, fill.shape[1] - 1)
    fig.suptitle("R4 isolated PLIC reversible-vortex evolution", fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _run_case(
    config: PlicVortexAuditConfig,
    *,
    resolution: int,
    courant_limit: float,
    capture: bool,
) -> tuple[dict[str, float | int | bool], dict[int, np.ndarray]]:
    shape = (resolution + 2, resolution + 2, 3)
    domain = PlicDomain.closed_box(shape, device=config.device)
    initial = _sample_circle(config, resolution)
    initial[domain.host] = 0.0
    exact_half_steps = config.normalized_half_period * resolution / courant_limit
    half_steps = math.ceil(exact_half_steps - 1.0e-12)
    maximum_courant = config.normalized_half_period * resolution / half_steps
    fields, maximum_divergence = _vortex_courant(shape, maximum_courant)
    positive = tuple(
        wp.array(field, dtype=float, device=config.device) for field in fields
    )
    negative = tuple(
        wp.array(-field, dtype=float, device=config.device) for field in fields
    )
    transport = PlicGeometricTransport(
        domain, endpoint_tolerance=config.endpoint_tolerance
    )
    fill = wp.array(initial, dtype=float, device=config.device)
    total_steps = 2 * half_steps
    sample_steps = {
        int(value)
        for value in np.rint(np.linspace(0, total_steps, 9)).astype(np.int64)
    }
    snapshots = {0: initial.copy()} if capture else {}
    maximum_volume_drift = 0.0
    maximum_bound_violation = 0.0
    snapped_cell_count = 0
    snapped_volume_delta = 0.0
    midpoint: np.ndarray | None = None
    started = time.perf_counter()
    for step in range(1, total_steps + 1):
        reverse = step > half_steps
        fill, diagnostics = transport.transport(
            fill,
            negative if reverse else positive,
            split_order=(1, 0) if reverse else (0, 1),
        )
        maximum_volume_drift = max(
            maximum_volume_drift, abs(diagnostics.relative_volume_drift)
        )
        snapped_cell_count += diagnostics.endpoint_snapped_cell_count
        snapped_volume_delta += diagnostics.endpoint_snapped_volume_delta
        if step == half_steps:
            midpoint = fill.numpy().copy()
        if capture and step in sample_steps:
            snapshots[step] = fill.numpy().copy()
    wp.synchronize_device(config.device)
    elapsed = time.perf_counter() - started
    final = fill.numpy()
    active_error = np.abs(final[:, :, 1] - initial[:, :, 1])
    initial_volume = float(np.sum(initial[:, :, 1], dtype=np.float64))
    if midpoint is None:
        raise RuntimeError("PLIC vortex midpoint was not captured")
    maximum_bound_violation = max(
        -float(np.min(final)),
        float(np.max(final)) - 1.0,
        0.0,
    )
    return (
        {
            "resolution": resolution,
            "courant_limit": courant_limit,
            "maximum_courant": maximum_courant,
            "half_steps": half_steps,
            "total_steps": total_steps,
            "normalized_half_period": maximum_courant
            * half_steps
            / resolution,
            "maximum_discrete_divergence": maximum_divergence,
            "maximum_step_relative_volume_drift": maximum_volume_drift,
            "relative_endpoint_snap_volume": snapped_volume_delta
            / max(initial_volume, 1.0),
            "endpoint_snapped_cell_count": snapped_cell_count,
            "initial_shape_anisotropy": _shape_anisotropy(initial),
            "midpoint_shape_anisotropy": _shape_anisotropy(midpoint),
            "midpoint_relative_l1_deformation": float(
                np.sum(np.abs(midpoint[:, :, 1] - initial[:, :, 1]))
                / initial_volume
            ),
            "relative_l1_shape_error": float(
                np.sum(active_error, dtype=np.float64) / initial_volume
            ),
            "maximum_absolute_shape_error": float(np.max(active_error)),
            "maximum_bound_violation": maximum_bound_violation,
            "elapsed_seconds": elapsed,
            "finite": bool(np.isfinite(final).all()),
        },
        snapshots,
    )


def _observed_orders(
    cases: list[dict[str, float | int | bool]], courant_limit: float
) -> list[float]:
    family = sorted(
        (
            case
            for case in cases
            if float(case["courant_limit"]) == courant_limit
        ),
        key=lambda case: int(case["resolution"]),
    )
    return [
        math.log(
            float(lower["relative_l1_shape_error"])
            / float(upper["relative_l1_shape_error"])
        )
        / math.log(int(upper["resolution"]) / int(lower["resolution"]))
        for lower, upper in itertools.pairwise(family)
    ]


def _fitted_order(
    cases: list[dict[str, float | int | bool]], courant_limit: float
) -> float:
    family = [
        case
        for case in cases
        if float(case["courant_limit"]) == courant_limit
    ]
    log_resolution = np.log(
        np.asarray([int(case["resolution"]) for case in family], dtype=np.float64)
    )
    log_error = np.log(
        np.asarray(
            [float(case["relative_l1_shape_error"]) for case in family],
            dtype=np.float64,
        )
    )
    slope = np.polyfit(log_resolution, log_error, 1)[0]
    return -float(slope)


def run(config: PlicVortexAuditConfig, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    cases: list[dict[str, float | int | bool]] = []
    representative_snapshots: dict[int, np.ndarray] = {}
    representative_key = (max(config.resolutions), max(config.courant_limits))
    for courant_limit in config.courant_limits:
        for resolution in config.resolutions:
            capture = (resolution, courant_limit) == representative_key
            case, snapshots = _run_case(
                config,
                resolution=resolution,
                courant_limit=courant_limit,
                capture=capture,
            )
            cases.append(case)
            if capture:
                representative_snapshots = snapshots

    orders = {
        f"{limit:.3f}": _observed_orders(cases, limit)
        for limit in config.courant_limits
    }
    fitted_orders = {
        f"{limit:.3f}": _fitted_order(cases, limit)
        for limit in config.courant_limits
    }
    monotonic = all(
        all(order > 0.0 for order in family) for family in orders.values()
    )
    minimum_fitted_order = min(fitted_orders.values())
    hard_gates = {
        "all_finite": all(bool(case["finite"]) for case in cases),
        "all_bounded": all(
            float(case["maximum_bound_violation"]) <= 1.0e-7 for case in cases
        ),
        "volume_drift_within_4e-6": all(
            float(case["maximum_step_relative_volume_drift"]) <= 4.0e-6
            for case in cases
        ),
        "midpoint_shape_anisotropy_at_least_1.25": all(
            float(case["midpoint_shape_anisotropy"]) >= 1.25
            for case in cases
        ),
        "shape_error_monotonically_decreases": monotonic,
        "minimum_fitted_order_at_least_0.5": minimum_fitted_order >= 0.5,
    }
    frame_paths: list[str] = []
    representative = next(
        case
        for case in cases
        if (
            int(case["resolution"]),
            float(case["courant_limit"]),
        )
        == representative_key
    )
    for step, snapshot in representative_snapshots.items():
        relative = f"frames/step-{step:04d}.png"
        _render_frame(
            output_dir / relative,
            snapshot,
            step=step,
            total_steps=int(representative["total_steps"]),
            maximum_courant=float(representative["maximum_courant"]),
        )
        frame_paths.append(relative)
    overview = "plic-vortex-convergence-overview.png"
    _render_overview(output_dir / overview, representative_snapshots)
    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    manifest: dict[str, object] = {
        "scene": "r4-isolated-plic-reversible-vortex-convergence",
        "config": asdict(config),
        "isolation": {
            "velocity_source": "analytic-discrete-divergence-free",
            "home_moments_consumed": False,
            "courant_projection_enabled": False,
            "topology_enabled": False,
            "mass_momentum_coupling_enabled": False,
            "infrastructure_reused": [],
        },
        "cases": cases,
        "pairwise_observed_l1_orders": orders,
        "fitted_l1_orders": fitted_orders,
        "minimum_fitted_l1_order": minimum_fitted_order,
        "hard_gates": hard_gates,
        "passed": all(hard_gates.values()),
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": config.device,
        },
        "metrics": metrics_path.name,
        "frames": frame_paths,
        "overview": overview,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not manifest["passed"]:
        raise RuntimeError(f"PLIC convergence audit failed: {hard_gates}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = PlicVortexAuditConfig(device=args.device)
    print(json.dumps(run(config, args.output), indent=2))


if __name__ == "__main__":
    main()
