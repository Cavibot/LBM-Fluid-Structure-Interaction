# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the R4.2 reversible-vortex PLIC transport audit."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.core import HomeCoreModel
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_fsl import FslWallMask
from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicGeometricTransport,
)


def _sample_circle(shape: tuple[int, int, int], samples: int = 16) -> np.ndarray:
    fill = np.zeros(shape, dtype=np.float32)
    center = np.asarray((shape[0] / 2.0 - 5.0, shape[1] / 2.0 + 3.0))
    offsets = (np.arange(samples, dtype=np.float64) + 0.5) / samples - 0.5
    for i in range(shape[0]):
        for j in range(shape[1]):
            inside = 0
            for dx in offsets:
                for dy in offsets:
                    inside += int(
                        np.linalg.norm(np.asarray((i + dx, j + dy)) - center)
                        <= 6.0
                    )
            fill[i, j, 1] = inside / (samples * samples)
    return fill


def _vortex_courant(
    shape: tuple[int, int, int], maximum_courant: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
    cx *= scale
    cy *= scale
    divergence = (
        cx[2:nx, 1 : ny - 1, 1]
        - cx[1 : nx - 1, 1 : ny - 1, 1]
        + cy[1 : nx - 1, 2:ny, 1]
        - cy[1 : nx - 1, 1 : ny - 1, 1]
    )
    return (
        cx.astype(np.float32),
        cy.astype(np.float32),
        cz.astype(np.float32),
        float(np.max(np.abs(divergence), initial=0.0)),
    )


def _render_frame(path: Path, fill: np.ndarray, step: int, phase: str) -> None:
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
    axis.contour(fill[:, :, 1].T, levels=[0.5], colors="#075985", linewidths=1.0)
    axis.set_title(f"R4.2 PLIC reversible vortex: {phase}, step {step}")
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
    for axis, (step, fill) in zip(axes.ravel(), snapshots.items()):
        axis.imshow(
            fill[:, :, 1].T,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            interpolation="nearest",
        )
        axis.contour(fill[:, :, 1].T, levels=[0.5], colors="#075985", linewidths=0.8)
        phase = "forward" if step <= 20 else "reverse"
        axis.set_title(f"step {step} ({phase})")
        axis.set_xlim(0, fill.shape[0] - 1)
        axis.set_ylim(0, fill.shape[1] - 1)
    fig.suptitle("R4.2 isolated PLIC reversible-vortex audit", fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(output_dir: Path, device: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    shape = (34, 34, 3)
    model = HomeCoreModel(fluid_grid_res=shape, device=device)
    walls = FslWallMask.closed_box(model)
    initial = _sample_circle(shape)
    initial[walls.host] = 0.0
    cx, cy, cz, maximum_divergence = _vortex_courant(shape, 0.18)
    positive = tuple(
        wp.array(field, dtype=float, device=device) for field in (cx, cy, cz)
    )
    negative = tuple(
        wp.array(-field, dtype=float, device=device) for field in (cx, cy, cz)
    )
    fill = wp.array(initial, dtype=float, device=device)
    transport = PlicGeometricTransport(model, walls)
    sample_steps = {0, 5, 10, 15, 20, 25, 30, 35, 40}
    snapshots = {0: initial.copy()}
    maximum_volume_drift = 0.0
    snapped_cell_count = 0
    snapped_volume_delta = 0.0
    started = time.perf_counter()
    for step in range(1, 41):
        reversing = step > 20
        fill, diagnostics = transport.transport(
            fill,
            negative if reversing else positive,
            split_order=(1, 0) if reversing else (0, 1),
        )
        maximum_volume_drift = max(
            maximum_volume_drift, abs(diagnostics.relative_volume_drift)
        )
        snapped_cell_count += diagnostics.endpoint_snapped_cell_count
        snapped_volume_delta += diagnostics.endpoint_snapped_volume_delta
        if step in sample_steps:
            snapshots[step] = fill.numpy().copy()
    wp.synchronize_device(device)
    elapsed = time.perf_counter() - started
    final = snapshots[40]
    active_error = np.abs(final[:, :, 1] - initial[:, :, 1])
    frame_paths: list[str] = []
    for step, snapshot in snapshots.items():
        relative = f"frames/step-{step:03d}.png"
        _render_frame(
            output_dir / relative,
            snapshot,
            step,
            "forward" if step <= 20 else "reverse",
        )
        frame_paths.append(relative)
    overview = "plic-vortex-overview.png"
    _render_overview(output_dir / overview, snapshots)
    manifest: dict[str, object] = {
        "scene": "r4.2-isolated-plic-reversible-vortex",
        "shape": shape,
        "forward_steps": 20,
        "reverse_steps": 20,
        "maximum_courant": 0.18,
        "maximum_discrete_divergence": maximum_divergence,
        "maximum_step_relative_volume_drift": maximum_volume_drift,
        "endpoint_snapped_cell_count": snapped_cell_count,
        "endpoint_snapped_volume_delta": snapped_volume_delta,
        "active_mean_absolute_error": float(np.mean(active_error)),
        "relative_l1_shape_error": float(
            np.sum(active_error) / np.sum(initial[:, :, 1])
        ),
        "maximum_absolute_shape_error": float(np.max(active_error)),
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "device": device,
        },
        "frames": frame_paths,
        "overview": overview,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
