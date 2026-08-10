# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Run FSL and reconstruct PLIC geometry strictly as read-only postprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicDomain,
    PlicGeometryReconstructor,
    PlicGeometryState,
)
from wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview import (
    GravityColumnConfig,
    run_scene,
)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _plic_segment(
    normal: np.ndarray,
    offset: float,
    center: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    nx, ny = (float(normal[0]), float(normal[1]))
    candidates: list[np.ndarray] = []
    tolerance = 1.0e-8
    if abs(ny) > tolerance:
        for x in (-0.5, 0.5):
            y = (offset - nx * x) / ny
            if -0.5 - tolerance <= y <= 0.5 + tolerance:
                candidates.append(np.asarray((x, np.clip(y, -0.5, 0.5))))
    if abs(nx) > tolerance:
        for y in (-0.5, 0.5):
            x = (offset - ny * y) / nx
            if -0.5 - tolerance <= x <= 0.5 + tolerance:
                candidates.append(np.asarray((np.clip(x, -0.5, 0.5), y)))
    unique: list[np.ndarray] = []
    for candidate in candidates:
        if not any(np.linalg.norm(candidate - value) < tolerance for value in unique):
            unique.append(candidate)
    if len(unique) < 2:
        return None
    first, second = max(
        (
            (a, b)
            for index, a in enumerate(unique)
            for b in unique[index + 1 :]
        ),
        key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])),
    )
    origin = np.asarray(center)
    return tuple(origin + first), tuple(origin + second)


def _segments_from_geometry(
    normal: np.ndarray,
    plane_offset: np.ndarray,
    valid: np.ndarray,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments = []
    center_z = valid.shape[2] // 2
    for x, y in zip(*np.nonzero(valid[:, :, center_z]), strict=True):
        segment = _plic_segment(
            normal[x, y, center_z],
            float(plane_offset[x, y, center_z]),
            (float(x), float(y)),
        )
        if segment is not None:
            segments.append(segment)
    return segments


def _render_geometry_frame(
    path: Path,
    fill: np.ndarray,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    step: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    center_z = fill.shape[2] // 2
    fig, axis = plt.subplots(figsize=(8.2, 5.0), dpi=180)
    axis.imshow(
        fill[:, :, center_z].T,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        interpolation="nearest",
        aspect="equal",
    )
    axis.add_collection(
        LineCollection(segments, colors="#ef4444", linewidths=0.75)
    )
    axis.set_xlim(0, fill.shape[0] - 1)
    axis.set_ylim(0, fill.shape[1] - 1)
    axis.set_title(f"FSL fill with read-only PLIC segments, step {step}")
    axis.set_xlabel("x lattice cell")
    axis.set_ylabel("y lattice cell")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_overview(
    path: Path,
    frames: list[
        tuple[
            int,
            np.ndarray,
            list[tuple[tuple[float, float], tuple[float, float]]],
        ]
    ],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 9.0), dpi=180)
    for axis, (step, fill, segments) in zip(
        axes.ravel(), frames, strict=True
    ):
        center_z = fill.shape[2] // 2
        axis.imshow(
            fill[:, :, center_z].T,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            interpolation="nearest",
            aspect="equal",
        )
        axis.add_collection(
            LineCollection(segments, colors="#ef4444", linewidths=0.45)
        )
        axis.set_xlim(0, fill.shape[0] - 1)
        axis.set_ylim(0, fill.shape[1] - 1)
        axis.set_title(f"step {step}")
    fig.suptitle("R5.1 FSL + read-only PLIC geometry", fontsize=15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(output_dir: Path, device: str) -> dict[str, object]:
    config = GravityColumnConfig(
        profile="r5.1-fsl-plus-plic-geometry",
        resolution_z=1,
        periodic_depth=True,
        physical_gravity_y=-5.0e-6,
        sample_steps=(0, 2000, 4000, 6000, 8000, 10000, 12500, 15000, 20000),
        save_volume_snapshots=True,
        device=device,
    )
    manifest = run_scene(config, output_dir)
    shape = (config.resolution_x, config.resolution_y, config.resolution_z)
    domain = PlicDomain(
        shape,
        device=device,
        closed_axes=(True, True, False),
    )
    geometry = PlicGeometryState(domain)
    reconstructor = PlicGeometryReconstructor(domain)
    geometry_dir = output_dir / "frames-plic-geometry"
    geometry_dir.mkdir(exist_ok=True)
    geometry_metrics: list[dict[str, object]] = []
    overview_frames = []
    for artifact in manifest["volume_snapshots"]:
        step = int(artifact["step"])
        with np.load(output_dir / artifact["file"], allow_pickle=False) as snapshot:
            fill = np.asarray(snapshot["fill"], dtype=np.float32)
        before = _array_sha256(fill)
        diagnostics = reconstructor.reconstruct_fill(
            wp.array(fill, dtype=float, device=device),
            geometry,
            interface_epsilon=4.0e-7,
        )
        after = _array_sha256(fill)
        if after != before:
            raise RuntimeError("read-only PLIC geometry changed the FSL snapshot")
        normal = geometry.normal.numpy()
        plane_offset = geometry.plane_offset.numpy()
        valid = geometry.valid.numpy().astype(bool)
        segments = _segments_from_geometry(normal, plane_offset, valid)
        relative = f"frames-plic-geometry/step-{step:06d}.png"
        _render_geometry_frame(
            output_dir / relative,
            fill,
            segments,
            step=step,
        )
        overview_frames.append((step, fill.copy(), segments))
        geometry_metrics.append(
            {
                "step": step,
                "snapshot_sha256_before": before,
                "snapshot_sha256_after": after,
                "state_unchanged": before == after,
                "interface_cell_count": diagnostics.interface_cell_count,
                "invalid_interface_count": diagnostics.invalid_interface_count,
                "maximum_interface_area": diagnostics.maximum_interface_area,
                "rendered_segment_count": len(segments),
                "frame": relative,
            }
        )
    overview = "fsl-plic-geometry-overview.png"
    _render_overview(output_dir / overview, overview_frames)
    manifest["variant"] = {
        "name": "FSL+PLIC-geometry",
        "plic_role": "read-only geometry and display",
        "mass_transport": "link-wise FSL",
        "home_moments_modified_by_plic": False,
        "topology_modified_by_plic": False,
        "all_snapshots_unchanged": all(
            bool(metric["state_unchanged"]) for metric in geometry_metrics
        ),
    }
    manifest["plic_geometry_metrics"] = geometry_metrics
    manifest["plic_geometry_overview"] = overview
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
