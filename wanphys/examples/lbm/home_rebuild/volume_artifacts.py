# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-neutral volume snapshots and preview surface extraction."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

SURFACE_METHOD = "padded-fill-marching-cubes-0.5"
SOLVER_AXIS_ORDER = ("x", "height_y", "depth_z")
RENDER_AXIS_ORDER = ("x", "depth_z", "height_y")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_volume_snapshot(
    path: str | Path,
    *,
    fill: np.ndarray,
    mass: np.ndarray,
    flags: np.ndarray,
    moments: np.ndarray,
    excess_mass: np.ndarray,
) -> dict[str, Any]:
    """Atomically save the common fields needed by diagnostics and rendering."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        fill=np.asarray(fill, dtype=np.float32),
        mass=np.asarray(mass, dtype=np.float32),
        flags=np.asarray(flags, dtype=np.int32),
        moments=np.asarray(moments, dtype=np.float32),
        excess_mass=np.asarray(excess_mass, dtype=np.float32),
        solver_axis_order=np.asarray(SOLVER_AXIS_ORDER),
    )
    os.replace(temporary, destination)
    return {
        "file": destination.name,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "shape": [int(value) for value in fill.shape],
        "solver_axis_order": list(SOLVER_AXIS_ORDER),
    }


def extract_fill_surface(
    snapshot_path: str | Path,
    output_path: str | Path,
    *,
    cell_size: float = 1.0,
    level: float = 0.5,
) -> dict[str, Any]:
    """Extract a common visualization surface with explicit solver-to-render axes."""

    try:
        from skimage.measure import marching_cubes  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError("surface extraction requires scikit-image") from error

    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        fill = np.asarray(snapshot["fill"], dtype=np.float32)
    if fill.ndim != 3 or min(fill.shape) < 2:
        raise ValueError("volume snapshot fill must be a three-dimensional field")

    padded = np.pad(fill, 1, mode="constant", constant_values=0.0)
    vertices, faces, _normals, _values = marching_cubes(
        padded,
        level=level,
        spacing=(cell_size, cell_size, cell_size),
        allow_degenerate=False,
    )
    vertices -= 0.5 * cell_size
    vertices = vertices[:, (0, 2, 1)]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# HOME rebuild common fill surface\n")
        stream.write(f"# method {SURFACE_METHOD}\n")
        stream.write("# render axes x depth_z height_y\n")
        for vertex in vertices:
            stream.write(f"v {vertex[0]:.8g} {vertex[1]:.8g} {vertex[2]:.8g}\n")
        for face in faces:
            stream.write(
                f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n"
            )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return {
        "file": destination.name,
        "method": SURFACE_METHOD,
        "level": level,
        "render_axis_order": list(RENDER_AXIS_ORDER),
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }
