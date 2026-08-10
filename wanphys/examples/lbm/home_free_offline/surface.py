# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Preview surface extraction kept separate from solver state."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .capture import sha256_file


PREVIEW_SURFACE_METHOD = "padded-fill-marching-cubes-0.5"


def extract_preview_surface(
    snapshot_path: str | Path,
    output_path: str | Path,
    *,
    cell_size: float,
) -> dict[str, Any]:
    """Export a temporary visualization mesh; this is not the final PLIC mesh."""

    try:
        from skimage.measure import marching_cubes
    except ImportError as error:
        raise RuntimeError("preview surface extraction requires scikit-image") from error

    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        fill = np.asarray(snapshot["fill"], dtype=np.float32)
    padded = np.pad(fill, 1, mode="constant", constant_values=0.0)
    vertices, faces, _normals, _values = marching_cubes(
        padded,
        level=0.5,
        spacing=(cell_size, cell_size, cell_size),
        allow_degenerate=False,
    )
    vertices -= 0.5 * cell_size

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# HOME-Free numerical preview surface\n")
        stream.write(f"# method {PREVIEW_SURFACE_METHOD}\n")
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
        "method": PREVIEW_SURFACE_METHOD,
        "final_plic_surface": False,
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }
