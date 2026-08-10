# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Atomic, render-only snapshots for HOME-Free offline previews."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .scene import HomeFreeOfflineScene


PREVIEW_SNAPSHOT_SCHEMA = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def capture_preview_snapshot(
    scene: HomeFreeOfflineScene,
    path: str | Path,
) -> dict[str, Any]:
    """Save one committed state without changing the simulation fields."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = scene.fluid.free_surface_state
    fluid = scene.fluid.fluid_state
    geometry = scene.fluid.geometric_stepper.geometry
    geometry.update(free, solid_phi=fluid.solid_phi, compute_curvature=False)
    scene.synchronize()

    shape = scene.config.fluid.resolution
    moments = fluid.moments.numpy().reshape(10, *shape)
    density = moments[0].copy()
    velocity = np.zeros(shape + (3,), dtype=np.float32)
    active = density > 0.0
    for component in range(3):
        velocity[..., component][active] = (
            moments[component + 1][active] / density[active]
        )
    metrics = scene.measure()
    metadata: dict[str, Any] = {
        "schema_version": PREVIEW_SNAPSHOT_SCHEMA,
        "scene": scene.config.scene.value,
        "level": scene.config.level.value,
        "step": scene.step_index,
        "physical_time": scene.physical_time,
        "config_hash": scene.config.config_hash,
        "numerically_qualified": False,
        "metrics": asdict(metrics),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            fill=free.fill_level.numpy(),
            flags=free.flags.numpy(),
            mass=free.mass.numpy(),
            density=density,
            velocity=velocity,
            plic_normal=geometry.normal.numpy(),
            plic_offset=geometry.plane_offset.numpy(),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    metadata["file"] = destination.name
    metadata["sha256"] = sha256_file(destination)
    metadata["bytes"] = destination.stat().st_size
    return metadata
