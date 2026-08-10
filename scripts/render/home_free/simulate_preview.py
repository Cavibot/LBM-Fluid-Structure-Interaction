#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate traceable dam-break preview snapshots and temporary meshes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

from wanphys.examples.lbm.home_free_offline import (
    OfflineSceneLevel,
    OfflineSceneName,
    make_scene_config,
)
from wanphys.examples.lbm.home_free_offline.convergence import (
    make_convergence_config,
)
from wanphys.examples.lbm.home_free_offline.capture import (
    atomic_write_json,
    capture_preview_snapshot,
    sha256_file,
)
from wanphys.examples.lbm.home_free_offline.factory import build_offline_scene
from wanphys.examples.lbm.home_free_offline.surface import extract_preview_surface


DEFAULT_PREVIEW_STEPS = (0, 300, 600, 900)


def make_preview_config(
    device: str,
    preview_steps: tuple[int, ...] = DEFAULT_PREVIEW_STEPS,
    max_lattice_speed: float | None = None,
    convergence_level: OfflineSceneLevel | None = None,
    kinematic_viscosity: float | None = None,
):
    if convergence_level is not None:
        base = make_convergence_config(
            OfflineSceneName.DAMBREAK,
            convergence_level,
            device=device,
        )
        fluid = base.fluid
        fluid = replace(
            fluid,
            max_lattice_speed=(
                fluid.max_lattice_speed
                if max_lattice_speed is None
                else max_lattice_speed
            ),
            kinematic_viscosity=(
                fluid.kinematic_viscosity
                if kinematic_viscosity is None
                else kinematic_viscosity
            ),
        )
        return replace(
            base,
            steps=max(preview_steps),
            output_every_steps=max(1, preview_steps[1] - preview_steps[0]),
            fluid=fluid,
        )
    base = make_scene_config(OfflineSceneName.DAMBREAK, device=device)
    fluid = replace(
        base.fluid,
        resolution=(48, 24, 32),
        cell_size=0.5,
        time_step=0.25,
        kinematic_viscosity=0.04,
        body_acceleration=(0.0, 0.0, -1.0e-3),
        periodic=(False, False, False),
        max_lattice_speed=(
            base.fluid.max_lattice_speed
            if max_lattice_speed is None
            else max_lattice_speed
        ),
    )
    return replace(
        base,
        level=OfflineSceneLevel.L1,
        steps=max(preview_steps),
        output_every_steps=300,
        initial_liquid_height=8.25,
        initial_liquid_x_range=(0.0, 8.25),
        fluid=fluid,
    )


def _source_tree_hash(repo: Path) -> str:
    digest = hashlib.sha256()
    roots = (
        repo / "wanphys/_src/fluid/fluid_grid/home_lbm",
        repo / "wanphys/examples/lbm/home_free_offline",
        repo / "scripts/render/home_free",
    )
    files = sorted(
        path
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    for path in files:
        relative = path.relative_to(repo).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--frame-steps",
        default=",".join(str(step) for step in DEFAULT_PREVIEW_STEPS),
        help="comma-separated committed steps to capture",
    )
    parser.add_argument(
        "--max-lattice-speed",
        type=float,
        default=None,
        help="optional unqualified preview-only speed limit",
    )
    parser.add_argument(
        "--convergence-level",
        choices=[level.value for level in OfflineSceneLevel],
        help="use the strict convergence configuration for this level",
    )
    parser.add_argument(
        "--kinematic-viscosity",
        type=float,
        help="unqualified preview-only physical kinematic viscosity override",
    )
    parser.add_argument(
        "--disable-speed-limit",
        action="store_true",
        help="unqualified preview only: continue until a non-speed hard error",
    )
    args = parser.parse_args()

    preview_steps = tuple(
        sorted({int(value.strip()) for value in args.frame_steps.split(",")})
    )
    if not preview_steps or preview_steps[0] != 0 or preview_steps[-1] <= 0:
        raise ValueError("frame steps must contain zero and at least one positive step")

    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    snapshots = output / "snapshots"
    meshes = output / "meshes-preview"
    snapshots.mkdir(parents=True, exist_ok=True)
    meshes.mkdir(parents=True, exist_ok=True)
    config = make_preview_config(
        args.device,
        preview_steps,
        args.max_lattice_speed,
        (
            None
            if args.convergence_level is None
            else OfflineSceneLevel(args.convergence_level)
        ),
        args.kinematic_viscosity,
    )
    scene = build_offline_scene(config)
    if args.disable_speed_limit:
        scene.fluid.model.max_lattice_speed = float("inf")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    manifest = {
        "schema_version": 1,
        "purpose": "early-visual-preview",
        "scene": config.scene.value,
        "numerically_qualified": False,
        "known_limit": (
            "This visual preview may use a relaxed or disabled speed limit and does "
            "not satisfy production acceptance."
        ),
        "runtime_speed_limit_disabled": bool(args.disable_speed_limit),
        "git_commit": commit,
        "source_tree_hash": _source_tree_hash(repo),
        "config": config.to_dict(),
        "derived_parameters": config.derived_parameters,
        "requested_frame_steps": list(preview_steps),
        "simulation_complete": False,
        "frames": [],
    }
    atomic_write_json(output / "manifest.json", manifest)

    targets = set(preview_steps)
    try:
        for step in range(max(preview_steps) + 1):
            if step in targets:
                stem = f"frame-{step:06d}"
                snapshot_path = snapshots / f"{stem}.npz"
                mesh_path = meshes / f"{stem}.obj"
                snapshot = capture_preview_snapshot(scene, snapshot_path)
                surface = extract_preview_surface(
                    snapshot_path,
                    mesh_path,
                    cell_size=config.fluid.cell_size,
                )
                manifest["frames"].append(
                    {
                        "step": step,
                        "physical_time": scene.physical_time,
                        "snapshot": snapshot,
                        "surface": surface,
                    }
                )
                atomic_write_json(output / "manifest.json", manifest)
                print(
                    json.dumps(
                        {
                            "step": step,
                            "physical_time": scene.physical_time,
                            "interface_cells": snapshot["metrics"]["interface_cells"],
                            "triangles": surface["triangles"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if step < max(preview_steps):
                scene.step()
                scene.synchronize()
    except Exception as error:
        manifest["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "failed_while_advancing_from_step": scene.step_index,
            "last_captured_step": manifest["frames"][-1]["step"],
        }
        manifest["complete"] = True
        atomic_write_json(output / "manifest.json", manifest)
        raise

    manifest["simulation_complete"] = True
    manifest["complete"] = True
    atomic_write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
