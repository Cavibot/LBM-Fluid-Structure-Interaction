#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Extract and render every common volume snapshot in a HOME rebuild run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from wanphys.examples.lbm.home_rebuild.volume_artifacts import (  # noqa: E402
    extract_fill_surface,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--view", choices=("side", "diagonal"), default="diagonal")
    parser.add_argument(
        "--style", choices=("diagnostic", "cinematic"), default="cinematic"
    )
    args = parser.parse_args()

    run = args.run.resolve()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = manifest.get("volume_snapshots", [])
    if not snapshots:
        raise RuntimeError("run manifest has no volume snapshots")

    mesh_dir = run / "meshes-common"
    frame_dir = run / f"frames-{args.style}-{args.view}"
    log_dir = run / f"logs-{args.style}-{args.view}"
    mesh_dir.mkdir(exist_ok=True)
    frame_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    scene_script = (
        Path(__file__).resolve().parents[1] / "home_free" / "blender_preview_scene.py"
    )
    config = manifest["config"]
    tank_extent = (
        float(config["resolution_x"]),
        float(config["resolution_z"]),
        float(config["resolution_y"]),
    )

    renders: list[dict[str, object]] = []
    for snapshot in snapshots:
        step = int(snapshot["step"])
        mesh_path = mesh_dir / f"step-{step:06d}.obj"
        surface = extract_fill_surface(run / snapshot["file"], mesh_path)
        output = frame_dir / f"frame-{step:06d}.png"
        log_path = log_dir / f"render-{step:06d}.log"
        command = [
            str(args.blender.resolve()),
            "--background",
            "--factory-startup",
            "--python",
            str(scene_script),
            "--",
            "--mesh",
            str(mesh_path),
            "--output",
            str(output),
            "--samples",
            str(args.samples),
            "--tank-extent",
            *(str(value) for value in tank_extent),
            "--camera-view",
            args.view,
            "--render-style",
            args.style,
            "--common-comparison",
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Blender failed at step {step}; see {log_path}")
        renders.append(
            {
                "step": step,
                "file": str(output.relative_to(run)),
                "sha256": sha256_file(output),
                "bytes": output.stat().st_size,
                "surface": surface,
                "log": str(log_path.relative_to(run)),
            }
        )
        print(output, flush=True)

    manifest["common_surface_render"] = {
        "view": args.view,
        "style": args.style,
        "samples": args.samples,
        "tank_extent_render_axes": list(tank_extent),
        "frames": renders,
        "complete": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
