#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Render every mesh listed by a completed preview manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from wanphys.examples.lbm.home_free_offline.capture import atomic_write_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--view", choices=("side", "diagonal"), default="side")
    parser.add_argument(
        "--style",
        choices=("diagnostic", "cinematic"),
        default="diagnostic",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError("preview simulation manifest is incomplete")
    if args.style == "diagnostic":
        frames_name = "frames-preview" if args.view == "side" else f"frames-{args.view}"
        logs_name = "logs" if args.view == "side" else f"logs-{args.view}"
    else:
        frames_name = f"frames-{args.style}-{args.view}"
        logs_name = f"logs-{args.style}-{args.view}"
    frames = run / frames_name
    logs = run / logs_name
    frames.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    blender_script = Path(__file__).with_name("blender_preview_scene.py").resolve()
    fluid = manifest["config"]["fluid"]
    tank_extent = [
        int(cells) * float(fluid["cell_size"])
        for cells in fluid["resolution"]
    ]

    for frame in manifest["frames"]:
        step = int(frame["step"])
        mesh = run / "meshes-preview" / frame["surface"]["file"]
        output = frames / f"frame-{step:06d}.png"
        log_path = logs / f"render-{step:06d}.log"
        command = [
            str(args.blender.resolve()),
            "--background",
            "--factory-startup",
            "--python",
            str(blender_script),
            "--",
            "--mesh",
            str(mesh),
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
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Blender failed for step {step}; see {log_path}")
        if args.style == "diagnostic":
            render_key = "preview_render" if args.view == "side" else f"{args.view}_render"
        else:
            render_key = f"{args.style}_{args.view}_render"
        frame[render_key] = {
            "file": output.name,
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "log": log_path.name,
        }
        atomic_write_json(manifest_path, manifest)
        print(output, flush=True)

    if args.style == "diagnostic":
        complete_key = "render_complete" if args.view == "side" else f"{args.view}_render_complete"
    else:
        complete_key = f"{args.style}_{args.view}_render_complete"
    manifest[complete_key] = True
    atomic_write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
