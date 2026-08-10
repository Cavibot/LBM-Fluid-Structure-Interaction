#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify a HOME-Free preview artifact without qualifying its numerics."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _log_value(text: str, key: str) -> Any:
    prefix = f"{key}="
    matches = [line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"render log must contain exactly one {key} marker")
    return ast.literal_eval(matches[0])


def _optional_log_value(text: str, key: str, default: Any) -> Any:
    prefix = f"{key}="
    matches = [line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)]
    if not matches:
        return default
    if len(matches) != 1:
        raise ValueError(f"render log must contain at most one {key} marker")
    return ast.literal_eval(matches[0])


def _check_bounds(
    bounds: tuple[list[float], list[float]],
    extent: tuple[float, float, float],
) -> None:
    lower, upper = (np.asarray(values, dtype=np.float64) for values in bounds)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("water bounds must contain two three-component vectors")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("water bounds contain non-finite coordinates")
    if np.any(lower < -1.0e-6) or np.any(upper > np.asarray(extent) + 1.0e-6):
        raise ValueError(f"water bounds {bounds} leave simulation extent {extent}")


def _as_bounds(value: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = (np.asarray(values, dtype=np.float64) for values in value)
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError(f"{name} must contain two three-component vectors")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return lower, upper


def verify_preview(
    run: Path,
    *,
    allow_partial: bool = False,
    view: str = "side",
    style: str = "diagnostic",
) -> dict[str, Any]:
    run = run.resolve()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    frame_reports: list[dict[str, Any]] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    require(bool(manifest.get("complete")), "manifest was not finalized")
    if style == "diagnostic":
        complete_key = "render_complete" if view == "side" else f"{view}_render_complete"
    else:
        complete_key = f"{style}_{view}_render_complete"
    require(bool(manifest.get(complete_key)), f"{view} render did not complete")
    if not allow_partial:
        require(bool(manifest.get("simulation_complete", True)), "simulation was partial")
    require(manifest.get("numerically_qualified") is False, "preview must remain unqualified")
    frames = manifest.get("frames", [])
    require(bool(frames), "manifest contains no frames")

    fluid = manifest["config"]["fluid"]
    resolution = tuple(int(value) for value in fluid["resolution"])
    cell_size = float(fluid["cell_size"])
    extent = tuple(value * cell_size for value in resolution)
    previous_step = -1
    initial_x_max = None
    final_x_max = None

    for frame in frames:
        step = int(frame["step"])
        require(step > previous_step, f"frame step {step} is not strictly increasing")
        previous_step = step
        try:
            snapshot = run / "snapshots" / frame["snapshot"]["file"]
            surface = run / "meshes-preview" / frame["surface"]["file"]
            if style == "diagnostic":
                render_key = "preview_render" if view == "side" else f"{view}_render"
            else:
                render_key = f"{style}_{view}_render"
            render = frame[render_key]
            if style == "diagnostic":
                frames_directory = "frames-preview" if view == "side" else f"frames-{view}"
                logs_directory = "logs" if view == "side" else f"logs-{view}"
            else:
                frames_directory = f"frames-{style}-{view}"
                logs_directory = f"logs-{style}-{view}"
            png = run / frames_directory / render["file"]
            log = run / logs_directory / render["log"]
            for path, expected in (
                (snapshot, frame["snapshot"]["sha256"]),
                (surface, frame["surface"]["sha256"]),
                (png, render["sha256"]),
            ):
                require(path.is_file(), f"step {step}: missing {path.name}")
                if path.is_file():
                    require(_sha256(path) == expected, f"step {step}: hash mismatch for {path.name}")

            with Image.open(png) as image:
                pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
                expected_size = (1280, 720) if style == "cinematic" else (960, 540)
                require(image.size == expected_size, f"step {step}: unexpected PNG size {image.size}")
                require(float(np.std(pixels)) > 8.0, f"step {step}: PNG has insufficient variance")
                require(float(np.mean(pixels)) > 2.0, f"step {step}: PNG is effectively black")

            log_text = log.read_text(encoding="utf-8")
            devices = _log_value(log_text, "HOME_FREE_OPTIX_DEVICES")
            forward = np.asarray(_log_value(log_text, "HOME_FREE_CAMERA_FORWARD"))
            up = np.asarray(_log_value(log_text, "HOME_FREE_CAMERA_UP"))
            bounds = _log_value(log_text, "HOME_FREE_WATER_BOUNDS")
            tank_bounds = _log_value(log_text, "HOME_FREE_TANK_INTERIOR_BOUNDS")
            frame_bounds = _log_value(log_text, "HOME_FREE_TANK_FRAME_BOUNDS")
            ortho_scale = float(_log_value(log_text, "HOME_FREE_CAMERA_ORTHO_SCALE"))
            logged_view = _optional_log_value(
                log_text,
                "HOME_FREE_CAMERA_VIEW",
                "side",
            )
            logged_style = _optional_log_value(
                log_text,
                "HOME_FREE_RENDER_STYLE",
                "diagnostic",
            )
            require(any("5090" in name for name in devices), f"step {step}: RTX 5090 was not used")
            require(logged_view == view, f"step {step}: expected {view} camera, got {logged_view}")
            require(logged_style == style, f"step {step}: expected {style} style, got {logged_style}")
            if view == "side":
                require(np.allclose(forward, (0.0, 1.0, 0.0), atol=1.0e-6), f"step {step}: camera forward axis changed")
                require(np.allclose(up, (0.0, 0.0, 1.0), atol=1.0e-6), f"step {step}: camera up axis changed")
            else:
                require(
                    all(abs(float(forward[axis])) > 0.2 for axis in range(3)),
                    f"step {step}: diagonal camera does not expose all three axes",
                )
                require(float(up[2]) > 0.5, f"step {step}: diagonal camera up axis is invalid")
            _check_bounds(bounds, extent)
            water_lower, water_upper = _as_bounds(bounds, "water bounds")
            tank_lower, tank_upper = _as_bounds(tank_bounds, "tank interior bounds")
            frame_lower, frame_upper = _as_bounds(frame_bounds, "tank frame bounds")
            require(
                np.all(water_lower[:2] >= tank_lower[:2] - 1.0e-6)
                and np.all(water_upper[:2] <= tank_upper[:2] + 1.0e-6),
                f"step {step}: water is outside the tank footprint",
            )
            require(
                abs(float(water_lower[2] - tank_lower[2])) <= 0.1 * cell_size,
                f"step {step}: water does not contact the tank bottom",
            )
            require(
                float(np.dot(np.asarray(fluid["body_acceleration"]), up)) < 0.0,
                f"step {step}: gravity is not downward in camera space",
            )
            require(
                frame_lower[0] < tank_lower[0]
                and frame_upper[0] > tank_upper[0]
                and frame_lower[2] < tank_lower[2]
                and frame_upper[2] > tank_upper[2],
                f"step {step}: tank frame does not surround the tank interior",
            )
            if style == "diagnostic":
                require(ortho_scale >= 1.1 * (frame_upper[2] - frame_lower[2]), f"step {step}: tank frame is vertically cropped")
            x_max = float(bounds[1][0])
            initial_x_max = x_max if initial_x_max is None else initial_x_max
            final_x_max = x_max
            frame_reports.append(
                {
                    "step": step,
                    "png": png.name,
                    "png_bytes": png.stat().st_size,
                    "water_bounds": bounds,
                    "tank_interior_bounds": tank_bounds,
                    "tank_frame_bounds": frame_bounds,
                    "optix_devices": devices,
                }
            )
        except Exception as error:
            failures.append(f"step {step}: {type(error).__name__}: {error}")

    if initial_x_max is not None and final_x_max is not None and len(frames) > 1:
        require(final_x_max > initial_x_max, "dam-break front did not advance in rendered frames")

    report = {
        "schema_version": 1,
        "purpose": "pipeline-smoke-only",
        "numerically_qualified": False,
        "run": str(run),
        "simulation_complete": bool(manifest.get("simulation_complete", True)),
        "allow_partial": allow_partial,
        "view": view,
        "style": style,
        "frames_verified": len(frame_reports),
        "failures": failures,
        "passed": not failures,
        "frames": frame_reports,
    }
    report_name = (
        "verification.json"
        if style == "diagnostic" and view == "side"
        else f"verification-{style}-{view}.json"
    )
    _atomic_write_json(run / report_name, report)
    if failures:
        raise RuntimeError("preview verification failed:\n- " + "\n- ".join(failures))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--view", choices=("side", "diagonal"), default="side")
    parser.add_argument("--style", choices=("diagnostic", "cinematic"), default="diagnostic")
    args = parser.parse_args()
    print(
        json.dumps(
            verify_preview(
                args.run,
                allow_partial=args.allow_partial,
                view=args.view,
                style=args.style,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
