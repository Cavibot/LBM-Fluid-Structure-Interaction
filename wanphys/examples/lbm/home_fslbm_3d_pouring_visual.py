# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay / live-view Home-FSLBM 3D pouring-water (``testMrLBM3D_bubble``) with wanphys SSFR.

Home-FSLBM writes 3D phi dumps via ``mlSavePhi``; this example loads them into
``ScreenSpaceFluidRenderer`` for interactive ray-march visualization.

Replay existing frames (default data dir = sibling ``Home-FSLBM/build/...``):

    uv run --extra examples python -m wanphys.examples.lbm.home_fslbm_3d_pouring_visual

Run the C++ solver and visualize as phi frames appear:

    uv run --extra examples python -m wanphys.examples.lbm.home_fslbm_3d_pouring_visual \\
        --run --max-frames 30

Build Home-FSLBM first (from repo root)::

    cd Home-FSLBM/build
    cmake .. -DFLR_WITH_CUDA=ON -DFLR_RUN_3D=ON
    cmake --build . --config Release

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm.io.home_fslbm_phi import (
    frame_index,
    list_phi_frames,
    load_phi_bin,
)
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer

FRAME_DT: float = 1.0 / 60.0
SSFR_THRESHOLD: float = 0.15
RAY_MARCH_STEPS: int = 512
CELL_SIZE: float = 1.0  # Home ``delta_x`` in testMrLBM3D_bubble.cpp


def _default_home_paths() -> tuple[Path, Path, Path]:
    """Return (exe, phi_dir, build_cwd) for sibling Home-FSLBM."""
    lbm_root = Path(__file__).resolve().parents[3]
    home = lbm_root.parent / "Home-FSLBM"
    build = home / "build"
    exe = build / "Release" / "lbm_flow_proj.exe"
    if not exe.is_file():
        exe = build / "lbm_flow_proj"
    # ``mlSavePhi`` uses ``../dataMR3D/...`` relative to ``build/`` cwd.
    phi_dir = home / "dataMR3D" / "ppm_ve_home_test_phi"
    if not phi_dir.is_dir():
        alt = build / "dataMR3D" / "ppm_ve_home_test_phi"
        if alt.is_dir():
            phi_dir = alt
    return exe, phi_dir, build


class HomeFslbm3DPouringVisual:
    def __init__(
        self,
        viewer: FluidViewerGL,
        *,
        data_dir: Path,
        exe: Path | None = None,
        build_cwd: Path | None = None,
        run_solver: bool = False,
        max_frames: int = 600,
        replay_fps: float = 30.0,
        cell_size: float = CELL_SIZE,
    ):
        self.viewer = viewer
        viewer._paused = not run_solver
        self._data_dir = Path(data_dir)
        self._exe = Path(exe) if exe is not None else None
        self._build_cwd = Path(build_cwd) if build_cwd is not None else None
        self._run_solver = bool(run_solver)
        self._max_frames = int(max_frames)
        self._replay_dt = 1.0 / max(float(replay_fps), 1.0)
        self._cell_size = float(cell_size)

        wp.init()
        self._device = "cuda:0" if wp.is_cuda_available() else "cpu"
        self._field: wp.array3d | None = None
        self._shape: tuple[int, int, int] | None = None

        self._proc: subprocess.Popen[str] | None = None
        self._frame_paths: list[Path] = []
        self._frame_cursor = 0
        self._last_loaded = -1
        self._sim_time = 0.0
        self._accum = 0.0
        self._live = self._run_solver

        self.ssfr = ScreenSpaceFluidRenderer(
            viewer=viewer,
            max_particles=1,
            particle_radius=0.01,
            device=self._device,
        )
        viewer.register_post_render_callback(lambda v: self.ssfr.render(v))

        if self._run_solver:
            if self._exe is None or not self._exe.is_file():
                raise FileNotFoundError(
                    f"Home-FSLBM executable not found: {self._exe}. "
                    "Build Home-FSLBM or pass --exe."
                )
            cwd = self._build_cwd or self._exe.parent.parent
            print(
                f"Starting Home-FSLBM: {self._exe} {self._max_frames} (cwd={cwd})",
                file=sys.stderr,
                flush=True,
            )
            self._proc = subprocess.Popen(
                [str(self._exe), str(self._max_frames)],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        else:
            self._frame_paths = list_phi_frames(self._data_dir)
            if not self._frame_paths:
                raise FileNotFoundError(
                    f"No phi*.bin under {self._data_dir}. "
                    "Re-run Home-FSLBM with mlSavePhi enabled, or use --run."
                )
            print(
                f"Replay {len(self._frame_paths)} phi frames from {self._data_dir}",
                file=sys.stderr,
                flush=True,
            )
            self._load_frame_path(self._frame_paths[0])

        print("Controls: [Space] unpause  [R] reset  [mouse] orbit", file=sys.stderr)

    def _load_frame_path(self, path: Path) -> None:
        nx, ny, nz, phi = load_phi_bin(path)
        if self._shape is None:
            self._shape = (nx, ny, nz)
            self._field = wp.zeros(self._shape, dtype=float, device=self._device)
        elif self._shape != (nx, ny, nz):
            raise ValueError(f"grid size changed: {self._shape} -> {(nx, ny, nz)}")
        assert self._field is not None
        self._field.assign(wp.array(phi, dtype=float, device=self._device))
        self._last_loaded = frame_index(path)

    def _poll_solver_output(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        while True:
            line = self._proc.stdout.readline()
            if not line:
                break
            print(line.rstrip(), file=sys.stderr, flush=True)

    def _try_load_live_frame(self) -> bool:
        for fidx in range(self._last_loaded + 1, self._max_frames + 2):
            path = self._data_dir / f"phi{fidx:05d}.bin"
            if path.is_file() and path.stat().st_size > 12:
                self._load_frame_path(path)
                return True
        return False

    def step(self) -> None:
        t0 = time.perf_counter()
        self._poll_solver_output()

        if self._live:
            if self._try_load_live_frame():
                self._sim_time = float(self._last_loaded) * self._replay_dt
            if self._proc is not None and self._proc.poll() is not None:
                self._live = False
                self._frame_paths = list_phi_frames(self._data_dir)
                print(
                    f"Home-FSLBM finished; {len(self._frame_paths)} phi frames on disk",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            self._accum += FRAME_DT
            while self._accum >= self._replay_dt and self._frame_paths:
                self._accum -= self._replay_dt
                if self._frame_cursor + 1 < len(self._frame_paths):
                    self._frame_cursor += 1
                    self._load_frame_path(self._frame_paths[self._frame_cursor])
                    self._sim_time = float(self._frame_cursor) * self._replay_dt
                else:
                    self._accum = 0.0
                    break

        wp.synchronize_device(self._device)
        _ = (time.perf_counter() - t0) * 1000.0

    def render(self) -> None:
        self.viewer.begin_frame(self._sim_time)
        if self.ssfr.available and self._field is not None:
            self.ssfr.set_density_field(
                density=self._field,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=self._cell_size,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        self.viewer.end_frame()


def main() -> None:
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer

    default_exe, default_phi, default_cwd = _default_home_paths()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path, default=default_phi)
    parser.add_argument("--exe", type=Path, default=default_exe)
    parser.add_argument("--build-cwd", type=Path, default=default_cwd)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Launch Home-FSLBM and visualize phi frames as they are written.",
    )
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--replay-fps", type=float, default=30.0)
    parser.add_argument("--cell-size", type=float, default=CELL_SIZE)
    pre_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    viewer, args = init_fluid_viewer()
    newton.examples.run(
        HomeFslbm3DPouringVisual(
            viewer,
            data_dir=pre_args.data_dir,
            exe=pre_args.exe,
            build_cwd=pre_args.build_cwd,
            run_solver=pre_args.run,
            max_frames=pre_args.max_frames,
            replay_fps=pre_args.replay_fps,
            cell_size=pre_args.cell_size,
        ),
        args,
    )


if __name__ == "__main__":
    main()
