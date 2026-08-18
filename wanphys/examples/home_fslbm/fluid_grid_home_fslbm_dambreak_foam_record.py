# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Offline MP4 recorder for HOME-FSLBM dam-break foam (default N=256).

Renders via headless ``FluidViewerGL`` + SSFR, reads frames with
``ViewerGL.get_frame``, and pipes RGB to ``ffmpeg`` (no screen capture).

Camera is a fixed elevated side view (侧上方) looking at the tank centre.

Example::

    uv run --extra examples python -m \\
        wanphys.examples.home_fslbm.fluid_grid_home_fslbm_dambreak_foam_record \\
        --duration 10 --fps 30 -o dambreak_foam_256.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_fslbm import HomeFslbmDomain, HomeFslbmModel
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer

from . import fluid_grid_home_fslbm_dambreak_foam as foam

# Defaults tuned for a 256³ offline render (override via CLI).
DEFAULT_N: int = 256
DEFAULT_DURATION_S: float = 45.0
DEFAULT_FPS: int = 60
DEFAULT_WIDTH: int = 1280
DEFAULT_HEIGHT: int = 720
DEFAULT_SUBSTEPS: int = 4
DEFAULT_CRF: int = 18
DEFAULT_OUTPUT: str = "dambreak_foam_256.mp4"
DEFAULT_GRAVITY_Z: float = -5.0e-4
DEFAULT_SURFACE_TENSION: float = 6.0 * 4e-3

# Elevated side / three-quarter view (Z-up): look toward domain centre.
CAMERA_PITCH: float = -45.0
CAMERA_YAW: float = -100.0


def _ray_march_steps(n: int) -> int:
    """Budget scales with domain diagonal (~√3 N) × oversampling."""
    return max(1600, int(np.ceil(np.sqrt(3.0) * float(n) * 4.5)))


def _setup_side_above_camera(viewer: FluidViewerGL, world_size: float) -> None:
    """Place camera above and to the side of the tank (+X/+Y elevated)."""
    viewer.set_camera(
        pos=wp.vec3(world_size * 0.75, world_size * 1.55, world_size * 1.25),
        pitch=CAMERA_PITCH,
        yaw=CAMERA_YAW,
    )


def _ffmpeg_candidates(explicit: str | None) -> list[str]:
    """Collect ffmpeg executables; prefer full GPL builds with libx264 when possible."""
    found: list[str] = []
    if explicit:
        found.append(explicit)
    which = shutil.which("ffmpeg")
    if which:
        found.append(which)
    # Windows: ``where`` may list additional shims (conda vs scoop).
    where = shutil.which("where")
    if where:
        try:
            raw = subprocess.check_output([where, "ffmpeg"], text=True, stderr=subprocess.DEVNULL)
            for line in raw.splitlines():
                path = line.strip()
                if path and path not in found:
                    found.append(path)
        except (subprocess.CalledProcessError, OSError):
            pass
    return found


def _ffmpeg_has_encoder(ffmpeg: str, name: str) -> bool:
    try:
        out = subprocess.check_output(
            [ffmpeg, "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except (subprocess.CalledProcessError, OSError):
        return False
    return name in out


def _resolve_ffmpeg(explicit: str | None) -> tuple[str, list[str]]:
    """Return (ffmpeg_path, video_encoder_args).

    Prefer libx264 across *all* candidates (scoop/full builds) before falling
    back to AV1/mpeg4 on a GPL-less conda ffmpeg.
    """
    candidates = _ffmpeg_candidates(explicit)
    if not candidates:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install a full build "
            "(e.g. scoop install ffmpeg) — conda ffmpeg is often built without libx264."
        )

    for ffmpeg in candidates:
        if _ffmpeg_has_encoder(ffmpeg, "libx264"):
            return ffmpeg, ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "CRF", "-preset", "medium"]
    for ffmpeg in candidates:
        if _ffmpeg_has_encoder(ffmpeg, "libaom-av1"):
            return ffmpeg, ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p", "-crf", "CRF"]
    for ffmpeg in candidates:
        if _ffmpeg_has_encoder(ffmpeg, "mpeg4"):
            return ffmpeg, ["-c:v", "mpeg4", "-q:v", "5"]

    raise RuntimeError(
        "No usable ffmpeg video encoder found (need libx264, libaom-av1, or mpeg4). "
        f"Tried: {candidates}"
    )


def _open_ffmpeg(
    output: Path,
    width: int,
    height: int,
    fps: int,
    crf: int,
    ffmpeg_path: str | None = None,
) -> subprocess.Popen:
    ffmpeg, enc = _resolve_ffmpeg(ffmpeg_path)
    enc = [str(crf) if a == "CRF" else a for a in enc]
    print(f"  ffmpeg={ffmpeg} encoder_args={enc}", flush=True)

    # Even dimensions required by yuv420p.
    if width % 2 or height % 2:
        raise ValueError(f"width/height must be even for yuv420p, got {width}x{height}")

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        *enc,
        str(output),
    ]
    # Capture stderr to a temp file so a long encode cannot deadlock on a full pipe.
    err_path = output.with_suffix(output.suffix + ".ffmpeg.log")
    err_f = open(err_path, "wb")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=err_f,
    )
    proc._wanphys_err_file = err_f  # type: ignore[attr-defined]
    proc._wanphys_err_path = err_path  # type: ignore[attr-defined]
    return proc


class DamBreakFoamRecorder:
    """Headless sim + SSFR + ffmpeg writer."""

    def __init__(
        self,
        viewer: FluidViewerGL,
        *,
        n: int,
        substeps: int,
        fps: int,
        gravity_z: float = DEFAULT_GRAVITY_Z,
        surface_tension: float = DEFAULT_SURFACE_TENSION,
    ):
        self.viewer = viewer
        self.n = int(n)
        self.substeps = int(substeps)
        self.frame_dt = 1.0 / float(fps)
        self.dh = foam.DH
        self.ray_steps = _ray_march_steps(self.n)
        self.gravity_z = float(gravity_z)
        self.surface_tension = float(surface_tension)

        if hasattr(viewer, "_paused"):
            viewer._paused = False
        if hasattr(viewer, "show_ui"):
            viewer.show_ui = False

        world_size = float(self.n) * self.dh
        dam_x = max(2, int(float(self.n) * foam.DAM_X_FRAC))
        dam_z = max(2, int(float(self.n) * foam.DAM_Z_FRAC))

        self.model = HomeFslbmModel(
            fluid_grid_res=(self.n, self.n, self.n),
            fluid_grid_cell_size=self.dh,
            omega=foam.OMEGA,
            gravity_x=0.0,
            gravity_y=0.0,
            gravity_z=self.gravity_z,
            surface_tension=self.surface_tension,
            disjoin_factor=foam.DISJOIN,
            atmosphere_open=foam.ATMOSPHERE_OPEN,
            enable_gas=foam.ENABLE_GAS,
            enable_disjoin=foam.ENABLE_DISJOIN,
        )
        print(
            f"HOME-FSLBM Dam-Break Foam Record: {self.n}^3, dh={self.dh}, "
            f"world={world_size:.3f}m, gz={self.gravity_z}, "
            f"sigma={self.surface_tension}, "
            f"atmosphere_open={foam.ATMOSPHERE_OPEN}, "
            f"enable_gas={foam.ENABLE_GAS}, dam x<{dam_x} z<{dam_z}, "
            f"ray_steps={self.ray_steps}",
            flush=True,
        )

        self.domain = HomeFslbmDomain(self.model)
        self.domain.create_state()
        self.sim_time = 0.0
        self.frame_count = 0
        self._last_ms = 0.0

        foam._setup_water_column(self.domain.state, self.n, self.n, self.n)
        foam._set_boundary_walls(self.domain.state, self.n, self.n, self.n)
        foam._init_rest_state(self.domain)
        self.domain.solver.init_bubbles(self.domain.state)
        foam._sync_double_buffer(self.domain)
        wp.synchronize_device(self.model._device)

        bc, atm_v, n_ent, _, _ = foam._bubble_stats(self.domain.state)
        print(f"  init bubbles={bc} atmV≈{atm_v:.1f} entrained={n_ent}", flush=True)

        target_gz = float(self.model.gravity_z)
        self.model.gravity_z = 0.0
        for s in range(foam.GRAVITY_RAMP_STEPS):
            self.model.gravity_z = target_gz * float(s + 1) / float(foam.GRAVITY_RAMP_STEPS)
            self.domain.step(1.0)
        self.model.gravity_z = target_gz
        wp.synchronize_device(self.model._device)

        _setup_side_above_camera(viewer, world_size)
        self.ssfr = ScreenSpaceFluidRenderer(
            viewer=viewer,
            max_particles=1,
            particle_radius=0.01,
            device=self.model._device,
        )
        viewer.register_post_render_callback(lambda v: self.ssfr.render(v))

    def step(self) -> None:
        t0 = time.perf_counter()
        for _ in range(self.substeps):
            self.domain.step(1.0)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000.0
        self.sim_time += self.frame_dt
        self.frame_count += 1

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self.ssfr.available:
            self.ssfr.set_density_field(
                density=self.domain.state.phi,
                grid_origin=(0.0, 0.0, 0.0),
                cell_size=self.dh,
                threshold=foam.SSFR_THRESHOLD,
                max_steps=self.ray_steps,
            )
        self.viewer.end_frame()

    def log_progress(self, i: int, total: int, *, rgb: np.ndarray | None = None) -> None:
        """Print status once per recorded video frame."""
        bc, atm_v, n_ent, _, _ = foam._bubble_stats(self.domain.state)
        frame_bit = ""
        if rgb is not None:
            n_unique = int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0])
            frame_bit = (
                f" mean={rgb.mean():.1f} std={rgb.std():.1f} "
                f"min={int(rgb.min())} max={int(rgb.max())} unique_rgb≈{n_unique}"
            )
            if i == 1 and (n_unique < 8 or float(rgb.std()) < 1.0):
                print(
                    "  WARNING: captured frame looks flat (sky-only?). "
                    "SSFR may not have reached get_frame().",
                    file=sys.stderr,
                    flush=True,
                )
        print(
            f"  [{i}/{total}] t={self.sim_time:.2f}s "
            f"bubbles={bc} entrained={n_ent} atmV={atm_v:.1f} "
            f"sim={self._last_ms:.0f}ms{frame_bit}",
            flush=True,
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record HOME-FSLBM dam-break foam to MP4 (headless GL → ffmpeg).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Target video duration in seconds (default {DEFAULT_DURATION_S}).",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Output frame rate (default {DEFAULT_FPS}).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output .mp4 path (default {DEFAULT_OUTPUT}).",
    )
    p.add_argument("--n", type=int, default=DEFAULT_N, help=f"Lattice resolution N³ (default {DEFAULT_N}).")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Frame width (default {DEFAULT_WIDTH}).")
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Frame height (default {DEFAULT_HEIGHT}).")
    p.add_argument(
        "--substeps",
        type=int,
        default=DEFAULT_SUBSTEPS,
        help=f"Lattice steps per video frame (default {DEFAULT_SUBSTEPS}).",
    )
    p.add_argument("--crf", type=int, default=DEFAULT_CRF, help=f"x264/AV1 CRF quality (default {DEFAULT_CRF}).")
    p.add_argument("--ffmpeg", type=str, default=None, help="Explicit path to ffmpeg executable.")
    p.add_argument("--device", type=str, default=None, help="Warp device, e.g. cuda:0.")
    p.add_argument(
        "--gravity-z",
        type=float,
        default=DEFAULT_GRAVITY_Z,
        help=f"Lattice gravity_z (default {DEFAULT_GRAVITY_Z}).",
    )
    p.add_argument(
        "--surface-tension",
        type=float,
        default=DEFAULT_SURFACE_TENSION,
        help=f"Surface tension (default {DEFAULT_SURFACE_TENSION}).",
    )
    p.add_argument(
        "--no-headless",
        action="store_true",
        help="Show a GL window while recording (default is headless).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.duration <= 0.0:
        raise SystemExit("--duration must be > 0")
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    if args.n < 16:
        raise SystemExit("--n must be >= 16")

    if args.device:
        wp.set_device(args.device)
    wp.init()

    num_frames = int(round(float(args.duration) * float(args.fps)))
    if num_frames < 1:
        raise SystemExit("duration*fps produced zero frames")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Recording {num_frames} frames @ {args.fps} fps "
        f"→ {args.duration:.3f}s video, {args.width}x{args.height}, "
        f"N={args.n}, out={output}",
        flush=True,
    )

    viewer = FluidViewerGL(
        width=int(args.width),
        height=int(args.height),
        vsync=False,
        headless=not args.no_headless,
    )

    recorder = DamBreakFoamRecorder(
        viewer,
        n=int(args.n),
        substeps=int(args.substeps),
        fps=int(args.fps),
        gravity_z=float(args.gravity_z),
        surface_tension=float(args.surface_tension),
    )

    # Warm up GL / SSFR before opening the encoder.
    recorder.render()
    if not recorder.ssfr.available:
        viewer.close()
        raise SystemExit("ScreenSpaceFluidRenderer failed to initialise; aborting record.")

    proc = _open_ffmpeg(
        output,
        int(args.width),
        int(args.height),
        int(args.fps),
        int(args.crf),
        ffmpeg_path=args.ffmpeg,
    )
    assert proc.stdin is not None
    frame_buf: wp.array | None = None
    t_wall0 = time.perf_counter()

    try:
        for i in range(1, num_frames + 1):
            if not viewer.is_running():
                print("Viewer closed early; stopping encode.", file=sys.stderr, flush=True)
                break
            recorder.step()
            recorder.render()
            frame_buf = viewer.get_frame(target_image=frame_buf, render_ui=False)
            rgb = np.ascontiguousarray(frame_buf.numpy())
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise RuntimeError(f"unexpected frame shape/dtype: {rgb.shape} {rgb.dtype}")
            # One frame → one write + flush so the encoder sees each frame promptly.
            proc.stdin.write(rgb.tobytes())
            proc.stdin.flush()
            recorder.log_progress(i, num_frames, rgb=rgb)
    except BrokenPipeError as exc:
        raise RuntimeError(f"ffmpeg pipe broken (see {getattr(proc, '_wanphys_err_path', '?')}): {exc}") from exc
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        ret = proc.wait()
        err_f = getattr(proc, "_wanphys_err_file", None)
        if err_f is not None:
            err_f.close()
        viewer.close()

    err_path = getattr(proc, "_wanphys_err_path", None)
    if ret != 0:
        err = ""
        if err_path is not None and Path(err_path).is_file():
            err = Path(err_path).read_text(encoding="utf-8", errors="replace")
        raise SystemExit(f"ffmpeg failed (exit {ret}): {err}")
    if err_path is not None:
        try:
            Path(err_path).unlink(missing_ok=True)
        except OSError:
            pass

    elapsed = time.perf_counter() - t_wall0
    print(
        f"Wrote {output} ({output.stat().st_size / (1024 * 1024):.1f} MiB) "
        f"in {elapsed:.1f}s wall time.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
