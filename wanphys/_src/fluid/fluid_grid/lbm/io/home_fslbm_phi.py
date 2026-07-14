# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Load Home-FSLBM ``mlSavePhi`` binary dumps for wanphys SSFR replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_phi_bin(path: str | Path) -> tuple[int, int, int, np.ndarray]:
    """Load one ``phi%05d.bin`` file written by Home-FSLBM ``mrSolver3D::mlSavePhi``.

    File layout: three 4-byte ints (Nx, Ny, Nz) followed by ``Nx*Ny*Nz``
    ``float32`` phi values in Home index order
    ``curind = z * Ny * Nx + y * Nx + x``.

    Returns ``(nx, ny, nz, phi)`` with ``phi.shape == (nx, ny, nz)``.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 12:
        raise ValueError(f"{path}: file too small ({len(raw)} bytes)")
    # Home writes ``int`` dims with ``fwrite(..., sizeof(float), ...)`` (4 bytes each).
    dims = np.frombuffer(raw[:12], dtype=np.int32)
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    flat = np.frombuffer(raw[12:], dtype=np.float32)
    expected = nx * ny * nz
    if flat.size != expected:
        raise ValueError(
            f"{path}: expected {expected} phi values, got {flat.size} "
            f"for grid ({nx}, {ny}, {nz})"
        )
    # Home z-major → wanphys (nx, ny, nz).
    phi = flat.reshape(nz, ny, nx).transpose(2, 1, 0)
    return nx, ny, nz, np.ascontiguousarray(phi, dtype=np.float32)


def list_phi_frames(data_dir: str | Path) -> list[Path]:
    """Sorted list of ``phi*.bin`` frames under ``data_dir``."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []
    frames = sorted(data_dir.glob("phi*.bin"))
    return frames


def frame_index(path: str | Path) -> int:
    """Parse ``phi00042.bin`` → 42."""
    stem = Path(path).stem
    if not stem.startswith("phi"):
        raise ValueError(f"not a Home phi frame: {path}")
    return int(stem[3:])
