# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for HOME-FSLBM Phase 1 unit tests."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running Phase-5 stability gates (e.g. 1e4-step volume)"
    )


def _load_float_txt(path: Path, dtype: type) -> np.ndarray:
    """Load one-value-per-line floats; normalize MSVC ``-nan(ind)`` tokens."""
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    norm = []
    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if low in ("-nan(ind)", "nan(ind)", "+nan(ind)", "-nan", "nan", "-1.#ind00", "1.#ind00"):
            norm.append("nan")
        elif low in ("-inf", "inf", "+inf", "1.#inf00", "-1.#inf00"):
            norm.append("-inf" if low.startswith("-") else "inf")
        else:
            norm.append(s)
    return np.loadtxt(StringIO("\n".join(norm)), dtype=dtype)

# ---------------------------------------------------------------------------
# Golden data loader
# ---------------------------------------------------------------------------

GOLDEN_DIR = Path(__file__).parent / "golden_data"


def load_golden(name: str) -> np.ndarray:
    """Load a golden data file (one float per line) as a numpy array."""
    return np.loadtxt(GOLDEN_DIR / f"{name}.txt", dtype=np.float32)


def load_surface_golden(scene_name: str) -> dict[str, np.ndarray]:
    """Load Phase 2 surface golden data (multi-file-per-directory .txt format).

    Each scene is a subdirectory under ``golden_data/`` containing one
    ``.txt`` file per field.

    Parameters
    ----------
    scene_name : str
        Subdirectory name, e.g. ``"droplet_r8"``.

    Returns
    -------
    dict
        ``"f_mom_post"``: (10*N,) float32 — HOME moments [mom0..mom9] interleaved
        ``"flag"``:       (N,) int32     — cell type bitfield (stored as int32)
        ``"mass"``:       (N,) float32   — VOF mass per cell
        ``"phi"``:        (N,) float32   — volume fraction per cell
        ``"tag_matrix"``: (N,) int32     — bubble ID tag per cell
    """
    scene_dir = GOLDEN_DIR / scene_name
    return {
        "f_mom_post": np.loadtxt(scene_dir / "f_mom_post.txt", dtype=np.float32),
        "flag":       np.loadtxt(scene_dir / "flag.txt", dtype=np.int32),
        "mass":       np.loadtxt(scene_dir / "mass.txt", dtype=np.float32),
        "phi":        np.loadtxt(scene_dir / "phi.txt", dtype=np.float32),
        "tag_matrix": np.loadtxt(scene_dir / "tag_matrix.txt", dtype=np.int32),
    }


# Optional per-scene bubble golden fields (one value per line, ref x-fastest flat).
_BUBBLE_OPTIONAL_FIELDS: dict[str, type] = {
    "input_matrix": np.int32,
    "label_matrix": np.int32,
    "flag": np.int32,
    "phi": np.float32,
    "delta_phi": np.float32,
    "tag_matrix": np.int32,
    "previous_tag": np.int32,
    "previous_merge_tag": np.int32,
    "merge_detector": np.int32,
    "bubble_volume": np.float64,
    "bubble_init_volume": np.float64,
    "bubble_rho": np.float64,
    # Phase 4 gas / foam
    "g_mom": np.float32,
    "delta_g": np.float32,
    "c_value": np.float32,
    "disjoin_force": np.float32,
    "pop_in": np.float32,
    "pop_out": np.float32,
    "moment_in": np.float32,
    "moment_out": np.float32,
    "uxuyuz": np.float32,
    "volume_rel_drift": np.float64,
    "volume_series": np.float64,
    "sum_disjoin": np.float32,
    "max_disjoin": np.float32,
    "com_distance": np.float32,
}


def bubble_golden_dir(scene_name: str) -> Path:
    """Return ``golden_data/<scene_name>/`` path."""
    return GOLDEN_DIR / scene_name


def bubble_golden_available(scene_name: str, required: list[str] | None = None) -> bool:
    """True if the scene directory exists and contains the required ``.txt`` files."""
    scene_dir = bubble_golden_dir(scene_name)
    if not scene_dir.is_dir():
        return False
    req = required or ["label_matrix"]
    return all((scene_dir / f"{name}.txt").is_file() for name in req)


def load_bubble_golden(scene_name: str) -> dict[str, np.ndarray | int | float]:
    """Load Phase 3 bubble golden data (multi-file-per-directory .txt).

    Reference dumps use x-fastest flat layout
    ``idx = x + nx*(y + ny*z)``.  Callers should reorder with
    :func:`reorder_ref_scalar_to_warp` before comparing to Warp ``array3d``.

    Scalar bookkeeping files (single value):
    ``bubble_count.txt``, ``label_num.txt``, ``merge_flag.txt``,
    ``split_flag.txt``, ``steps.txt``, ``nx.txt``, ``ny.txt``, ``nz.txt``.
    """
    scene_dir = bubble_golden_dir(scene_name)
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Bubble golden scene not found: {scene_dir}")

    out: dict[str, np.ndarray | int | float] = {}
    for name, dtype in _BUBBLE_OPTIONAL_FIELDS.items():
        path = scene_dir / f"{name}.txt"
        if path.is_file():
            if dtype in (np.float32, np.float64):
                out[name] = _load_float_txt(path, dtype)
            else:
                out[name] = np.loadtxt(path, dtype=dtype)

    for scalar in (
        "bubble_count",
        "label_num",
        "merge_flag",
        "split_flag",
        "steps",
        "nx",
        "ny",
        "nz",
    ):
        path = scene_dir / f"{scalar}.txt"
        if path.is_file():
            val = np.loadtxt(path)
            out[scalar] = int(np.asarray(val).reshape(-1)[0])

    return out


def reorder_ref_scalar_to_warp(
    golden_flat: np.ndarray, nx: int, ny: int, nz: int
) -> np.ndarray:
    """Convert ref x-fastest flat field → Warp ``array3d`` layout ``[x,y,z]``.

    Reference: ``flat[x + nx*(y + ny*z)]``.
    Warp ``.numpy()`` on ``array3d(nx,ny,nz)`` is C-order with shape ``(nx,ny,nz)``.
    """
    g = np.asarray(golden_flat).reshape(-1)
    expected = nx * ny * nz
    if g.size != expected:
        raise ValueError(
            f"flat size {g.size} != nx*ny*nz={expected} ({nx}x{ny}x{nz})"
        )
    # C-reshape of x-fastest → (nz, ny, nx) then transpose to (nx, ny, nz)
    g3d = g.reshape((nz, ny, nx))
    return np.transpose(g3d, (2, 1, 0))

# ---------------------------------------------------------------------------
# Deferred imports — warp must be imported after pytest collection
# to avoid triggering kernel compilation during test discovery.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _warp():
    """Session-scoped warp handle (imported lazily)."""
    import warp as wp

    return wp


@pytest.fixture(scope="session")
def _constants():
    """Session-scoped constants module."""
    from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C

    return C


@pytest.fixture(scope="session")
def default_model(_warp, _constants):
    """32^3 HomeFslbmModel — follows Warp default device for consistency."""
    from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel

    return HomeFslbmModel(
        fluid_grid_res=(32, 32, 32),
        fluid_grid_cell_size=1.0,
        omega=1.0,
        turbulence_radius=3,
    )


@pytest.fixture
def default_state(_warp, default_model):
    """Fresh HomeFslbmState allocated from default_model."""
    from wanphys._src.fluid.fluid_grid.home_fslbm.state import HomeFslbmState

    return HomeFslbmState(default_model)


@pytest.fixture
def default_domain(_warp, default_model):
    """HomeFslbmDomain with state created."""
    from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
    from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver

    domain = HomeFslbmDomain(default_model, solver=HomeFslbmSolver(default_model))
    domain.create_state()
    return domain
