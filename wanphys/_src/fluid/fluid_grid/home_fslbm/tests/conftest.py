# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for HOME-FSLBM Phase 1 unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
