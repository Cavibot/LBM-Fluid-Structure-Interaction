# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for HOME-FSLBM Phase 1 unit tests."""

from __future__ import annotations

import pytest

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
