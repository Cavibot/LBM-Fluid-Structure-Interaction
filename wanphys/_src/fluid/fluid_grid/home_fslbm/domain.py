# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM simulation domain with double-buffered state.

Reference:
    [REF] Home-FSLBM inc/3D/cpu/mrSolver3D.h
"""

from __future__ import annotations

import warp as wp

from wanphys._src.core.domain import Domain

from .model import HomeFSLbmModel
from .solver import HomeFSLbmSolver
from .state import HomeFSLbmState


class HomeFSLbmDomain(Domain):
    """HOME-FSLBM free-surface fluid simulation domain.

    Owns the model (static config), solver (timestepping), and
    double-buffered state (current / next).  The domain manages
    the MomSwap (pointer exchange) between the two state buffers
    after each step.

    Example
    -------
    >>> model = HomeFSLbmModel(fluid_grid_res=(64, 64, 64),
    ...                        fluid_grid_cell_size=0.1, tau=0.55)
    >>> domain = HomeFSLbmDomain(model)
    >>> domain.create_state()
    >>> domain.step(dt=1.0)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: HomeFSLbmModel,
        solver: HomeFSLbmSolver | None = None,
    ) -> None:
        self._model: HomeFSLbmModel = model
        self._solver: HomeFSLbmSolver = solver or HomeFSLbmSolver(model)

        # Double-buffered state (lazy, created via create_state)
        self._state_in: HomeFSLbmState | None = None
        self._state_out: HomeFSLbmState | None = None

    # ------------------------------------------------------------------
    # Domain protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique domain identifier for composite simulations."""
        return "fluid_grid_home_fslbm"

    @property
    def model(self) -> HomeFSLbmModel:
        """Static HOME-FSLBM configuration."""
        return self._model

    @property
    def solver(self) -> HomeFSLbmSolver:
        """HOME-FSLBM solver."""
        return self._solver

    @property
    def state(self) -> HomeFSLbmState:
        """Current (active) simulation state."""
        if self._state_in is None:
            self.create_state()
        return self._state_in

    def create_state(self) -> HomeFSLbmState:
        """Allocate the double-buffered GPU state from the model.

        Returns the newly created active state.
        """
        self._state_in = HomeFSLbmState(self._model)
        self._state_out = HomeFSLbmState(self._model)
        return self._state_in

    def step(self, dt: float, contacts: object = None) -> None:
        """Advance the domain by *dt*.

        Args:
            dt: Timestep in seconds (ignored; LBM uses lattice dt=1).
            contacts: Ignored (reserved for rigid-body coupling).
        """
        if self._state_in is None:
            self.create_state()

        self._solver.step(self._state_in, self._state_out, dt)

        # Swap entire state objects (same as LbmDomain)
        # [REF]: MomSwap(fMom, fMomPost) in mrSolver3D_step2Kernel
        self._state_in, self._state_out = self._state_out, self._state_in

    def pre_step(self, dt: float) -> None:
        """Hook called before each step (no-op)."""
        pass

    def post_step(self, dt: float) -> None:
        """Hook called after each step (no-op)."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _swap_moments(self) -> None:
        """Swap the f_mom / f_mom_post pointers between the two states.

        This implements the [REF] MomSwap operation: after the solver
        writes results to state_out.f_mom_post, we exchange pointers
        so that state_in.f_mom always holds the latest moments.
        """
        self._state_in.f_mom, self._state_out.f_mom_post = (
            self._state_out.f_mom_post,
            self._state_in.f_mom,
        )
