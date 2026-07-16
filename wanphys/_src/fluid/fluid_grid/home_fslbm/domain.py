# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME-FSLBM simulation domain.

Defines :class:`HomeFslbmDomain`, the top-level entry point for running
HOME-FSLBM free-surface simulations.  Owns the model (static config),
solver (timestepping), and double-buffered state (current / next).
"""

from __future__ import annotations

from wanphys._src.core.domain import Domain

from .model import HomeFslbmModel
from .solver import HomeFslbmSolver
from .state import HomeFslbmState


class HomeFslbmDomain(Domain):
    """HOME-FSLBM fluid simulation domain with double-buffered state.

    This is the top-level entry point for running HOME-FSLBM simulations.
    It owns the model (static config), solver (timestepping), and
    double-buffered state (current / next).

    Example
    -------
    >>> model = HomeFslbmModel.default_32()
    >>> domain = HomeFslbmDomain(model)
    >>> domain.create_state()
    >>> domain.solver.initialize_equilibrium(domain.state)
    >>> for _ in range(100):
    ...     domain.step(dt=1.0)
    """

    def __init__(
        self,
        model: HomeFslbmModel,
        solver: HomeFslbmSolver | None = None,
    ) -> None:
        self._model: HomeFslbmModel = model
        self._solver: HomeFslbmSolver = solver or HomeFslbmSolver(model)

        # Double-buffered state (created lazily or via create_state)
        self._state_in: HomeFslbmState | None = None
        self._state_out: HomeFslbmState | None = None

    # ------------------------------------------------------------------
    # Domain protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique domain identifier for composite simulations."""
        return "fluid_grid_home_fslbm"

    @property
    def model(self) -> HomeFslbmModel:
        """Static HOME-FSLBM configuration."""
        return self._model

    @property
    def solver(self) -> HomeFslbmSolver:
        """HOME-FSLBM solver (timestepping logic)."""
        return self._solver

    @property
    def state(self) -> HomeFslbmState:
        """Current (active) simulation state."""
        if self._state_in is None:
            self.create_state()
        return self._state_in

    def create_state(self) -> HomeFslbmState:
        """Allocate the double-buffered GPU state from the model.

        Returns the newly created active state.
        """
        self._state_in = HomeFslbmState(self._model)
        self._state_out = HomeFslbmState(self._model)
        return self._state_in

    def step(
        self,
        dt: float,
        contacts: object = None,
    ) -> None:
        """Advance the domain by one timestep.

        The domain reads from the current state buffer, writes to the back
        buffer, and swaps them internally.
        """
        self._solver.step(
            self._state_in,
            self._state_out,
            dt,
            contacts=contacts,
        )
        # Swap buffers
        self._state_in, self._state_out = self._state_out, self._state_in

    def pre_step(self, dt: float) -> None:
        """Hook called before each step."""
        pass

    def post_step(self, dt: float) -> None:
        """Hook called after each step."""
        pass
