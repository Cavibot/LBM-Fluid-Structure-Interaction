# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""D3Q19 Lattice Boltzmann method - simulation domain."""

from __future__ import annotations

from wanphys._src.core.domain import Domain

from .model import LbmModel
from .solver import LbmSolver
from .state import LbmState


class LbmDomain(Domain):
    """D3Q19 LBM fluid simulation domain with double-buffered state.

    This is the top-level entry point for running LBM simulations.
    It owns the model (static config), solver (timestepping), and
    double-buffered state (current / next).

    Example
    -------
    >>> model = LbmModel(fluid_grid_res=(64, 64, 64),
    ...                  fluid_grid_cell_size=0.1,
    ...                  tau=0.55)
    >>> domain = LbmDomain(model)
    >>> domain.create_state()
    >>> domain.solver.initialize_equilibrium(domain.state,
    ...                                      rho0=1.0,
    ...                                      u0=(0.0, 0.0, 0.0))
    >>> for _ in range(100):
    ...     domain.step(dt=1.0)
    """

    def __init__(
        self,
        model: LbmModel,
        solver: LbmSolver | None = None,
    ) -> None:
        self._model: LbmModel = model
        self._solver: LbmSolver = solver or LbmSolver(model)

        # Double-buffered state (created lazily or via create_state)
        self._state_in: LbmState | None = None
        self._state_out: LbmState | None = None

    # ------------------------------------------------------------------
    # Domain protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique domain identifier for composite simulations."""
        return "fluid_grid_lbm"

    @property
    def model(self) -> LbmModel:
        """Static LBM configuration."""
        return self._model

    @property
    def solver(self) -> LbmSolver:
        """LBM solver (timestepping logic)."""
        return self._solver

    @property
    def state(self) -> LbmState:
        """Current (active) simulation state."""
        if self._state_in is None:
            self.create_state()
        return self._state_in

    def create_state(self) -> LbmState:
        """Allocate the double-buffered GPU state from the model.

        Returns the newly created active state.
        """
        self._state_in = LbmState(self._model)
        self._state_out = LbmState(self._model)
        return self._state_in

    def step(
        self,
        dt: float,
        contacts: object = None,
    ) -> None:
        """Advance the domain by *dt* (accepted but ignored - see :class:`LbmSolver`)."""
        self._solver.step(
            self._state_in,
            self._state_out,
            dt,
            contacts=contacts,
        )
        # Swap buffers
        self._state_in, self._state_out = self._state_out, self._state_in

    def pre_step(self, dt: float) -> None:
        """Hook called before each step (no-op by default)."""
        pass

    def post_step(self, dt: float) -> None:
        """Hook called after each step (no-op by default)."""
        pass
