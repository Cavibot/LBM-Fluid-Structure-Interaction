# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Double-buffered HOME-LBM domain."""

from __future__ import annotations

from wanphys._src.core.domain import Domain

from .model import HomeLbmModel
from .solver import HomeLbmSolver
from .state import HomeLbmState


class HomeLbmDomain(Domain):
    def __init__(self, model: HomeLbmModel, solver: HomeLbmSolver | None = None) -> None:
        self._model = model
        self._solver = solver or HomeLbmSolver(model)
        self._state_in: HomeLbmState | None = None
        self._state_out: HomeLbmState | None = None

    @property
    def name(self) -> str:
        return "fluid_grid_home_lbm"

    @property
    def model(self) -> HomeLbmModel:
        return self._model

    @property
    def solver(self) -> HomeLbmSolver:
        return self._solver

    @property
    def state(self) -> HomeLbmState:
        if self._state_in is None:
            self.create_state()
        assert self._state_in is not None
        return self._state_in

    def create_state(self) -> HomeLbmState:
        self._state_in = HomeLbmState(self._model)
        self._state_out = HomeLbmState(self._model)
        return self._state_in

    def step(self, dt: float, contacts: object = None) -> None:
        if self._state_in is None or self._state_out is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None
        self._solver.step(self._state_in, self._state_out, dt, contacts=contacts)
        self._state_in, self._state_out = self._state_out, self._state_in
