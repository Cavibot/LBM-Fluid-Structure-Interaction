# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Double-buffered domain for the isolated periodic HOME core."""

from __future__ import annotations

from wanphys._src.core.domain import Domain

from .model import HomeCoreModel
from .solver import HomeCoreSolver
from .state import HomeCoreState


class HomeCoreDomain(Domain):
    def __init__(self, model: HomeCoreModel, solver: HomeCoreSolver | None = None) -> None:
        if solver is not None and solver.model is not model:
            raise ValueError("an injected HOME core solver must own the same model")
        self._model = model
        self._solver = solver or HomeCoreSolver(model)
        self._state_in: HomeCoreState | None = None
        self._state_out: HomeCoreState | None = None

    @property
    def name(self) -> str:
        return "fluid_grid_home_rebuild_core"

    @property
    def model(self) -> HomeCoreModel:
        return self._model

    @property
    def solver(self) -> HomeCoreSolver:
        return self._solver

    @property
    def state(self) -> HomeCoreState:
        if self._state_in is None:
            self.create_state()
        assert self._state_in is not None
        return self._state_in

    def create_state(self) -> HomeCoreState:
        self._state_in = HomeCoreState(self._model)
        self._state_out = HomeCoreState(self._model)
        return self._state_in

    def step(self, dt: float, contacts: object = None) -> None:
        del contacts
        if self._state_in is None or self._state_out is None:
            self.create_state()
        assert self._state_in is not None and self._state_out is not None
        self._solver.step(self._state_in, self._state_out, dt)
        self._state_in, self._state_out = self._state_out, self._state_in
