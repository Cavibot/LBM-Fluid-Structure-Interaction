from __future__ import annotations

from .model import FluidGridModel
from .state import FluidGridState
from .solver import FluidGridSolver
from wanphys._src.core.domain import Domain

class FluidGridDomain(Domain):
    def __init__(
        self,
        model: FluidGridModel,
        solver: FluidGridSolver | None = None,
    ):
        self._model = model
        self._solver = solver or FluidGridSolver(model)
        self._state_in: FluidGridState | None = None
        self._state_out: FluidGridState | None = None

    @property
    def name(self) -> str:
        return "fluid_grid"

    @property
    def model(self) -> FluidGridModel:
        return self._model
    
    @property
    def solver(self) -> FluidGridSolver:
        return self._solver

    @property
    def state(self) -> FluidGridState:
        if self._state_in is None:
            self.create_state()
        return self._state_in

    def create_state(self) -> FluidGridState:
        self._state_in = FluidGridState(self._model)
        self._state_out = FluidGridState(self._model)
        return self._state_in

    def step(
        self,
        dt: float,
        contacts=None,
    ) -> None:
        self._solver.step(self._state_in, self._state_out, dt)
        # Swap buffers
        self._state_in, self._state_out = self._state_out, self._state_in
