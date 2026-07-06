from __future__ import annotations

from wanphys._src.core.domain import Domain

from .model import FluidGridApicModel
from .solver import FluidGridApicSolver
from .state import FluidGridApicState


class FluidGridApicDomain(Domain):
    def __init__(
        self,
        model: FluidGridApicModel,
        solver: FluidGridApicSolver,
    ) -> None:
        self._model = model
        self._solver = solver or FluidGridApicSolver(model)
        self._state_in: FluidGridApicState | None = None
        self._state_out: FluidGridApicState | None = None

    @property
    def name(self) -> str:
        return "fluid_grid_apic"

    @property
    def model(self) -> FluidGridApicModel:
        return self._model

    @property
    def solver(self) -> FluidGridApicSolver:
        return self._solver
    
    @property
    def state(self) -> FluidGridApicState:
        if self._state_in is None:
            self.create_state()
        return self._state_in

    def create_state(self) -> FluidGridApicState:
        self._state_in = FluidGridApicState(self._model)
        self._state_out = FluidGridApicState(self._model)
        return self._state_in

    def step(
        self,
        dt: float,
        contacts=None,
    ) -> None:
        self._solver.step(self._state_in, self._state_out, dt)
        # Swap buffers
        self._state_in, self._state_out = self._state_out, self._state_in
