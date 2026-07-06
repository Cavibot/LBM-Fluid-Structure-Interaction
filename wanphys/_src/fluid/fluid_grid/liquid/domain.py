from __future__ import annotations

from .model import FluidGridLiquidModel
from .state import FluidGridLiquidState
from .solver import FluidGridLiquidSolver
from ..base import FluidGridDomainBase
 
class FluidGridLiquidDomain(FluidGridDomainBase):
    def __init__(self,
                 model: FluidGridLiquidModel,
                 solver: FluidGridLiquidSolver) -> None:
        self._model = model
        self._solver = solver or FluidGridLiquidSolver(model)
    
    @property
    def name(self)->str:
        return "fluid_grid_liquid"
    
    @property
    def model(self)->FluidGridLiquidModel:
        return self._model
    
    @property
    def solver(self)->FluidGridLiquidSolver:
        return self._solver
    
    @property
    def state(self)->FluidGridLiquidState:
        if self._state_in is None:
            self.create_state()
        return self._state_in
    
    def create_state(self)->None:
        self._state_in = FluidGridLiquidState(self._model)
        self._state_out = FluidGridLiquidState(self._model)
        return self._state_in
    
    def step(
        self,
        dt: float,
        contacts=None,
    ) -> None:
        self._solver.step(self._state_in, self._state_out, dt)
        self._state_in, self._state_out = self._state_out, self._state_in
