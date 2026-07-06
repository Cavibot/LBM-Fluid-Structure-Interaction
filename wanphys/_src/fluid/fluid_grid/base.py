from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import warp as wp

from wanphys._src.core.domain import DomainModel, DomainSolver, DomainState, Domain


@dataclass
class FluidGridModelBase(DomainModel):
    # Grid Config
    fluid_grid_res: tuple = (32.0, 32.0, 32.0)
    fluid_grid_cell_size: float = 0.1

    # Pressure solver config
    pressure_iteration: int = 20
    pressure_solver: str = "jacobi"
    pcg_max_iterations: int = 32
    pcg_tolerance: float = 1.0e-5
    pcg_check_interval: int = 0
    mgpcg_smoother_iterations: int = 2
    mgpcg_coarse_iterations: int = 2
    device: Optional[str] = None
    _device: Any = field(init=False, repr=False)

    def __post_init__(self):
        if self.device is None:
            self._device = wp.get_device()
        else:
            self._device = wp.get_device(self.device)

    @property
    def resolution(self) -> tuple:
        return self.fluid_grid_res

    @property
    def cell_size(self) -> float:
        return self.fluid_grid_cell_size

    @property
    def nx(self) -> int:
        return self.fluid_grid_res[0]

    @property
    def ny(self) -> int:
        return self.fluid_grid_res[1]

    @property
    def nz(self) -> int:
        return self.fluid_grid_res[2]

    @property
    def dh(self) -> float:
        return self.fluid_grid_cell_size

    @property
    def pressure_iter(self) -> int:
        return self.pressure_iteration


class FluidGridStateBase(DomainState):
    def __init__(self, model: FluidGridModelBase, requires_grad: bool = False) -> None:
        self.model = model

        nx = int(model.fluid_grid_res[0])
        ny = int(model.fluid_grid_res[1])
        nz = int(model.fluid_grid_res[2])
        self.res = (nx, ny, nz)
        self.device = model._device
        self.requires_grad = requires_grad

        self.vel_u = wp.zeros((nx + 1, ny, nz), dtype=float, device=self.device, requires_grad=requires_grad)
        self.vel_v = wp.zeros((nx, ny + 1, nz), dtype=float, device=self.device, requires_grad=requires_grad)
        self.vel_w = wp.zeros((nx, ny, nz + 1), dtype=float, device=self.device, requires_grad=requires_grad)

        self.pressure = wp.zeros((nx, ny, nz), dtype=float, device=self.device, requires_grad=requires_grad)
        self.density = wp.zeros((nx, ny, nz), dtype=float, device=self.device, requires_grad=requires_grad)
        self.solid_phi = wp.zeros((nx, ny, nz), dtype=float, device=self.device, requires_grad=requires_grad)
        self.solid_phi.fill_(1000.0)
        self.solid_body_id = wp.full((nx, ny, nz), -1, dtype=wp.int32, device=self.device)

    def clear(self):
        self.vel_u.zero_()
        self.vel_v.zero_()
        self.vel_w.zero_()
        self.pressure.zero_()
        self.density.zero_()
        self.solid_phi.fill_(1000.0)
        self.solid_body_id.fill_(-1)

    def clone(self) -> "FluidGridStateBase":
        """Create a deep copy of the state."""
        new_state = FluidGridStateBase(self.model, requires_grad=self.requires_grad)

        wp.copy(new_state.vel_u, self.vel_u)
        wp.copy(new_state.vel_v, self.vel_v)
        wp.copy(new_state.vel_w, self.vel_w)
        wp.copy(new_state.pressure, self.pressure)
        wp.copy(new_state.density, self.density)
        wp.copy(new_state.solid_phi, self.solid_phi)
        wp.copy(new_state.solid_body_id, self.solid_body_id)

        return new_state


class FluidGridSolverBase(DomainSolver):
    @abstractmethod
    def step(
        self,
        state_in: FluidGridStateBase,
        state_out: FluidGridStateBase,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        ...


class FluidGridMacSolverBase(FluidGridSolverBase, ABC):
    """Shared MAC-grid solver pipeline for smoke and liquid variants."""

    def __init__(self, model: FluidGridModelBase) -> None:
        self.model = model
        self.nx = model.nx
        self.ny = model.ny
        self.nz = model.nz
        self.device = model._device

        self.div_array = wp.zeros((self.nx, self.ny, self.nz), dtype=float, device=self.device)
        self.u_aux = wp.zeros((self.nx + 1, self.ny, self.nz), dtype=float, device=self.device)
        self.v_aux = wp.zeros((self.nx, self.ny + 1, self.nz), dtype=float, device=self.device)
        self.w_aux = wp.zeros((self.nx, self.ny, self.nz + 1), dtype=float, device=self.device)

        self._pressure_solver_method = str(model.pressure_solver).strip().lower()
        self._pressure_solver_impl = self._build_pressure_solver(self._pressure_solver_method)

    def step(
        self,
        state_in: FluidGridStateBase,
        state_out: FluidGridStateBase,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        self._prepare_step(state_in, state_out, dt, contacts=contacts, control=control)
        self._advect_velocity(state_in, state_out, dt)
        self._after_velocity_advection(state_in, state_out, dt, contacts=contacts, control=control)
        self._compute_divergence(state_out, dt)
        self._solve_pressure(state_in, state_out, dt)
        self._project_velocity(state_out, dt)
        self._enforce_boundary(state_out)
        self._after_pressure_projection(state_in, state_out, dt, contacts=contacts, control=control)

    def _prepare_step(
        self,
        state_in: FluidGridStateBase,
        state_out: FluidGridStateBase,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Hook for variant-specific work before velocity advection."""

    def _after_velocity_advection(
        self,
        state_in: FluidGridStateBase,
        state_out: FluidGridStateBase,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Hook for forces or extra operators before pressure solve."""

    def _after_pressure_projection(
        self,
        state_in: FluidGridStateBase,
        state_out: FluidGridStateBase,
        dt: float,
        contacts: Any | None = None,
        control: Any | None = None,
    ) -> None:
        """Hook for scalar advection or bookkeeping after projection."""

    def _advect_velocity(self, state_in: FluidGridStateBase, state_out: FluidGridStateBase, dt: float) -> None:
        advect_u, advect_v, advect_w = self._velocity_advect_kernels()

        wp.launch(
            advect_u,
            dim=(self.nx + 1, self.ny, self.nz),
            inputs=[state_in.vel_u, state_in.vel_v, state_in.vel_w, self.u_aux, dt, self.model.dh, self.nx, self.ny, self.nz],
        )
        wp.launch(
            advect_v,
            dim=(self.nx, self.ny + 1, self.nz),
            inputs=[state_in.vel_u, state_in.vel_v, state_in.vel_w, self.v_aux, dt, self.model.dh, self.nx, self.ny, self.nz],
        )
        wp.launch(
            advect_w,
            dim=(self.nx, self.ny, self.nz + 1),
            inputs=[state_in.vel_u, state_in.vel_v, state_in.vel_w, self.w_aux, dt, self.model.dh, self.nx, self.ny, self.nz],
        )

        wp.copy(state_out.vel_u, self.u_aux)
        wp.copy(state_out.vel_v, self.v_aux)
        wp.copy(state_out.vel_w, self.w_aux)

    def _solve_pressure(self, state_in: FluidGridStateBase, state_out: FluidGridStateBase, dt: float) -> None:
        # Allow changing `model.pressure_solver` at runtime without recreating the solver.
        method = str(self.model.pressure_solver).strip().lower()
        if method != self._pressure_solver_method:
            self.set_pressure_solver(method)
        self._pressure_solver_impl.solve(self, state_in, state_out, dt)

    def _build_pressure_solver(self, method: str) -> Any:
        from .pressure_solver import build_pressure_solver

        return build_pressure_solver(method, self)

    def set_pressure_solver(self, method: str) -> None:
        normalized = str(method).strip().lower()
        self.model.pressure_solver = normalized
        self._pressure_solver_method = normalized
        self._pressure_solver_impl = self._build_pressure_solver(normalized)

    @abstractmethod
    def _velocity_advect_kernels(self) -> tuple[Any, Any, Any]:
        """Return MAC velocity advection kernels for u, v, w."""

    @abstractmethod
    def _compute_divergence(self, state: FluidGridStateBase, dt: float) -> None:
        """Populate ``self.div_array`` from the current state."""

    @abstractmethod
    def _pressure_kernel(self) -> Any:
        """Return the pressure iteration kernel used by Jacobi solver."""

    @abstractmethod
    def _pressure_iteration_inputs(
        self,
        pressure_src: Any,
        pressure_dst: Any,
        state: FluidGridStateBase,
        dt: float,
    ) -> list[Any]:
        """Return launch inputs for one pressure Jacobi iteration."""

    @abstractmethod
    def _pressure_apply_operator_kernel(self) -> Any:
        """Return kernel for matrix-free pressure operator ``y = A x``."""

    @abstractmethod
    def _pressure_apply_operator_inputs(
        self,
        x: Any,
        y: Any,
        state: FluidGridStateBase,
    ) -> list[Any]:
        """Return launch inputs for ``_pressure_apply_operator_kernel``."""

    @abstractmethod
    def _pressure_build_inv_diag_kernel(self) -> Any:
        """Return kernel that fills inverse diagonal preconditioner."""

    @abstractmethod
    def _pressure_build_inv_diag_inputs(
        self,
        inv_diag: Any,
        state: FluidGridStateBase,
    ) -> list[Any]:
        """Return launch inputs for ``_pressure_build_inv_diag_kernel``."""

    @abstractmethod
    def _project_velocity(self, state: FluidGridStateBase, dt: float) -> None:
        """Apply pressure projection back to the MAC velocity field."""

    @abstractmethod
    def _enforce_boundary(self, state: FluidGridStateBase) -> None:
        """Apply solid and domain boundary conditions."""


class FluidGridDomainBase(Domain):
    """Domain for grid-based Eulerian fluid simulation (Collocated Grid).

    This domain encapsulates the Model, State creation, and Solver logic.
    State is managed internally with double buffering.

    Example:
        >>> model = FluidGridModel(fluid3d_res=128, fluid3d_cell_size=0.1)
        >>> fluid = FluidGridDomain(model)
        >>> fluid.create_state()
        >>> fluid.step(dt=1/60)
    """

    def __init__(
        self,
        model: FluidGridModelBase,
        solver: FluidGridSolverBase | None = None,
    ):
        """Initialize fluid domain.

        Args:
            model: FluidGridModel configuration.
            solver: Fluid solver instance (optional).
        """
        self._model = model
        self._solver = solver or FluidGridSolver(model)

        # Double-buffered state (lazily created)
        self._state_in: FluidGridStateBase | None = None
        self._state_out: FluidGridStateBase | None = None

    @property
    def name(self) -> str:
        """Domain identifier."""
        return "fluid_grid"

    @property
    def model(self) -> FluidGridModelBase:
        """Fluid model configuration."""
        return self._model

    @property
    def solver(self) -> FluidGridSolverBase:
        """Fluid solver instance."""
        return self._solver

    @property
    def state(self) -> FluidGridStateBase:
        """Current simulation state (active buffer)."""
        if self._state_in is None:
            self.create_state()
        return self._state_in

    def create_state(self) -> None:
        """Initialize internal double-buffered states from the model."""
        self._state_in = FluidGridStateBase(self._model)
        self._state_out = FluidGridStateBase(self._model)

    def step(
        self,
        dt: float,
        contacts=None,
    ) -> None:
        """Advance fluid simulation by dt.

        Args:
            dt: Timestep in seconds.
            contacts: Unused, accepted for API compatibility.
        """
        self._solver.step(self._state_in, self._state_out, dt)
        self._state_in, self._state_out = self._state_out, self._state_in
