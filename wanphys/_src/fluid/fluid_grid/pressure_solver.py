from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import warp as wp

if TYPE_CHECKING:
    from .base import FluidGridMacSolverBase, FluidGridStateBase


@wp.kernel
def _build_pressure_rhs_kernel(
    div: wp.array3d(dtype=float),
    rhs: wp.array3d(dtype=float),
    scale: float,
):
    i, j, k = wp.tid()
    rhs[i, j, k] = -div[i, j, k] * scale


@wp.kernel
def _combine_axpby_kernel(
    a: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    out: wp.array3d(dtype=float),
    alpha: float,
    beta: float,
):
    i, j, k = wp.tid()
    out[i, j, k] = alpha * a[i, j, k] + beta * b[i, j, k]


@wp.kernel
def _inplace_axpy_kernel(
    y: wp.array3d(dtype=float),
    x: wp.array3d(dtype=float),
    alpha: float,
):
    i, j, k = wp.tid()
    y[i, j, k] = y[i, j, k] + alpha * x[i, j, k]


@wp.kernel
def _inplace_axpy_scalar_kernel(
    y: wp.array3d(dtype=float),
    x: wp.array3d(dtype=float),
    alpha: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    y[i, j, k] = y[i, j, k] + alpha[0] * x[i, j, k]


@wp.kernel
def _pcg_update_xr_kernel(
    x: wp.array3d(dtype=float),
    r: wp.array3d(dtype=float),
    p: wp.array3d(dtype=float),
    ap: wp.array3d(dtype=float),
    alpha: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    a = alpha[0]
    x[i, j, k] = x[i, j, k] + a * p[i, j, k]
    r[i, j, k] = r[i, j, k] - a * ap[i, j, k]


@wp.kernel
def _pcg_update_direction_kernel(
    p: wp.array3d(dtype=float),
    z: wp.array3d(dtype=float),
    beta: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    p[i, j, k] = z[i, j, k] + beta[0] * p[i, j, k]


@wp.kernel
def _apply_inv_diag_kernel(
    src: wp.array3d(dtype=float),
    inv_diag: wp.array3d(dtype=float),
    dst: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    dst[i, j, k] = src[i, j, k] * inv_diag[i, j, k]


@wp.kernel
def _dot_kernel(
    a: wp.array3d(dtype=float),
    b: wp.array3d(dtype=float),
    out: wp.array(dtype=float),
):
    i, j, k = wp.tid()
    wp.atomic_add(out, 0, a[i, j, k] * b[i, j, k])


@wp.kernel
def _safe_ratio_kernel(
    out: wp.array(dtype=float),
    num: wp.array(dtype=float),
    den: wp.array(dtype=float),
    eps: float,
):
    d = den[0]
    if wp.abs(d) <= eps:
        out[0] = 0.0
    else:
        out[0] = num[0] / d


@wp.kernel
def _scalar_copy_kernel(dst: wp.array(dtype=float), src: wp.array(dtype=float)):
    dst[0] = src[0]


class PressureLinearSolver(ABC):
    """Strategy interface for pressure linear solves.

    Each strategy owns and allocates only its required buffers.
    """

    def __init__(self, solver: "FluidGridMacSolverBase") -> None:
        self._nx = solver.nx
        self._ny = solver.ny
        self._nz = solver.nz
        self._device = solver.device

    @property
    def grid_dim(self) -> tuple[int, int, int]:
        return (self._nx, self._ny, self._nz)

    @abstractmethod
    def solve(
        self,
        solver: "FluidGridMacSolverBase",
        state_in: "FluidGridStateBase",
        state_out: "FluidGridStateBase",
        dt: float,
    ) -> None:
        ...


class JacobiPressureSolver(PressureLinearSolver):
    def __init__(self, solver: "FluidGridMacSolverBase") -> None:
        super().__init__(solver)
        self._pressure_aux = wp.zeros(self.grid_dim, dtype=float, device=self._device)

    def solve(
        self,
        solver: "FluidGridMacSolverBase",
        state_in: "FluidGridStateBase",
        state_out: "FluidGridStateBase",
        dt: float,
    ) -> None:
        wp.copy(state_out.pressure, state_in.pressure)
        p_src = state_out.pressure
        p_dst = self._pressure_aux

        for _ in range(solver.model.pressure_iteration):
            wp.launch(
                solver._pressure_kernel(),
                dim=self.grid_dim,
                inputs=solver._pressure_iteration_inputs(p_src, p_dst, state_out, dt),
            )
            p_src, p_dst = p_dst, p_src

        if p_src.ptr != state_out.pressure.ptr:
            wp.copy(state_out.pressure, p_src)


class PcgPressureSolver(PressureLinearSolver):
    _EPS = 1.0e-20

    def __init__(self, solver: "FluidGridMacSolverBase") -> None:
        super().__init__(solver)
        self._pressure_rhs = wp.zeros(self.grid_dim, dtype=float, device=self._device)
        self._r = wp.zeros(self.grid_dim, dtype=float, device=self._device)
        self._z = wp.zeros(self.grid_dim, dtype=float, device=self._device)
        self._p = wp.zeros(self.grid_dim, dtype=float, device=self._device)
        self._ap = wp.zeros(self.grid_dim, dtype=float, device=self._device)
        self._inv_diag = wp.zeros(self.grid_dim, dtype=float, device=self._device)

        # Device-side scalars to avoid per-iteration CPU synchronization.
        self._s_rz_old = wp.zeros(1, dtype=float, device=self._device)
        self._s_rz_new = wp.zeros(1, dtype=float, device=self._device)
        self._s_p_ap = wp.zeros(1, dtype=float, device=self._device)
        self._s_rr = wp.zeros(1, dtype=float, device=self._device)
        self._s_alpha = wp.zeros(1, dtype=float, device=self._device)
        self._s_beta = wp.zeros(1, dtype=float, device=self._device)

    def solve(
        self,
        solver: "FluidGridMacSolverBase",
        state_in: "FluidGridStateBase",
        state_out: "FluidGridStateBase",
        dt: float,
    ) -> None:
        wp.copy(state_out.pressure, state_in.pressure)

        scale = solver.model.dh * solver.model.dh / dt
        wp.launch(
            _build_pressure_rhs_kernel,
            dim=self.grid_dim,
            inputs=[solver.div_array, self._pressure_rhs, scale],
        )

        self._apply_operator(solver, state_out.pressure, self._ap, state_out)
        wp.launch(
            _combine_axpby_kernel,
            dim=self.grid_dim,
            inputs=[self._pressure_rhs, self._ap, self._r, 1.0, -1.0],
        )

        self._build_inv_diag(solver, state_out)
        self._apply_preconditioner(solver, self._r, self._z, state_out)
        wp.copy(self._p, self._z)

        self._dot_to(self._s_rz_old, self._r, self._z)

        # Optional low-frequency residual check to balance speed and robustness.
        check_interval = max(0, int(solver.model.pcg_check_interval))
        rhs_norm_sq = 1.0
        tol_sq = 0.0
        if check_interval > 0:
            self._dot_to(self._s_rr, self._pressure_rhs, self._pressure_rhs)
            rhs_norm_sq = float(self._s_rr.numpy()[0])
            tol_sq = (solver.model.pcg_tolerance * solver.model.pcg_tolerance) * rhs_norm_sq
            if rhs_norm_sq <= self._EPS:
                return

        max_iters = max(1, int(solver.model.pcg_max_iterations))

        for it in range(max_iters):
            self._apply_operator(solver, self._p, self._ap, state_out)
            self._dot_to(self._s_p_ap, self._p, self._ap)
            self._safe_ratio(self._s_alpha, self._s_rz_old, self._s_p_ap)

            wp.launch(
                _pcg_update_xr_kernel,
                dim=self.grid_dim,
                inputs=[state_out.pressure, self._r, self._p, self._ap, self._s_alpha],
            )

            if check_interval > 0 and (it + 1) % check_interval == 0:
                self._dot_to(self._s_rr, self._r, self._r)
                rr = float(self._s_rr.numpy()[0])
                if rr <= tol_sq:
                    break

            self._apply_preconditioner(solver, self._r, self._z, state_out)
            self._dot_to(self._s_rz_new, self._r, self._z)
            self._safe_ratio(self._s_beta, self._s_rz_new, self._s_rz_old)

            wp.launch(
                _pcg_update_direction_kernel,
                dim=self.grid_dim,
                inputs=[self._p, self._z, self._s_beta],
            )
            wp.launch(_scalar_copy_kernel, dim=1, inputs=[self._s_rz_old, self._s_rz_new])

    def _apply_operator(
        self,
        solver: "FluidGridMacSolverBase",
        x: Any,
        y: Any,
        state: "FluidGridStateBase",
    ) -> None:
        wp.launch(
            solver._pressure_apply_operator_kernel(),
            dim=self.grid_dim,
            inputs=solver._pressure_apply_operator_inputs(x, y, state),
        )

    def _build_inv_diag(self, solver: "FluidGridMacSolverBase", state: "FluidGridStateBase") -> None:
        wp.launch(
            solver._pressure_build_inv_diag_kernel(),
            dim=self.grid_dim,
            inputs=solver._pressure_build_inv_diag_inputs(self._inv_diag, state),
        )

    def _apply_preconditioner(
        self,
        solver: "FluidGridMacSolverBase",
        r: Any,
        z: Any,
        state: "FluidGridStateBase",
    ) -> None:
        del solver, state
        wp.launch(
            _apply_inv_diag_kernel,
            dim=self.grid_dim,
            inputs=[r, self._inv_diag, z],
        )

    def _dot_to(self, out_scalar: Any, a: Any, b: Any) -> None:
        out_scalar.zero_()
        wp.launch(
            _dot_kernel,
            dim=self.grid_dim,
            inputs=[a, b, out_scalar],
        )

    def _safe_ratio(self, out_scalar: Any, num_scalar: Any, den_scalar: Any) -> None:
        wp.launch(
            _safe_ratio_kernel,
            dim=1,
            inputs=[out_scalar, num_scalar, den_scalar, self._EPS],
        )


class MgpcgPressureSolver(PcgPressureSolver):
    """MGPCG-style solver.

    Current implementation keeps matrix-free PCG outer loop and uses
    multi-sweep smoothers as the preconditioner hook.
    """

    def __init__(self, solver: "FluidGridMacSolverBase") -> None:
        super().__init__(solver)
        self._smooth_residual = wp.zeros(self.grid_dim, dtype=float, device=self._device)

    def _apply_preconditioner(
        self,
        solver: "FluidGridMacSolverBase",
        r: Any,
        z: Any,
        state: "FluidGridStateBase",
    ) -> None:
        z.zero_()

        pre_sweeps = max(1, int(solver.model.mgpcg_smoother_iterations))
        coarse_sweeps = max(0, int(solver.model.mgpcg_coarse_iterations))

        for _ in range(pre_sweeps):
            self._smooth_once(solver, r, z, state, omega=0.8)

        for _ in range(coarse_sweeps):
            self._smooth_once(solver, r, z, state, omega=1.0)

    def _smooth_once(
        self,
        solver: "FluidGridMacSolverBase",
        rhs: Any,
        z: Any,
        state: "FluidGridStateBase",
        omega: float,
    ) -> None:
        self._apply_operator(solver, z, self._ap, state)
        wp.launch(
            _combine_axpby_kernel,
            dim=self.grid_dim,
            inputs=[rhs, self._ap, self._smooth_residual, 1.0, -1.0],
        )
        wp.launch(
            _apply_inv_diag_kernel,
            dim=self.grid_dim,
            inputs=[self._smooth_residual, self._inv_diag, self._ap],
        )
        wp.launch(
            _inplace_axpy_kernel,
            dim=self.grid_dim,
            inputs=[z, self._ap, omega],
        )


def build_pressure_solver(method: str, solver: "FluidGridMacSolverBase") -> PressureLinearSolver:
    mode = method.lower().strip()
    aliases = {
        "jacobi": "jacobi",
        "pcg": "pcg",
        "cg": "pcg",
        "mgpcg": "mgpcg",
        "mg-cg": "mgpcg",
        "multigrid_pcg": "mgpcg",
    }
    mode = aliases.get(mode, mode)

    if mode == "jacobi":
        return JacobiPressureSolver(solver)
    if mode == "pcg":
        return PcgPressureSolver(solver)
    if mode == "mgpcg":
        return MgpcgPressureSolver(solver)
    raise ValueError(f"Unsupported pressure solver '{method}'. Choose from: jacobi, pcg, mgpcg")



