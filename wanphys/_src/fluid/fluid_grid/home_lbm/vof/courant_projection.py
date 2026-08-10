# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Pressure projection for the frozen face Courants used by geometric VOF."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..model import HomeLbmModel
from . import courant_projection_kernels as kernels
from .flags import HomeFreeCellFlag


@dataclass(frozen=True)
class HomeFreeCourantProjectionDiagnostics:
    active_cell_count: int
    corrected_face_count: int
    iteration_count: int
    initial_max_divergence: float
    projected_max_divergence: float
    relative_residual: float
    maximum_face_correction: float
    refinement_iteration_count: int


@dataclass(frozen=True)
class HomeFreeCourantProjectionResult:
    face_courant: tuple[wp.array, wp.array, wp.array]
    diagnostics: HomeFreeCourantProjectionDiagnostics


@dataclass(frozen=True)
class CourantProjectionReference:
    face_courant: tuple[np.ndarray, np.ndarray, np.ndarray]
    pressure: np.ndarray
    initial_max_divergence: float
    projected_max_divergence: float


def project_face_courant_reference(
    face_courant: tuple[np.ndarray, np.ndarray, np.ndarray],
    flags: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool],
) -> CourantProjectionReference:
    """Dense oracle with gas Dirichlet and impermeable-solid Neumann faces."""

    category = np.asarray(flags, dtype=np.int32)
    if category.ndim != 3:
        raise ValueError("projection flags must be three-dimensional")
    shape = category.shape
    active = np.isin(
        category,
        (int(HomeFreeCellFlag.INTERFACE), int(HomeFreeCellFlag.LIQUID)),
    )
    faces = tuple(np.asarray(field, dtype=np.float64).copy() for field in face_courant)
    for axis, field in enumerate(faces):
        expected = list(shape)
        expected[axis] += 1
        if field.shape != tuple(expected) or not np.isfinite(field).all():
            raise ValueError("projection face Courants have invalid shape or values")
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = 0
        upper[axis] = expected[axis] - 1
        if periodic[axis]:
            if not np.array_equal(field[tuple(lower)], field[tuple(upper)]):
                raise ValueError("periodic projection faces must be duplicated exactly")
        elif np.any(field[tuple(lower)] != 0.0) or np.any(
            field[tuple(upper)] != 0.0
        ):
            raise ValueError("closed projection boundary faces must be zero")

    solid = category == int(HomeFreeCellFlag.SOLID)
    for axis, field in enumerate(faces):
        unique_shape = list(field.shape)
        if periodic[axis]:
            unique_shape[axis] -= 1
        for face_index in np.ndindex(tuple(unique_shape)):
            face = face_index[axis]
            if not periodic[axis] and face in (0, shape[axis]):
                continue
            lower = list(face_index)
            upper = list(face_index)
            lower[axis] = (face - 1) % shape[axis]
            upper[axis] = face % shape[axis]
            if solid[tuple(lower)] or solid[tuple(upper)]:
                field[face_index] = 0.0
        if periodic[axis]:
            first = [slice(None)] * 3
            duplicate = [slice(None)] * 3
            first[axis] = 0
            duplicate[axis] = shape[axis]
            field[tuple(duplicate)] = field[tuple(first)]

    divergence = (
        faces[0][1:] - faces[0][:-1]
        + faces[1][:, 1:] - faces[1][:, :-1]
        + faces[2][:, :, 1:] - faces[2][:, :, :-1]
    )
    indices = [tuple(index) for index in np.argwhere(active)]
    if not indices:
        raise ValueError("Courant projection requires at least one active cell")
    row = {index: number for number, index in enumerate(indices)}
    matrix = np.zeros((len(indices), len(indices)), dtype=np.float64)
    rhs = np.empty(len(indices), dtype=np.float64)

    def neighbor(index: tuple[int, int, int], axis: int, sign: int):
        result = list(index)
        result[axis] += sign
        if result[axis] < 0 or result[axis] >= shape[axis]:
            if not periodic[axis]:
                return None
            result[axis] %= shape[axis]
        return tuple(result)

    for number, index in enumerate(indices):
        rhs[number] = -divergence[index]
        for axis in range(3):
            for sign in (-1, 1):
                adjacent = neighbor(index, axis, sign)
                if adjacent is None:
                    continue
                if solid[adjacent]:
                    continue
                matrix[number, number] += 1.0
                if active[adjacent]:
                    matrix[number, row[adjacent]] -= 1.0
    try:
        pressure_values = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Courant projection operator is singular") from error
    pressure = np.zeros(shape, dtype=np.float64)
    for index, value in zip(indices, pressure_values, strict=True):
        pressure[index] = value

    projected = []
    for axis, source in enumerate(faces):
        destination = source.copy()
        for face_index in np.ndindex(source.shape):
            face = face_index[axis]
            if not periodic[axis] and face in (0, shape[axis]):
                continue
            lower = list(face_index)
            upper = list(face_index)
            lower[axis] = (face - 1) % shape[axis]
            upper[axis] = face % shape[axis]
            lower_index = tuple(lower)
            upper_index = tuple(upper)
            if solid[lower_index] or solid[upper_index]:
                destination[face_index] = 0.0
                continue
            if active[lower_index] or active[upper_index]:
                destination[face_index] -= (
                    pressure[upper_index] - pressure[lower_index]
                )
        projected.append(destination)
    projected_tuple = tuple(projected)
    projected_divergence = (
        projected_tuple[0][1:] - projected_tuple[0][:-1]
        + projected_tuple[1][:, 1:] - projected_tuple[1][:, :-1]
        + projected_tuple[2][:, :, 1:] - projected_tuple[2][:, :, :-1]
    )
    return CourantProjectionReference(
        face_courant=projected_tuple,
        pressure=pressure,
        initial_max_divergence=float(np.max(np.abs(divergence[active]))),
        projected_max_divergence=float(
            np.max(np.abs(projected_divergence[active]))
        ),
    )


class HomeFreeCourantProjector:
    """Matrix-free Jacobi-PCG projection on HOME-Free active cells."""

    _EPSILON = 1.0e-30
    _DOT_CHUNK_SIZE = 128

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        max_iterations: int = 160,
        relative_tolerance: float = 1.0e-5,
        absolute_divergence_tolerance: float = 2.0e-8,
        check_interval: int = 8,
        require_projected_divergence_limit: bool = True,
        warm_start: bool = True,
        use_cuda_graph: bool = True,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("projection max_iterations must be positive")
        if check_interval < 1:
            raise ValueError("projection check_interval must be positive")
        if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
            raise ValueError("projection relative_tolerance must be positive")
        if (
            not math.isfinite(absolute_divergence_tolerance)
            or absolute_divergence_tolerance <= 0.0
        ):
            raise ValueError("projection absolute tolerance must be positive")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.device = model._device
        self.max_iterations = int(max_iterations)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_divergence_tolerance = float(
            absolute_divergence_tolerance
        )
        self.check_interval = int(check_interval)
        self.require_projected_divergence_limit = bool(
            require_projected_divergence_limit
        )
        self.warm_start = bool(warm_start)
        self._has_pressure_guess = False
        self.use_cuda_graph = bool(use_cuda_graph)
        self._pcg_iteration_graph = None
        self._pcg_graph_flags_ptr: int | None = None
        self.pressure = wp.zeros(self.res, dtype=float, device=self.device)
        self.rhs = wp.zeros_like(self.pressure)
        self.residual = wp.zeros_like(self.pressure)
        self.preconditioned = wp.zeros_like(self.pressure)
        self.direction = wp.zeros_like(self.pressure)
        self.operator_direction = wp.zeros_like(self.pressure)
        self.inverse_diagonal = wp.zeros_like(self.pressure)
        self.divergence = wp.zeros_like(self.pressure)
        self.projected_faces = tuple(
            wp.zeros(
                tuple(
                    self.res[index] + (1 if index == axis else 0)
                    for index in range(3)
                ),
                dtype=float,
                device=self.device,
            )
            for axis in range(3)
        )
        self._counts = wp.zeros(2, dtype=wp.int32, device=self.device)
        self._invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._initial_max_divergence = wp.zeros(
            1, dtype=float, device=self.device
        )
        self._projected_max_divergence = wp.zeros(
            1, dtype=float, device=self.device
        )
        self._maximum_correction = wp.zeros(1, dtype=float, device=self.device)
        self._rz_old = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._rz_new = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._p_ap = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._rr = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._rhs_norm = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._residual_limit = wp.zeros(
            1, dtype=wp.float64, device=self.device
        )
        self._alpha = wp.zeros(1, dtype=float, device=self.device)
        self._beta = wp.zeros(1, dtype=float, device=self.device)
        self._element_count = math.prod(self.res)
        partial_count = max(
            1,
            math.ceil(self._element_count / self._DOT_CHUNK_SIZE),
        )
        self._dot_partial_a = wp.zeros(
            partial_count, dtype=wp.float64, device=self.device
        )
        self._dot_partial_b = wp.zeros_like(self._dot_partial_a)

    @property
    def has_pressure_guess(self) -> bool:
        return self._has_pressure_guess

    def reset_pressure_guess(self) -> None:
        self.pressure.zero_()
        self._has_pressure_guess = False

    def capture_pressure_guess(self, destination: wp.array) -> bool:
        self._validate_pressure_history(destination)
        wp.copy(destination, self.pressure)
        return self._has_pressure_guess

    def restore_pressure_guess(self, source: wp.array, exists: bool) -> None:
        self._validate_pressure_history(source)
        wp.copy(self.pressure, source)
        self._has_pressure_guess = bool(exists)

    def _validate_pressure_history(self, pressure: wp.array) -> None:
        if (
            tuple(pressure.shape) != self.res
            or pressure.dtype != wp.float32
            or pressure.device != self.device
        ):
            raise ValueError("projection pressure history must match the projector")

    def project(
        self,
        face_courant: tuple[wp.array, wp.array, wp.array],
        flags: wp.array,
    ) -> HomeFreeCourantProjectionResult:
        self._validate_inputs(face_courant, flags)
        if not self.warm_start or not self._has_pressure_guess:
            self.pressure.zero_()
        self._counts.zero_()
        self._invalid.zero_()
        self._initial_max_divergence.zero_()
        wp.launch(
            kernels.build_projection_system_kernel,
            dim=self.res,
            inputs=[
                *face_courant,
                flags,
                self.rhs,
                self.inverse_diagonal,
                self.divergence,
                self._counts,
                self._initial_max_divergence,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                *self.res,
            ],
            device=self.device,
        )
        self._dot(self._rhs_norm, self.rhs, self.rhs)
        wp.synchronize_device(self.device)
        counts = self._counts.numpy()
        active_count = int(counts[0])
        if active_count == 0:
            raise RuntimeError("Courant projection has no active cells")
        initial_max = float(self._initial_max_divergence.numpy()[0])
        rhs_norm_squared = float(self._rhs_norm.numpy()[0])
        if not math.isfinite(rhs_norm_squared):
            raise FloatingPointError("Courant projection RHS is non-finite")
        if initial_max <= self.absolute_divergence_tolerance:
            self.pressure.zero_()
            self._has_pressure_guess = False
            self._counts.zero_()
            self._maximum_correction.zero_()
            for axis in range(3):
                wp.launch(
                    kernels.project_face_courant_kernel,
                    dim=face_courant[axis].shape,
                    inputs=[
                        face_courant[axis],
                        self.pressure,
                        flags,
                        self.projected_faces[axis],
                        self._counts,
                        self._maximum_correction,
                        axis,
                        int(self.model.periodic[axis]),
                        *self.res,
                    ],
                    device=self.device,
                )
            return HomeFreeCourantProjectionResult(
                face_courant=self.projected_faces,
                diagnostics=HomeFreeCourantProjectionDiagnostics(
                    active_cell_count=active_count,
                    corrected_face_count=0,
                    iteration_count=0,
                    initial_max_divergence=initial_max,
                    projected_max_divergence=initial_max,
                    relative_residual=0.0,
                    maximum_face_correction=0.0,
                    refinement_iteration_count=0,
                ),
            )

        if self.warm_start and self._has_pressure_guess:
            self._apply_operator(
                self.pressure, self.operator_direction, flags
            )
            wp.launch(
                kernels.combine_kernel,
                dim=self.res,
                inputs=[
                    self.rhs,
                    self.operator_direction,
                    self.residual,
                    1.0,
                    -1.0,
                ],
                device=self.device,
            )
        else:
            wp.copy(self.residual, self.rhs)
        wp.launch(
            kernels.apply_preconditioner_kernel,
            dim=self.res,
            inputs=[self.residual, self.inverse_diagonal, self.preconditioned],
            device=self.device,
        )
        wp.copy(self.direction, self.preconditioned)
        self._dot(self._rz_old, self.residual, self.preconditioned)
        residual_limit = (
            self.relative_tolerance * self.relative_tolerance * rhs_norm_squared
        )
        self._residual_limit.fill_(residual_limit)
        iterations = 0
        while iterations < self.max_iterations:
            block_size = min(
                self.check_interval, self.max_iterations - iterations
            )
            if block_size == self.check_interval and self._graph_enabled():
                self._launch_pcg_graph_block(flags)
            else:
                for _ in range(block_size):
                    self._pcg_iteration(flags)
            iterations += block_size
            self._dot(self._rr, self.residual, self.residual)
            wp.synchronize_device(self.device)
            if float(self._rr.numpy()[0]) <= residual_limit:
                break

        self._counts.zero_()
        self._maximum_correction.zero_()
        for axis in range(3):
            wp.launch(
                kernels.project_face_courant_kernel,
                dim=face_courant[axis].shape,
                inputs=[
                    face_courant[axis],
                    self.pressure,
                    flags,
                    self.projected_faces[axis],
                    self._counts,
                    self._maximum_correction,
                    axis,
                    int(self.model.periodic[axis]),
                    *self.res,
                ],
                device=self.device,
            )
        self._projected_max_divergence.zero_()
        wp.launch(
            kernels.measure_projected_divergence_kernel,
            dim=self.res,
            inputs=[*self.projected_faces, flags, self._projected_max_divergence],
            device=self.device,
        )
        self._dot(self._rr, self.residual, self.residual)
        wp.synchronize_device(self.device)
        invalid = int(self._invalid.numpy()[0])
        relative_residual = math.sqrt(
            max(float(self._rr.numpy()[0]), 0.0) / rhs_norm_squared
        )
        projected_max = float(self._projected_max_divergence.numpy()[0])
        allowed_divergence = max(
            self.absolute_divergence_tolerance,
            self.relative_tolerance * initial_max,
        )
        if invalid:
            self._has_pressure_guess = False
            raise RuntimeError(f"Courant projection PCG had {invalid} breakdowns")
        if relative_residual > self.relative_tolerance:
            self._has_pressure_guess = False
            raise RuntimeError(
                "Courant projection PCG did not converge: relative residual "
                f"{relative_residual:.6g} after {iterations} iterations"
            )
        if (
            self.require_projected_divergence_limit
            and projected_max > allowed_divergence
        ):
            self._has_pressure_guess = False
            raise RuntimeError(
                "Courant projection left excessive divergence: "
                f"{projected_max:.6g} > {allowed_divergence:.6g}"
            )
        self._has_pressure_guess = True
        counts = self._counts.numpy()
        return HomeFreeCourantProjectionResult(
            face_courant=self.projected_faces,
            diagnostics=HomeFreeCourantProjectionDiagnostics(
                active_cell_count=active_count,
                corrected_face_count=int(counts[1]),
                iteration_count=iterations,
                initial_max_divergence=initial_max,
                projected_max_divergence=projected_max,
                relative_residual=relative_residual,
                maximum_face_correction=float(self._maximum_correction.numpy()[0]),
                refinement_iteration_count=0,
            ),
        )

    def _apply_operator(
        self, source: wp.array, destination: wp.array, flags: wp.array
    ) -> None:
        wp.launch(
            kernels.apply_projection_operator_kernel,
            dim=self.res,
            inputs=[
                source,
                flags,
                self.inverse_diagonal,
                destination,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                *self.res,
            ],
            device=self.device,
        )

    def _graph_enabled(self) -> bool:
        return self.use_cuda_graph and bool(getattr(self.device, "is_cuda", False))

    def _launch_pcg_graph_block(self, flags: wp.array) -> None:
        flags_ptr = int(flags.ptr)
        if (
            self._pcg_iteration_graph is None
            or self._pcg_graph_flags_ptr != flags_ptr
        ):
            with wp.ScopedCapture(
                device=self.device, force_module_load=False
            ) as capture:
                for _ in range(self.check_interval):
                    self._pcg_iteration(flags)
            self._pcg_iteration_graph = capture.graph
            self._pcg_graph_flags_ptr = flags_ptr
        wp.capture_launch(self._pcg_iteration_graph)

    def _pcg_iteration(self, flags: wp.array) -> None:
        self._apply_operator(self.direction, self.operator_direction, flags)
        self._dot(self._p_ap, self.direction, self.operator_direction)
        self._ratio(self._alpha, self._rz_old, self._p_ap)
        wp.launch(
            kernels.update_solution_residual_kernel,
            dim=self.res,
            inputs=[
                self.pressure,
                self.residual,
                self.direction,
                self.operator_direction,
                self._alpha,
            ],
            device=self.device,
        )
        wp.launch(
            kernels.apply_preconditioner_kernel,
            dim=self.res,
            inputs=[
                self.residual,
                self.inverse_diagonal,
                self.preconditioned,
            ],
            device=self.device,
        )
        self._dot(self._rz_new, self.residual, self.preconditioned)
        self._ratio(self._beta, self._rz_new, self._rz_old)
        wp.launch(
            kernels.update_direction_kernel,
            dim=self.res,
            inputs=[self.direction, self.preconditioned, self._beta],
            device=self.device,
        )
        wp.launch(
            kernels.copy_double_scalar_kernel,
            dim=1,
            inputs=[self._rz_old, self._rz_new],
            device=self.device,
        )

    def _dot(self, result: wp.array, a: wp.array, b: wp.array) -> None:
        count = math.ceil(self._element_count / self._DOT_CHUNK_SIZE)
        wp.launch(
            kernels.dot_chunk_kernel,
            dim=count,
            inputs=[
                a,
                b,
                self._dot_partial_a,
                self._element_count,
                self.res[1],
                self.res[2],
                self._DOT_CHUNK_SIZE,
            ],
            device=self.device,
        )
        source = self._dot_partial_a
        destination = self._dot_partial_b
        while count > 1:
            reduced_count = math.ceil(count / self._DOT_CHUNK_SIZE)
            wp.launch(
                kernels.reduce_double_chunks_kernel,
                dim=reduced_count,
                inputs=[source, destination, count, self._DOT_CHUNK_SIZE],
                device=self.device,
            )
            source, destination = destination, source
            count = reduced_count
        wp.launch(
            kernels.copy_double_scalar_kernel,
            dim=1,
            inputs=[result, source],
            device=self.device,
        )

    def _ratio(
        self,
        result: wp.array,
        numerator: wp.array,
        denominator: wp.array,
    ) -> None:
        wp.launch(
            kernels.ratio_kernel,
            dim=1,
            inputs=[
                result,
                numerator,
                denominator,
                self._invalid,
                self._EPSILON,
                self._residual_limit,
            ],
            device=self.device,
        )

    def _validate_inputs(
        self,
        face_courant: tuple[wp.array, wp.array, wp.array],
        flags: wp.array,
    ) -> None:
        if len(face_courant) != 3:
            raise ValueError("Courant projection requires all three face fields")
        if (
            tuple(flags.shape) != self.res
            or flags.device != self.device
            or flags.dtype != wp.int32
        ):
            raise ValueError("Courant projection flags must match the HOME model")
        for axis, field in enumerate(face_courant):
            expected = tuple(
                self.res[index] + (1 if index == axis else 0)
                for index in range(3)
            )
            if (
                tuple(field.shape) != expected
                or field.device != self.device
                or field.dtype != wp.float32
            ):
                raise ValueError(
                    f"Courant projection axis {axis} field has invalid ownership"
                )
