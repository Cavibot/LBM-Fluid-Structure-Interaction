# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""HOME face-Courant builder and warm-started matrix-free PCG projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from ..core import HomeCoreModel, HomeCoreState
from ..free_surface_fsl import FslState, FslWallMask
from . import courant_kernels


@dataclass(frozen=True)
class FaceCourantDiagnostics:
    maximum_courant: float
    maximum_active_divergence: float


@dataclass(frozen=True)
class CourantProjectionDiagnostics:
    active_cell_count: int
    iteration_count: int
    initial_maximum_divergence: float
    projected_maximum_divergence: float
    relative_residual: float
    maximum_face_correction: float


class HomeFaceCourantBuilder:
    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        *,
        maximum_courant: float = 0.95,
    ) -> None:
        if walls.model is not model:
            raise ValueError("Courant builder and walls must share one HOME model")
        self.model = model
        self.walls = walls
        self.res = walls.res
        self.maximum_courant = float(maximum_courant)
        self.face_courant = tuple(
            wp.zeros(
                tuple(size + (1 if index == axis else 0) for index, size in enumerate(self.res)),
                dtype=float,
                device=model._device,
            )
            for axis in range(3)
        )
        self._invalid = wp.zeros(1, dtype=wp.int32, device=model._device)
        self._maximum = wp.zeros(1, dtype=float, device=model._device)
        self._maximum_divergence = wp.zeros(1, dtype=float, device=model._device)
        self.last_diagnostics: FaceCourantDiagnostics | None = None

    def build(
        self, fluid: HomeCoreState, fsl: FslState
    ) -> tuple[wp.array, wp.array, wp.array]:
        if fluid.model is not self.model or fsl.model is not self.model:
            raise ValueError("Courant builder and HOME-FSL states must match")
        self._invalid.zero_()
        self._maximum.zero_()
        for axis in range(3):
            wp.launch(
                courant_kernels.build_face_courant_kernel,
                dim=self.face_courant[axis].shape,
                inputs=[
                    fluid.moments,
                    fsl.flags,
                    self.walls.device,
                    self.face_courant[axis],
                    self._invalid,
                    self._maximum,
                    axis,
                    int(not self.walls.closed_axes[axis]),
                    *self.res,
                    fluid.cell_count,
                ],
                device=self.model._device,
            )
        self._maximum_divergence.zero_()
        wp.launch(
            courant_kernels.measure_divergence_kernel,
            dim=self.res,
            inputs=[
                *self.face_courant,
                fsl.flags,
                self.walls.device,
                self._maximum_divergence,
            ],
            device=self.model._device,
        )
        invalid = int(self._invalid.numpy()[0])
        diagnostics = FaceCourantDiagnostics(
            float(self._maximum.numpy()[0]),
            float(self._maximum_divergence.numpy()[0]),
        )
        self.last_diagnostics = diagnostics
        if invalid:
            raise FloatingPointError(f"HOME face builder found {invalid} invalid faces")
        if diagnostics.maximum_courant > self.maximum_courant:
            raise FloatingPointError(
                f"HOME face Courant {diagnostics.maximum_courant} exceeds "
                f"{self.maximum_courant}"
            )
        return self.face_courant


class HomeCourantProjector:
    _CHUNK = 128
    _EPSILON = 1.0e-30

    def __init__(
        self,
        model: HomeCoreModel,
        walls: FslWallMask,
        *,
        maximum_iterations: int = 80,
        relative_tolerance: float = 1.0e-5,
        absolute_divergence_tolerance: float = 2.0e-8,
        check_interval: int = 8,
    ) -> None:
        if walls.model is not model:
            raise ValueError("Courant projector and walls must share one HOME model")
        self.model = model
        self.walls = walls
        self.res = walls.res
        self.maximum_iterations = int(maximum_iterations)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_divergence_tolerance = float(absolute_divergence_tolerance)
        self.check_interval = int(check_interval)
        if self.maximum_iterations < 1 or self.check_interval < 1:
            raise ValueError("projection iterations and check interval must be positive")
        self.pressure = wp.zeros(self.res, dtype=float, device=model._device)
        self.rhs = wp.zeros_like(self.pressure)
        self.residual = wp.zeros_like(self.pressure)
        self.true_residual = wp.zeros_like(self.pressure)
        self.preconditioned = wp.zeros_like(self.pressure)
        self.direction = wp.zeros_like(self.pressure)
        self.operator_direction = wp.zeros_like(self.pressure)
        self.inverse_diagonal = wp.zeros_like(self.pressure)
        self.divergence = wp.zeros_like(self.pressure)
        self.projected_faces = tuple(
            wp.zeros(
                tuple(size + (1 if index == axis else 0) for index, size in enumerate(self.res)),
                dtype=float,
                device=model._device,
            )
            for axis in range(3)
        )
        self._active_count = wp.zeros(1, dtype=wp.int32, device=model._device)
        self._invalid = wp.zeros(1, dtype=wp.int32, device=model._device)
        self._initial_maximum = wp.zeros(1, dtype=float, device=model._device)
        self._projected_maximum = wp.zeros(1, dtype=float, device=model._device)
        self._maximum_correction = wp.zeros(1, dtype=float, device=model._device)
        self._rz_old = wp.zeros(1, dtype=wp.float64, device=model._device)
        self._rz_new = wp.zeros(1, dtype=wp.float64, device=model._device)
        self._p_ap = wp.zeros(1, dtype=wp.float64, device=model._device)
        self._rr = wp.zeros(1, dtype=wp.float64, device=model._device)
        self._rhs_norm = wp.zeros(1, dtype=wp.float64, device=model._device)
        self._alpha = wp.zeros(1, dtype=float, device=model._device)
        self._beta = wp.zeros(1, dtype=float, device=model._device)
        self._element_count = math.prod(self.res)
        partial_count = max(1, math.ceil(self._element_count / self._CHUNK))
        self._partial_a = wp.zeros(partial_count, dtype=wp.float64, device=model._device)
        self._partial_b = wp.zeros_like(self._partial_a)
        self._has_pressure_guess = False
        self.last_diagnostics: CourantProjectionDiagnostics | None = None

    def project(
        self,
        face_courant: tuple[wp.array, wp.array, wp.array],
        fsl: FslState,
    ) -> tuple[wp.array, wp.array, wp.array]:
        if fsl.model is not self.model:
            raise ValueError("Courant projector and FSL state must match")
        self._active_count.zero_()
        self._invalid.zero_()
        self._initial_maximum.zero_()
        wp.launch(
            courant_kernels.build_projection_system_kernel,
            dim=self.res,
            inputs=[
                *face_courant,
                fsl.flags,
                self.walls.device,
                self.rhs,
                self.inverse_diagonal,
                self.divergence,
                self._active_count,
                self._initial_maximum,
                *(int(not closed) for closed in self.walls.closed_axes),
                *self.res,
            ],
            device=self.model._device,
        )
        self._dot(self._rhs_norm, self.rhs, self.rhs)
        active_count = int(self._active_count.numpy()[0])
        initial_maximum = float(self._initial_maximum.numpy()[0])
        rhs_norm = float(self._rhs_norm.numpy()[0])
        if active_count == 0 or not math.isfinite(rhs_norm):
            raise FloatingPointError("Courant projection has no valid active system")
        if initial_maximum <= self.absolute_divergence_tolerance:
            self.pressure.zero_()
            self._has_pressure_guess = False
            self._project_faces(face_courant, fsl)
            diagnostics = CourantProjectionDiagnostics(
                active_count,
                0,
                initial_maximum,
                float(self._projected_maximum.numpy()[0]),
                0.0,
                float(self._maximum_correction.numpy()[0]),
            )
            self.last_diagnostics = diagnostics
            if diagnostics.projected_maximum_divergence > self.absolute_divergence_tolerance:
                raise FloatingPointError(
                    "zero-residual Courant projection violated the divergence limit"
                )
            return self.projected_faces
        if not self._has_pressure_guess:
            self.pressure.zero_()
            wp.copy(self.residual, self.rhs)
        else:
            self._apply_operator(self.pressure, self.operator_direction, fsl)
            wp.launch(
                courant_kernels.combine_kernel,
                dim=self.res,
                inputs=[self.rhs, self.operator_direction, self.residual, 1.0, -1.0],
                device=self.model._device,
            )
        wp.launch(
            courant_kernels.precondition_kernel,
            dim=self.res,
            inputs=[self.residual, self.inverse_diagonal, self.preconditioned],
            device=self.model._device,
        )
        wp.copy(self.direction, self.preconditioned)
        self._dot(self._rz_old, self.residual, self.preconditioned)
        residual_limit = self.relative_tolerance**2 * rhs_norm
        iterations = 0
        while iterations < self.maximum_iterations:
            self._pcg_iteration(fsl)
            iterations += 1
            if iterations % self.check_interval == 0:
                self._measure_true_residual(fsl)
                if float(self._rr.numpy()[0]) <= residual_limit:
                    break
                self._dot(self._rz_new, self.residual, self.residual)
                if float(self._rz_new.numpy()[0]) <= residual_limit:
                    self._restart_from_true_residual()
        if iterations % self.check_interval != 0:
            self._measure_true_residual(fsl)
        self._project_faces(face_courant, fsl)
        invalid = int(self._invalid.numpy()[0])
        relative_residual = math.sqrt(max(float(self._rr.numpy()[0]), 0.0) / max(rhs_norm, self._EPSILON))
        projected_maximum = float(self._projected_maximum.numpy()[0])
        diagnostics = CourantProjectionDiagnostics(
            active_count,
            iterations,
            initial_maximum,
            projected_maximum,
            relative_residual,
            float(self._maximum_correction.numpy()[0]),
        )
        self.last_diagnostics = diagnostics
        allowed = max(
            self.absolute_divergence_tolerance,
            self.relative_tolerance * initial_maximum,
        )
        if invalid or projected_maximum > allowed:
            self._has_pressure_guess = False
            raise FloatingPointError(
                "Courant PCG failed: "
                f"invalid={invalid}, residual={relative_residual}, divergence={projected_maximum}"
            )
        self._has_pressure_guess = True
        return self.projected_faces

    def _project_faces(
        self,
        face_courant: tuple[wp.array, wp.array, wp.array],
        fsl: FslState,
    ) -> None:
        self._maximum_correction.zero_()
        for axis in range(3):
            wp.launch(
                courant_kernels.project_face_kernel,
                dim=face_courant[axis].shape,
                inputs=[
                    face_courant[axis],
                    self.pressure,
                    fsl.flags,
                    self.walls.device,
                    self.projected_faces[axis],
                    self._maximum_correction,
                    axis,
                    int(not self.walls.closed_axes[axis]),
                    *self.res,
                ],
                device=self.model._device,
            )
        self._projected_maximum.zero_()
        wp.launch(
            courant_kernels.measure_divergence_kernel,
            dim=self.res,
            inputs=[
                *self.projected_faces,
                fsl.flags,
                self.walls.device,
                self._projected_maximum,
            ],
            device=self.model._device,
        )

    def _apply_operator(
        self, source: wp.array, destination: wp.array, fsl: FslState
    ) -> None:
        wp.launch(
            courant_kernels.apply_operator_kernel,
            dim=self.res,
            inputs=[
                source,
                fsl.flags,
                self.walls.device,
                self.inverse_diagonal,
                destination,
                *(int(not closed) for closed in self.walls.closed_axes),
                *self.res,
            ],
            device=self.model._device,
        )

    def _measure_true_residual(self, fsl: FslState) -> None:
        self._apply_operator(self.pressure, self.operator_direction, fsl)
        wp.launch(
            courant_kernels.combine_kernel,
            dim=self.res,
            inputs=[
                self.rhs,
                self.operator_direction,
                self.true_residual,
                1.0,
                -1.0,
            ],
            device=self.model._device,
        )
        self._dot(self._rr, self.true_residual, self.true_residual)

    def _restart_from_true_residual(self) -> None:
        wp.copy(self.residual, self.true_residual)
        wp.launch(
            courant_kernels.precondition_kernel,
            dim=self.res,
            inputs=[self.residual, self.inverse_diagonal, self.preconditioned],
            device=self.model._device,
        )
        wp.copy(self.direction, self.preconditioned)
        self._dot(self._rz_old, self.residual, self.preconditioned)

    def _pcg_iteration(self, fsl: FslState) -> None:
        self._apply_operator(self.direction, self.operator_direction, fsl)
        self._dot(self._p_ap, self.direction, self.operator_direction)
        self._ratio(self._alpha, self._rz_old, self._p_ap)
        wp.launch(
            courant_kernels.update_solution_residual_kernel,
            dim=self.res,
            inputs=[
                self.pressure,
                self.residual,
                self.direction,
                self.operator_direction,
                self._alpha,
            ],
            device=self.model._device,
        )
        wp.launch(
            courant_kernels.precondition_kernel,
            dim=self.res,
            inputs=[self.residual, self.inverse_diagonal, self.preconditioned],
            device=self.model._device,
        )
        self._dot(self._rz_new, self.residual, self.preconditioned)
        self._ratio(self._beta, self._rz_new, self._rz_old)
        wp.launch(
            courant_kernels.update_direction_kernel,
            dim=self.res,
            inputs=[self.direction, self.preconditioned, self._beta],
            device=self.model._device,
        )
        wp.launch(
            courant_kernels.copy_scalar_kernel,
            dim=1,
            inputs=[self._rz_old, self._rz_new],
            device=self.model._device,
        )

    def _dot(self, result: wp.array, a: wp.array, b: wp.array) -> None:
        count = math.ceil(self._element_count / self._CHUNK)
        wp.launch(
            courant_kernels.dot_chunk_kernel,
            dim=count,
            inputs=[
                a,
                b,
                self._partial_a,
                self._element_count,
                self.res[1],
                self.res[2],
                self._CHUNK,
            ],
            device=self.model._device,
        )
        source = self._partial_a
        destination = self._partial_b
        while count > 1:
            reduced_count = math.ceil(count / self._CHUNK)
            wp.launch(
                courant_kernels.reduce_chunks_kernel,
                dim=reduced_count,
                inputs=[source, destination, count, self._CHUNK],
                device=self.model._device,
            )
            source, destination = destination, source
            count = reduced_count
        wp.launch(
            courant_kernels.copy_scalar_kernel,
            dim=1,
            inputs=[result, source],
            device=self.model._device,
        )

    def _ratio(
        self,
        result: wp.array,
        numerator: wp.array,
        denominator: wp.array,
    ) -> None:
        wp.launch(
            courant_kernels.scalar_ratio_kernel,
            dim=1,
            inputs=[result, numerator, denominator, self._invalid, self._EPSILON],
            device=self.model._device,
        )
