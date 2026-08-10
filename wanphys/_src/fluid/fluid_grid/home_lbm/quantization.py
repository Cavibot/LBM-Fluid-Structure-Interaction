# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit error budget and saturation audit for HOME moment quantization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .constants import MOMENT_COUNT, MOMENT_NAMES
from .model import HomeLbmModel
from . import quantization_kernels


INT16_MAX = 32767


@dataclass(frozen=True)
class HomeMomentQuantizationSpec:
    """Per-component symmetric int16 coding around explicit offsets."""

    offsets: tuple[float, ...]
    bounds: tuple[float, ...]
    scales: tuple[float, ...]
    maximum_roundtrip_error: tuple[float, ...]
    density_variation_fraction: float

    @classmethod
    def from_model(
        cls,
        model: HomeLbmModel,
        *,
        density_variation_fraction: float = 0.25,
    ) -> HomeMomentQuantizationSpec:
        if not 0.0 < density_variation_fraction < 1.0:
            raise ValueError("density variation fraction must lie in (0, 1)")
        rho0 = float(model.reference_density)
        rho_max = rho0 * (1.0 + density_variation_fraction)
        momentum_bound = rho_max * float(model.max_lattice_speed)
        bounds = (
            rho0 * density_variation_fraction,
            momentum_bound,
            momentum_bound,
            momentum_bound,
            (2.0 / 3.0) * rho_max,
            (2.0 / 3.0) * rho_max,
            (2.0 / 3.0) * rho_max,
            rho_max,
            rho_max,
            rho_max,
        )
        offsets = (rho0,) + (0.0,) * (MOMENT_COUNT - 1)
        scales = tuple(bound / INT16_MAX for bound in bounds)
        return cls(
            offsets=offsets,
            bounds=bounds,
            scales=scales,
            maximum_roundtrip_error=tuple(0.5 * scale for scale in scales),
            density_variation_fraction=float(density_variation_fraction),
        )


@dataclass(frozen=True)
class HomeMomentQuantizationDiagnostics:
    saturation_counts: tuple[int, ...]
    invalid_counts: tuple[int, ...]

    @property
    def saturation_count(self) -> int:
        return sum(self.saturation_counts)

    @property
    def invalid_count(self) -> int:
        return sum(self.invalid_counts)


class HomeMomentQuantizationAuditor:
    """Audit coding ranges on-device without altering the FP32 solver state."""

    def __init__(
        self,
        model: HomeLbmModel,
        spec: HomeMomentQuantizationSpec | None = None,
    ) -> None:
        self.model = model
        self.device = model._device
        self.cell_count = int(model.nx) * int(model.ny) * int(model.nz)
        self.spec = spec or HomeMomentQuantizationSpec.from_model(model)
        self._offsets = wp.array(
            self.spec.offsets, dtype=float, device=self.device
        )
        self._bounds = wp.array(self.spec.bounds, dtype=float, device=self.device)
        self._saturation_counts = wp.zeros(
            MOMENT_COUNT, dtype=wp.int32, device=self.device
        )
        self._invalid_counts = wp.zeros_like(self._saturation_counts)
        self._all_active = wp.ones(
            (int(model.nx), int(model.ny), int(model.nz)),
            dtype=wp.int32,
            device=self.device,
        )

    def audit(
        self,
        moments: wp.array,
        *,
        active_flags: wp.array | None = None,
        raise_on_saturation: bool = True,
    ) -> HomeMomentQuantizationDiagnostics:
        if (
            moments.dtype != wp.float32
            or moments.device != self.device
            or int(moments.size) != MOMENT_COUNT * self.cell_count
        ):
            raise ValueError("quantization audit moments must match the HOME model")
        flags = self._all_active if active_flags is None else active_flags
        if (
            tuple(flags.shape) != self._all_active.shape
            or flags.dtype != wp.int32
            or flags.device != self.device
        ):
            raise ValueError("quantization audit flags must match the HOME model")
        self._saturation_counts.zero_()
        self._invalid_counts.zero_()
        wp.launch(
            quantization_kernels.audit_moment_quantization_ranges_kernel,
            dim=self.cell_count,
            inputs=[
                moments,
                self._offsets,
                self._bounds,
                self._saturation_counts,
                self._invalid_counts,
                flags,
                int(active_flags is not None),
                int(self.model.ny),
                int(self.model.nz),
                self.cell_count,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        diagnostics = HomeMomentQuantizationDiagnostics(
            saturation_counts=tuple(
                int(value) for value in self._saturation_counts.numpy()
            ),
            invalid_counts=tuple(
                int(value) for value in self._invalid_counts.numpy()
            ),
        )
        if diagnostics.invalid_count:
            raise FloatingPointError(
                "HOME quantization audit found non-finite moments: "
                + _format_counts(diagnostics.invalid_counts)
            )
        if raise_on_saturation and diagnostics.saturation_count:
            raise OverflowError(
                "HOME int16 quantization range would saturate: "
                + _format_counts(diagnostics.saturation_counts)
            )
        return diagnostics


def quantize_dequantize_home_moments(
    moments: np.ndarray,
    spec: HomeMomentQuantizationSpec,
) -> tuple[np.ndarray, HomeMomentQuantizationDiagnostics]:
    """Apply the candidate int16 coding in NumPy for error-budget studies."""

    values = np.asarray(moments, dtype=np.float64)
    if values.shape[-1] != MOMENT_COUNT:
        raise ValueError(f"HOME moments must end with {MOMENT_COUNT} components")
    offsets = np.asarray(spec.offsets, dtype=np.float64)
    scales = np.asarray(spec.scales, dtype=np.float64)
    normalized = (values - offsets) / scales
    invalid = ~np.isfinite(normalized)
    saturation = np.abs(normalized) > INT16_MAX
    coded = np.clip(np.rint(normalized), -INT16_MAX, INT16_MAX).astype(np.int16)
    reconstructed = coded.astype(np.float64) * scales + offsets
    axes = tuple(range(values.ndim - 1))
    diagnostics = HomeMomentQuantizationDiagnostics(
        saturation_counts=tuple(
            int(value) for value in np.count_nonzero(saturation, axis=axes)
        ),
        invalid_counts=tuple(
            int(value) for value in np.count_nonzero(invalid, axis=axes)
        ),
    )
    return reconstructed, diagnostics


def _format_counts(counts: tuple[int, ...]) -> str:
    return ", ".join(
        f"{name}={count}" for name, count in zip(MOMENT_NAMES, counts) if count
    )
