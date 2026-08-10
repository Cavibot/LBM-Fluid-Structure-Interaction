# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Bulk momentum-strain reconstruction for full Bogner FSL closure."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from ..model import HomeLbmModel
from ..state import HomeLbmState
from . import fsl_stress_kernels
from . import kernels as vof_kernels
from ..constants import D3Q27_DIRECTIONS


@dataclass(frozen=True)
class FslBulkStrainReference:
    strain: np.ndarray
    donor_count: np.ndarray
    maximum_condition_number: float


@dataclass(frozen=True)
class FslBulkStrainDiagnostics:
    invalid_moment_count: int
    insufficient_donor_cell_count: int
    ill_conditioned_cell_count: int
    reconstructed_cell_count: int


@dataclass(frozen=True)
class FslStressClosureDiagnostics:
    bulk: FslBulkStrainDiagnostics
    invalid_normal_strain_count: int
    invalid_boundary_strain_count: int
    no_support_link_count: int


@dataclass(frozen=True)
class FslNormalStrainReference:
    normal_strain: np.ndarray
    no_support_link_count: int


class HomeFreeFslBulkStrainBuilder:
    """Reconstruct ``S=1/2(grad(j)+grad(j)^T)`` on active HOME nodes."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        maximum_fit_condition: float = 1.0e3,
    ) -> None:
        if not math.isfinite(maximum_fit_condition) or maximum_fit_condition <= 1.0:
            raise ValueError("maximum_fit_condition must be finite and greater than one")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.device = model._device
        self.maximum_fit_condition = float(maximum_fit_condition)
        self.bulk_strain = wp.zeros(
            6 * self.stride, dtype=float, device=self.device
        )
        self.donor_count = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._counts = wp.zeros(4, dtype=wp.int32, device=self.device)
        self.last_diagnostics: FslBulkStrainDiagnostics | None = None

    def build(self, fluid_state: HomeLbmState, active: wp.array) -> wp.array:
        if fluid_state.model is not self.model:
            raise ValueError("FSL strain builder and HOME state must share a model")
        if tuple(active.shape) != self.res or active.device != self.device:
            raise ValueError("active must match the FSL strain grid and device")
        if active.dtype != wp.int32:
            raise TypeError("active has the wrong Warp dtype")
        self._counts.zero_()
        wp.launch(
            fsl_stress_kernels.reconstruct_fsl_bulk_strain_kernel,
            dim=self.res,
            inputs=[
                fluid_state.moments,
                active,
                self.bulk_strain,
                self.donor_count,
                self._counts,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                self.maximum_fit_condition,
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        counts = self._counts.numpy()
        diagnostics = FslBulkStrainDiagnostics(
            invalid_moment_count=int(counts[0]),
            insufficient_donor_cell_count=int(counts[1]),
            ill_conditioned_cell_count=int(counts[2]),
            reconstructed_cell_count=int(counts[3]),
        )
        self.last_diagnostics = diagnostics
        if diagnostics.invalid_moment_count:
            raise FloatingPointError(
                "FSL bulk strain found "
                f"{diagnostics.invalid_moment_count} invalid moment samples"
            )
        if diagnostics.insufficient_donor_cell_count:
            raise RuntimeError(
                "FSL bulk strain found "
                f"{diagnostics.insufficient_donor_cell_count} cells with fewer than four donors"
            )
        if diagnostics.ill_conditioned_cell_count:
            raise RuntimeError(
                "FSL bulk strain found "
                f"{diagnostics.ill_conditioned_cell_count} rank-deficient or ill-conditioned fits"
            )
        return self.bulk_strain


class HomeFreeFslStressClosure:
    """Build all link tensors required by the full Bogner ``D`` term."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        maximum_fit_condition: float = 1.0e3,
        no_support_policy: str = "error",
    ) -> None:
        if no_support_policy not in ("error", "only_missing"):
            raise ValueError("no_support_policy must be 'error' or 'only_missing'")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.stride = int(np.prod(self.res))
        self.link_stride = 27 * self.stride
        self.device = model._device
        self.no_support_policy = no_support_policy
        self._policy_code = 0 if no_support_policy == "error" else 1
        self.bulk_builder = HomeFreeFslBulkStrainBuilder(
            model, maximum_fit_condition=maximum_fit_condition
        )
        self.normal_strain = wp.zeros(
            self.link_stride, dtype=float, device=self.device
        )
        self.boundary_strain = wp.zeros(
            6 * self.link_stride, dtype=float, device=self.device
        )
        self._normal_counts = wp.zeros(2, dtype=wp.int32, device=self.device)
        self._invalid_projection = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.last_diagnostics: FslStressClosureDiagnostics | None = None

    def build(
        self,
        fluid_state: HomeLbmState,
        active: wp.array,
        link_status: wp.array,
        plane_owner: wp.array,
        fraction: wp.array,
        interface_normal: wp.array,
        gas_density: wp.array,
        directions: wp.array,
    ) -> wp.array:
        if fluid_state.model is not self.model:
            raise ValueError("FSL stress closure and HOME state must share a model")
        self._validate_grid(active, wp.int32, "active")
        self._validate_grid(interface_normal, wp.vec3, "interface_normal")
        self._validate_grid(gas_density, wp.float32, "gas_density")
        self._validate_links(link_status, wp.int32, "link_status")
        self._validate_links(plane_owner, wp.int32, "plane_owner")
        self._validate_links(fraction, wp.float32, "fraction")
        if tuple(directions.shape) != (27,) or directions.device != self.device:
            raise ValueError("directions must contain the 27 lattice vectors")
        if directions.dtype != wp.vec3:
            raise TypeError("directions has the wrong Warp dtype")
        bulk = self.bulk_builder.build(fluid_state, active)
        assert self.bulk_builder.last_diagnostics is not None
        self._normal_counts.zero_()
        self._invalid_projection.zero_()
        wp.launch(
            fsl_stress_kernels.build_fsl_normal_strain_kernel,
            dim=self.link_stride,
            inputs=[
                fluid_state.moments,
                active,
                link_status,
                plane_owner,
                fraction,
                gas_density,
                directions,
                self.normal_strain,
                self._normal_counts,
                self._policy_code,
                self.model.lattice_viscosity,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.launch(
            vof_kernels.extrapolate_fsl_boundary_strain_kernel,
            dim=self.link_stride,
            inputs=[
                bulk,
                interface_normal,
                active,
                link_status,
                plane_owner,
                fraction,
                self.normal_strain,
                directions,
                self.boundary_strain,
                self._invalid_projection,
                self._policy_code,
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                *self.res,
                self.stride,
            ],
            device=self.device,
        )
        wp.synchronize_device(self.device)
        normal_counts = self._normal_counts.numpy()
        diagnostics = FslStressClosureDiagnostics(
            bulk=self.bulk_builder.last_diagnostics,
            invalid_normal_strain_count=int(normal_counts[0]),
            invalid_boundary_strain_count=int(self._invalid_projection.numpy()[0]),
            no_support_link_count=int(normal_counts[1]),
        )
        self.last_diagnostics = diagnostics
        if diagnostics.invalid_normal_strain_count:
            raise RuntimeError(
                "FSL normal stress construction found "
                f"{diagnostics.invalid_normal_strain_count} invalid links"
            )
        if diagnostics.invalid_boundary_strain_count:
            raise RuntimeError(
                "FSL boundary strain projection found "
                f"{diagnostics.invalid_boundary_strain_count} invalid links"
            )
        return self.boundary_strain

    def _validate_grid(self, field: wp.array, dtype: object, name: str) -> None:
        if tuple(field.shape) != self.res or field.device != self.device:
            raise ValueError(f"{name} must match the FSL stress grid and device")
        if field.dtype != dtype:
            raise TypeError(f"{name} has the wrong Warp dtype")

    def _validate_links(self, field: wp.array, dtype: object, name: str) -> None:
        if tuple(field.shape) != (self.link_stride,) or field.device != self.device:
            raise ValueError(f"{name} must match the FSL stress links and device")
        if field.dtype != dtype:
            raise TypeError(f"{name} has the wrong Warp dtype")


def reconstruct_fsl_bulk_strain(
    moments: np.ndarray,
    active: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    maximum_fit_condition: float = 1.0e3,
) -> FslBulkStrainReference:
    """Weighted affine oracle for the symmetric momentum gradient."""

    values = np.asarray(moments, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    if values.ndim != 4 or values.shape[-1] != 10:
        raise ValueError("FSL bulk strain moments must have grid shape + (10,)")
    shape = values.shape[:-1]
    if mask.shape != shape:
        raise ValueError("FSL bulk strain active mask must match the HOME grid")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not math.isfinite(maximum_fit_condition) or maximum_fit_condition <= 1.0:
        raise ValueError("maximum_fit_condition must be finite and greater than one")
    if not np.isfinite(values[mask]).all() or np.any(values[..., 0][mask] <= 0.0):
        raise ValueError("active FSL moments must be finite with positive density")
    result = np.zeros(shape + (3, 3), dtype=np.float64)
    donor_count = np.zeros(shape, dtype=np.int32)
    maximum_condition = 0.0

    def neighbor(index: tuple[int, int, int], delta: tuple[int, int, int]):
        candidate = [index[axis] + delta[axis] for axis in range(3)]
        for axis in range(3):
            if candidate[axis] < 0 or candidate[axis] >= shape[axis]:
                if periodic[axis]:
                    candidate[axis] %= shape[axis]
                else:
                    return None
        return tuple(candidate)

    for index in map(tuple, np.argwhere(mask)):
        rows: list[tuple[float, float, float, float]] = []
        samples: list[np.ndarray] = []
        weights: list[float] = []
        for displacement in np.ndindex((5, 5, 5)):
            delta = tuple(component - 2 for component in displacement)
            donor = neighbor(index, delta)
            if donor is None or not mask[donor]:
                continue
            rows.append((1.0, float(delta[0]), float(delta[1]), float(delta[2])))
            samples.append(values[donor][1:4])
            radius_squared = float(sum(component * component for component in delta))
            weights.append(1.0 / (1.0 + radius_squared))
        donor_count[index] = len(rows)
        if len(rows) < 4:
            raise RuntimeError(
                f"FSL active cell {index} has only {len(rows)} affine donors"
            )
        sample = np.asarray(samples, dtype=np.float64)
        design = np.asarray(rows, dtype=np.float64)
        root_weight = np.sqrt(np.asarray(weights, dtype=np.float64))
        weighted_design = root_weight[:, None] * design
        singular_values = np.linalg.svd(weighted_design, compute_uv=False)
        if singular_values[-1] <= 1.0e-12 * singular_values[0]:
            raise RuntimeError(f"FSL active cell {index} affine donor fit is rank deficient")
        condition = float(singular_values[0] / singular_values[-1])
        if condition > maximum_fit_condition:
            raise RuntimeError(
                f"FSL active cell {index} affine donor condition {condition:.6g} "
                f"exceeds {maximum_fit_condition:.6g}"
            )
        if np.all(sample == sample[0]):
            maximum_condition = max(maximum_condition, condition)
            continue
        coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_design, root_weight[:, None] * sample, rcond=None
        )
        if rank != 4 or not np.isfinite(coefficients).all():
            raise RuntimeError(f"FSL active cell {index} affine momentum fit failed")
        gradient = coefficients[1:4].T
        result[index] = 0.5 * (gradient + gradient.T)
        maximum_condition = max(maximum_condition, condition)
    return FslBulkStrainReference(
        strain=result,
        donor_count=donor_count,
        maximum_condition_number=maximum_condition,
    )


def fsl_normal_strain_targets(
    moments: np.ndarray,
    active: np.ndarray,
    link_status: np.ndarray,
    plane_owner: np.ndarray,
    fraction: np.ndarray,
    gas_density: np.ndarray,
    *,
    lattice_viscosity: float,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    no_support_policy: str = "error",
) -> FslNormalStrainReference:
    """Apply the normal free-surface stress condition to every FSL link."""

    values = np.asarray(moments, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    status = np.asarray(link_status, dtype=np.int32)
    owner = np.asarray(plane_owner, dtype=np.int32)
    delta = np.asarray(fraction, dtype=np.float64)
    gas = np.asarray(gas_density, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 10:
        raise ValueError("FSL normal stress moments must have grid shape + (10,)")
    shape = values.shape[:-1]
    if mask.shape != shape or gas.shape != shape:
        raise ValueError("FSL normal stress grid fields must share one shape")
    if (
        status.shape != shape + (27,)
        or owner.shape != status.shape
        or delta.shape != status.shape
    ):
        raise ValueError("FSL normal stress link fields must have grid shape + (27,)")
    if len(periodic) != 3:
        raise ValueError("periodic must contain three axis flags")
    if not math.isfinite(lattice_viscosity) or lattice_viscosity <= 0.0:
        raise ValueError("lattice_viscosity must be finite and positive")
    if no_support_policy not in ("error", "only_missing"):
        raise ValueError("no_support_policy must be 'error' or 'only_missing'")
    result = np.zeros(status.shape, dtype=np.float64)
    no_support = 0
    directions = D3Q27_DIRECTIONS.astype(np.int32)

    def neighbor(index: tuple[int, int, int], displacement: np.ndarray):
        candidate = [index[axis] + int(displacement[axis]) for axis in range(3)]
        for axis in range(3):
            if candidate[axis] < 0 or candidate[axis] >= shape[axis]:
                if periodic[axis]:
                    candidate[axis] %= shape[axis]
                else:
                    return None
        return tuple(candidate)

    for index in map(tuple, np.argwhere(mask)):
        for q in range(1, 27):
            link_status_value = int(status[index + (q,)])
            if link_status_value == -7:
                no_support += 1
                if no_support_policy == "error":
                    raise RuntimeError("FSL normal stress encountered NO_SUPPORT")
                continue
            if link_status_value not in (1, 2):
                continue
            support = neighbor(index, directions[q])
            if support is None or not mask[support]:
                raise RuntimeError("FSL normal stress link has no active support")
            owner_value = int(owner[index + (q,)])
            if owner_value == 1:
                owner_index = index
            elif owner_value == 2:
                owner_index = neighbor(index, -directions[q])
                if owner_index is None:
                    raise RuntimeError("FSL normal stress owner lies outside the domain")
            else:
                raise ValueError("FSL normal stress link has an invalid plane owner")
            link_delta = float(delta[index + (q,)])
            local_rho = float(values[index + (0,)])
            support_rho = float(values[support + (0,)])
            gas_rho = float(gas[owner_index])
            if (
                not math.isfinite(link_delta)
                or not 0.0 <= link_delta <= 1.0
                or not math.isfinite(local_rho)
                or local_rho <= 0.0
                or not math.isfinite(support_rho)
                or support_rho <= 0.0
                or not math.isfinite(gas_rho)
                or gas_rho <= 0.0
            ):
                raise ValueError("FSL normal stress inputs must be finite and physical")
            rho_boundary = (1.0 + link_delta) * local_rho - link_delta * support_rho
            target = (rho_boundary - gas_rho) / (6.0 * lattice_viscosity)
            if not math.isfinite(target):
                raise ValueError("FSL normal stress target is non-finite")
            result[index + (q,)] = target
    return FslNormalStrainReference(
        normal_strain=result,
        no_support_link_count=no_support,
    )
