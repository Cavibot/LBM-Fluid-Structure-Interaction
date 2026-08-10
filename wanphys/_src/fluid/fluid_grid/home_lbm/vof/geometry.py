# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Device owner for reconstructed PLIC interface geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

from ..model import HomeLbmModel
from . import geometry_kernels
from .state import HomeFreeState


@dataclass(frozen=True)
class HomeFreeGeometryDiagnostics:
    interface_cell_count: int
    curvature_required_cell_count: int
    curvature_skipped_cell_count: int
    invalid_stencil_count: int
    undefined_normal_count: int
    invalid_plane_offset_count: int
    invalid_interface_area_count: int
    insufficient_curvature_neighbor_count: int
    ill_conditioned_curvature_count: int
    invalid_curvature_count: int
    wetting_cell_count: int
    invalid_wall_normal_count: int
    undefined_contact_tangent_count: int
    unconfigured_wetting_cell_count: int


class HomeFreeInterfaceGeometry:
    """Construct PLIC planes without assuming an unconfigured wetting model."""

    def __init__(
        self,
        model: HomeLbmModel,
        *,
        max_condition_number: float = 1.0e3,
        contact_angle_degrees: float | None = None,
    ) -> None:
        if not math.isfinite(max_condition_number) or max_condition_number <= 1.0:
            raise ValueError("max_condition_number must be greater than one")
        self.model = model
        self.res = (int(model.nx), int(model.ny), int(model.nz))
        self.device = model._device
        self.max_condition_number = float(max_condition_number)
        if contact_angle_degrees is not None and (
            not math.isfinite(contact_angle_degrees)
            or not 0.0 < contact_angle_degrees < 180.0
        ):
            raise ValueError("contact_angle_degrees must be finite and in (0, 180)")
        self.contact_angle_degrees = (
            None if contact_angle_degrees is None else float(contact_angle_degrees)
        )
        self.normal = wp.zeros(self.res, dtype=wp.vec3, device=self.device)
        self.plane_offset = wp.zeros(self.res, dtype=float, device=self.device)
        self.interface_area = wp.zeros(self.res, dtype=float, device=self.device)
        self.valid = wp.zeros(self.res, dtype=wp.int32, device=self.device)
        self.curvature = wp.zeros(self.res, dtype=float, device=self.device)
        self.curvature_valid = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self.curvature_required = wp.zeros(
            self.res, dtype=wp.int32, device=self.device
        )
        self._empty_solid_phi = wp.full(
            self.res, 1000.0, dtype=float, device=self.device
        )
        self._counts = wp.zeros(13, dtype=wp.int32, device=self.device)

    def update(
        self,
        state: HomeFreeState,
        *,
        solid_phi: wp.array | None = None,
        compute_curvature: bool = True,
    ) -> HomeFreeGeometryDiagnostics:
        """Reconstruct PLIC geometry and, optionally, time-level curvature.

        Directionally split VOF advection needs a new plane after each sweep,
        but those intermediate states are not physical time levels.  Their
        curvature must therefore neither be evaluated nor used for a capillary
        boundary condition.
        """
        if state.res != self.res or state.device != self.device:
            raise ValueError("HOME-Free state and interface geometry must match")
        if solid_phi is not None and (
            tuple(solid_phi.shape) != self.res or solid_phi.device != self.device
        ):
            raise ValueError("solid_phi and interface geometry must match")
        geometry_solid_phi = (
            self._empty_solid_phi if solid_phi is None else solid_phi
        )
        theta = (
            0.0
            if self.contact_angle_degrees is None
            else math.radians(self.contact_angle_degrees)
        )
        self._counts.zero_()
        wp.launch(
            geometry_kernels.construct_plic_geometry_kernel,
            dim=self.res,
            inputs=[
                state.fill_level,
                state.flags,
                geometry_solid_phi,
                self.normal,
                self.plane_offset,
                self.interface_area,
                self.valid,
                self._counts,
                int(self.contact_angle_degrees is not None),
                math.cos(theta),
                math.sin(theta),
                int(self.model.periodic[0]),
                int(self.model.periodic[1]),
                int(self.model.periodic[2]),
                *self.res,
            ],
            device=self.device,
        )
        if compute_curvature:
            wp.launch(
                geometry_kernels.construct_plic_curvature_kernel,
                dim=self.res,
                inputs=[
                    state.fill_level,
                    state.flags,
                    self.normal,
                    self.plane_offset,
                    self.valid,
                    self.curvature,
                    self.curvature_valid,
                    self.curvature_required,
                    self._counts,
                    self.max_condition_number,
                    int(self.model.periodic[0]),
                    int(self.model.periodic[1]),
                    int(self.model.periodic[2]),
                    *self.res,
                ],
                device=self.device,
            )
        else:
            self.curvature.zero_()
            self.curvature_valid.zero_()
            self.curvature_required.zero_()
        wp.synchronize_device(self.device)
        counts = self._counts.numpy()
        diagnostics = HomeFreeGeometryDiagnostics(
            interface_cell_count=int(counts[3]),
            curvature_required_cell_count=int(counts[8]),
            curvature_skipped_cell_count=int(counts[3] - counts[8]),
            invalid_stencil_count=int(counts[0]),
            undefined_normal_count=int(counts[1]),
            invalid_plane_offset_count=int(counts[2]),
            invalid_interface_area_count=int(counts[7]),
            insufficient_curvature_neighbor_count=int(counts[4]),
            ill_conditioned_curvature_count=int(counts[5]),
            invalid_curvature_count=int(counts[6]),
            wetting_cell_count=int(counts[9]),
            invalid_wall_normal_count=int(counts[10]),
            undefined_contact_tangent_count=int(counts[11]),
            unconfigured_wetting_cell_count=int(counts[12]),
        )
        if diagnostics.invalid_stencil_count:
            raise RuntimeError(
                "PLIC geometry found "
                f"{diagnostics.invalid_stencil_count} interface stencils crossing "
                "invalid fill or solid fields"
            )
        if diagnostics.unconfigured_wetting_cell_count:
            raise RuntimeError(
                "PLIC geometry found "
                f"{diagnostics.unconfigured_wetting_cell_count} interface cells adjacent "
                "to solid without an explicitly configured contact angle"
            )
        if diagnostics.invalid_wall_normal_count:
            raise RuntimeError(
                "PLIC wetting geometry found "
                f"{diagnostics.invalid_wall_normal_count} cells with undefined solid_phi wall normals"
            )
        if diagnostics.undefined_contact_tangent_count:
            raise RuntimeError(
                "PLIC wetting geometry found "
                f"{diagnostics.undefined_contact_tangent_count} cells with undefined wall-tangent interface direction"
            )
        if diagnostics.undefined_normal_count:
            raise RuntimeError(
                "PLIC geometry found "
                f"{diagnostics.undefined_normal_count} interface cells with undefined normals"
            )
        if diagnostics.invalid_plane_offset_count:
            raise FloatingPointError(
                "PLIC geometry produced "
                f"{diagnostics.invalid_plane_offset_count} invalid plane offsets"
            )
        if diagnostics.invalid_interface_area_count:
            raise FloatingPointError(
                "PLIC geometry produced "
                f"{diagnostics.invalid_interface_area_count} nonpositive fractional-fill "
                "or non-finite interface areas"
            )
        if diagnostics.insufficient_curvature_neighbor_count:
            raise RuntimeError(
                "PLIC curvature found "
                f"{diagnostics.insufficient_curvature_neighbor_count} interface cells "
                "with fewer than five interface neighbors"
            )
        if diagnostics.ill_conditioned_curvature_count:
            raise RuntimeError(
                "PLIC curvature found "
                f"{diagnostics.ill_conditioned_curvature_count} rank-deficient or "
                "ill-conditioned interface fits"
            )
        if diagnostics.invalid_curvature_count:
            raise FloatingPointError(
                "PLIC geometry produced "
                f"{diagnostics.invalid_curvature_count} non-finite curvatures"
            )
        return diagnostics
