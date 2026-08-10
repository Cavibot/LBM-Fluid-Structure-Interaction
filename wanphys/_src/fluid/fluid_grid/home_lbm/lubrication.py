# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Resolved-to-subgrid sphere-wall lubrication for HOME-LBM coupling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Iterable

import numpy as np
import warp as wp

from newton._src.geometry.types import GeoType

from .model import HomeLbmModel

if TYPE_CHECKING:
    from wanphys._src.rigid.domain import RigidDomain


@dataclass(frozen=True)
class SphereWallLubricationConfig:
    """Ten Cate sphere-wall correction and roughness contact closure."""

    cutoff_cells: float = 1.0
    contact_gap_cells: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.cutoff_cells) or self.cutoff_cells <= 0.0:
            raise ValueError("cutoff_cells must be finite and positive")
        if (
            not math.isfinite(self.contact_gap_cells)
            or self.contact_gap_cells <= 0.0
            or self.contact_gap_cells >= self.cutoff_cells
        ):
            raise ValueError("contact_gap_cells must be in (0, cutoff_cells)")


@wp.func
def _wall_normal(axis: int, sign: float) -> wp.vec3:
    if axis == 0:
        return wp.vec3(sign, 0.0, 0.0)
    if axis == 1:
        return wp.vec3(0.0, sign, 0.0)
    return wp.vec3(0.0, 0.0, sign)


@wp.func
def _axis_periodic(axis: int, periodic_x: int, periodic_y: int, periodic_z: int) -> bool:
    if axis == 0:
        return periodic_x != 0
    if axis == 1:
        return periodic_y != 0
    return periodic_z != 0


@wp.kernel
def _compute_sphere_wall_correction(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com_local: wp.array(dtype=wp.vec3),
    shape_transform: wp.array(dtype=wp.transform),
    sphere_shape: wp.array(dtype=wp.int32),
    sphere_radius: wp.array(dtype=float),
    domain_extent: wp.vec3,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    dynamic_viscosity: float,
    fluid_density: float,
    body_acceleration: wp.vec3,
    cutoff_gap: float,
    contact_gap: float,
    correction_force: wp.array(dtype=wp.vec3),
    correction_torque: wp.array(dtype=wp.vec3),
    minimum_gap: wp.array(dtype=float),
    active_wall_count: wp.array(dtype=wp.int32),
):
    body = wp.tid()
    shape = sphere_shape[body]
    if shape < 0:
        correction_force[body] = wp.vec3(0.0)
        correction_torque[body] = wp.vec3(0.0)
        minimum_gap[body] = float(1.0e20)
        active_wall_count[body] = 0
        return

    sphere_center_local = wp.transform_get_translation(shape_transform[shape])
    sphere_center = wp.transform_point(body_q[body], sphere_center_local)
    body_center = wp.transform_point(body_q[body], body_com_local[body])
    lever = sphere_center - body_center
    twist = body_qd[body]
    sphere_velocity = wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), lever)
    radius = sphere_radius[body]
    coefficient_scale = 6.0 * wp.pi * dynamic_viscosity * radius * radius
    displaced_volume = 4.0 * wp.pi * radius * radius * radius / 3.0
    force = -fluid_density * displaced_volume * body_acceleration
    closest_gap = float(1.0e20)
    active_count = 0

    for axis in range(3):
        if not _axis_periodic(axis, periodic_x, periodic_y, periodic_z):
            lower_normal = _wall_normal(axis, 1.0)
            lower_gap = sphere_center[axis] - radius
            closest_gap = wp.min(closest_gap, lower_gap)
            if lower_gap < cutoff_gap:
                resolved_gap = wp.max(lower_gap, contact_gap)
                coefficient = coefficient_scale * (1.0 / resolved_gap - 1.0 / cutoff_gap)
                normal_velocity = wp.dot(sphere_velocity, lower_normal)
                force -= coefficient * normal_velocity * lower_normal
                active_count += 1

            upper_normal = _wall_normal(axis, -1.0)
            upper_gap = domain_extent[axis] - sphere_center[axis] - radius
            closest_gap = wp.min(closest_gap, upper_gap)
            if upper_gap < cutoff_gap:
                resolved_gap = wp.max(upper_gap, contact_gap)
                coefficient = coefficient_scale * (1.0 / resolved_gap - 1.0 / cutoff_gap)
                normal_velocity = wp.dot(sphere_velocity, upper_normal)
                force -= coefficient * normal_velocity * upper_normal
                active_count += 1

    correction_force[body] = force
    correction_torque[body] = wp.cross(lever, force)
    minimum_gap[body] = closest_gap
    active_wall_count[body] = active_count


@wp.kernel
def _add_sphere_wall_correction(
    correction_force: wp.array(dtype=wp.vec3),
    correction_torque: wp.array(dtype=wp.vec3),
    sphere_shape: wp.array(dtype=wp.int32),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    body = wp.tid()
    if sphere_shape[body] >= 0:
        wp.atomic_add(
            body_f,
            body,
            wp.spatial_vector(correction_force[body], correction_torque[body]),
        )


@wp.kernel
def _project_sphere_wall_contact(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com_local: wp.array(dtype=wp.vec3),
    shape_transform: wp.array(dtype=wp.transform),
    sphere_shape: wp.array(dtype=wp.int32),
    sphere_radius: wp.array(dtype=float),
    domain_extent: wp.vec3,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    contact_gap: float,
    contact_count: wp.array(dtype=wp.int32),
):
    body = wp.tid()
    shape = sphere_shape[body]
    if shape < 0:
        contact_count[body] = 0
        return

    transform = body_q[body]
    sphere_center_local = wp.transform_get_translation(shape_transform[shape])
    sphere_center = wp.transform_point(transform, sphere_center_local)
    body_center = wp.transform_point(transform, body_com_local[body])
    lever = sphere_center - body_center
    twist = body_qd[body]
    linear_velocity = wp.spatial_top(twist)
    angular_velocity = wp.spatial_bottom(twist)
    radius = sphere_radius[body]
    translation = wp.vec3(0.0)
    count = 0

    for axis in range(3):
        if not _axis_periodic(axis, periodic_x, periodic_y, periodic_z):
            lower_normal = _wall_normal(axis, 1.0)
            lower_gap = sphere_center[axis] - radius
            if lower_gap < contact_gap:
                translation += (contact_gap - lower_gap) * lower_normal
                center_velocity = linear_velocity + wp.cross(angular_velocity, lever)
                normal_velocity = wp.dot(center_velocity, lower_normal)
                if normal_velocity < 0.0:
                    linear_velocity -= normal_velocity * lower_normal
                count += 1

            upper_normal = _wall_normal(axis, -1.0)
            upper_gap = domain_extent[axis] - sphere_center[axis] - radius
            if upper_gap < contact_gap:
                translation += (contact_gap - upper_gap) * upper_normal
                center_velocity = linear_velocity + wp.cross(angular_velocity, lever)
                normal_velocity = wp.dot(center_velocity, upper_normal)
                if normal_velocity < 0.0:
                    linear_velocity -= normal_velocity * upper_normal
                count += 1

    if count > 0:
        position = wp.transform_get_translation(transform) + translation
        rotation = wp.transform_get_rotation(transform)
        body_q[body] = wp.transform(position, rotation)
        body_qd[body] = wp.spatial_vector(linear_velocity, angular_velocity)
    contact_count[body] = count


class SphereWallLubrication:
    """GPU correction for one exact sphere per selected rigid body."""

    def __init__(
        self,
        fluid_model: HomeLbmModel,
        rigid_domain: RigidDomain,
        config: SphereWallLubricationConfig,
        body_ids: Iterable[int] | None = None,
        rigid_substeps: int = 1,
    ) -> None:
        self.model = fluid_model
        self.rigid_domain = rigid_domain
        self.config = config
        self.device = fluid_model._device
        rigid_model = rigid_domain.model
        if rigid_model.device != self.device:
            raise ValueError("HOME and rigid domains must use the same device")

        body_count = int(rigid_model.body_count)
        selected = np.asarray(
            list(range(body_count)) if body_ids is None else list(body_ids),
            dtype=np.int32,
        )
        if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
            raise ValueError("lubrication body_ids must be one-dimensional and unique")
        if np.any(selected < 0) or np.any(selected >= body_count):
            raise ValueError("lubrication body_ids contains an index outside the rigid model")

        shape_body = rigid_model.shape_body.numpy()
        shape_type = rigid_model.shape_type.numpy()
        shape_scale = rigid_model.shape_scale.numpy()
        inverse_mass = rigid_model.body_inv_mass.numpy()
        sphere_shape = np.full(body_count, -1, dtype=np.int32)
        sphere_radius = np.zeros(body_count, dtype=np.float32)
        for body in selected:
            if inverse_mass[body] <= 0.0:
                raise ValueError(
                    f"sphere-wall lubrication requires dynamic coupled body {int(body)}"
                )
            shapes = np.flatnonzero(shape_body == body)
            if len(shapes) != 1 or int(shape_type[shapes[0]]) != int(GeoType.SPHERE):
                raise ValueError(
                    "sphere-wall lubrication requires exactly one sphere shape "
                    f"on coupled body {int(body)}"
                )
            shape = int(shapes[0])
            radius = float(shape_scale[shape, 0])
            if not math.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"coupled sphere body {int(body)} has invalid radius")
            sphere_shape[body] = shape
            sphere_radius[body] = radius

        self.body_count = body_count
        self.sphere_shape = wp.array(sphere_shape, dtype=wp.int32, device=self.device)
        self.sphere_radius = wp.array(sphere_radius, dtype=float, device=self.device)
        self.correction_force = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.correction_torque = wp.zeros(body_count, dtype=wp.vec3, device=self.device)
        self.minimum_gap = wp.full(body_count, 1.0e20, dtype=float, device=self.device)
        self.active_wall_count = wp.zeros(body_count, dtype=wp.int32, device=self.device)
        self.contact_count = wp.zeros(body_count, dtype=wp.int32, device=self.device)
        self._periodic = tuple(int(flag) for flag in fluid_model.periodic)
        self._extent = wp.vec3(
            *(
                float(resolution) * float(fluid_model.fluid_grid_cell_size)
                for resolution in fluid_model.fluid_grid_res
            )
        )
        self._cutoff_gap = config.cutoff_cells * float(fluid_model.fluid_grid_cell_size)
        self._contact_gap = config.contact_gap_cells * float(fluid_model.fluid_grid_cell_size)
        self._dynamic_viscosity = (
            float(fluid_model.reference_density) * float(fluid_model.kinematic_viscosity)
        )
        self._validate_explicit_stability(selected, sphere_radius, rigid_substeps)

    def add_to_rigid_forces(
        self,
        body_q: wp.array,
        body_qd: wp.array,
        body_f: wp.array,
    ) -> None:
        """Evaluate the unresolved resistance and add its physical wrench."""

        self._validate_state_arrays(body_q, body_qd)
        if body_f.device != self.device or body_f.dtype != wp.spatial_vector:
            raise ValueError("body_f must be a spatial-vector array on the coupling device")
        if len(body_f) != self.body_count:
            raise ValueError("body_f must contain one entry per rigid body")
        wp.launch(
            _compute_sphere_wall_correction,
            dim=self.body_count,
            inputs=[
                body_q,
                body_qd,
                self.rigid_domain.model.body_com,
                self.rigid_domain.model.shape_transform,
                self.sphere_shape,
                self.sphere_radius,
                self._extent,
                *self._periodic,
                self._dynamic_viscosity,
                float(self.model.reference_density),
                wp.vec3(*self.model.body_acceleration),
                self._cutoff_gap,
                self._contact_gap,
            ],
            outputs=[
                self.correction_force,
                self.correction_torque,
                self.minimum_gap,
                self.active_wall_count,
            ],
            device=self.device,
        )
        wp.launch(
            _add_sphere_wall_correction,
            dim=self.body_count,
            inputs=[
                self.correction_force,
                self.correction_torque,
                self.sphere_shape,
                body_f,
            ],
            device=self.device,
        )

    def project_contacts(self, body_q: wp.array, body_qd: wp.array) -> None:
        """Enforce the configured roughness gap with zero normal restitution."""

        self._validate_state_arrays(body_q, body_qd)
        wp.launch(
            _project_sphere_wall_contact,
            dim=self.body_count,
            inputs=[
                body_q,
                body_qd,
                self.rigid_domain.model.body_com,
                self.rigid_domain.model.shape_transform,
                self.sphere_shape,
                self.sphere_radius,
                self._extent,
                *self._periodic,
                self._contact_gap,
            ],
            outputs=[self.contact_count],
            device=self.device,
        )

    def _validate_state_arrays(self, body_q: wp.array, body_qd: wp.array) -> None:
        if body_q.device != self.device or body_qd.device != self.device:
            raise ValueError("rigid state arrays must be on the coupling device")
        if body_q.dtype != wp.transform or body_qd.dtype != wp.spatial_vector:
            raise ValueError("body_q/body_qd must use transform/spatial-vector dtypes")
        if len(body_q) != self.body_count or len(body_qd) != self.body_count:
            raise ValueError("rigid state arrays must contain one entry per body")

    def _validate_explicit_stability(
        self,
        selected: np.ndarray,
        sphere_radius: np.ndarray,
        rigid_substeps: int,
    ) -> None:
        if rigid_substeps < 1:
            raise ValueError("rigid_substeps must be at least one")
        inverse_mass = self.rigid_domain.model.body_inv_mass.numpy()
        active_axes = sum(not flag for flag in self.model.periodic)
        rigid_dt = float(self.model.time_step) / rigid_substeps
        for body in selected:
            if inverse_mass[body] <= 0.0 or active_axes == 0:
                continue
            radius = float(sphere_radius[body])
            maximum_coefficient = (
                active_axes
                * 6.0
                * math.pi
                * self._dynamic_viscosity
                * radius**2
                * (1.0 / self._contact_gap - 1.0 / self._cutoff_gap)
            )
            damping_number = maximum_coefficient * rigid_dt * float(inverse_mass[body])
            if damping_number >= 1.0:
                raise ValueError(
                    "sphere-wall lubrication is explicitly unstable for coupled "
                    f"body {int(body)}: C_max*dt/m={damping_number:.6g}; "
                    "increase rigid_substeps or contact_gap_cells"
                )
