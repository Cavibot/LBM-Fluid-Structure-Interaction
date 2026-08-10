# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Kinematic midpoint/end transform samples for partitioned HOME-FSI."""

from __future__ import annotations

import warp as wp


@wp.kernel
def _measure_surface_displacement(
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_radius_bound: wp.array(dtype=float),
    maximum: wp.array(dtype=float),
    time_step_over_cell_size: float,
):
    body = wp.tid()
    velocity = body_qd[body]
    linear_speed = wp.length(wp.spatial_top(velocity))
    angular_speed = wp.length(wp.spatial_bottom(velocity))
    displacement = (linear_speed + angular_speed * body_radius_bound[body]) * time_step_over_cell_size
    wp.atomic_max(maximum, 0, displacement)


@wp.func
def _predict_constant_twist(
    transform: wp.transform,
    velocity: wp.spatial_vector,
    center_local: wp.vec3,
    dt: float,
) -> wp.transform:
    position = wp.transform_get_translation(transform)
    rotation = wp.transform_get_rotation(transform)
    linear = wp.spatial_top(velocity)
    angular = wp.spatial_bottom(velocity)
    center_world = wp.transform_point(transform, center_local) + linear * dt

    angular_speed = wp.length(angular)
    next_rotation = rotation
    if angular_speed > 1.0e-12:
        increment = wp.quat_from_axis_angle(angular / angular_speed, angular_speed * dt)
        next_rotation = wp.normalize(increment * rotation)
    next_position = center_world - wp.quat_rotate(next_rotation, center_local)
    return wp.transform(next_position, next_rotation)


@wp.kernel
def _sample_rigid_motion(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com_local: wp.array(dtype=wp.vec3),
    body_q_half: wp.array(dtype=wp.transform),
    body_q_end: wp.array(dtype=wp.transform),
    dt: float,
):
    body = wp.tid()
    transform = body_q[body]
    velocity = body_qd[body]
    center_local = body_com_local[body]
    body_q_half[body] = _predict_constant_twist(transform, velocity, center_local, 0.5 * dt)
    body_q_end[body] = _predict_constant_twist(transform, velocity, center_local, dt)


class RigidMotionPredictor:
    """Store constant-twist samples at the fluid step midpoint and endpoint."""

    def __init__(self, body_count: int, device: wp.DeviceLike = None) -> None:
        if body_count < 0:
            raise ValueError(f"body_count must be non-negative, got {body_count}")
        self.body_count = int(body_count)
        self.device = wp.get_device(device)
        self.body_q_half = wp.zeros(self.body_count, dtype=wp.transform, device=self.device)
        self.body_q_end = wp.zeros(self.body_count, dtype=wp.transform, device=self.device)
        self.maximum_lattice_displacement = wp.zeros(1, dtype=float, device=self.device)

    def sample(
        self,
        body_q: wp.array,
        body_qd: wp.array,
        body_com_local: wp.array,
        dt: float,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if body_q.device != self.device or body_qd.device != self.device:
            raise ValueError("rigid state and motion-predictor devices must match")
        if body_com_local.device != self.device:
            raise ValueError("rigid centers and motion-predictor devices must match")
        if body_q.dtype != wp.transform or body_qd.dtype != wp.spatial_vector:
            raise ValueError("body_q/body_qd must use transform/spatial_vector dtypes")
        if body_com_local.dtype != wp.vec3:
            raise ValueError("body_com_local must use vec3 dtype")
        if len(body_q) != self.body_count or len(body_qd) != self.body_count:
            raise ValueError("rigid state arrays must contain one entry per body")
        if len(body_com_local) != self.body_count:
            raise ValueError("body_com_local must contain one entry per body")
        if self.body_count:
            wp.launch(
                _sample_rigid_motion,
                dim=self.body_count,
                inputs=[
                    body_q,
                    body_qd,
                    body_com_local,
                    self.body_q_half,
                    self.body_q_end,
                    float(dt),
                ],
                device=self.device,
            )

    def measure_lattice_displacement(
        self,
        body_qd: wp.array,
        body_radius_bound: wp.array,
        cell_size: float,
        time_step: float,
    ) -> float:
        if cell_size <= 0.0 or time_step <= 0.0:
            raise ValueError("cell_size and time_step must be positive")
        if body_qd.device != self.device or body_radius_bound.device != self.device:
            raise ValueError("velocity, radius, and motion-predictor devices must match")
        if body_qd.dtype != wp.spatial_vector or body_radius_bound.dtype != wp.float32:
            raise ValueError("body_qd/body_radius_bound must use spatial_vector/float dtypes")
        if len(body_qd) != self.body_count or len(body_radius_bound) != self.body_count:
            raise ValueError("velocity and radius arrays must contain one entry per body")
        self.maximum_lattice_displacement.zero_()
        if self.body_count:
            wp.launch(
                _measure_surface_displacement,
                dim=self.body_count,
                inputs=[
                    body_qd,
                    body_radius_bound,
                    self.maximum_lattice_displacement,
                    float(time_step / cell_size),
                ],
                device=self.device,
            )
        wp.synchronize_device(self.device)
        return float(self.maximum_lattice_displacement.numpy()[0])

    def validate_lattice_displacement(
        self,
        body_qd: wp.array,
        body_radius_bound: wp.array,
        cell_size: float,
        time_step: float,
        limit: float = 0.5,
    ) -> float:
        if not 0.0 < limit <= 1.0:
            raise ValueError("displacement limit must be in (0, 1]")
        displacement = self.measure_lattice_displacement(
            body_qd, body_radius_bound, cell_size, time_step
        )
        if displacement > limit:
            raise ValueError(
                f"predicted rigid surface displacement {displacement} cells exceeds limit {limit}; "
                "reduce the physical HOME time step"
            )
        return displacement
