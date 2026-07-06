"""Analytic stabilizing force helpers for staged dam-break MEM demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HybridForceBreakdown:
    """Separated analytic force terms used by hybrid dam-break MEM demos."""

    force: np.ndarray
    gravity_force_z: float
    buoyancy_z: float
    vertical_drag_z: float
    flow_drag_x: float
    flow_drag_y: float
    mass: float
    volume: float
    analytic_force_scale: float = 1.0
    unscaled_gravity_accel_z: float = 0.0
    scaled_gravity_accel_z: float = 0.0
    buoyancy_accel_z: float = 0.0
    vertical_drag_accel_z: float = 0.0
    flow_drag_accel_x: float = 0.0
    flow_drag_accel_y: float = 0.0

    @property
    def gravity_z(self) -> float:
        """Compatibility alias for the unscaled gravity force term, not acceleration."""
        return self.gravity_force_z


def compute_hybrid_force_breakdown(
    *,
    sphere_radius: float,
    sphere_density: float,
    rho_water: float,
    gravity_z: float,
    water_contact_fraction: float,
    body_velocity: tuple[float, float, float],
    local_flow_velocity: tuple[float, float, float],
    analytic_force_scale: float,
    vertical_drag_scale: float,
    flow_drag_scale: float,
) -> HybridForceBreakdown:
    """Compute analytic stabilizing force terms for one rigid sphere.

    The vertical force is ``(gravity_force_z + buoyancy_z) * analytic_force_scale + vertical_drag_z``.
    ``vertical_drag_z`` keeps a 0.02 contact floor so dry moving bodies retain a small stabilizing damping term.
    """
    radius: float = float(sphere_radius)
    volume: float = 4.0 * float(np.pi) * radius * radius * radius / 3.0
    mass: float = max(1.0e-8, float(sphere_density) * volume)
    gravity_mag: float = abs(float(gravity_z))
    contact: float = min(1.0, max(0.0, float(water_contact_fraction)))

    buoyancy_z: float = float(rho_water) * volume * contact * gravity_mag
    weight_z: float = -mass * gravity_mag
    vertical_drag_z: float = -float(vertical_drag_scale) * float(body_velocity[2]) * max(contact, 0.02)

    flow_delta_x: float = float(local_flow_velocity[0]) - float(body_velocity[0])
    flow_delta_y: float = float(local_flow_velocity[1]) - float(body_velocity[1])
    flow_drag_x: float = mass * float(flow_drag_scale) * contact * flow_delta_x
    flow_drag_y: float = mass * float(flow_drag_scale) * contact * flow_delta_y

    scale: float = float(analytic_force_scale)
    unscaled_gravity_accel_z: float = weight_z / mass
    scaled_gravity_accel_z: float = weight_z * scale / mass
    buoyancy_accel_z: float = buoyancy_z * scale / mass
    vertical_drag_accel_z: float = vertical_drag_z / mass
    flow_drag_accel_x: float = flow_drag_x / mass
    flow_drag_accel_y: float = flow_drag_y / mass

    force_z: float = (weight_z + buoyancy_z) * scale + vertical_drag_z
    force: np.ndarray = np.asarray([flow_drag_x, flow_drag_y, force_z, 0.0, 0.0, 0.0], dtype=np.float64)
    return HybridForceBreakdown(
        force=force,
        gravity_force_z=weight_z,
        buoyancy_z=buoyancy_z,
        vertical_drag_z=vertical_drag_z,
        flow_drag_x=flow_drag_x,
        flow_drag_y=flow_drag_y,
        mass=mass,
        volume=volume,
        analytic_force_scale=scale,
        unscaled_gravity_accel_z=unscaled_gravity_accel_z,
        scaled_gravity_accel_z=scaled_gravity_accel_z,
        buoyancy_accel_z=buoyancy_accel_z,
        vertical_drag_accel_z=vertical_drag_accel_z,
        flow_drag_accel_x=flow_drag_accel_x,
        flow_drag_accel_y=flow_drag_accel_y,
    )
