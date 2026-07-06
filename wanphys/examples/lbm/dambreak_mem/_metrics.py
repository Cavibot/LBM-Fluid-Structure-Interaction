"""Metrics and validation helpers for staged dam-break MEM demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DambreakMemMetrics:
    """CPU-side M1 diagnostics for visual and headless acceptance."""

    frame: int
    sim_time: float
    body_position: np.ndarray
    body_velocity: np.ndarray
    analytic_force: np.ndarray
    mem_force: np.ndarray
    density_all_finite: bool
    density_min: float
    density_max: float
    flow_speed_max: float
    water_contact_fraction: float
    initial_water_contact_fraction: float
    displacement_norm: float
    coupling_mode: str
    sphere_density: float
    min_body_z: float
    horizontal_displacement: float
    max_water_contact_fraction: float = 0.0
    fluid_gravity_z: float = 0.0
    unscaled_gravity_accel_z: float = 0.0
    scaled_gravity_accel_z: float = 0.0
    buoyancy_accel_z: float = 0.0
    vertical_drag_accel_z: float = 0.0
    flow_drag_accel_x: float = 0.0
    flow_drag_accel_y: float = 0.0


def _array_is_finite(values: np.ndarray) -> bool:
    """Return true when every value is finite."""
    return bool(np.all(np.isfinite(values)))


def validate_m1_metrics(
    metrics: DambreakMemMetrics,
    *,
    require_contact: bool,
    require_mem: bool,
) -> None:
    """Raise a clear error if M1 acceptance metrics are invalid."""
    if not bool(metrics.density_all_finite):
        raise ValueError("density field contains nonfinite values")
    if not np.isfinite(float(metrics.density_min)) or not np.isfinite(float(metrics.density_max)):
        raise ValueError("density range is not finite")
    if not np.isfinite(float(metrics.flow_speed_max)):
        raise ValueError("flow_speed_max is not finite")
    if not np.isfinite(float(metrics.sphere_density)):
        raise ValueError("sphere_density is not finite")
    if not np.isfinite(float(metrics.min_body_z)):
        raise ValueError("min_body_z is not finite")
    if not np.isfinite(float(metrics.horizontal_displacement)):
        raise ValueError("horizontal_displacement is not finite")
    if not _array_is_finite(metrics.body_position):
        raise ValueError("body_position is not finite")
    if not _array_is_finite(metrics.body_velocity):
        raise ValueError("body_velocity is not finite")
    if not _array_is_finite(metrics.analytic_force):
        raise ValueError("analytic_force is not finite")
    if not _array_is_finite(metrics.mem_force):
        raise ValueError("mem_force is not finite")
    audit_values: tuple[float, ...] = (
        float(metrics.fluid_gravity_z),
        float(metrics.unscaled_gravity_accel_z),
        float(metrics.scaled_gravity_accel_z),
        float(metrics.buoyancy_accel_z),
        float(metrics.vertical_drag_accel_z),
        float(metrics.flow_drag_accel_x),
        float(metrics.flow_drag_accel_y),
    )
    if not bool(np.all(np.isfinite(np.asarray(audit_values, dtype=np.float64)))):
        raise ValueError("force audit metrics are not finite")
    if not np.isfinite(float(metrics.water_contact_fraction)):
        raise ValueError("water_contact_fraction is not finite")
    if not np.isfinite(float(metrics.initial_water_contact_fraction)):
        raise ValueError("initial_water_contact_fraction is not finite")
    if not np.isfinite(float(metrics.max_water_contact_fraction)):
        raise ValueError("max_water_contact_fraction is not finite")
    if not np.isfinite(float(metrics.displacement_norm)):
        raise ValueError("displacement_norm is not finite")
    contact_fraction: float = max(float(metrics.water_contact_fraction), float(metrics.max_water_contact_fraction))
    if require_contact and contact_fraction <= float(metrics.initial_water_contact_fraction):
        raise ValueError("water contact did not increase")
    if require_mem and float(np.linalg.norm(metrics.mem_force[:3])) <= 0.0:
        raise ValueError("mem_force did not become nonzero")
