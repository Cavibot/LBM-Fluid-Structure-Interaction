"""Headless M1 dam-break MEM scene helpers."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from wanphys._src.rigid.state import RigidState
from wanphys.examples.lbm._lbm_falling_sphere_scene import SPHERE_DENSITY_PRESETS
from wanphys.examples.lbm.fluid_grid_lbm_dambreak_falling_sphere_visual import (
    DamBreakFallingSphereConfig,
    DamBreakFallingSphereMetrics,
    DamBreakFallingSphereScene,
    build_dambreak_falling_sphere_scene,
    collect_dambreak_falling_sphere_metrics,
    update_dambreak_display_density,
)
from wanphys.examples.lbm.dambreak_mem._forces import HybridForceBreakdown, compute_hybrid_force_breakdown
from wanphys.examples.lbm.dambreak_mem._metrics import DambreakMemMetrics, validate_m1_metrics

COUPLING_MODE_ANALYTIC: str = "analytic"
COUPLING_MODE_HYBRID: str = "hybrid"
COUPLING_MODE_MEM: str = "mem"
COUPLING_MODES: tuple[str, str, str] = (COUPLING_MODE_ANALYTIC, COUPLING_MODE_HYBRID, COUPLING_MODE_MEM)


@dataclass(frozen=True)
class DambreakMemM1Config:
    """Configuration for the staged M1 dam-break MEM smoke scene."""

    grid_res: tuple[int, int, int] = (32, 24, 28)
    cell_size: float = 0.04
    tau: float = 1.0
    dam_x_fraction: float = 0.32
    dam_z_fraction: float = 0.85
    pool_z_fraction: float = 0.52
    sphere_radius: float = 0.14
    sphere_start_fraction: tuple[float, float, float] = (0.58, 0.5, 0.76)
    force_scale: float = 15.0
    coupling_mode: str = COUPLING_MODE_HYBRID
    tracer_count: int = 0
    analytic_force_scale: float = 100.0
    vertical_drag_scale: float = 0.2
    flow_drag_scale: float = 8.0
    density_preset: str = "light"
    sphere_density: float | None = None
    device: str | None = None


@dataclass
class DambreakMemM1Scene:
    """Runtime handles and retained diagnostics for the M1 scene."""

    base: DamBreakFallingSphereScene
    config: DambreakMemM1Config
    frame: int
    sim_time: float
    last_analytic_force: np.ndarray
    last_force_breakdown: HybridForceBreakdown | None
    max_water_contact_fraction: float


def _base_config_from_m1(config: DambreakMemM1Config) -> DamBreakFallingSphereConfig:
    base_field_names: set[str] = {field.name for field in fields(DamBreakFallingSphereConfig)}
    values: dict[str, object] = {
        field.name: getattr(config, field.name)
        for field in fields(DambreakMemM1Config)
        if field.name in base_field_names and field.name != "sphere_density"
    }
    values["sphere_density"] = resolve_m1_sphere_density(config)
    base_config: DamBreakFallingSphereConfig = DamBreakFallingSphereConfig(**values)
    return base_config


def resolve_m1_sphere_density(config: DambreakMemM1Config) -> float:
    """Resolve the shared sphere density from an explicit value or preset name."""
    explicit_density: float | None = config.sphere_density
    if explicit_density is not None:
        density: float = float(explicit_density)
        if not bool(np.isfinite(density)) or density <= 0.0:
            raise ValueError("sphere_density must be a finite positive value")
        return density

    density_preset: str = str(config.density_preset)
    if density_preset in SPHERE_DENSITY_PRESETS:
        return float(SPHERE_DENSITY_PRESETS[density_preset])

    valid_presets: str = ", ".join(sorted(SPHERE_DENSITY_PRESETS))
    raise ValueError(f"unknown density preset {density_preset!r}; expected one of: {valid_presets}")


def normalize_coupling_mode(mode: str) -> str:
    """Return a supported coupling mode or raise a clear configuration error."""
    coupling_mode: str = str(mode)
    if coupling_mode not in COUPLING_MODES:
        valid_modes: str = ", ".join(COUPLING_MODES)
        raise ValueError(f"unsupported coupling_mode {coupling_mode!r}; expected one of: {valid_modes}")
    return coupling_mode


def build_m1_scene(config: DambreakMemM1Config) -> DambreakMemM1Scene:
    """Build the M1 dam-break scene with analytic, hybrid, or MEM-only feedback."""
    coupling_mode: str = normalize_coupling_mode(config.coupling_mode)
    base_config: DamBreakFallingSphereConfig = _base_config_from_m1(config)
    base: DamBreakFallingSphereScene = build_dambreak_falling_sphere_scene(base_config)
    two_way: bool = coupling_mode in (COUPLING_MODE_HYBRID, COUPLING_MODE_MEM)
    base.coupling.set_two_way_feedback_enabled(two_way, force_scale=float(config.force_scale))
    base.coupling.set_rigid_dynamics_enabled(True)
    initial_metrics: DamBreakFallingSphereMetrics = collect_dambreak_falling_sphere_metrics(base)
    last_analytic_force: np.ndarray = np.zeros(6, dtype=np.float64)
    scene: DambreakMemM1Scene = DambreakMemM1Scene(
        base=base,
        config=config,
        frame=0,
        sim_time=0.0,
        last_analytic_force=last_analytic_force,
        last_force_breakdown=None,
        max_water_contact_fraction=float(initial_metrics.water_contact_fraction),
    )
    return scene


def _apply_m1_analytic_force(scene: DambreakMemM1Scene) -> None:
    """Apply the retained analytic force component for analytic and hybrid modes."""
    coupling_mode: str = normalize_coupling_mode(scene.config.coupling_mode)
    if coupling_mode == COUPLING_MODE_MEM:
        scene.last_analytic_force: np.ndarray = np.zeros(6, dtype=np.float64)
        scene.last_force_breakdown: HybridForceBreakdown | None = None
        return

    base_metrics: DamBreakFallingSphereMetrics = collect_dambreak_falling_sphere_metrics(scene.base)
    base_config: DamBreakFallingSphereConfig = scene.base.config
    breakdown: HybridForceBreakdown = compute_hybrid_force_breakdown(
        sphere_radius=float(base_config.sphere_radius),
        sphere_density=float(base_config.sphere_density),
        rho_water=float(base_config.rho_water),
        gravity_z=float(base_config.gravity_z),
        water_contact_fraction=float(base_metrics.water_contact_fraction),
        body_velocity=tuple(float(value) for value in base_metrics.body_velocity[:3]),
        local_flow_velocity=tuple(float(value) for value in base_metrics.local_flow_velocity[:3]),
        analytic_force_scale=float(base_config.analytic_force_scale),
        vertical_drag_scale=float(base_config.vertical_drag_scale),
        flow_drag_scale=float(base_config.flow_drag_scale),
    )
    scene.last_force_breakdown: HybridForceBreakdown | None = breakdown
    scene.last_analytic_force: np.ndarray = np.asarray(breakdown.force, dtype=np.float64).copy()
    force_tuple: tuple[float, float, float, float, float, float] = tuple(
        float(value) for value in scene.last_analytic_force
    )
    scene.base.rigid_domain.state.add_body_force(scene.base.body_id, force_tuple)


def _advance_m1_hybrid_scene(scene: DambreakMemM1Scene, dt: float) -> None:
    """Advance hybrid M1 with MEM retained for diagnostics and analytic force driving rigid motion."""
    scene.base.coupling.set_rigid_dynamics_enabled(False)
    try:
        scene.base.coupling.step(dt=float(dt))
    finally:
        scene.base.coupling.set_rigid_dynamics_enabled(True)

    scene.base.rigid_domain.state.clear_forces()
    force_tuple: tuple[float, float, float, float, float, float] = tuple(
        float(value) for value in scene.last_analytic_force
    )
    scene.base.rigid_domain.state.add_body_force(scene.base.body_id, force_tuple)
    scene.base.rigid_domain.step(float(dt))
    consumed_rigid_state: RigidState = scene.base.rigid_domain._state_out
    consumed_rigid_state.clear_forces()


def collect_m1_metrics(scene: DambreakMemM1Scene) -> DambreakMemMetrics:
    """Collect M1 headless diagnostics from the base scene and retained MEM wrench."""
    coupling_mode: str = normalize_coupling_mode(scene.config.coupling_mode)
    base_config: DamBreakFallingSphereConfig = scene.base.config
    base_metrics: DamBreakFallingSphereMetrics = collect_dambreak_falling_sphere_metrics(scene.base)
    mem_force: np.ndarray = np.asarray(
        scene.base.coupling.get_last_lbm_feedback_wrench(scene.base.body_id),
        dtype=np.float64,
    ).copy()
    water_contact_fraction: float = float(base_metrics.water_contact_fraction)
    scene.max_water_contact_fraction: float = max(float(scene.max_water_contact_fraction), water_contact_fraction)
    displacement: np.ndarray = base_metrics.body_position - base_metrics.initial_body_position
    displacement_norm: float = float(np.linalg.norm(displacement))
    flow_speed_max: float = float(np.linalg.norm(base_metrics.local_flow_velocity[:3]))
    breakdown: HybridForceBreakdown | None = scene.last_force_breakdown
    fluid_gravity_z: float = float(scene.base.config.gravity_z)
    unscaled_gravity_accel_z: float = 0.0
    scaled_gravity_accel_z: float = 0.0
    buoyancy_accel_z: float = 0.0
    vertical_drag_accel_z: float = 0.0
    flow_drag_accel_x: float = 0.0
    flow_drag_accel_y: float = 0.0
    if breakdown is not None:
        unscaled_gravity_accel_z: float = float(breakdown.unscaled_gravity_accel_z)
        scaled_gravity_accel_z: float = float(breakdown.scaled_gravity_accel_z)
        buoyancy_accel_z: float = float(breakdown.buoyancy_accel_z)
        vertical_drag_accel_z: float = float(breakdown.vertical_drag_accel_z)
        flow_drag_accel_x: float = float(breakdown.flow_drag_accel_x)
        flow_drag_accel_y: float = float(breakdown.flow_drag_accel_y)
    metrics: DambreakMemMetrics = DambreakMemMetrics(
        frame=int(scene.frame),
        sim_time=float(scene.sim_time),
        body_position=base_metrics.body_position.copy(),
        body_velocity=base_metrics.body_velocity.copy(),
        analytic_force=scene.last_analytic_force.copy(),
        mem_force=mem_force,
        density_all_finite=bool(base_metrics.density_all_finite),
        density_min=float(base_metrics.density_min),
        density_max=float(base_metrics.density_max),
        flow_speed_max=flow_speed_max,
        water_contact_fraction=water_contact_fraction,
        initial_water_contact_fraction=float(base_metrics.initial_water_contact_fraction),
        displacement_norm=displacement_norm,
        coupling_mode=coupling_mode,
        sphere_density=float(base_config.sphere_density),
        min_body_z=float(base_metrics.min_body_z),
        horizontal_displacement=float(base_metrics.horizontal_displacement),
        max_water_contact_fraction=float(scene.max_water_contact_fraction),
        fluid_gravity_z=fluid_gravity_z,
        unscaled_gravity_accel_z=unscaled_gravity_accel_z,
        scaled_gravity_accel_z=scaled_gravity_accel_z,
        buoyancy_accel_z=buoyancy_accel_z,
        vertical_drag_accel_z=vertical_drag_accel_z,
        flow_drag_accel_x=flow_drag_accel_x,
        flow_drag_accel_y=flow_drag_accel_y,
    )
    return metrics


def step_m1_scene(scene: DambreakMemM1Scene, dt: float) -> DambreakMemMetrics:
    """Advance the M1 scene one timestep and return current metrics."""
    coupling_mode: str = normalize_coupling_mode(scene.config.coupling_mode)
    _apply_m1_analytic_force(scene)
    if coupling_mode == COUPLING_MODE_HYBRID:
        _advance_m1_hybrid_scene(scene, dt=float(dt))
    else:
        scene.base.coupling.step(dt=float(dt))
    update_dambreak_display_density(scene.base)
    scene.frame: int = int(scene.frame) + 1
    scene.sim_time: float = float(scene.sim_time) + float(dt)
    metrics: DambreakMemMetrics = collect_m1_metrics(scene)
    return metrics


def run_m1_headless(
    config: DambreakMemM1Config,
    *,
    num_frames: int,
    dt: float,
    print_every: int,
    require_contact: bool,
    require_mem: bool,
) -> DambreakMemMetrics:
    """Run the M1 scene without a viewer and return final diagnostics."""
    normalize_coupling_mode(config.coupling_mode)
    scene: DambreakMemM1Scene = build_m1_scene(config)
    metrics: DambreakMemMetrics = collect_m1_metrics(scene)
    frame_index: int
    for frame_index in range(int(num_frames)):
        metrics: DambreakMemMetrics = step_m1_scene(scene, dt=float(dt))
        validate_m1_metrics(metrics, require_contact=False, require_mem=False)
        if int(print_every) > 0 and (frame_index + 1) % int(print_every) == 0:
            body_position: list[float] = [float(value) for value in metrics.body_position]
            body_velocity: list[float] = [float(value) for value in metrics.body_velocity[:3]]
            analytic_norm: float = float(np.linalg.norm(metrics.analytic_force[:3]))
            mem_norm: float = float(np.linalg.norm(metrics.mem_force[:3]))
            print(
                f"m1 frame={metrics.frame:5d} t={metrics.sim_time:7.4f} "
                f"mode={metrics.coupling_mode} pos={body_position} vel={body_velocity} "
                f"density=[{metrics.density_min:.6f}, {metrics.density_max:.6f}] "
                f"contact={metrics.water_contact_fraction:.6f} "
                f"initial_contact={metrics.initial_water_contact_fraction:.6f} "
                f"max_contact={metrics.max_water_contact_fraction:.6f} "
                f"sphere_density={metrics.sphere_density:.3f} "
                f"analytic={analytic_norm:.6f} mem={mem_norm:.6f} "
                f"disp={metrics.displacement_norm:.6f} "
                f"min_z={metrics.min_body_z:.6f} "
                f"hdisp={metrics.horizontal_displacement:.6f}"
            )

    validate_m1_metrics(metrics, require_contact=require_contact, require_mem=require_mem)
    return metrics

