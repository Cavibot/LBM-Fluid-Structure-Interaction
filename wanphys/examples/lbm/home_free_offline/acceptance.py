# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical acceptance gates shared by tests and offline production runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

from .config import HomeFreeOfflineConfig, OfflineSceneName
from .factory import build_offline_scene
from .scene import OfflineSceneMetrics


@dataclass(frozen=True, slots=True)
class OfflineSceneAcceptance:
    scene: str
    level: str
    samples: int
    maximum_relative_mass_error: float
    maximum_projected_divergence: float
    maximum_speed: float
    maximum_mach: float
    minimum_density: float
    maximum_density: float
    topology_change_count: int
    maximum_strong_coupling_iterations: int
    maximum_strong_coupling_residual: float
    initial_liquid_centroid: tuple[float, float, float]
    final_liquid_centroid: tuple[float, float, float]
    initial_body_position: tuple[float, float, float] | None
    final_body_position: tuple[float, float, float] | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _numeric_values(value: object) -> Iterable[float]:
    if value is None or isinstance(value, (str, bytes, bool)):
        return
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _numeric_values(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _numeric_values(item)


def _physical_speed(config: HomeFreeOfflineConfig, lattice_speed: float) -> float:
    return lattice_speed * config.fluid.cell_size / config.fluid.time_step


def evaluate_scene_trace(
    config: HomeFreeOfflineConfig,
    trace: Iterable[OfflineSceneMetrics],
) -> OfflineSceneAcceptance:
    samples = tuple(trace)
    if len(samples) != config.steps + 1:
        raise ValueError("acceptance trace must contain the initial state and every step")
    if tuple(item.step for item in samples) != tuple(range(config.steps + 1)):
        raise ValueError("acceptance trace steps must be contiguous and start at zero")
    if any(not math.isfinite(value) for item in samples for value in _numeric_values(asdict(item))):
        raise RuntimeError("offline scene acceptance found a non-finite metric")

    evolved = samples[1:]
    violations: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    max_mass_error = max(item.relative_mass_error for item in evolved)
    max_divergence = max(item.projected_max_divergence for item in evolved)
    max_speed = max(item.maximum_speed for item in evolved)
    max_physical_speed = _physical_speed(config, max_speed)
    max_mach = max(item.maximum_mach for item in evolved)
    min_density = min(item.minimum_density for item in evolved)
    max_density = max(item.maximum_density for item in evolved)
    topology_changes = sum(
        item.gas_to_interface_cells
        + item.liquid_to_interface_cells
        + item.interface_to_gas_cells
        + item.interface_to_liquid_cells
        for item in evolved
    )
    max_strong_iterations = max(
        item.strong_coupling_iterations for item in evolved
    )
    max_strong_residual = max(item.strong_coupling_residual for item in evolved)

    require(max_mass_error < 2.0e-5, "relative represented-mass error reached 2e-5")
    require(max_divergence < 2.0e-8, "projected Courant divergence reached 2e-8")
    require(max_speed < config.fluid.max_lattice_speed, "lattice speed exceeded its configured limit")
    require(max_mach < config.fluid.max_lattice_speed * math.sqrt(3.0), "Mach limit was exceeded")
    require(min_density > 0.0, "nonpositive fluid density was reported")
    require(max(item.maximum_fill for item in evolved) <= 1.0, "fill exceeded one")
    require(min(item.maximum_fill for item in evolved) > 0.0, "the liquid phase vanished")
    require(topology_changes > 0, "the free surface had no topology transitions")
    require(sum(item.invalid_fluid_cells for item in evolved) == 0, "invalid fluid cells were reported")
    require(sum(item.missing_cut_links for item in evolved) == 0, "cut links were missing")
    require(sum(item.direct_liquid_gas_links for item in evolved) == 0, "direct liquid-gas links were reported")
    require(sum(item.invalid_rigid_flags for item in evolved) == 0, "rigid cut links had invalid phase flags")
    require(sum(item.dry_nonzero_impulses for item in evolved) == 0, "dry cut links carried impulse")
    require(max(item.queue_relative_mass_error for item in evolved) < 2.0e-5, "queue mass did not close")
    require(max(item.queue_relative_momentum_error for item in evolved) < 2.0e-5, "queue momentum did not close")
    require(max(item.transition_relative_mass_error for item in evolved) < 2.0e-5, "rigid remap mass did not close")
    require(
        max(item.transition_relative_momentum_error for item in evolved) < 2.0e-5,
        "rigid remap momentum did not close",
    )

    initial = samples[0]
    final = samples[-1]
    if config.scene is OfflineSceneName.DAMBREAK:
        require(
            final.occupied_maximum[0] >= initial.occupied_maximum[0] + 1,
            "dam-break front did not advance by one cell",
        )
        require(
            final.liquid_centroid[0] > initial.liquid_centroid[0] + 0.1,
            "dam-break liquid centroid did not move downstream",
        )
        require(
            final.liquid_centroid[2] < initial.liquid_centroid[2] - 0.1,
            "dam-break liquid centroid did not descend",
        )
        require(
            max_physical_speed > 1.0e-2,
            "dam-break remained effectively static",
        )
    else:
        coupling = config.coupling
        assert coupling is not None
        require(max(item.wet_links for item in evolved) > 0, "sphere never established wet cut links")
        require(max_strong_iterations > 1, "strong coupling never required an iteration")
        require(
            max_strong_iterations <= coupling.strong_max_iterations,
            "strong coupling exceeded its iteration budget",
        )
        require(
            max_strong_residual <= coupling.strong_tolerance,
            "strong coupling residual exceeded its tolerance",
        )
        require(
            max(item.maximum_lattice_displacement for item in evolved) < 0.5,
            "rigid body moved too far in one lattice step",
        )
        assert initial.body_position is not None and final.body_position is not None
        assert initial.body_velocity is not None and final.body_velocity is not None
        if config.scene is OfflineSceneName.DAMBREAK_SPHERE:
            require(
                final.body_position[0] > initial.body_position[0] + 0.5,
                "impact sphere did not move downstream",
            )
            require(
                final.liquid_centroid[0] > initial.liquid_centroid[0] + 1.0,
                "impact flow did not traverse the sphere",
            )
        else:
            require(
                final.body_position[2] < initial.body_position[2] - 1.0,
                "entry sphere did not cross the free surface",
            )
            require(
                abs(final.body_velocity[2]) < 0.8 * abs(initial.body_velocity[2]),
                "entry sphere did not decelerate in the liquid",
            )
            require(
                max(item.fresh_cells + item.dead_cells for item in evolved) > 0,
                "entry sphere produced no phase remap",
            )

    if violations:
        raise RuntimeError("offline scene acceptance failed:\n- " + "\n- ".join(violations))

    return OfflineSceneAcceptance(
        scene=config.scene.value,
        level=config.level.value,
        samples=len(samples),
        maximum_relative_mass_error=max_mass_error,
        maximum_projected_divergence=max_divergence,
        maximum_speed=max_speed,
        maximum_mach=max_mach,
        minimum_density=min_density,
        maximum_density=max_density,
        topology_change_count=topology_changes,
        maximum_strong_coupling_iterations=max_strong_iterations,
        maximum_strong_coupling_residual=max_strong_residual,
        initial_liquid_centroid=initial.liquid_centroid,
        final_liquid_centroid=final.liquid_centroid,
        initial_body_position=initial.body_position,
        final_body_position=final.body_position,
    )


def run_scene_acceptance(config: HomeFreeOfflineConfig) -> OfflineSceneAcceptance:
    scene = build_offline_scene(config)
    trace = [scene.measure()]
    for _ in range(config.steps):
        scene.step()
        scene.synchronize()
        trace.append(scene.measure())
    return evaluate_scene_trace(config, trace)
