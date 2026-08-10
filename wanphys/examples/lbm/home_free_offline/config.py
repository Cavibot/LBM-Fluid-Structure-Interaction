# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Serializable scene contracts for HOME-Free offline simulation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


SCHEMA_VERSION = 2


class OfflineSceneName(str, Enum):
    DAMBREAK = "dambreak"
    DAMBREAK_SPHERE = "dambreak-sphere"
    SPHERE_ENTRY = "sphere-entry"


class OfflineSceneLevel(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"


def _finite_vector(values: tuple[float, ...], length: int, name: str) -> None:
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain {length} finite values")


@dataclass(frozen=True, slots=True)
class FluidConfig:
    resolution: tuple[int, int, int]
    cell_size: float
    time_step: float
    reference_density: float
    kinematic_viscosity: float
    body_acceleration: tuple[float, float, float]
    periodic: tuple[bool, bool, bool]
    max_lattice_speed: float
    contact_angle_degrees: float
    surface_tension: float = 0.0
    project_courant: bool = True
    projection_max_iterations: int | None = None

    def __post_init__(self) -> None:
        if len(self.resolution) != 3 or any(value <= 0 for value in self.resolution):
            raise ValueError("resolution must contain three positive integers")
        if len(self.periodic) != 3:
            raise ValueError("periodic must contain one flag per axis")
        for name in (
            "cell_size",
            "time_step",
            "reference_density",
            "kinematic_viscosity",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        _finite_vector(self.body_acceleration, 3, "body_acceleration")
        if not math.isfinite(self.max_lattice_speed) or not (
            0.0 < self.max_lattice_speed < 1.0 / math.sqrt(3.0)
        ):
            raise ValueError("max_lattice_speed must be below the lattice sound speed")
        if not math.isfinite(self.contact_angle_degrees) or not (
            0.0 < self.contact_angle_degrees < 180.0
        ):
            raise ValueError("contact_angle_degrees must be in (0, 180)")
        if not math.isfinite(self.surface_tension) or self.surface_tension < 0.0:
            raise ValueError("surface_tension must be finite and nonnegative")
        if self.projection_max_iterations is not None and self.projection_max_iterations < 1:
            raise ValueError("projection_max_iterations must be positive when provided")


@dataclass(frozen=True, slots=True)
class SphereConfig:
    position: tuple[float, float, float]
    radius: float
    density: float
    initial_velocity: tuple[float, float, float]
    gravity: float
    label: str

    def __post_init__(self) -> None:
        _finite_vector(self.position, 3, "sphere.position")
        _finite_vector(self.initial_velocity, 3, "sphere.initial_velocity")
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("sphere.radius must be finite and positive")
        if not math.isfinite(self.density) or self.density <= 0.0:
            raise ValueError("sphere.density must be finite and positive")
        if not math.isfinite(self.gravity):
            raise ValueError("sphere.gravity must be finite")
        if not self.label:
            raise ValueError("sphere.label must not be empty")


@dataclass(frozen=True, slots=True)
class CouplingConfig:
    rigid_substeps: int
    strong_max_iterations: int
    strong_tolerance: float
    strong_relaxation: float

    def __post_init__(self) -> None:
        if self.rigid_substeps < 1:
            raise ValueError("rigid_substeps must be at least one")
        if self.strong_max_iterations < 1:
            raise ValueError("strong_max_iterations must be at least one")
        if not math.isfinite(self.strong_tolerance) or self.strong_tolerance <= 0.0:
            raise ValueError("strong_tolerance must be finite and positive")
        if not math.isfinite(self.strong_relaxation) or not (
            0.0 < self.strong_relaxation <= 1.0
        ):
            raise ValueError("strong_relaxation must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class CameraHint:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    focal_length_mm: float = 50.0

    def __post_init__(self) -> None:
        _finite_vector(self.position, 3, "camera.position")
        _finite_vector(self.target, 3, "camera.target")
        if not math.isfinite(self.focal_length_mm) or self.focal_length_mm <= 0.0:
            raise ValueError("camera.focal_length_mm must be finite and positive")
        if self.position == self.target:
            raise ValueError("camera position and target must differ")


@dataclass(frozen=True, slots=True)
class HomeFreeOfflineConfig:
    scene: OfflineSceneName
    level: OfflineSceneLevel
    steps: int
    output_every_steps: int
    device: str
    fluid: FluidConfig
    camera: CameraHint
    initial_liquid_height: float
    initial_liquid_x_range: tuple[float, float] | None
    initial_fluid_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sphere: SphereConfig | None = None
    coupling: CouplingConfig | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scene schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.output_every_steps < 1:
            raise ValueError("output_every_steps must be positive")
        if not self.device:
            raise ValueError("device must not be empty")
        _finite_vector(self.initial_fluid_velocity, 3, "initial_fluid_velocity")
        domain_x = self.fluid.resolution[0] * self.fluid.cell_size
        domain_z = self.fluid.resolution[2] * self.fluid.cell_size
        if not math.isfinite(self.initial_liquid_height) or not (
            0.0 < self.initial_liquid_height <= domain_z
        ):
            raise ValueError("initial_liquid_height must lie inside the physical domain")
        if self.initial_liquid_x_range is not None:
            _finite_vector(self.initial_liquid_x_range, 2, "initial_liquid_x_range")
            lower, upper = self.initial_liquid_x_range
            if not 0.0 <= lower < upper <= domain_x:
                raise ValueError("initial_liquid_x_range must lie inside the physical domain")
        if self.scene is OfflineSceneName.SPHERE_ENTRY and self.initial_liquid_x_range is not None:
            raise ValueError("sphere-entry must initialize a full-width liquid pool")
        if self.scene is not OfflineSceneName.SPHERE_ENTRY and self.initial_liquid_x_range is None:
            raise ValueError(f"{self.scene.value} requires an initial liquid x range")
        has_rigid = self.sphere is not None and self.coupling is not None
        if (self.sphere is None) != (self.coupling is None):
            raise ValueError("sphere and coupling must either both be set or both be absent")
        if self.scene is OfflineSceneName.DAMBREAK and has_rigid:
            raise ValueError("the pure dam-break scene must not contain a rigid body")
        if self.scene is not OfflineSceneName.DAMBREAK and not has_rigid:
            raise ValueError(f"{self.scene.value} requires a sphere and coupling config")
        if self.sphere is not None:
            for axis, coordinate in enumerate(self.sphere.position):
                extent = self.fluid.resolution[axis] * self.fluid.cell_size
                if coordinate - self.sphere.radius <= 0.0 or coordinate + self.sphere.radius >= extent:
                    raise ValueError(
                        f"sphere must start strictly inside axis {axis} domain extent (0, {extent})"
                    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["scene"] = self.scene.value
        result["level"] = self.level.value
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.to_json(indent=None).encode("utf-8")).hexdigest()

    @property
    def derived_parameters(self) -> dict[str, Any]:
        fluid = self.fluid
        dx = fluid.cell_size
        dt = fluid.time_step
        lattice_viscosity = fluid.kinematic_viscosity * dt / (dx * dx)
        acceleration_magnitude = math.sqrt(
            sum(component * component for component in fluid.body_acceleration)
        )
        characteristic_velocity = math.sqrt(
            acceleration_magnitude * self.initial_liquid_height
        )
        characteristic_velocity = max(
            characteristic_velocity,
            math.sqrt(sum(value * value for value in self.initial_fluid_velocity)),
            (
                0.0
                if self.sphere is None
                else math.sqrt(
                    sum(value * value for value in self.sphere.initial_velocity)
                )
            ),
        )
        reynolds = (
            characteristic_velocity
            * self.initial_liquid_height
            / fluid.kinematic_viscosity
        )
        froude = (
            None
            if acceleration_magnitude == 0.0 or characteristic_velocity == 0.0
            else characteristic_velocity
            / math.sqrt(acceleration_magnitude * self.initial_liquid_height)
        )
        weber = (
            None
            if fluid.surface_tension == 0.0
            else fluid.reference_density
            * characteristic_velocity**2
            * self.initial_liquid_height
            / fluid.surface_tension
        )
        result: dict[str, Any] = {
            "physical_extent": tuple(
                cells * dx for cells in fluid.resolution
            ),
            "velocity_unit": dx / dt,
            "lattice_viscosity": lattice_viscosity,
            "shear_relaxation_time": 0.5 + 3.0 * lattice_viscosity,
            "shear_omega": 1.0 / (0.5 + 3.0 * lattice_viscosity),
            "lattice_acceleration": tuple(
                value * dt * dt / dx for value in fluid.body_acceleration
            ),
            "lattice_surface_tension": (
                fluid.surface_tension
                * dt
                * dt
                / (fluid.reference_density * dx**3)
            ),
            "configured_max_mach": fluid.max_lattice_speed * math.sqrt(3.0),
            "characteristic_length": self.initial_liquid_height,
            "characteristic_velocity": characteristic_velocity,
            "reynolds": reynolds,
            "froude": froude,
            "weber": weber,
        }
        if self.sphere is not None:
            sphere_volume = 4.0 * math.pi * self.sphere.radius**3 / 3.0
            sphere_mass = self.sphere.density * sphere_volume
            result.update(
                {
                    "sphere_diameter_cells": 2.0 * self.sphere.radius / dx,
                    "sphere_density_ratio": (
                        self.sphere.density / fluid.reference_density
                    ),
                    "sphere_mass": sphere_mass,
                    "sphere_inertia": 0.4 * sphere_mass * self.sphere.radius**2,
                }
            )
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HomeFreeOfflineConfig:
        fluid_data = dict(data["fluid"])
        fluid_data["resolution"] = tuple(fluid_data["resolution"])
        fluid_data["body_acceleration"] = tuple(fluid_data["body_acceleration"])
        fluid_data["periodic"] = tuple(fluid_data["periodic"])
        camera_data = dict(data["camera"])
        camera_data["position"] = tuple(camera_data["position"])
        camera_data["target"] = tuple(camera_data["target"])
        sphere_data = data.get("sphere")
        if sphere_data is not None:
            sphere_data = dict(sphere_data)
            sphere_data["position"] = tuple(sphere_data["position"])
            sphere_data["initial_velocity"] = tuple(sphere_data["initial_velocity"])
        return cls(
            scene=OfflineSceneName(data["scene"]),
            level=OfflineSceneLevel(data["level"]),
            steps=int(data["steps"]),
            output_every_steps=int(data["output_every_steps"]),
            device=str(data["device"]),
            fluid=FluidConfig(**fluid_data),
            camera=CameraHint(**camera_data),
            initial_liquid_height=float(data["initial_liquid_height"]),
            initial_liquid_x_range=(
                None
                if data.get("initial_liquid_x_range") is None
                else tuple(data["initial_liquid_x_range"])
            ),
            initial_fluid_velocity=tuple(data["initial_fluid_velocity"]),
            sphere=None if sphere_data is None else SphereConfig(**sphere_data),
            coupling=(
                None
                if data.get("coupling") is None
                else CouplingConfig(**data["coupling"])
            ),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

    @classmethod
    def from_json(cls, value: str) -> HomeFreeOfflineConfig:
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("scene JSON root must be an object")
        return cls.from_dict(data)


def make_scene_config(
    scene: OfflineSceneName | str,
    *,
    level: OfflineSceneLevel | str = OfflineSceneLevel.L0,
    device: str = "cpu",
) -> HomeFreeOfflineConfig:
    scene = OfflineSceneName(scene)
    level = OfflineSceneLevel(level)
    if level is not OfflineSceneLevel.L0:
        raise ValueError(
            f"{level.value} presets are not frozen; complete P2 convergence before use"
        )

    coupling = CouplingConfig(
        rigid_substeps=2,
        strong_max_iterations=10,
        strong_tolerance=2.0e-6,
        strong_relaxation=0.8,
    )
    if scene is OfflineSceneName.DAMBREAK:
        return HomeFreeOfflineConfig(
            scene=scene,
            level=level,
            steps=100,
            output_every_steps=4,
            device=device,
            fluid=FluidConfig(
                resolution=(24, 4, 16),
                cell_size=1.0,
                time_step=1.0,
                reference_density=1.0,
                kinematic_viscosity=0.08,
                body_acceleration=(0.0, 0.0, -2.0e-4),
                periodic=(False, True, False),
                max_lattice_speed=0.2,
                contact_angle_degrees=90.0,
            ),
            camera=CameraHint(position=(34.0, -26.0, 25.0), target=(12.0, 2.0, 7.0)),
            initial_liquid_height=8.5,
            initial_liquid_x_range=(0.0, 8.5),
        )
    if scene is OfflineSceneName.DAMBREAK_SPHERE:
        return HomeFreeOfflineConfig(
            scene=scene,
            level=level,
            steps=40,
            output_every_steps=1,
            device=device,
            fluid=FluidConfig(
                resolution=(28, 14, 14),
                cell_size=1.0,
                time_step=1.0,
                reference_density=1.0,
                kinematic_viscosity=0.2,
                body_acceleration=(0.0, 0.0, 0.0),
                periodic=(True, True, True),
                max_lattice_speed=0.18,
                contact_angle_degrees=90.0,
            ),
            camera=CameraHint(position=(39.0, -27.0, 24.0), target=(14.0, 7.0, 7.0)),
            initial_liquid_height=14.0,
            initial_liquid_x_range=(2.5, 11.5),
            initial_fluid_velocity=(0.06, 0.0, 0.0),
            sphere=SphereConfig(
                position=(16.0, 7.0, 7.0),
                radius=3.0,
                density=1.0,
                initial_velocity=(0.0, 0.0, 0.0),
                gravity=0.0,
                label="impact_sphere",
            ),
            coupling=coupling,
        )
    return HomeFreeOfflineConfig(
        scene=scene,
        level=level,
        steps=20,
        output_every_steps=1,
        device=device,
        fluid=FluidConfig(
            resolution=(24, 24, 22),
            cell_size=1.0,
            time_step=1.0,
            reference_density=1.0,
            kinematic_viscosity=0.2,
            body_acceleration=(0.0, 0.0, 0.0),
            periodic=(True, True, False),
            max_lattice_speed=0.2,
            contact_angle_degrees=90.0,
        ),
        camera=CameraHint(position=(38.0, -34.0, 29.0), target=(12.0, 12.0, 10.0)),
        initial_liquid_height=10.5,
        initial_liquid_x_range=None,
        sphere=SphereConfig(
            position=(12.0, 12.0, 14.8),
            radius=3.0,
            density=2.0,
            initial_velocity=(0.0, 0.0, -0.15),
            gravity=0.0,
            label="entering_sphere",
        ),
        coupling=coupling,
    )
