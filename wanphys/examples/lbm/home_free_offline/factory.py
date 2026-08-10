# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Build verified L0 HOME-Free scenes from versioned configs."""

from __future__ import annotations

import numpy as np

from wanphys._src.fluid.fluid_grid.home_lbm import (
    HomeFreeGeometricDomain,
    HomeFreeRigidCoupling,
    HomeLbmModel,
)
from wanphys.rigid import (
    RigidDomain,
    RigidModelBuilder,
    ShapeConfig,
    create_semiimplicit_solver,
)

from .config import HomeFreeOfflineConfig, OfflineSceneName
from .scene import HomeFreeOfflineScene


def _interval_fill(count: int, cell_size: float, lower: float, upper: float) -> np.ndarray:
    cell_lower = np.arange(count, dtype=np.float64) * cell_size
    cell_upper = cell_lower + cell_size
    overlap = np.minimum(cell_upper, upper) - np.maximum(cell_lower, lower)
    return np.clip(overlap / cell_size, 0.0, 1.0).astype(np.float32)


def build_rectangular_liquid_fill(
    shape: tuple[int, int, int],
    *,
    cell_size: float,
    height: float,
    x_range: tuple[float, float] | None,
) -> np.ndarray:
    """Rasterize a full-width pool or rectangular column in physical units."""

    fill_x = np.ones(shape[0], dtype=np.float32)
    if x_range is not None:
        fill_x = _interval_fill(shape[0], cell_size, *x_range)
    fill_z = _interval_fill(shape[2], cell_size, 0.0, height)
    return np.broadcast_to(
        fill_x[:, None, None] * fill_z[None, None, :], shape
    ).copy()


def build_dambreak_fill(shape: tuple[int, int, int]) -> np.ndarray:
    """Create the public L0 dam-break column with physical-unit rasterization."""

    return build_rectangular_liquid_fill(
        shape,
        cell_size=1.0,
        height=shape[2] // 2 + 0.5,
        x_range=(0.0, shape[0] // 3 + 0.5),
    )


def _build_model(config: HomeFreeOfflineConfig) -> HomeLbmModel:
    fluid = config.fluid
    return HomeLbmModel(
        fluid_grid_res=fluid.resolution,
        fluid_grid_cell_size=fluid.cell_size,
        time_step=fluid.time_step,
        reference_density=fluid.reference_density,
        kinematic_viscosity=fluid.kinematic_viscosity,
        body_acceleration=fluid.body_acceleration,
        periodic=fluid.periodic,
        max_lattice_speed=fluid.max_lattice_speed,
        device=config.device,
    )


def _build_rigid(
    config: HomeFreeOfflineConfig,
) -> tuple[RigidDomain, int]:
    if config.sphere is None or config.coupling is None:
        raise ValueError(f"{config.scene.value} requires rigid configuration")
    sphere = config.sphere
    builder = RigidModelBuilder(gravity=sphere.gravity)
    body = builder.add_body(position=sphere.position, label=sphere.label)
    builder.add_shape_sphere(
        body,
        radius=sphere.radius,
        cfg=ShapeConfig(density=sphere.density, has_shape_collision=True),
    )
    builder.set_ground_plane(enabled=True, label="tank_floor")
    rigid_model = builder.finalize(device=config.device)
    rigid = RigidDomain(
        rigid_model,
        solver=create_semiimplicit_solver(rigid_model, angular_damping=0.0),
    )
    rigid.create_state()
    body_qd = rigid.state.body_qd
    if body_qd is None:
        raise RuntimeError("rigid builder did not allocate body velocities")
    velocity = np.zeros((rigid_model.body_count, 6), dtype=np.float32)
    velocity[body, :3] = np.asarray(sphere.initial_velocity, dtype=np.float32)
    body_qd.assign(velocity)
    return rigid, body


def build_offline_scene(config: HomeFreeOfflineConfig) -> HomeFreeOfflineScene:
    model = _build_model(config)
    fluid = HomeFreeGeometricDomain(
        model,
        contact_angle_degrees=config.fluid.contact_angle_degrees,
        surface_tension=config.fluid.surface_tension,
        project_courant=config.fluid.project_courant,
        projection_max_iterations=config.fluid.projection_max_iterations,
    )

    rigid = None
    coupling = None
    body = None
    if config.sphere is not None:
        rigid, body = _build_rigid(config)
        coupling_config = config.coupling
        assert coupling_config is not None
        coupling = HomeFreeRigidCoupling(
            fluid,
            rigid,
            body_ids=(body,),
            rigid_substeps=coupling_config.rigid_substeps,
            strong_coupling_max_iterations=coupling_config.strong_max_iterations,
            strong_coupling_tolerance=coupling_config.strong_tolerance,
            strong_coupling_relaxation=coupling_config.strong_relaxation,
        )

    shape = config.fluid.resolution
    fill = build_rectangular_liquid_fill(
        shape,
        cell_size=config.fluid.cell_size,
        height=config.initial_liquid_height,
        x_range=config.initial_liquid_x_range,
    )
    if config.scene is OfflineSceneName.DAMBREAK:
        fluid.initialize_anchored_hydrostatic_lattice(
            fill,
            reference_coordinate=(
                0.5 * shape[0],
                0.5 * shape[1],
                shape[2] / 2 + 1.0,
            ),
        )
    elif config.scene is OfflineSceneName.DAMBREAK_SPHERE:
        fluid.initialize_uniform_lattice(
            fill,
            velocity=config.initial_fluid_velocity,
        )
    else:
        vertical_fill = fill[0, 0]
        interface_indices = np.flatnonzero(
            (vertical_fill > 0.0) & (vertical_fill < 1.0)
        )
        if len(interface_indices) != 1:
            raise ValueError(
                "sphere-entry water height must cut exactly one layer of cells"
            )
        fluid.initialize_planar_hydrostatic_lattice(
            fill,
            interface_axis=2,
            interface_index=int(interface_indices[0]),
            gas_direction=1,
        )

    initial_mass = float(
        np.sum(fluid.free_surface_state.mass.numpy(), dtype=np.float64)
    )
    return HomeFreeOfflineScene(
        config=config,
        fluid=fluid,
        initial_fill=fill,
        initial_mass=initial_mass,
        rigid=rigid,
        coupling=coupling,
        body_index=body,
    )
