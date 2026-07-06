# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid body model - isolates Newton dependency for future replacement."""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

import numpy as np
import warp as wp

from newton import Model as NewtonModel
from newton import eval_fk as _newton_eval_fk
from newton._src.sim.contacts import Contacts as NewtonContacts
from newton._src.sim.state import State as NewtonState
from newton._src.utils.heightfield import HeightfieldData

from .state import RigidState


class RigidModel:
    """Rigid body model wrapping scene configuration.

    RigidModel provides a stable WanPhys API for rigid body simulation,
    isolating the Newton dependency. This allows future migration to
    custom WanPhys implementations without breaking client code.

    WanPhys owns model arrays directly. Newton receives a zero-copy shell
    only through internal solver / viewer compatibility bridges.
    """

    _REQUEST_ATTRS: ClassVar[set[str]] = {"_requested_state_attributes", "_requested_contact_attributes"}
    _SUPPORTED_STATE_ATTRIBUTES: ClassVar[set[str]] = {
        "body_qdd",
        "body_parent_f",
        "mujoco:qfrc_actuator",
    }

    _newton_model_field_names: list[str]
    _requested_state_attributes: list[str]
    _requested_contact_attributes: set[str]
    _backend: NewtonModel | None
    _rigid_contact_max: int

    # Direct WanPhys-owned fields copied from the bridge model.
    particle_q: wp.array | None
    particle_qd: wp.array | None
    particle_mass: wp.array | None
    particle_inv_mass: wp.array | None
    particle_radius: wp.array | None
    particle_flags: wp.array | None
    particle_world: wp.array | None
    particle_world_start: wp.array | None
    particle_colors: wp.array | None
    shape_transform: wp.array | None
    shape_body: wp.array | None
    shape_flags: wp.array | None
    shape_material_ke: wp.array | None
    shape_material_kd: wp.array | None
    shape_material_kf: wp.array | None
    shape_material_ka: wp.array | None
    shape_material_mu: wp.array | None
    shape_material_restitution: wp.array | None
    shape_material_mu_torsional: wp.array | None
    shape_material_mu_rolling: wp.array | None
    shape_material_kh: wp.array | None
    shape_gap: wp.array | None
    shape_type: wp.array | None
    shape_is_solid: wp.array | None
    shape_margin: wp.array | None
    shape_source_ptr: wp.array | None
    shape_scale: wp.array | None
    shape_collision_group: wp.array | None
    shape_collision_radius: wp.array | None
    shape_contact_pairs: wp.array | None
    shape_world: wp.array | None
    shape_world_start: wp.array | None
    shape_heightfield_data: wp.array | None
    heightfield_elevation_data: wp.array | None
    sdf_data: wp.array | None
    shape_sdf_index: wp.array | None
    sdf_block_coords: wp.array | None
    sdf_index2blocks: wp.array | None
    shape_collision_aabb_lower: wp.array | None
    shape_collision_aabb_upper: wp.array | None
    _shape_voxel_resolution: wp.array | None
    muscle_start: wp.array | None
    muscle_params: wp.array | None
    muscle_bodies: wp.array | None
    muscle_points: wp.array | None
    muscle_activations: wp.array | None
    body_q: wp.array | None
    body_qd: wp.array | None
    body_com: wp.array | None
    body_inertia: wp.array | None
    body_inv_inertia: wp.array | None
    body_mass: wp.array | None
    body_inv_mass: wp.array | None
    body_flags: wp.array | None
    body_world: wp.array | None
    body_world_start: wp.array | None
    joint_q: wp.array | None
    joint_qd: wp.array | None
    joint_f: wp.array | None
    joint_target_pos: wp.array | None
    joint_target_vel: wp.array | None
    joint_act: wp.array | None
    joint_type: wp.array | None
    joint_articulation: wp.array | None
    joint_parent: wp.array | None
    joint_child: wp.array | None
    joint_ancestor: wp.array | None
    joint_X_p: wp.array | None
    joint_X_c: wp.array | None
    joint_axis: wp.array | None
    joint_armature: wp.array | None
    joint_target_mode: wp.array | None
    joint_target_ke: wp.array | None
    joint_target_kd: wp.array | None
    joint_effort_limit: wp.array | None
    joint_velocity_limit: wp.array | None
    joint_friction: wp.array | None
    joint_dof_dim: wp.array | None
    joint_enabled: wp.array | None
    joint_limit_lower: wp.array | None
    joint_limit_upper: wp.array | None
    joint_limit_ke: wp.array | None
    joint_limit_kd: wp.array | None
    joint_q_start: wp.array | None
    joint_qd_start: wp.array | None
    joint_world: wp.array | None
    joint_world_start: wp.array | None
    joint_dof_world_start: wp.array | None
    joint_coord_world_start: wp.array | None
    joint_constraint_world_start: wp.array | None
    articulation_start: wp.array | None
    articulation_world: wp.array | None
    articulation_world_start: wp.array | None
    gravity: wp.array | None
    equality_constraint_type: wp.array | None
    equality_constraint_body1: wp.array | None
    equality_constraint_body2: wp.array | None
    equality_constraint_anchor: wp.array | None
    equality_constraint_torquescale: wp.array | None
    equality_constraint_relpose: wp.array | None
    equality_constraint_joint1: wp.array | None
    equality_constraint_joint2: wp.array | None
    equality_constraint_polycoef: wp.array | None
    equality_constraint_enabled: wp.array | None
    equality_constraint_world: wp.array | None
    equality_constraint_world_start: wp.array | None
    constraint_mimic_joint0: wp.array | None
    constraint_mimic_joint1: wp.array | None
    constraint_mimic_coef0: wp.array | None
    constraint_mimic_coef1: wp.array | None
    constraint_mimic_enabled: wp.array | None
    constraint_mimic_world: wp.array | None

    requires_grad: bool
    world_count: int
    particle_max_radius: Any
    particle_ke: float
    particle_kd: float
    particle_kf: float
    particle_mu: float
    particle_cohesion: float
    particle_adhesion: float
    particle_grid: Any
    particle_max_velocity: float
    shape_filter: Any
    shape_contact_pair_count: int
    spring_indices: Any
    spring_rest_length: Any
    spring_stiffness: Any
    spring_damping: Any
    spring_control: Any
    spring_constraint_lambdas: Any
    tri_indices: Any
    tri_poses: Any
    tri_activations: Any
    tri_materials: Any
    tri_areas: Any
    edge_indices: Any
    edge_rest_angle: Any
    edge_rest_length: Any
    edge_bending_properties: Any
    edge_constraint_lambdas: Any
    tet_indices: Any
    tet_poses: Any
    tet_activations: Any
    tet_materials: Any
    joint_twist_lower: Any
    joint_twist_upper: Any
    max_joints_per_articulation: int
    max_dofs_per_articulation: int
    soft_contact_ke: float
    soft_contact_kd: float
    soft_contact_kf: float
    soft_contact_mu: float
    soft_contact_restitution: float
    up_axis: Any
    particle_count: int
    body_count: int
    shape_count: int
    joint_count: int
    tri_count: int
    tet_count: int
    edge_count: int
    spring_count: int
    muscle_count: int
    articulation_count: int
    joint_dof_count: int
    joint_coord_count: int
    joint_constraint_count: int
    equality_constraint_count: int
    constraint_mimic_count: int
    body_colors: Any
    _collision_pipeline: Any

    shape_label: list[str]
    body_shapes: dict[int, list[int]]
    shape_source: list[Any]
    shape_collision_filter_pairs: set[Any]
    sdf_volume: list[Any]
    sdf_coarse_volume: list[Any]
    body_label: list[str]
    joint_label: list[str]
    articulation_label: list[str]
    equality_constraint_label: list[str]
    constraint_mimic_label: list[str]
    particle_color_groups: list[Any]
    body_color_groups: list[Any]
    device: Any
    attribute_frequency: dict[str, Any]
    custom_frequency_counts: dict[str, Any]
    attribute_assignment: dict[str, Any]
    actuators: list[Any]

    def __init__(self, newton_model: NewtonModel):
        """Initialize rigid model from the current bridge model.

        Args:
            newton_model: Bridge model supplied by RigidModelBuilder. Direct
                construction is retained for migration compatibility.
        """
        self._requested_state_attributes: list[str] = list(
            getattr(newton_model, "_requested_state_attributes", ())
        )
        self._requested_contact_attributes: set[str] = set(
            getattr(newton_model, "_requested_contact_attributes", set())
        )
        self._newton_model_field_names: list[str] = []
        self._backend: NewtonModel | None = None
        self._copy_owned_model_fields(newton_model)
        self._ensure_direct_field_defaults()

    @property
    def num_worlds(self) -> int:
        """Compatibility alias for examples migrated from Newton terminology."""
        return self.world_count

    def get_attribute_frequency(self, name: str) -> Any:
        """Return the registered frequency for a model attribute."""
        frequency = self.attribute_frequency.get(name)
        if frequency is None:
            raise KeyError(f"Attribute frequency of '{name}' is not known")
        return frequency

    # ------------------------------------------------------------------
    # WANPHYS ownership helpers
    # ------------------------------------------------------------------

    def _copy_owned_model_fields(self, newton_model: NewtonModel) -> None:
        """Clone bridge model fields into direct WanPhys-owned attributes."""
        model_items = list(vars(newton_model).items())
        for attr, value in model_items:
            if attr in self._REQUEST_ATTRS:
                continue

            self._newton_model_field_names.append(attr)
            if attr == "rigid_contact_max":
                self._rigid_contact_max: int = int(value or 0)
                continue

            copied_value = self._copy_model_value(value)
            setattr(self, attr, copied_value)

    def _copy_model_value(self, value: Any) -> Any:
        """Copy one model field while preserving owned GPU buffer semantics."""
        if isinstance(value, wp.array):
            cloned_value = wp.clone(value)
            return cloned_value
        if isinstance(value, list):
            list_value = list(value)
            return list_value
        if isinstance(value, dict):
            dict_value = dict(value)
            return dict_value
        if isinstance(value, set):
            set_value = set(value)
            return set_value
        return value

    def _ensure_direct_field_defaults(self) -> None:
        """Fill optional fields that rigid collision code expects as arrays."""
        if not hasattr(self, "shape_heightfield_data") or self.shape_heightfield_data is None:
            self.shape_heightfield_data = wp.zeros(0, dtype=HeightfieldData, device=self.device)
            self._register_newton_model_field("shape_heightfield_data")
        if not hasattr(self, "heightfield_elevation_data") or self.heightfield_elevation_data is None:
            self.heightfield_elevation_data = wp.zeros(0, dtype=wp.float32, device=self.device)
            self._register_newton_model_field("heightfield_elevation_data")

    def _register_newton_model_field(self, attr: str) -> None:
        if attr not in self._newton_model_field_names:
            self._newton_model_field_names.append(attr)

    def as_newton_model(self) -> NewtonModel:
        """Return a ``newton.Model`` compatible with solvers and viewers.

        The returned model aliases the owned warp arrays (zero copy).

        Returns:
            Newton Model instance.
        """
        warnings.warn(
            "as_newton_model() 是临时内部接口，将在 Newton 移除后失效。"
            "请使用 WanPhys 原生 API 代替。",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._backend is not None:
            return self._backend

        m = NewtonModel(device=self.device)
        for attr in self._newton_model_field_names:
            value = getattr(self, attr)
            setattr(m, attr, value)
        m._requested_state_attributes = set(self._requested_state_attributes)
        m._requested_contact_attributes = set(self._requested_contact_attributes)
        self._backend = m
        return m

    @property
    def rigid_contact_max(self) -> int:
        """Maximum rigid contact capacity configured on the model."""
        return self._rigid_contact_max

    @rigid_contact_max.setter
    def rigid_contact_max(self, value: int) -> None:
        self._rigid_contact_max = int(value)
        if self._backend is not None:
            self._backend.rigid_contact_max = int(value)

    def set_gravity(
        self,
        gravity: tuple[float, float, float] | list | wp.vec3 | np.ndarray,
        world: int | None = None,
    ) -> None:
        """Set gravity for all worlds or one world."""
        gravity_np = np.asarray(gravity, dtype=np.float32)
        if world is not None:
            if gravity_np.shape != (3,):
                raise ValueError("Expected single gravity vector (3,) when world is specified")
            if world < 0 or world >= self.world_count:
                raise IndexError(f"world {world} out of range [0, {self.world_count})")
            current = self.gravity.numpy()
            current[world] = gravity_np
            self.gravity.assign(current)
        elif gravity_np.ndim == 1:
            self.gravity.fill_(gravity_np)
        else:
            if len(gravity_np) != self.world_count:
                raise ValueError(f"Expected {self.world_count} gravity vectors, got {len(gravity_np)}")
            self.gravity.assign(gravity_np)

        if self._backend is not None:
            self._backend.set_gravity(gravity, world=world)

    def get_body_mass(self, body_idx: int) -> float:
        """Return the mass of a rigid body.

        Args:
            body_idx: Index of the body.

        Returns:
            Mass in kg. Returns 0.0 for static (infinite-mass) bodies.
        """
        inv_mass_arr = self.body_inv_mass
        inv_mass = float(inv_mass_arr.numpy()[body_idx])
        return 1.0 / inv_mass if inv_mass != 0.0 else 0.0

    def request_state_attributes(self, *attributes: str) -> None:
        """Request optional state arrays for subsequently created states."""
        NewtonState.validate_extended_attributes(attributes)
        unsupported = set(attributes).difference(self._SUPPORTED_STATE_ATTRIBUTES)
        if unsupported:
            allowed = ", ".join(sorted(self._SUPPORTED_STATE_ATTRIBUTES))
            bad = ", ".join(sorted(unsupported))
            raise ValueError(f"Unsupported RigidState attribute(s): {bad}. Allowed: {allowed}.")
        for attr in attributes:
            if attr not in self._requested_state_attributes:
                self._requested_state_attributes.append(attr)

        if self._backend is not None:
            self._backend.request_state_attributes(*attributes)

    def get_requested_state_attributes(self) -> list[str]:
        """Return optional state arrays requested through this RigidModel."""
        return list(self._requested_state_attributes)

    def request_contact_attributes(self, *attributes: str) -> None:
        """Request optional contact arrays for subsequently created contacts."""
        NewtonContacts.validate_extended_attributes(attributes)
        self._requested_contact_attributes.update(attributes)

        if self._backend is not None:
            self._backend.request_contact_attributes(*attributes)

    def get_requested_contact_attributes(self) -> set[str]:
        """Return requested contact attribute names."""
        return set(self._requested_contact_attributes)

    def _add_custom_attributes(
        self,
        destination: object,
        assignment,
        requires_grad: bool = False,
        clone_arrays: bool = True,
    ) -> None:
        """Delegate custom attribute attachment to the backing Newton model."""
        self.as_newton_model()._add_custom_attributes(
            destination,
            assignment,
            requires_grad=requires_grad,
            clone_arrays=clone_arrays,
        )

    # ------------------------------------------------------------------
    # State factory
    # ------------------------------------------------------------------

    def state(self) -> RigidState:
        """Create a new state with initial conditions from the model.

        Returns:
            RigidState initialized with model's default configuration.
        """
        return RigidState.create_owned(
            body_count=self.body_count,
            joint_coord_count=self.joint_coord_count,
            joint_dof_count=self.joint_dof_count,
            particle_count=self.particle_count,
            device=self.device,
            init_from=self,
            requested_state_attributes=self.get_requested_state_attributes(),
            requires_grad=self.requires_grad,
        )

    def control(self):
        """Create a control input object for setting joint targets.

        Returns:
            Control object for joint motor targets and actuator inputs.
        """
        warnings.warn(
            "control() 直接返回 newton.Control 对象，属于临时内部接口。"
            "后续将由 WanPhys 原生 Control 抽象替代。",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._newton_backend.control()

    # ------------------------------------------------------------------
    # Viewer & FK helpers
    # ------------------------------------------------------------------

    def setup_viewer(self, viewer) -> None:
        """Register this model with a Newton viewer.

        Args:
            viewer: Newton viewer instance (GL, USD, Rerun, or Null).
        """
        warnings.warn(
            "setup_viewer() 依赖 Newton Viewer API，属于临时内部接口。"
            "待 WanPhys 自有 Viewer 适配层完成后将移除。",
            DeprecationWarning,
            stacklevel=2,
        )
        viewer.set_model(self._newton_backend)

    def eval_forward_kinematics(self, state: RigidState) -> None:
        """Evaluate forward kinematics, updating body transforms from joint state.

        Args:
            state: State whose ``body_q`` / ``body_qd`` will be updated
                   in-place based on ``joint_q`` / ``joint_qd``.
        """
        warnings.warn(
            "eval_forward_kinematics() 当前委托给 Newton 实现，属于临时内部接口。"
            "后续将由 WanPhys 原生 Warp 正运动学内核替代。",
            DeprecationWarning,
            stacklevel=2,
        )
        _newton_eval_fk(
            self._newton_backend,
            state.joint_q,
            state.joint_qd,
            state.as_newton_state(),
        )

    # ------------------------------------------------------------------
    # Newton escape hatches
    # ------------------------------------------------------------------

    @property
    def _newton_backend(self) -> NewtonModel:
        """Access underlying Newton model (for migration/debugging).

        In WANPHYS mode, builds and caches the zero-copy shell on first access.

        Warning:
            This is a temporary escape hatch. Code using this property
            will break when Newton is replaced. Use sparingly.
        """
        warnings.warn(
            "_newton_backend 是临时内部接口，供迁移与调试使用。"
            "Newton 移除后此属性将不再可用，请勿在业务代码中依赖。",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.as_newton_model()

    @classmethod
    def from_newton(cls, newton_model: NewtonModel) -> RigidModel:
        """Create RigidModel from Newton Model (explicit factory).

        Args:
            newton_model: Newton Model instance.

        Returns:
            RigidModel wrapping the Newton model.
        """
        warnings.warn(
            "from_newton() 是临时工厂方法，依赖 Newton Model 作为入口。"
            "后续将由 RigidModelBuilder.finalize() 的 WanPhys 原生实现替代。",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(newton_model)
