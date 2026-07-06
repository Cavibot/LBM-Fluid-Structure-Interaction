# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys scene builder — stable API wrapping Newton's ModelBuilder.

RigidModelBuilder provides a WanPhys-native interface for constructing
rigid body scenes.  Internally it delegates to Newton's ModelBuilder,
but the public surface is stable and scoped to the features that
WanPhys demos actually exercise.

Example:
    >>> from wanphys.rigid import RigidModelBuilder
    >>> builder = RigidModelBuilder(up_axis=2)
    >>> body = builder.add_body(position=(0, 3, 0), label="box")
    >>> builder.add_shape_box(body, hx=0.2, hy=0.2, hz=0.2)
    >>> model = builder.finalize()
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import warp as wp

import newton

if TYPE_CHECKING:
    from .model import RigidModel


# ---------------------------------------------------------------------------
# ShapeConfig — thin wrapper around Newton's ShapeConfig
# ---------------------------------------------------------------------------


@dataclass
class ShapeConfig:
    """Configuration for collision shapes.

    This is a WanPhys-stable wrapper around Newton's
    ``ModelBuilder.ShapeConfig``.  Only the most commonly used fields
    are exposed; extend as needed.
    """

    density: float = 1000.0
    mu: float = 0.5
    restitution: float = 0.0
    ke: float = 1.0e3
    kd: float = 100.0
    kf: float = 1000.0
    ka: float = 0.0
    is_visible: bool = True
    is_solid: bool = True
    collision_group: int = 1
    collision_filter_parent: bool = True
    has_shape_collision: bool = True
    has_particle_collision: bool = True
    margin: float = 0.0
    gap: float | None = None

    def _to_newton(self) -> newton.ModelBuilder.ShapeConfig:
        """Convert to Newton ShapeConfig."""
        return newton.ModelBuilder.ShapeConfig(
            density=self.density,
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            ka=self.ka,
            mu=self.mu,
            restitution=self.restitution,
            margin=self.margin,
            gap=self.gap,
            is_visible=self.is_visible,
            is_solid=self.is_solid,
            collision_group=self.collision_group,
            collision_filter_parent=self.collision_filter_parent,
            has_shape_collision=self.has_shape_collision,
            has_particle_collision=self.has_particle_collision,
        )


# ---------------------------------------------------------------------------
# RigidModelBuilder
# ---------------------------------------------------------------------------


class RigidModelBuilder:
    """WanPhys scene builder.  Internally delegates to Newton's ModelBuilder.

    All ``add_*`` methods mirror Newton's signatures closely so that
    migration from raw Newton code is straightforward.  The key
    difference is that ``finalize()`` returns a :class:`RigidModel`
    instead of a bare ``newton.Model``.

    Args:
        up_axis: Up-axis index (0 = X, 1 = Y, 2 = Z).  Default 2 (Z-up),
            matching Newton's default.
        gravity: Gravity magnitude along the up-axis (negative = downward).
    """

    def __init__(
        self,
        up_axis: int = 2,
        gravity: float = -9.81,
    ) -> None:
        self._newton_builder: newton.ModelBuilder = newton.ModelBuilder(up_axis=up_axis, gravity=gravity)

    # ------------------------------------------------------------------
    # Direct access to Newton builder (escape hatch for advanced usage)
    # ------------------------------------------------------------------

    @property
    def _backend(self) -> newton.ModelBuilder:
        """Access underlying Newton ModelBuilder (temporary escape hatch)."""
        return self._newton_builder

    # ------------------------------------------------------------------
    # Default shape config
    # ------------------------------------------------------------------

    @property
    def default_shape_config(self) -> ShapeConfig:
        """Return a default ShapeConfig (convenience factory)."""
        return ShapeConfig()

    def set_default_shape_config(self, cfg: ShapeConfig) -> None:
        """Set the default shape config used by subsequent shape additions."""
        self._newton_builder.default_shape_cfg = cfg._to_newton()

    @property
    def default_shape_cfg(self) -> newton.ModelBuilder.ShapeConfig:
        """Mutable default shape config for importer-heavy examples."""
        return self._newton_builder.default_shape_cfg

    @property
    def default_joint_cfg(self) -> newton.ModelBuilder.JointDofConfig:
        """Mutable default joint config for importer-heavy examples."""
        return self._newton_builder.default_joint_cfg

    @property
    def default_body_armature(self) -> float:
        """Default armature applied to subsequently imported bodies."""
        return self._newton_builder.default_body_armature

    @default_body_armature.setter
    def default_body_armature(self, value: float) -> None:
        self._newton_builder.default_body_armature = value

    def set_default_joint_config(
        self,
        *,
        armature: float | None = None,
        limit_ke: float | None = None,
        limit_kd: float | None = None,
        target_ke: float | None = None,
        target_kd: float | None = None,
        friction: float | None = None,
    ) -> None:
        """Set commonly used default joint DOF parameters."""
        cfg = self._newton_builder.default_joint_cfg
        if armature is not None:
            cfg.armature = armature
        if limit_ke is not None:
            cfg.limit_ke = limit_ke
        if limit_kd is not None:
            cfg.limit_kd = limit_kd
        if target_ke is not None:
            cfg.target_ke = target_ke
        if target_kd is not None:
            cfg.target_kd = target_kd
        if friction is not None:
            cfg.friction = friction

    # ------------------------------------------------------------------
    # Bodies
    # ------------------------------------------------------------------

    def add_body(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] | None = None,
        xform: Any | None = None,
        mass: float = 0.0,
        label: str | None = None,
    ) -> int:
        """Add a free rigid body.

        You can specify the pose either via *position*/*rotation* or
        via a pre-built *xform* (``wp.transform``).  If *xform* is
        given it takes precedence.

        Args:
            position: World position (x, y, z).
            rotation: Quaternion (qx, qy, qz, qw).  Defaults to identity.
            xform: Warp transform (overrides position/rotation if given).
            mass: Mass hint (usually inferred from shapes' density).
            label: Optional string label for the body.

        Returns:
            Body index.
        """
        if xform is None:
            quat = wp.quat_identity() if rotation is None else rotation
            xform = wp.transform(position, quat)
        return self._newton_builder.add_body(xform=xform, mass=mass, label=label)

    # ------------------------------------------------------------------
    # Links & Articulations
    # ------------------------------------------------------------------

    def add_link(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float, float] | None = None,
        xform: Any | None = None,
        mass: float = 0.0,
        label: str | None = None,
    ) -> int:
        """Add a link (body without automatic free-joint).

        Use this when building articulations where joints are added
        explicitly via :meth:`add_joint_revolute` etc.

        Returns:
            Link (body) index.
        """
        if xform is None:
            quat = wp.quat_identity() if rotation is None else rotation
            xform = wp.transform(position, quat)
        return self._newton_builder.add_link(xform=xform, mass=mass, label=label)

    def add_articulation(self, joints: list[int], label: str | None = None) -> int:
        """Register a set of joints as an articulation.

        Args:
            joints: List of joint indices (must be contiguous and monotonically increasing).
            label: Optional label for the articulation.

        Returns:
            Articulation index.
        """
        return self._newton_builder.add_articulation(joints, label=label)

    # ------------------------------------------------------------------
    # Shapes
    # ------------------------------------------------------------------

    @staticmethod
    def _cfg(cfg: ShapeConfig | None) -> newton.ModelBuilder.ShapeConfig | None:
        """Convert optional WanPhys ShapeConfig to Newton's."""
        if cfg is None:
            return None
        return cfg._to_newton()

    def add_shape_box(
        self,
        body: int,
        hx: float = 0.5,
        hy: float = 0.5,
        hz: float = 0.5,
        xform: Any | None = None,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        """Add a box collision shape.

        Args:
            body: Body index to attach the shape to.
            hx, hy, hz: Half-extents along each axis.
            xform: Local transform of the shape relative to the body.
            cfg: Shape configuration (density, friction, etc.).
            label: Optional label for the shape.

        Returns:
            Shape index.
        """
        return self._newton_builder.add_shape_box(
            body=body, hx=hx, hy=hy, hz=hz, xform=xform, cfg=self._cfg(cfg), label=label
        )

    def add_shape_sphere(
        self,
        body: int,
        radius: float = 0.5,
        xform: Any | None = None,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        """Add a sphere collision shape.

        Args:
            body: Body index.
            radius: Sphere radius.
            xform: Local transform relative to the body.
            cfg: Shape configuration.
            label: Optional label.

        Returns:
            Shape index.
        """
        return self._newton_builder.add_shape_sphere(body=body, radius=radius, xform=xform, cfg=self._cfg(cfg), label=label)

    def add_shape_capsule(
        self,
        body: int,
        radius: float = 0.5,
        half_height: float = 0.5,
        xform: Any | None = None,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        """Add a capsule collision shape.

        Args:
            body: Body index.
            radius: Capsule radius.
            half_height: Half of the capsule's cylindrical section height.
            xform: Local transform relative to the body.
            cfg: Shape configuration.
            label: Optional label.

        Returns:
            Shape index.
        """
        return self._newton_builder.add_shape_capsule(
            body=body, radius=radius, half_height=half_height, xform=xform, cfg=self._cfg(cfg), label=label
        )

    def add_shape_cylinder(
        self,
        body: int,
        radius: float = 0.5,
        half_height: float = 0.5,
        xform: Any | None = None,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        """Add a cylinder collision shape.

        Args:
            body: Body index.
            radius: Cylinder radius.
            half_height: Half-height of the cylinder along Z.
            xform: Local transform relative to the body.
            cfg: Shape configuration.
            label: Optional label.

        Returns:
            Shape index.
        """
        return self._newton_builder.add_shape_cylinder(
            body=body, radius=radius, half_height=half_height, xform=xform, cfg=self._cfg(cfg), label=label
        )

    def add_shape_cone(
        self,
        body: int,
        radius: float = 1.0,
        half_height: float = 0.5,
        xform: Any | None = None,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        """Add a cone collision shape."""
        return self._newton_builder.add_shape_cone(
            body=body,
            radius=radius,
            half_height=half_height,
            xform=xform,
            cfg=self._cfg(cfg),
            label=label,
        )

    def add_shape_ellipsoid(
        self,
        body: int,
        a: float = 1.0,
        b: float = 0.75,
        c: float = 0.5,
        xform: Any | None = None,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        """Add an ellipsoid collision shape."""
        return self._newton_builder.add_shape_ellipsoid(
            body=body,
            a=a,
            b=b,
            c=c,
            xform=xform,
            cfg=self._cfg(cfg),
            label=label,
        )

    def add_shape_mesh(
        self,
        body: int,
        mesh: Any = None,
        xform: Any | None = None,
        scale: Any | None = None,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        """Add a triangle-mesh collision shape.

        Args:
            body: Body index.
            mesh: Newton Mesh object (vertices + triangles).
            xform: Local transform relative to the body.
            scale: Scale vector (Vec3 or 3-tuple).
            cfg: Shape configuration.
            label: Optional label.

        Returns:
            Shape index.
        """
        return self._newton_builder.add_shape_mesh(
            body=body, mesh=mesh, xform=xform, scale=scale, cfg=self._cfg(cfg), label=label
        )

    def add_shape_convex_hull(
        self,
        body: int,
        mesh: Any = None,
        xform: Any | None = None,
        scale: Any | None = None,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        """Add a convex hull collision shape."""
        return self._newton_builder.add_shape_convex_hull(
            body=body,
            mesh=mesh,
            xform=xform,
            scale=scale,
            cfg=self._cfg(cfg),
            label=label,
        )

    def add_shape_heightfield(
        self,
        heightfield: Any,
        body: int = -1,
        xform: Any | None = None,
        scale: Any | None = None,
        cfg: ShapeConfig | None = None,
        is_static: bool = True,
        label: str | None = None,
    ) -> int:
        """Add a static heightfield collision shape."""
        return self._newton_builder.add_shape(
            body=body,
            type=newton.GeoType.HFIELD,
            xform=xform,
            src=heightfield,
            scale=scale,
            cfg=self._cfg(cfg),
            is_static=is_static,
            label=label,
        )

    def add_shape_plane(
        self,
        plane: Any | None = (0.0, 0.0, 1.0, 0.0),
        xform: Any | None = None,
        width: float = 10.0,
        length: float = 10.0,
        body: int = -1,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        """Add a plane collision shape."""
        return self._newton_builder.add_shape_plane(
            plane=plane,
            xform=xform,
            width=width,
            length=length,
            body=body,
            cfg=self._cfg(cfg),
            label=label,
        )

    def add_site(
        self,
        body: int,
        xform: Any | None = None,
        type: int | None = None,
        scale: Any = (0.01, 0.01, 0.01),
        label: str | None = None,
        visible: bool = False,
    ) -> int:
        """Add a non-colliding reference site shape."""
        kwargs = {}
        if type is not None:
            kwargs["type"] = type
        return self._newton_builder.add_site(
            body=body,
            xform=xform,
            scale=scale,
            label=label,
            visible=visible,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Joints
    # ------------------------------------------------------------------

    def add_joint_revolute(
        self,
        parent: int,
        child: int,
        axis: tuple[float, float, float] | Any = (0.0, 0.0, 1.0),
        parent_xform: Any | None = None,
        child_xform: Any | None = None,
        friction: float | None = None,
        armature: float | None = None,
        target_ke: float | None = None,
        target_kd: float | None = None,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        label: str | None = None,
    ) -> int:
        """Add a revolute (hinge) joint.

        Args:
            parent: Parent body index (-1 for world).
            child: Child body index.
            axis: Rotation axis in parent frame.
            parent_xform: Transform from parent body to joint frame.
            child_xform: Transform from child body to joint frame.
            friction: Joint friction coefficient.
            armature: Joint armature (regularization).
            target_ke: Proportional gain for position target.
            target_kd: Derivative gain for velocity target.
            limit_lower: Lower angle limit (radians).
            limit_upper: Upper angle limit (radians).
            label: Optional label.

        Returns:
            Joint index.
        """
        kwargs = {}
        if parent_xform is not None:
            kwargs["parent_xform"] = parent_xform
        if child_xform is not None:
            kwargs["child_xform"] = child_xform
        if friction is not None:
            kwargs["friction"] = friction
        if armature is not None:
            kwargs["armature"] = armature
        if target_ke is not None:
            kwargs["target_ke"] = target_ke
        if target_kd is not None:
            kwargs["target_kd"] = target_kd
        if limit_lower is not None:
            kwargs["limit_lower"] = limit_lower
        if limit_upper is not None:
            kwargs["limit_upper"] = limit_upper

        return self._newton_builder.add_joint_revolute(
            parent=parent,
            child=child,
            axis=axis,
            label=label,
            **kwargs,
        )

    def add_joint_fixed(
        self,
        parent: int,
        child: int,
        parent_xform: Any | None = None,
        child_xform: Any | None = None,
        label: str | None = None,
    ) -> int:
        """Add a fixed (rigid) joint.

        Args:
            parent: Parent body index (-1 for world).
            child: Child body index.
            parent_xform: Transform from parent body to joint frame.
            child_xform: Transform from child body to joint frame.
            label: Optional label.

        Returns:
            Joint index.
        """
        kwargs = {}
        if parent_xform is not None:
            kwargs["parent_xform"] = parent_xform
        if child_xform is not None:
            kwargs["child_xform"] = child_xform

        return self._newton_builder.add_joint_fixed(
            parent=parent,
            child=child,
            label=label,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Ground plane
    # ------------------------------------------------------------------

    def set_ground_plane(
        self,
        enabled: bool = True,
        cfg: ShapeConfig | None = None,
        label: str | None = None,
    ) -> int | None:
        """Add a ground plane.

        Args:
            enabled: Whether to add the ground plane.
            cfg: Shape configuration for the ground.
            label: Optional label.

        Returns:
            Shape index, or None if ``enabled`` is False.
        """
        if not enabled:
            return None
        return self._newton_builder.add_ground_plane(cfg=self._cfg(cfg), label=label)

    def add_ground_plane(self, cfg: ShapeConfig | None = None, label: str | None = None) -> int:
        """Add a ground plane (shorthand matching Newton's API).

        Returns:
            Shape index.
        """
        return self._newton_builder.add_ground_plane(cfg=self._cfg(cfg), label=label)

    # ------------------------------------------------------------------
    # Particles
    # ------------------------------------------------------------------

    def add_particles(
        self,
        positions: Sequence | None = None,
        velocities: Sequence | None = None,
        *,
        pos: Sequence | None = None,
        vel: Sequence | None = None,
        mass: Sequence | float = 1.0,
        radius: Sequence | float | None = 0.1,
        flags: Sequence | None = None,
    ) -> None:
        """Add a group of particles (e.g. for rendering demos).

        Args:
            positions: List of (x, y, z) positions.
            velocities: List of (vx, vy, vz) velocities.  Defaults to zero.
            pos: Newton-compatible alias for positions.
            vel: Newton-compatible alias for velocities.
            mass: Per-particle mass (scalar or list).
            radius: Per-particle radius (scalar, list, or None).
            flags: Optional per-particle flags forwarded to the builder.
        """
        if positions is None:
            positions = pos
        if velocities is None:
            velocities = vel
        if positions is None:
            raise ValueError("add_particles requires positions or pos")

        n = len(positions)
        if velocities is None:
            velocities = [(0.0, 0.0, 0.0)] * n

        # Normalize scalar mass/radius to lists
        if isinstance(mass, (int, float)):
            mass = [float(mass)] * n
        if radius is None:
            radius = [0.1] * n
        elif isinstance(radius, (int, float)):
            radius = [float(radius)] * n

        self._newton_builder.add_particles(
            pos=list(positions),
            vel=list(velocities),
            mass=list(mass),
            radius=list(radius),
            flags=flags,
        )

    # ------------------------------------------------------------------
    # USD loading (delegate)
    # ------------------------------------------------------------------

    def add_usd(self, filename: str, **kwargs: Any) -> None:
        """Load scene from a USD file.

        This delegates to Newton's USD importer.

        Args:
            filename: Path to USD/USDA/USDC file.
            **kwargs: Importer options forwarded to Newton's USD importer.
        """
        self._newton_builder.add_usd(filename, **kwargs)

    def add_mjcf(self, filename: str, **kwargs: Any) -> None:
        """Load scene from a MuJoCo MJCF file."""
        self._newton_builder.add_mjcf(filename, **kwargs)

    def add_urdf(self, source: Any, **kwargs: Any) -> None:
        """Load a URDF into this builder."""
        self._newton_builder.add_urdf(source, **kwargs)

    def register_custom_attributes(self, solver_cls: Any) -> None:
        """Register solver-specific model attributes on this builder."""
        solver_cls.register_custom_attributes(self._newton_builder)

    def begin_world(
        self,
        label: str | None = None,
        attributes: dict[str, Any] | None = None,
        gravity: Any | None = None,
    ) -> None:
        """Begin a world context for subsequently added entities."""
        self._newton_builder.begin_world(label=label, attributes=attributes, gravity=gravity)

    def end_world(self) -> None:
        """End the active world context."""
        self._newton_builder.end_world()

    def add_builder(
        self,
        builder: RigidModelBuilder,
        xform: Any | None = None,
        label_prefix: str | None = None,
    ) -> None:
        """Copy entities from another RigidModelBuilder into this builder."""
        self._newton_builder.add_builder(builder._newton_builder, xform=xform, label_prefix=label_prefix)

    def add_world(self, builder: RigidModelBuilder, **kwargs: Any) -> None:
        """Copy another builder as a separate world."""
        self._newton_builder.add_world(builder._newton_builder, **kwargs)

    def replicate(
        self,
        builder: RigidModelBuilder,
        world_count: int,
        spacing: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Replicate a builder across multiple worlds."""
        self._newton_builder.replicate(builder._newton_builder, world_count, spacing=spacing)

    def approximate_meshes(self, method: str = "bounding_box", **kwargs: Any) -> None:
        """Approximate imported meshes using Newton's importer helper."""
        self._newton_builder.approximate_meshes(method, **kwargs)

    # ------------------------------------------------------------------
    # Joint initial state access
    # ------------------------------------------------------------------

    @property
    def joint_q(self) -> list:
        """Joint position coordinate list (mutable, set before finalize).

        Use to set initial joint angles/positions before building:
            >>> builder.joint_q[joint_idx] = angle
        """
        return self._newton_builder.joint_q

    @joint_q.setter
    def joint_q(self, value: list[float]) -> None:
        self._newton_builder.joint_q = value

    @property
    def joint_label(self) -> list[str]:
        """Joint labels accumulated before finalize."""
        return self._newton_builder.joint_label

    @property
    def body_key(self) -> list[str]:
        """Body labels accumulated before finalize."""
        return getattr(self._newton_builder, "body_key", self._newton_builder.body_label)

    @property
    def body_count(self) -> int:
        """Number of bodies accumulated before finalize."""
        return self._newton_builder.body_count

    @property
    def joint_dof_count(self) -> int:
        """Current joint DOF count accumulated before finalize."""
        return self._newton_builder.joint_dof_count

    @property
    def joint_coord_count(self) -> int:
        """Current joint coordinate count accumulated before finalize."""
        return self._newton_builder.joint_coord_count

    @property
    def joint_target_ke(self) -> list[float]:
        """Joint target stiffness list accumulated before finalize."""
        return self._newton_builder.joint_target_ke

    @property
    def joint_target_kd(self) -> list[float]:
        """Joint target damping list accumulated before finalize."""
        return self._newton_builder.joint_target_kd

    @property
    def joint_target_pos(self) -> list[float]:
        """Joint target position list accumulated before finalize."""
        return self._newton_builder.joint_target_pos

    @property
    def joint_effort_limit(self) -> list[float]:
        """Joint effort limit list accumulated before finalize."""
        return self._newton_builder.joint_effort_limit

    @property
    def joint_armature(self) -> list[float]:
        """Joint armature list accumulated before finalize."""
        return self._newton_builder.joint_armature

    @property
    def shape_count(self) -> int:
        """Number of shapes accumulated before finalize."""
        return self._newton_builder.shape_count

    @property
    def shape_key(self) -> list[str]:
        """Shape labels accumulated before finalize."""
        return getattr(self._newton_builder, "shape_key", self._newton_builder.shape_label)

    @property
    def shape_body(self) -> list[int]:
        """Shape-to-body mapping accumulated before finalize."""
        return self._newton_builder.shape_body

    @property
    def shape_flags(self) -> list[int]:
        """Shape flags accumulated before finalize."""
        return self._newton_builder.shape_flags

    @property
    def shape_type(self) -> list[Any]:
        """Shape types accumulated before finalize."""
        return self._newton_builder.shape_type

    @property
    def shape_scale(self) -> list[Any]:
        """Shape scales accumulated before finalize."""
        return self._newton_builder.shape_scale

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def finalize(
        self,
        device: Any | None = None,
        requires_grad: bool = False,
    ) -> RigidModel:
        """Build and return a :class:`RigidModel`.

        Args:
            device: Warp device to allocate model data on. Defaults to the current Warp device.
            requires_grad: Enable gradient computation on model arrays.

        Returns:
            A :class:`RigidModel` wrapping the finalized Newton model.
        """
        from .model import RigidModel

        if device is None:
            device = wp.get_device()
        newton_model = self._newton_builder.finalize(device=device, requires_grad=requires_grad)
        return RigidModel(newton_model)
