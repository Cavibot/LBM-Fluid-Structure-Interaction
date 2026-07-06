from __future__ import annotations

"""WanPhys-owned MuJoCo custom attribute registration helpers."""

import warp as wp

import newton
from newton._src.core.types import vec5


def _resolve_builder(builder):
    """Accept either a Newton builder or a WanPhys builder wrapper."""
    return getattr(builder, "_backend", builder)


def register_mujoco_custom_attributes(builder) -> None:
    """Register MuJoCo custom attributes required by WanPhys' compiler."""
    builder = _resolve_builder(builder)
    custom_attribute = newton.ModelBuilder.CustomAttribute
    custom_frequency = newton.ModelBuilder.CustomFrequency
    pair_frequency = "mujoco:pair"

    # Custom frequencies must be registered before attributes that reference them.
    builder.add_custom_frequency(custom_frequency(name="pair", namespace="mujoco"))

    builder.add_custom_attribute(
        custom_attribute(
            name="condim",
            frequency=newton.Model.AttributeFrequency.SHAPE,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.int32,
            default=3,
            namespace="mujoco",
            usd_attribute_name="mjc:condim",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="geom_priority",
            frequency=newton.Model.AttributeFrequency.SHAPE,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.int32,
            default=0,
            namespace="mujoco",
            usd_attribute_name="mjc:priority",
            mjcf_attribute_name="priority",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="geom_solimp",
            frequency=newton.Model.AttributeFrequency.SHAPE,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=vec5,
            default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
            namespace="mujoco",
            usd_attribute_name="mjc:solimp",
            mjcf_attribute_name="solimp",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="geom_solmix",
            frequency=newton.Model.AttributeFrequency.SHAPE,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.float32,
            default=1.0,
            namespace="mujoco",
            usd_attribute_name="mjc:solmix",
            mjcf_attribute_name="solmix",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="geom_gap",
            frequency=newton.Model.AttributeFrequency.SHAPE,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.float32,
            default=0.0,
            namespace="mujoco",
            usd_attribute_name="mjc:gap",
            mjcf_attribute_name="gap",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="limit_margin",
            frequency=newton.Model.AttributeFrequency.JOINT_DOF,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.float32,
            default=0.0,
            namespace="mujoco",
            usd_attribute_name="mjc:margin",
            mjcf_attribute_name="margin",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="solimplimit",
            frequency=newton.Model.AttributeFrequency.JOINT_DOF,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=vec5,
            default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
            namespace="mujoco",
            usd_attribute_name="mjc:solimplimit",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="solreffriction",
            frequency=newton.Model.AttributeFrequency.JOINT_DOF,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.vec2,
            default=wp.vec2(0.02, 1.0),
            namespace="mujoco",
            usd_attribute_name="mjc:solreffriction",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="solimpfriction",
            frequency=newton.Model.AttributeFrequency.JOINT_DOF,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=vec5,
            default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
            namespace="mujoco",
            usd_attribute_name="mjc:solimpfriction",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="gravcomp",
            frequency=newton.Model.AttributeFrequency.BODY,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.float32,
            default=0.0,
            namespace="mujoco",
            usd_attribute_name="mjc:gravcomp",
            mjcf_attribute_name="gravcomp",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="dof_passive_stiffness",
            frequency=newton.Model.AttributeFrequency.JOINT_DOF,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.float32,
            default=0.0,
            namespace="mujoco",
            usd_attribute_name="mjc:stiffness",
            mjcf_attribute_name="stiffness",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="dof_passive_damping",
            frequency=newton.Model.AttributeFrequency.JOINT_DOF,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.float32,
            default=0.0,
            namespace="mujoco",
            usd_attribute_name="mjc:damping",
            mjcf_attribute_name="damping",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="jnt_actgravcomp",
            frequency=newton.Model.AttributeFrequency.JOINT_DOF,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.bool,
            default=False,
            namespace="mujoco",
            usd_attribute_name="mjc:actuatorgravcomp",
            mjcf_attribute_name="actuatorgravcomp",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="eq_solref",
            frequency=newton.Model.AttributeFrequency.EQUALITY_CONSTRAINT,
            assignment=newton.Model.AttributeAssignment.MODEL,
            dtype=wp.vec2,
            default=wp.vec2(0.02, 1.0),
            namespace="mujoco",
            usd_attribute_name="mjc:solref",
            mjcf_attribute_name="solref",
        )
    )

    builder.add_custom_attribute(
        custom_attribute(
            name="pair_world",
            frequency=pair_frequency,
            dtype=wp.int32,
            default=0,
            namespace="mujoco",
            references="world",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="pair_geom1",
            frequency=pair_frequency,
            dtype=wp.int32,
            default=-1,
            namespace="mujoco",
            references="shape",
            mjcf_attribute_name="geom1",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="pair_geom2",
            frequency=pair_frequency,
            dtype=wp.int32,
            default=-1,
            namespace="mujoco",
            references="shape",
            mjcf_attribute_name="geom2",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="pair_condim",
            frequency=pair_frequency,
            dtype=wp.int32,
            default=3,
            namespace="mujoco",
            mjcf_attribute_name="condim",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="pair_solref",
            frequency=pair_frequency,
            dtype=wp.vec2,
            default=wp.vec2(0.02, 1.0),
            namespace="mujoco",
            mjcf_attribute_name="solref",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="pair_solreffriction",
            frequency=pair_frequency,
            dtype=wp.vec2,
            default=wp.vec2(0.02, 1.0),
            namespace="mujoco",
            mjcf_attribute_name="solreffriction",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="pair_solimp",
            frequency=pair_frequency,
            dtype=vec5,
            default=vec5(0.9, 0.95, 0.001, 0.5, 2.0),
            namespace="mujoco",
            mjcf_attribute_name="solimp",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="pair_margin",
            frequency=pair_frequency,
            dtype=wp.float32,
            default=0.0,
            namespace="mujoco",
            mjcf_attribute_name="margin",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="pair_gap",
            frequency=pair_frequency,
            dtype=wp.float32,
            default=0.0,
            namespace="mujoco",
            mjcf_attribute_name="gap",
        )
    )
    builder.add_custom_attribute(
        custom_attribute(
            name="pair_friction",
            frequency=pair_frequency,
            dtype=vec5,
            default=vec5(1.0, 1.0, 0.005, 0.0001, 0.0001),
            namespace="mujoco",
            mjcf_attribute_name="friction",
        )
    )
