from __future__ import annotations

"""WanPhys MuJoCo runtime update flags."""

from enum import IntEnum


class SolverNotifyFlags(IntEnum):
    JOINT_PROPERTIES = 1 << 0
    JOINT_DOF_PROPERTIES = 1 << 1
    BODY_PROPERTIES = 1 << 2
    BODY_INERTIAL_PROPERTIES = 1 << 3
    SHAPE_PROPERTIES = 1 << 4
    MODEL_PROPERTIES = 1 << 5
    EQUALITY_CONSTRAINT_PROPERTIES = 1 << 6

    ALL = (
        JOINT_PROPERTIES
        | JOINT_DOF_PROPERTIES
        | BODY_PROPERTIES
        | BODY_INERTIAL_PROPERTIES
        | SHAPE_PROPERTIES
        | MODEL_PROPERTIES
        | EQUALITY_CONSTRAINT_PROPERTIES
    )
