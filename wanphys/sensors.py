# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Sensor interface for WanPhys simulations.

This module provides sensor abstractions for measuring simulation quantities
like contact forces between bodies or shapes.

Example:
    >>> from wanphys.sensors import Sensor, SensorContact
    >>> from wanphys.rigid import RigidDomain, RigidModelBuilder
    >>> builder = RigidModelBuilder()
    >>> model = builder.finalize()
    >>> domain = RigidDomain(model)
    >>> # Load sensor from YAML config
    >>> sensor = Sensor(domain, "path/to/sensor_config.yaml")
    >>> # Or create directly
    >>> contact_sensor = SensorContact(domain, sensing_obj_bodies="gripper*")
"""

from wanphys._src.sensors import Sensor, SensorContact, SensorIMU, SensorFrameTransform

__all__ = [
    "Sensor",
    "SensorContact",
    "SensorIMU",
    "SensorFrameTransform"
]
