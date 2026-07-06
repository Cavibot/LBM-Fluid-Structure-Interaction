# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Composite simulation abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod


class CompositeSimulation(ABC):
    """Abstract base class for multi-domain physics simulations.

    Subclass this to implement a named, reusable coupling algorithm.
    Each subclass owns its domains directly (as attributes) and implements
    the full physics step in ``step()``.  The base class tracks simulation
    time via ``_time`` and exposes it through the ``time`` property.

    Example (custom coupling subclass)::

        class MyFluidRigidCoupling(CompositeSimulation):
            def __init__(self, rigid, fluid):
                super().__init__()
                self.rigid = rigid
                self.fluid = fluid

            def step(self, dt):
                self.fluid.step(dt)
                # ... coupling logic ...
                self.rigid.step(dt)
                self._time += dt

            def reset(self):
                super().reset()
                self.rigid.create_state()
                self.fluid.create_state()
    """

    def __init__(self) -> None:
        self._time: float = 0.0

    @property
    def time(self) -> float:
        """Current simulation time in seconds."""
        return self._time

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance the simulation by one timestep.

        Args:
            dt: Timestep in seconds.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._time = 0.0
