# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""WanPhys-native particle fluid builder — no Newton dependency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class ParticleFluidData:
    """Result of :meth:`ParticleFluidBuilder.finalize`."""

    particle_q: wp.array
    """Particle positions, shape ``(N,)``, dtype ``wp.vec3``."""

    particle_qd: wp.array
    """Particle velocities, shape ``(N,)``, dtype ``wp.vec3``."""

    particle_count: int
    """Number of particles."""

    particle_mass: wp.array
    """Particle masses, shape ``(N,)``, dtype ``float32``."""

    particle_flags: wp.array
    """Particle flags, shape ``(N,)``, dtype ``int32``."""

    particle_radius: wp.array
    """Particle radii, shape ``(N,)``, dtype ``float32``."""


# ParticleFlags.ACTIVE == 1 in Newton; use literal to avoid the import.
_PARTICLE_FLAG_ACTIVE = 1


class ParticleFluidBuilder:
    """Accumulates particle data and converts to GPU arrays.

    Drop-in replacement for ``newton.ModelBuilder`` when the only goal is to
    emit particles (pos, vel, mass, radius).  Call :meth:`add_particle` for
    each particle, then :meth:`finalize` to get a :class:`ParticleFluidData`.
    """

    def __init__(self):
        self.particle_q: list = []
        self.particle_qd: list = []
        self.particle_mass: list[float] = []
        self.particle_radius: list[float] = []
        self.particle_flags: list[int] = []

    @property
    def particle_count(self) -> int:
        """Number of particles added so far."""
        return len(self.particle_q)

    def add_particle(
        self,
        pos: tuple[float, float, float],
        vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mass: float = 1.0,
        radius: float = 0.01,
        flags: int = _PARTICLE_FLAG_ACTIVE,
    ) -> int:
        """Append one particle. Returns the particle index."""
        self.particle_q.append((float(pos[0]), float(pos[1]), float(pos[2])))
        self.particle_qd.append((float(vel[0]), float(vel[1]), float(vel[2])))
        self.particle_mass.append(float(mass))
        self.particle_radius.append(float(radius))
        self.particle_flags.append(int(flags))
        return len(self.particle_q) - 1

    def finalize(self, device: str | None = None) -> ParticleFluidData:
        """Convert accumulated lists to GPU arrays.

        Args:
            device: Warp device (string or ``wp.Device``). Defaults to the
                current default device.

        Returns:
            A :class:`ParticleFluidData` with all arrays on *device*.
        """
        n = self.particle_count
        if device is None:
            device = wp.get_device()

        q_np = np.array(self.particle_q, dtype=np.float32).reshape(n, 3)
        qd_np = np.array(self.particle_qd, dtype=np.float32).reshape(n, 3)
        mass_np = np.array(self.particle_mass, dtype=np.float32)
        radius_np = np.array(self.particle_radius, dtype=np.float32)
        flags_np = np.array(self.particle_flags, dtype=np.int32)

        return ParticleFluidData(
            particle_q=wp.array(q_np, dtype=wp.vec3, device=device),
            particle_qd=wp.array(qd_np, dtype=wp.vec3, device=device),
            particle_count=n,
            particle_mass=wp.array(mass_np, dtype=wp.float32, device=device),
            particle_flags=wp.array(flags_np, dtype=wp.int32, device=device),
            particle_radius=wp.array(radius_np, dtype=wp.float32, device=device),
        )
