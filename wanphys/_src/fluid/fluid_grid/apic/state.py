from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..base import FluidGridStateBase

if TYPE_CHECKING:
    from .model import FluidGridApicModel


class FluidGridApicState(FluidGridStateBase):
    def __init__(self, model: FluidGridApicModel, requires_grad: bool = False) -> None:
        super().__init__(model, requires_grad)

        nx, ny, nz = self.res

        # Liquid occupancy mask: 0 = air, 1 = fluid, 2 = solid.
        self.cell_type = wp.zeros((nx, ny, nz), dtype=wp.int32, device=self.device)

        # APIC particles
        self.particle_q = wp.zeros(
            model.particle_count,
            dtype=wp.vec3,
            device=self.device,
            requires_grad=requires_grad,
        )
        self.particle_v = wp.zeros(
            model.particle_count,
            dtype=wp.vec3,
            device=self.device,
            requires_grad=requires_grad,
        )
        self.particle_c = wp.zeros(
            model.particle_count,
            dtype=wp.mat33,
            device=self.device,
            requires_grad=requires_grad,
        )

        # Embedded solid face velocities. These stay zero for static solids and
        # let APIC reuse the liquid kernels that now expect moving-boundary data.
        self.vel_solid_u = wp.zeros((nx + 1, ny, nz), dtype=float, device=self.device)
        self.vel_solid_v = wp.zeros((nx, ny + 1, nz), dtype=float, device=self.device)
        self.vel_solid_w = wp.zeros((nx, ny, nz + 1), dtype=float, device=self.device)

