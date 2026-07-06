from __future__ import annotations

from typing import TYPE_CHECKING, Optional
import warp as wp
if TYPE_CHECKING:
    from .model import FluidGridLiquidModel

from ..base import FluidGridStateBase


class FluidGridLiquidState(FluidGridStateBase):
    def __init__(self, model: FluidGridLiquidModel, requires_grad: bool = False) -> None:
        super().__init__(model, requires_grad)

        nx, ny, nz = self.res

        # Liquid occupancy mask: 0 = air, 1 = fluid, 2 = solid.
        self.cell_type = wp.zeros((nx, ny, nz), dtype=wp.int32, device=self.device)

        # particles for tracing fluid surface
        self.particle_q = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)
        self.particle_v = wp.zeros(model.particle_count, dtype=wp.vec3, device=self.device, requires_grad=requires_grad)

        # Rigid body surface velocity embedded at MAC faces.
        # Set by GridLiquidRigidCoupling before each fluid step so the
        # pressure solve enforces the correct moving-boundary condition.
        self.vel_solid_u = wp.zeros((nx + 1, ny, nz), dtype=float, device=self.device)
        self.vel_solid_v = wp.zeros((nx, ny + 1, nz), dtype=float, device=self.device)
        self.vel_solid_w = wp.zeros((nx, ny, nz + 1), dtype=float, device=self.device)

    def clear(self) -> None:
        super().clear()
        self.cell_type.zero_()
        self.particle_q.zero_()
        self.particle_v.zero_()
        self.vel_solid_u.zero_()
        self.vel_solid_v.zero_()
        self.vel_solid_w.zero_()

    def clone(self) -> FluidGridLiquidState:
        new_state = FluidGridLiquidState(self.model, requires_grad=self.particle_q.requires_grad)
        new_state.cell_type = self.cell_type.clone()
        new_state.particle_q = self.particle_q.clone()
        new_state.particle_v = self.particle_v.clone()
        new_state.vel_solid_u = self.vel_solid_u.clone()
        new_state.vel_solid_v = self.vel_solid_v.clone()
        new_state.vel_solid_w = self.vel_solid_w.clone()
        wp.copy(new_state.solid_body_id, self.solid_body_id)
        return new_state
    