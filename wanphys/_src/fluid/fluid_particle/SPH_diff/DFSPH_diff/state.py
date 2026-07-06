# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH_diff state - time-varying data for differentiable WCSPH baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp



from ..kernels import MaterialMarks, RigidBodies
from ..model import SPHDiffModel
from ..rigid_fluid_coupling import compute_rigid_cm_mass_kernel, compute_rigid_inertia_kernel, finalize_rigid_cm_kernel, finalize_rigid_inertia_kernel
from ..state import SPHDiffState


if TYPE_CHECKING:
    from newton import State as NewtonState



class DFSPHDiffState(SPHDiffState):
    """
    DFSPH专用State，增加DFSPH所需的中间变量和多步缓存。
    """

    def __init__(self, newton_state: "NewtonState", model: SPHDiffModel, rbs=None,material_marks=None, object_id=None, particle_x0=None, sim_steps=100):
        super().__init__(newton_state, model, rbs, material_marks=material_marks, object_id=object_id, particle_x0=particle_x0)
        n = newton_state.particle_q.shape[0]
        device = model._device
        requires_grad = bool(model.requires_grad)

        self.dfsph_factor = wp.zeros(n, dtype=float, device=device, requires_grad=False)
        self.density_error_accum = wp.zeros(1, dtype=float, device=device, requires_grad=False)
        self.sim_steps = sim_steps

        self.density_change = wp.zeros(n, dtype=float, device=device, requires_grad=True)
        self.density_adv = wp.zeros(n, dtype=float, device=device, requires_grad=True)

    @property
    def alpha(self):
        return self.dfsph_factor

    @alpha.setter
    def alpha(self, value):
        self.dfsph_factor = value


    def clear_grad(self):
        super().clear_grad()
        if hasattr(self.dfsph_factor, 'grad') and self.dfsph_factor.grad:
            self.dfsph_factor.grad.zero_()
        if hasattr(self.density_change, 'grad') and self.density_change.grad:
            self.density_change.grad.zero_()
        if hasattr(self.density_adv, 'grad') and self.density_adv.grad:
            self.density_adv.grad.zero_()
        if hasattr(self, 'density_error_accum') and hasattr(self.density_error_accum, 'zero_'):
            self.density_error_accum.zero_()
