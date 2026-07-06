# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""SPHDiff domain - Weakly Compressible SPH domain implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton import Model as NewtonModel

from wanphys.core import Domain
from ..kernels import MaterialMarks, MaterialType
from ..domain import SPHDiffDomain
from ..model import SPHDiffModel
import warp as wp
import numpy as np

from .state import DFSPHDiffState
from .solver import DFSPHDiffSolver

class DFSPHDiffDomain(SPHDiffDomain):
    """
    DFSPH专用Domain，重载create_state和solver属性，分别返回DFSPHDiffState和DFSPH专用solver。
    """

    def __init__(
        self,
        newton_model: NewtonModel,
        sphdiff_model: SPHDiffModel | None = None,
        solver: DFSPHDiffSolver | None = None,
        target_time: float | None = None,
        target_timesteps: int | None = None,
        dt: float | None = None,
        task: object | None = None,
    ):
        # Use DFSPH-specific solver by default
        if sphdiff_model is None:
            sphdiff_model = SPHDiffModel(device=newton_model.device)

        if solver is None:
            # Pass the raw newton.Model directly to the solver (no deprecated shim).
            solver = DFSPHDiffSolver(newton_model, sphdiff_model)

        super().__init__(
            newton_model,
            sphdiff_model,
            solver,
            target_time=target_time,
            target_timesteps=target_timesteps,
            dt=dt,
            task=task,
        )

    def create_state(self):
        """返回DFSPHDiffState实例。"""
        newton_state = self._model_adapter.state()

        particle_x0 = wp.clone(newton_state.particle_q) if hasattr(newton_state, "particle_q") else None

        particle_count = newton_state.particle_count
        num_rigid_bodies = newton_state.body_count

        material_marks = MaterialMarks()
        particle_mass = self._model_adapter.particle_mass 
        mass_np = particle_mass.numpy() if hasattr(particle_mass, "numpy") else particle_mass
        material_arr = np.where(np.abs(mass_np) != 0.0, MaterialType.FLUID, MaterialType.SOLID)
        # particles are often mass=0, so mass-based classification is incorrect.
        is_dynamic_arr = np.where(material_arr == MaterialType.SOLID, 1, np.where(mass_np != 0.0, 1, 0)).astype(np.int32)
        material_marks.material = wp.array(material_arr, dtype=int, device=newton_state.particle_q.device)
        material_marks.is_dynamic = wp.array(is_dynamic_arr, dtype=int, device=newton_state.particle_q.device)
        object_id_np = np.zeros(particle_count, dtype=np.int32)
        object_id_np[material_marks.material.numpy() == int(MaterialType.SOLID)] = 1
        object_id = wp.array(object_id_np, dtype=wp.int32, device=newton_state.particle_q.device)
        state = DFSPHDiffState(newton_state, self._sphdiff_model, rbs=None,material_marks=material_marks,object_id=object_id,particle_x0=particle_x0)
        n = state._state.particle_q.shape[0]
        rho_np = np.zeros(n, dtype=np.float32)
        boundary_idx_np = state.boundary_indices.numpy() if state.boundary_indices is not None else np.array([], dtype=np.int32)
        if boundary_idx_np.size > 0:
            rho_np[boundary_idx_np] = float(state._model.rigid_density)
        state._rho = wp.array(rho_np, dtype=wp.float32, device=state._model._device, requires_grad=True)

        self._state_in = state
        self._state_out = DFSPHDiffState(
            newton_state,
            self._sphdiff_model,
            rbs=None,
            material_marks=material_marks,
            object_id=object_id,
            particle_x0=particle_x0,
        )
        self._state_out._rho = wp.array(rho_np, dtype=wp.float32, device=state._model._device, requires_grad=True)

        return self._state_in

    @property
    def solver(self):
        """返回DFSPH专用solver（如有），否则返回父类solver。"""
        return self._solver

    @property
    def task(self):
        """返回域内部持有的优化任务。"""
        return self._task

    # Runtime trigger/reset logic is inherited from SPHDiffDomain.
   