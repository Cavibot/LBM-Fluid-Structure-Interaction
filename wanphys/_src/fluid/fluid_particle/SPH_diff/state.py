# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH_diff state - time-varying data for differentiable WCSPH baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..dfsph.state import DFSPHState


from .kernels import MaterialMarks, MaterialType, RigidBodies
from .model import SPHDiffModel
from .rigid_fluid_coupling import compute_rigid_cm_mass_kernel, compute_rigid_inertia_kernel, finalize_rigid_cm_kernel, finalize_rigid_inertia_kernel

import numpy as np

if TYPE_CHECKING:
    from newton import State as NewtonState


class SPHDiffState(DFSPHState):
    """Time-varying state for SPH_diff.

    Reuses the DFSPH state layout for the shared fluid buffers and keeps
    compatibility with wrapped ``newton.State`` access.
    """
    def __init__(self, newton_state: NewtonState, model: SPHDiffModel, rbs=None, material_marks=None, object_id=None, particle_x0=None):
        # Keep a reference to the underlying newton.State
        self._state = newton_state
        self._model = model

        n = newton_state.particle_q.shape[0]
        device = model._device
        super().__init__(n, device, model)

        # Reuse parent particle-state attributes, but bind them to the wrapped newton.State
        # so downstream Newton-based rendering/collision paths remain unchanged.
        self._particle_q = newton_state.particle_q
        self._particle_qd = newton_state.particle_qd
        self._particle_f = newton_state.particle_f

        requires_grad = bool(model.requires_grad)
        if requires_grad:
            self._rho = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
            self._pressure = wp.zeros(n, dtype=wp.float32, device=device, requires_grad=True)
        self._a = wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self._viscous_forces = wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self._pressure_forces = wp.zeros(n, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self._debug_val = wp.zeros(n, dtype=wp.float32, device=device)
        particle_radius = 0.25 * model.h
        particle_diameter = 2.0 * particle_radius
        self.m_V0 = 0.8 * particle_diameter ** 3
        self._m_V = wp.full(n, self.m_V0, dtype=wp.float32, device=device)

        legacy_material_marks = material_marks
        self._particle_x0 = particle_x0

        # Keep rigid/boundary particles in dedicated arrays (same names as dfsph state).
        self._boundary_indices = wp.empty(shape=(0,), dtype=wp.int32, device=device)
        self._boundary_object_id = wp.empty(shape=(0,), dtype=wp.int32, device=device)
        boundary_body_id_np = np.zeros(n, dtype=np.int32)
        particle_flags_np = np.ones(n, dtype=np.int32)
        boundary_idx: np.ndarray | None = None
        if object_id is not None:
            obj_np = object_id.numpy() if hasattr(object_id, "numpy") else np.asarray(object_id)
            boundary_idx = np.where(obj_np > 0)[0].astype(np.int32)
            if legacy_material_marks is None:
                legacy_material_marks = MaterialMarks()
                legacy_material_marks.material = wp.array(np.where(obj_np > 0, int(MaterialType.SOLID), int(MaterialType.FLUID)), dtype=int, device=device)
                legacy_material_marks.is_dynamic = wp.array(np.ones(n, dtype=np.int32), dtype=int, device=device)
        elif material_marks is not None and hasattr(material_marks, "material"):
            mat_np = material_marks.material.numpy()
            boundary_idx = np.where(mat_np == int(MaterialType.SOLID))[0].astype(np.int32)

        if boundary_idx is not None and boundary_idx.size > 0:
            self._boundary_indices = wp.array(boundary_idx, dtype=wp.int32, device=device)
            q_np = newton_state.particle_q.numpy()[boundary_idx]
            self._boundary_q = wp.array(q_np, dtype=wp.vec3, device=device)
            psi_val = float(max(self.m_V0 * max(model.rest_density, 1.0) * 3.0, 1.0e-8))
            self._boundary_psi = wp.full(boundary_idx.size, psi_val, dtype=wp.float32, device=device)
            if object_id is not None:
                obj_np = object_id.numpy() if hasattr(object_id, "numpy") else np.asarray(object_id)
                self._boundary_object_id = wp.array(obj_np[boundary_idx].astype(np.int32), dtype=wp.int32, device=device)
            else:
                self._boundary_object_id = wp.full(boundary_idx.size, 1, dtype=wp.int32, device=device)
            boundary_body_id_np[boundary_idx] = 1
            particle_flags_np[boundary_idx] = 0

        self._boundary_body_id = wp.array(boundary_body_id_np, dtype=wp.int32, device=device)
        self._particle_flags = wp.array(particle_flags_np, dtype=wp.int32, device=device)
        self._material_marks = legacy_material_marks

        self._num_objects = 2
        self.particle_max_num = self._state.particle_q.shape[0]

        rbs = RigidBodies()
        rbs.rigid_rest_cm = wp.zeros(self.num_objects, dtype=wp.vec3)
        rbs.rigid_x = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        rbs.rigid_v0 = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        rbs.rigid_v = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        rbs.rigid_force = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        rbs.rigid_torque = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        rbs.rigid_omega = wp.zeros(self.num_objects, dtype=wp.vec3, requires_grad=True)
        rbs.rigid_omega0 = wp.zeros(self.num_objects, dtype=wp.vec3)
        rbs.rigid_mass = wp.zeros(self.num_objects, dtype=wp.float32)
        rbs.rigid_inv_mass = wp.zeros(self.num_objects, dtype=wp.float32)
        rbs.rigid_inertia = wp.zeros(self.num_objects, dtype=wp.mat33)
        rbs.rigid_inertia0 = wp.zeros(self.num_objects, dtype=wp.mat33)
        rbs.rigid_inv_inertia = wp.zeros(self.num_objects, dtype=wp.mat33)
        quat_data = np.zeros((self.num_objects, 4), dtype=np.float32)
        quat_data[:, 3] = 1.0
        rbs.rigid_quaternion = wp.array(quat_data, dtype=wp.quat)

        self._rbs = rbs

    @classmethod
    def from_arrays(
        cls,
        q: wp.array,
        qd: wp.array,
        model: SPHDiffModel,
    ) -> "SPHDiffState":
        raise NotImplementedError("SPHDiffState must be constructed from a Newton state wrapper.")

    def initialize_rigid_bodies(self):
        """Initialize rigid body properties and inertia kernels."""

        # 更新 _num_objects
        self._num_objects = 2
        # Re-init accumulators so repeated calls (e.g., reset hooks) are idempotent.
        self._rbs.rigid_mass.zero_()
        self._rbs.rigid_rest_cm.zero_()
        self._rbs.rigid_force.zero_()
        self._rbs.rigid_torque.zero_()
        self._rbs.rigid_inertia.zero_()
        self._rbs.rigid_inv_inertia.zero_()
        self._rbs.rigid_inv_mass.zero_()
        wp.launch(
            kernel=compute_rigid_cm_mass_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.object_id,
                self._state.particle_q,
                self.m_V,
                self.rho,
                self._rbs.rigid_mass,
                self._rbs.rigid_rest_cm,
                self.m_V0
            ]
        )
        wp.launch(
            kernel=finalize_rigid_cm_kernel,
            dim=self._num_objects,
            inputs=[
                self._rbs.rigid_mass,
                self._rbs.rigid_rest_cm,
                self._num_objects
            ]
        )
        wp.copy(self._rbs.rigid_x, self._rbs.rigid_rest_cm)
        rigid_inertia_accum_flat = wp.zeros(self._num_objects * 9, dtype=float)
        wp.launch(
            kernel=compute_rigid_inertia_kernel,
            dim=self.particle_max_num,
            inputs=[
                self.object_id,
                self._state.particle_q,
                self.m_V,
                self.rho,
                self._rbs.rigid_rest_cm,
                rigid_inertia_accum_flat,
                self.m_V0
            ]
        )
        wp.launch(
            kernel=finalize_rigid_inertia_kernel,
            dim=self._num_objects,
            inputs=[
                self._rbs.rigid_mass,
                rigid_inertia_accum_flat,
                self._rbs.rigid_inertia,
                self._rbs.rigid_inv_inertia,
                self._rbs.rigid_inv_mass,
                self._rbs.rigid_inertia0,
                self._num_objects
            ]
        )
        wp.copy(self._rbs.rigid_inertia0, self._rbs.rigid_inertia)

    @property
    def m_V(self) -> wp.array:
        return self._m_V

    @property
    def rho(self) -> wp.array:
        return self._rho

    @property
    def pressure(self) -> wp.array:
        return self._pressure

    @property
    def a(self) -> wp.array:
        return self._a

    @property
    def viscous_forces(self) -> wp.array:
        return self._viscous_forces

    @property
    def pressure_forces(self) -> wp.array:
        return self._pressure_forces

    @property
    def debug_val(self) -> wp.array:
        return self._debug_val

    @property
    def material_marks(self) -> MaterialMarks | None:
        return self._material_marks

    @property
    def particle_flags(self) -> wp.array:
        return self._particle_flags

    @property
    def boundary_body_id(self) -> wp.array:
        return self._boundary_body_id

    @boundary_body_id.setter
    def boundary_body_id(self, value: wp.array):
        self._boundary_body_id = value

    @property
    def object_id(self) -> wp.array | None:
        return self._boundary_body_id

    @property
    def boundary_q(self) -> wp.array:
        return self._boundary_q

    @boundary_q.setter
    def boundary_q(self, value: wp.array):
        self._boundary_q = value

    @property
    def boundary_psi(self) -> wp.array:
        return self._boundary_psi

    @boundary_psi.setter
    def boundary_psi(self, value: wp.array):
        self._boundary_psi = value

    @property
    def boundary_indices(self) -> wp.array:
        return self._boundary_indices

    @boundary_indices.setter
    def boundary_indices(self, value: wp.array):
        self._boundary_indices = value

    @property
    def boundary_object_id(self) -> wp.array:
        return self._boundary_object_id

    @boundary_object_id.setter
    def boundary_object_id(self, value: wp.array):
        self._boundary_object_id = value

    @property
    def rbs(self) -> RigidBodies | None:
        return self._rbs

    @property
    def particle_x0(self) -> wp.array | None:
        return self._particle_x0

    @property
    def num_objects(self) -> int:
        return self._num_objects

    def zero_scratch_buffers(self) -> None:  # TODO: verify whether this is necessary
        super().zero_scratch_buffers()
        self._rho.zero_()
        self._pressure.zero_()
        self._a.zero_()
        self._viscous_forces.zero_()
        self._pressure_forces.zero_()
        self._debug_val.zero_()

    def clear_grad(self) -> None:
        """Clear gradients for all differentiable state buffers."""
        for arr in (
            self._rho,
            self._pressure,
            self._a,
            self._viscous_forces,
            self._pressure_forces,
            self._m_V,
            self._rbs.rigid_x,
            self._rbs.rigid_v0,
            self._rbs.rigid_v,
            self._rbs.rigid_force,
            self._rbs.rigid_torque,
            self._rbs.rigid_omega,
        ):
            grad = getattr(arr, "grad", None)
            if grad is not None:
                grad.zero_()

    
    def print_rigid_info(self) -> None:
        if self.num_objects > 0:
            masses = self.rbs.rigid_mass.numpy()
            pos = self.rbs.rigid_x.numpy()
            vel = self.rbs.rigid_v.numpy()
            omega = self.rbs.rigid_omega.numpy()
            quat = self.rbs.rigid_quaternion.numpy()
            force = self.rbs.rigid_force.numpy()
            torque = self.rbs.rigid_torque.numpy()
            rest_cm = self.rbs.rigid_rest_cm.numpy()

            print(f"[rbs] num={self.num_objects}")
            for i in range(0, self.num_objects):
                q = quat[i]
                w = float(np.clip(q[3], -1.0, 1.0))
                rot_angle_deg = float(np.degrees(2.0 * np.arccos(w)))
                print(
                    f" id={i} mass={masses[i]:.6f} pos={pos[i]} rest_cm={rest_cm[i]}\n"
                    f" vel={vel[i]} omega={omega[i]} force={force[i]} torque={torque[i]}\n"
                    f" quat={q} rot_angle_deg={rot_angle_deg:.3f}"
                )

    def __getattr__(self, name: str):
        """Proxy unknown attribute access to the underlying newton.State."""
        return getattr(self._state, name)

    @property
    def newton_state(self):
        """Legacy name expected by examples: return underlying newton.State."""
        return self._state