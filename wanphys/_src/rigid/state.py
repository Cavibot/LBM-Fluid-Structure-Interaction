# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid body state owned by WanPhys.

``RigidState`` stores simulation arrays directly in WanPhys and provides a
lazy ``newton.State`` view only for temporary solver / viewer compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from newton import Model, State


@wp.kernel
def _apply_body_forces_kernel(
    body_f: wp.array(dtype=wp.spatial_vector),
    indices: wp.array(dtype=int),
    forces: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    wp.atomic_add(body_f, indices[tid], forces[tid])


class RigidState:
    """Rigid body simulation state with WanPhys-owned arrays.

    Attributes:
        body_q: Rigid body transforms, shape (body_count,), dtype transform.
        body_qd: Rigid body spatial velocities, shape (body_count,), dtype spatial_vector.
        body_f: Rigid body spatial forces, shape (body_count,), dtype spatial_vector.
        joint_q: Joint position coordinates, shape (joint_coord_count,), dtype float.
        joint_qd: Joint velocity coordinates, shape (joint_dof_count,), dtype float.
    """

    def __init__(
        self,
        *,
        body_q: wp.array | None,
        body_qd: wp.array | None,
        body_f: wp.array | None,
        joint_q: wp.array | None,
        joint_qd: wp.array | None,
        particle_q: wp.array | None = None,
        particle_qd: wp.array | None = None,
        particle_f: wp.array | None = None,
        body_qdd: wp.array | None = None,
        body_parent_f: wp.array | None = None,
        mujoco_qfrc_actuator: wp.array | None = None,
    ) -> None:
        """Create a WanPhys-owned rigid state."""
        self._body_q: wp.array | None = body_q
        self._body_qd: wp.array | None = body_qd
        self._body_f: wp.array | None = body_f
        self._body_qdd: wp.array | None = body_qdd
        self._body_parent_f: wp.array | None = body_parent_f
        self._joint_q: wp.array | None = joint_q
        self._joint_qd: wp.array | None = joint_qd
        self._particle_q: wp.array | None = particle_q
        self._particle_qd: wp.array | None = particle_qd
        self._particle_f: wp.array | None = particle_f
        self._mujoco_qfrc_actuator: wp.array | None = mujoco_qfrc_actuator
        self._newton_view: State | None = None

    @classmethod
    def create_owned(
        cls,
        body_count: int,
        joint_coord_count: int,
        joint_dof_count: int,
        device: str | wp.Device,
        particle_count: int = 0,
        init_from: Any | None = None,
        requested_state_attributes: list[str] | tuple[str, ...] | None = None,
        requires_grad: bool = False,
    ) -> RigidState:
        """Allocate WanPhys-owned arrays.

        Args:
            body_count: Number of rigid bodies.
            joint_coord_count: Number of generalized position coordinates.
            joint_dof_count: Number of generalized velocity coordinates.
            device: Warp device for allocation.
            particle_count: Number of particles.
            init_from: Optional model-like object to copy initial values from.
            requested_state_attributes: Optional extended state attributes
                to allocate, e.g. ``body_qdd``, ``body_parent_f``,
                or ``mujoco:qfrc_actuator``.
            requires_grad: Whether owned arrays should require gradients.
        """
        requested = set(requested_state_attributes or ())

        if body_count > 0:
            body_q = wp.zeros(
                body_count, dtype=wp.transform, device=device, requires_grad=requires_grad
            )
            body_qd = wp.zeros(
                body_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad
            )
            body_f = wp.zeros(
                body_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad
            )
        else:
            body_q = None
            body_qd = None
            body_f = None

        if body_count > 0 and "body_qdd" in requested:
            body_qdd = wp.zeros(
                body_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad
            )
        else:
            body_qdd = None

        if body_count > 0 and "body_parent_f" in requested:
            body_parent_f = wp.zeros(
                body_count, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad
            )
        else:
            body_parent_f = None

        if joint_coord_count > 0:
            joint_q = wp.zeros(
                joint_coord_count, dtype=float, device=device, requires_grad=requires_grad
            )
        else:
            joint_q = None

        if joint_dof_count > 0:
            joint_qd = wp.zeros(
                joint_dof_count, dtype=float, device=device, requires_grad=requires_grad
            )
        else:
            joint_qd = None

        if "mujoco:qfrc_actuator" in requested and joint_qd is not None:
            mujoco_qfrc_actuator = wp.zeros_like(joint_qd, requires_grad=requires_grad)
        else:
            mujoco_qfrc_actuator = None

        if particle_count > 0:
            particle_q = wp.zeros(
                particle_count, dtype=wp.vec3, device=device, requires_grad=requires_grad
            )
            particle_qd = wp.zeros(
                particle_count, dtype=wp.vec3, device=device, requires_grad=requires_grad
            )
            particle_f = wp.zeros(
                particle_count, dtype=wp.vec3, device=device, requires_grad=requires_grad
            )
        else:
            particle_q = None
            particle_qd = None
            particle_f = None

        rs = cls(
            body_q=body_q,
            body_qd=body_qd,
            body_f=body_f,
            body_qdd=body_qdd,
            body_parent_f=body_parent_f,
            joint_q=joint_q,
            joint_qd=joint_qd,
            particle_q=particle_q,
            particle_qd=particle_qd,
            particle_f=particle_f,
            mujoco_qfrc_actuator=mujoco_qfrc_actuator,
        )

        if init_from is not None:
            ref_body_q = getattr(init_from, "body_q", None)
            ref_body_qd = getattr(init_from, "body_qd", None)
            ref_joint_q = getattr(init_from, "joint_q", None)
            ref_joint_qd = getattr(init_from, "joint_qd", None)
            ref_particle_q = getattr(init_from, "particle_q", None)
            ref_particle_qd = getattr(init_from, "particle_qd", None)
            if rs._body_q is not None and ref_body_q is not None:
                wp.copy(rs._body_q, ref_body_q)
            if rs._body_qd is not None and ref_body_qd is not None:
                wp.copy(rs._body_qd, ref_body_qd)
            if rs._joint_q is not None and ref_joint_q is not None:
                wp.copy(rs._joint_q, ref_joint_q)
            if rs._joint_qd is not None and ref_joint_qd is not None:
                wp.copy(rs._joint_qd, ref_joint_qd)
            if rs._particle_q is not None and ref_particle_q is not None:
                wp.copy(rs._particle_q, ref_particle_q)
            if rs._particle_qd is not None and ref_particle_qd is not None:
                wp.copy(rs._particle_qd, ref_particle_qd)

        return rs

    @property
    def body_q(self) -> wp.array | None:
        """Rigid body transforms (7-DOF: position + quaternion)."""
        return self._body_q

    @property
    def body_qd(self) -> wp.array | None:
        """Rigid body spatial velocities."""
        return self._body_qd

    @property
    def body_f(self) -> wp.array | None:
        """Rigid body spatial forces (external wrenches in world frame)."""
        return self._body_f

    @property
    def body_qdd(self) -> wp.array | None:
        """Optional rigid body spatial accelerations."""
        return self._body_qdd

    @property
    def body_parent_f(self) -> wp.array | None:
        """Optional incoming parent interaction forces."""
        return self._body_parent_f

    @property
    def joint_q(self) -> wp.array | None:
        """Joint position coordinates."""
        return self._joint_q

    @property
    def joint_qd(self) -> wp.array | None:
        """Joint velocity coordinates."""
        return self._joint_qd

    @property
    def particle_q(self) -> wp.array | None:
        """Particle positions."""
        return self._particle_q

    @property
    def particle_qd(self) -> wp.array | None:
        """Particle velocities."""
        return self._particle_qd

    @property
    def particle_f(self) -> wp.array | None:
        """Particle forces."""
        return self._particle_f

    @property
    def mujoco_qfrc_actuator(self) -> wp.array | None:
        """Optional MuJoCo actuator generalized forces, one value per joint DOF."""
        return self._mujoco_qfrc_actuator

    def as_newton_state(self) -> State:
        """Return a ``newton.State`` compatible with solvers and viewers.

        The returned state aliases the owned warp arrays (zero copy).
        """
        if self._newton_view is None:
            newton_state = State()
            newton_state.body_q = self._body_q
            newton_state.body_qd = self._body_qd
            newton_state.body_f = self._body_f
            newton_state.body_qdd = self._body_qdd
            newton_state.body_parent_f = self._body_parent_f
            newton_state.joint_q = self._joint_q
            newton_state.joint_qd = self._joint_qd
            newton_state.particle_q = self._particle_q
            newton_state.particle_qd = self._particle_qd
            newton_state.particle_f = self._particle_f
            if self._mujoco_qfrc_actuator is not None:
                newton_state.mujoco = Model.AttributeNamespace("mujoco")
                newton_state.mujoco.qfrc_actuator = self._mujoco_qfrc_actuator
            self._newton_view = newton_state

        return self._newton_view

    def _sync_from_newton_state(self) -> None:
        """Adopt arrays that a Newton solver rebound on the cached state view."""
        if self._newton_view is None:
            return

        # Newton differentiable solvers intentionally replace some State arrays
        # instead of writing in-place, so gradients remain connected to the new
        # buffers. WanPhys must adopt those replacement arrays; copying them back
        # into old buffers would undo the reason Newton allocated them.
        self._body_q = self._newton_view.body_q
        self._body_qd = self._newton_view.body_qd
        self._body_f = self._newton_view.body_f
        self._body_qdd = self._newton_view.body_qdd
        self._body_parent_f = self._newton_view.body_parent_f
        self._joint_q = self._newton_view.joint_q
        self._joint_qd = self._newton_view.joint_qd
        self._particle_q = self._newton_view.particle_q
        self._particle_qd = self._newton_view.particle_qd
        self._particle_f = self._newton_view.particle_f
        mujoco_namespace = getattr(self._newton_view, "mujoco", None)
        self._mujoco_qfrc_actuator = getattr(mujoco_namespace, "qfrc_actuator", None)

    @property
    def body_count(self) -> int:
        """Number of rigid bodies in the simulation."""
        return self._body_q.shape[0] if self._body_q is not None else 0

    @property
    def joint_coord_count(self) -> int:
        """Number of generalized joint position coordinates."""
        return self._joint_q.shape[0] if self._joint_q is not None else 0

    @property
    def joint_dof_count(self) -> int:
        """Number of generalized joint velocity coordinates."""
        return self._joint_qd.shape[0] if self._joint_qd is not None else 0

    @property
    def particle_count(self) -> int:
        """Number of particles in the simulation."""
        return self._particle_q.shape[0] if self._particle_q is not None else 0

    @property
    def requires_grad(self) -> bool:
        """Check if state arrays have gradient computation enabled."""
        if self._particle_q is not None:
            return bool(self._particle_q.requires_grad)
        if self._body_q is not None:
            return bool(self._body_q.requires_grad)
        return False

    def clear_forces(self) -> None:
        """Clear all accumulated forces (required by DomainState protocol)."""
        if self._body_f is not None:
            self._body_f.zero_()
        if self._particle_f is not None:
            self._particle_f.zero_()

    def assign(self, other: RigidState) -> None:
        """Copy state arrays from another RigidState."""
        if self.body_q is not None and other.body_q is not None:
            wp.copy(self.body_q, other.body_q)
        if self.body_qd is not None and other.body_qd is not None:
            wp.copy(self.body_qd, other.body_qd)
        if self.body_f is not None and other.body_f is not None:
            wp.copy(self.body_f, other.body_f)
        if self.body_qdd is not None and other.body_qdd is not None:
            wp.copy(self.body_qdd, other.body_qdd)
        if self.body_parent_f is not None and other.body_parent_f is not None:
            wp.copy(self.body_parent_f, other.body_parent_f)
        if self.joint_q is not None and other.joint_q is not None:
            wp.copy(self.joint_q, other.joint_q)
        if self.joint_qd is not None and other.joint_qd is not None:
            wp.copy(self.joint_qd, other.joint_qd)
        if self.particle_q is not None and other.particle_q is not None:
            wp.copy(self.particle_q, other.particle_q)
        if self.particle_qd is not None and other.particle_qd is not None:
            wp.copy(self.particle_qd, other.particle_qd)
        if self.particle_f is not None and other.particle_f is not None:
            wp.copy(self.particle_f, other.particle_f)
        if self.mujoco_qfrc_actuator is not None and other.mujoco_qfrc_actuator is not None:
            wp.copy(self.mujoco_qfrc_actuator, other.mujoco_qfrc_actuator)

    def get_body_transform(self, body_idx: int) -> np.ndarray:
        """Get world transform for a specific body.

        Returns:
            Numpy array of shape (7,) containing [px, py, pz, qx, qy, qz, qw].
        """
        return self.body_q.numpy()[body_idx]

    def get_body_position(self, body_idx: int) -> np.ndarray:
        """Get world position for a specific body.

        Returns:
            Numpy array of shape (3,) containing [px, py, pz].
        """
        transform = self.body_q.numpy()[body_idx]
        return transform[:3]

    def get_body_rotation(self, body_idx: int) -> np.ndarray:
        """Get world rotation quaternion for a specific body.

        Returns:
            Numpy array of shape (4,) containing [qx, qy, qz, qw].
        """
        transform = self.body_q.numpy()[body_idx]
        return transform[3:]

    def get_body_velocity(self, body_idx: int) -> np.ndarray:
        """Get spatial velocity for a specific body.

        Returns:
            Numpy array of shape (6,) containing [vx, vy, vz, wx, wy, wz].
        """
        return self.body_qd.numpy()[body_idx]

    def get_body_linear_velocity(self, body_idx: int) -> np.ndarray:
        """Get linear velocity for a specific body.

        Returns:
            Numpy array of shape (3,) containing [vx, vy, vz].
        """
        spatial_vel = self.body_qd.numpy()[body_idx]
        return spatial_vel[:3]

    def get_body_angular_velocity(self, body_idx: int) -> np.ndarray:
        """Get angular velocity for a specific body.

        Returns:
            Numpy array of shape (3,) containing [wx, wy, wz].
        """
        spatial_vel = self.body_qd.numpy()[body_idx]
        return spatial_vel[3:]

    def get_body_force(self, body_idx: int) -> np.ndarray:
        """Get spatial force for a specific body.

        Returns:
            Numpy array of shape (6,) containing [fx, fy, fz, tx, ty, tz].
        """
        return self.body_f.numpy()[body_idx]

    def apply_body_forces(
        self,
        body_indices: "wp.array[int]",
        forces: "wp.array[wp.spatial_vector]",
    ) -> None:
        """Accumulate forces onto multiple bodies in a single kernel launch.

        Uses atomic_add, so forces from multiple calls accumulate. Call
        clear_forces() before each step to reset the buffer.

        Args:
            body_indices: 1-D warp array of body indices, dtype=int,
                          on the same device as this state.
            forces: 1-D warp array of spatial forces, dtype=wp.spatial_vector,
                    same length as body_indices. Each entry is (fx, fy, fz, tx, ty, tz).

        Raises:
            RuntimeError: If body_f is None (body_count == 0).
            ValueError: If body_indices and forces have different lengths, or
                        if either array is on a different device than this state.
        """
        if self.body_f is None:
            raise RuntimeError("apply_body_forces: state has no bodies (body_f is None)")
        if len(body_indices) != len(forces):
            raise ValueError(
                f"apply_body_forces: body_indices length ({len(body_indices)}) "
                f"!= forces length ({len(forces)})"
            )
        if body_indices.device != self.body_f.device or forces.device != self.body_f.device:
            raise ValueError(
                f"apply_body_forces: all arrays must be on the same device "
                f"(state={self.body_f.device}, body_indices={body_indices.device}, "
                f"forces={forces.device})"
            )
        wp.launch(
            _apply_body_forces_kernel,
            dim=len(body_indices),
            inputs=[self.body_f, body_indices, forces],
            device=self.body_f.device,
        )

    def add_body_force(self, body_idx: int, force: tuple[float, float, float, float, float, float]) -> None:
        """Accumulate a spatial force onto a single body.

        Convenience wrapper around :meth:`apply_body_forces` for the common
        single-body case. Internally allocates two single-element arrays and
        calls the batched kernel.

        Args:
            body_idx: Index of the body to apply force to.
            force: Tuple of (fx, fy, fz, tx, ty, tz).
        """
        if self.body_f is None:
            raise RuntimeError("add_body_force: state has no bodies (body_f is None)")
        device = self.body_f.device
        indices = wp.array([body_idx], dtype=int, device=device)
        forces = wp.array([force], dtype=wp.spatial_vector, device=device)
        self.apply_body_forces(indices, forces)

    def set_body_force(self, body_idx: int, force: tuple[float, float, float, float, float, float]) -> None:
        """Deprecated alias for :meth:`add_body_force`."""
        self.add_body_force(body_idx, force)
