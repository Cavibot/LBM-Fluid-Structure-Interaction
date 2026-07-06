from __future__ import annotations

import warp as wp

from .kernels import (
    convert_warp_coords_to_mj_kernel,
    convert_mj_coords_to_warp_kernel,
    convert_body_xforms_to_warp_kernel,
    eval_articulation_fk,
)


class MujocoStateBridgeMixin:
    def _update_mjc_data(self, newton_state):
        """Convert the input state into MuJoCo runtime ``qpos``/``qvel`` buffers."""

        if self.mjw_data is None:
            return

        qpos = self.mjw_data.qpos
        qvel = self.mjw_data.qvel
        nworld = self.mjw_data.nworld

        joints_per_world = self.model.joint_count // nworld

        wp.launch(
            convert_warp_coords_to_mj_kernel,
            dim=(nworld, joints_per_world),
            inputs=[
                newton_state.joint_q,
                newton_state.joint_qd,
                joints_per_world,
                self.model.up_axis,
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
            ],
            outputs=[qpos, qvel],
            device=self.model.device,
        )

    def _update_newton_state(self, newton_state, eval_fk: bool = True):
        """Pull MuJoCo runtime results back into the output state view.

        Joint coordinates are always synchronized from ``qpos``/``qvel``.
        Body transforms are then updated either by evaluating WanPhys FK
        from the recovered joint state or by copying MuJoCo body transforms
        directly when ``eval_fk`` is disabled.
        """

        if self.mjw_data is None:
            return

        qpos = self.mjw_data.qpos
        qvel = self.mjw_data.qvel
        xpos = self.mjw_data.xpos
        xquat = self.mjw_data.xquat
        nworld = self.mjw_data.nworld

        joints_per_world = self.model.joint_count // nworld

        wp.launch(
            convert_mj_coords_to_warp_kernel,
            dim=(nworld, joints_per_world),
            inputs=[
                qpos,
                qvel,
                joints_per_world,
                int(self.model.up_axis),
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
            ],
            outputs=[
                newton_state.joint_q,
                newton_state.joint_qd,
            ],
            device=self.model.device,
        )

        if eval_fk:
            # Rebuild body pose/velocity from the synchronized joint state so
            # the output stays consistent with WanPhys articulation semantics.
            wp.launch(
                kernel=eval_articulation_fk,
                dim=self.model.articulation_count,
                inputs=[
                    self.model.articulation_start,
                    newton_state.joint_q,
                    newton_state.joint_qd,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.model.joint_type,
                    self.model.joint_parent,
                    self.model.joint_child,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                    self.model.joint_axis,
                    self.model.joint_dof_dim,
                    self.model.body_com,
                ],
                outputs=[
                    newton_state.body_q,
                    newton_state.body_qd,
                ],
                device=self.model.device,
            )
        else:
            if self.mjc_body_to_source_body is None:
                return

            nbody = self.mjc_body_to_source_body.shape[1]

            # Fast path: copy MuJoCo world transforms directly without running FK.
            wp.launch(
                convert_body_xforms_to_warp_kernel,
                dim=(nworld, nbody),
                inputs=[
                    self.mjc_body_to_source_body,
                    xpos,
                    xquat,
                ],
                outputs=[newton_state.body_q],
                device=self.model.device,
            )
