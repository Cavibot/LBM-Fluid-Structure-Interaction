from __future__ import annotations

import warp as wp

from .kernels import (
    apply_mjc_body_f_kernel,
    apply_mjc_control_kernel,
    apply_mjc_qfrc_kernel,
)


class MujocoControlBridgeMixin:
    def _apply_mjc_control(self, source_state, control):
        """Push control inputs and external forces into MuJoCo runtime buffers.

        The bridge writes three categories of inputs before stepping:
        - actuator targets into ``self.mjw_data.ctrl``
        - generalized joint forces into ``self.mjw_data.qfrc_applied``
        - body spatial forces into ``self.mjw_data.xfrc_applied``
        """

        if self.mjw_data is None:
            return

        # If neither high-level control nor body forces are present, there is
        # nothing to push into the runtime for this step.
        if control is None:
            if getattr(source_state, "body_f", None) is None:
                return
        else:
            if getattr(control, "joint_f", None) is None and getattr(source_state, "body_f", None) is None:
                return

        model = self._source_model
        data = self.mjw_data

        ctrl = data.ctrl
        qfrc = data.qfrc_applied
        xfrc = data.xfrc_applied

        nworld = data.nworld
        joints_per_world = model.joint_count // nworld
        bodies_per_world = model.body_count // nworld

        # 1) Convert WanPhys joint targets into MuJoCo actuator-space control values.
        if control is not None and self.mjc_actuator_to_source_axis is not None:
            nu = self.mjc_actuator_to_source_axis.shape[1]

            wp.launch(
                apply_mjc_control_kernel,
                dim=(nworld, nu),
                inputs=[
                    self.mjc_actuator_to_source_axis,
                    control.joint_target_pos,
                    control.joint_target_vel,
                ],
                outputs=[
                    ctrl,
                ],
                device=model.device,
            )

        # 2) Convert per-joint generalized forces into MuJoCo's qfrc_applied layout.
        if control is not None and getattr(control, "joint_f", None) is not None:
            wp.launch(
                    apply_mjc_qfrc_kernel,
                    dim=(nworld, joints_per_world),
                    inputs=[
                    source_state.body_q,
                    control.joint_f,
                    model.joint_type,
                    model.body_com,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    joints_per_world,
                    bodies_per_world,
                ],
                outputs=[
                    qfrc,
                ],
                device=model.device,
            )

        # 3) Copy body-space wrench inputs into MuJoCo's xfrc_applied buffer.
        if getattr(source_state, "body_f", None) is not None and self.mjc_body_to_source_body is not None:
            nbody = self.mjc_body_to_source_body.shape[1]

            wp.launch(
                apply_mjc_body_f_kernel,
                dim=(nworld, nbody),
                inputs=[
                    self.mjc_body_to_source_body,
                    source_state.body_f,
                ],
                outputs=[
                    xfrc,
                ],
                device=model.device,
            )
