from __future__ import annotations

import warp as wp

from .flags import SolverNotifyFlags
from .kernels import (
    update_axis_properties_kernel,
    update_body_inertia_kernel,
    update_body_mass_ipos_kernel,
    update_dof_properties_kernel,
    update_eq_properties_kernel,
    update_geom_properties_kernel,
    update_jnt_properties_kernel,
    update_joint_transforms_kernel,
    update_mocap_transforms_kernel,
    update_model_properties_kernel,
)

class MujocoSyncMixin:
    def notify_model_changed(self, flags: int):
        """Dispatch model edits to the matching MuJoCo runtime update paths."""
        if flags & SolverNotifyFlags.BODY_INERTIAL_PROPERTIES:
            self.update_model_inertial_properties()
        if flags & SolverNotifyFlags.JOINT_PROPERTIES:
            self.update_joint_properties()
        if flags & SolverNotifyFlags.JOINT_DOF_PROPERTIES:
            self.update_joint_dof_properties()
        if flags & SolverNotifyFlags.SHAPE_PROPERTIES:
            self.update_geom_properties()
        if flags & SolverNotifyFlags.MODEL_PROPERTIES:
            self.update_model_properties()
        if flags & SolverNotifyFlags.EQUALITY_CONSTRAINT_PROPERTIES:
            self.update_eq_properties()

    def update_model_inertial_properties(self):
        """Refresh body mass, COM, inertia, and gravity-compensation buffers."""
        if self.model.body_count == 0:
            return

        mujoco_attrs = getattr(self.model, "mujoco", None)
        gravcomp = getattr(mujoco_attrs, "gravcomp", None) if mujoco_attrs is not None else None

        nworld = self.mjc_body_to_source_body.shape[0]
        nbody = self.mjc_body_to_source_body.shape[1]

        wp.launch(
            update_body_mass_ipos_kernel,
            dim=(nworld, nbody),
            inputs=[
                self.mjc_body_to_source_body,
                self.model.body_com,
                self.model.body_mass,
                gravcomp,
                self.model.up_axis,
            ],
            outputs=[
                self.mjw_model.body_ipos,
                self.mjw_model.body_mass,
                self.mjw_model.body_gravcomp,
            ],
            device=self.model.device,
        )

        wp.launch(
            update_body_inertia_kernel,
            dim=(nworld, nbody),
            inputs=[
                self.mjc_body_to_source_body,
                self.model.body_inertia,
            ],
            outputs=[
                self.mjw_model.body_inertia,
                self.mjw_model.body_iquat,
            ],
            device=self.model.device,
        )


    def update_joint_dof_properties(self):
        """Refresh actuator gains plus per-DOF friction, damping, and limits."""
        if self.model.joint_dof_count == 0:
            return

        # MuJoCo actuators cache target stiffness/damping separately from the
        # joint DOF arrays, so update them first when actuator mappings exist.
        if self.mjc_actuator_to_source_axis is not None:
            nworld = self.mjc_actuator_to_source_axis.shape[0]
            nu = self.mjc_actuator_to_source_axis.shape[1]

            wp.launch(
                update_axis_properties_kernel,
                dim=(nworld, nu),
                inputs=[
                    self.mjc_actuator_to_source_axis,
                    self.model.joint_target_ke,
                    self.model.joint_target_kd,
                ],
                outputs=[
                    self.mjw_model.actuator_biasprm,
                    self.mjw_model.actuator_gainprm,
                ],
                device=self.model.device,
            )

        mujoco_attrs = getattr(self.model, "mujoco", None)
        joint_damping = getattr(mujoco_attrs, "dof_passive_damping", None) if mujoco_attrs is not None else None
        dof_solimp = getattr(mujoco_attrs, "solimpfriction", None) if mujoco_attrs is not None else None
        dof_solref = getattr(mujoco_attrs, "solreffriction", None) if mujoco_attrs is not None else None

        nworld = self.mjc_dof_to_source_dof.shape[0]
        nv = self.mjc_dof_to_source_dof.shape[1]

        wp.launch(
            update_dof_properties_kernel,
            dim=(nworld, nv),
            inputs=[
                self.mjc_dof_to_source_dof,
                self.model.joint_armature,
                self.model.joint_friction,
                joint_damping,
                dof_solimp,
                dof_solref,
            ],
            outputs=[
                self.mjw_model.dof_armature,
                self.mjw_model.dof_frictionloss,
                self.mjw_model.dof_damping,
                self.mjw_model.dof_solimp,
                self.mjw_model.dof_solref,
            ],
            device=self.model.device,
        )

        solimplimit = getattr(mujoco_attrs, "solimplimit", None) if mujoco_attrs is not None else None
        joint_dof_limit_margin = getattr(mujoco_attrs, "limit_margin", None) if mujoco_attrs is not None else None
        joint_stiffness = getattr(mujoco_attrs, "dof_passive_stiffness", None) if mujoco_attrs is not None else None

        njnt = self.mjc_jnt_to_source_dof.shape[1]

        wp.launch(
            update_jnt_properties_kernel,
            dim=(nworld, njnt),
            inputs=[
                self.mjc_jnt_to_source_dof,
                self.model.joint_limit_ke,
                self.model.joint_limit_kd,
                self.model.joint_limit_lower,
                self.model.joint_limit_upper,
                self.model.joint_effort_limit,
                solimplimit,
                joint_stiffness,
                joint_dof_limit_margin,
            ],
            outputs=[
                self.mjw_model.jnt_solimp,
                self.mjw_model.jnt_solref,
                self.mjw_model.jnt_stiffness,
                self.mjw_model.jnt_margin,
                self.mjw_model.jnt_range,
                self.mjw_model.jnt_actfrcrange,
            ],
            device=self.model.device,
        )


    def update_joint_properties(self):
        """Refresh joint frame definitions and mocap transforms."""
        if self.model.joint_count == 0:
            return

        # Fixed-base or mocap-driven bodies store their target transforms in
        # separate mocap buffers that must track the source model.
        if self.mjc_mocap_to_source_joint is not None:
            nworld = self.mjc_mocap_to_source_joint.shape[0]
            nmocap = self.mjc_mocap_to_source_joint.shape[1]

            wp.launch(
                update_mocap_transforms_kernel,
                dim=(nworld, nmocap),
                inputs=[
                    self.mjc_mocap_to_source_joint,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                ],
                outputs=[
                    self.mjw_data.mocap_pos,
                    self.mjw_data.mocap_quat,
                ],
                device=self.model.device,
            )

        if self.mjc_jnt_to_source_joint is not None and self.mjc_jnt_to_source_joint.shape[1] > 0:
            nworld = self.mjc_jnt_to_source_joint.shape[0]
            njnt = self.mjc_jnt_to_source_joint.shape[1]

            wp.launch(
                update_joint_transforms_kernel,
                dim=(nworld, njnt),
                inputs=[
                    self.mjc_jnt_to_source_joint,
                    self.mjc_jnt_to_source_dof,
                    self.mjw_model.jnt_bodyid,
                    self.mjw_model.jnt_type,
                    self.model.joint_X_p,
                    self.model.joint_X_c,
                    self.model.joint_axis,
                ],
                outputs=[
                    self.mjw_model.jnt_pos,
                    self.mjw_model.jnt_axis,
                    self.mjw_model.body_pos,
                    self.mjw_model.body_quat,
                ],
                device=self.model.device,
            )


    def update_geom_properties(self):
        """Refresh geom shape, pose, and material/contact properties."""
        num_geoms = self.mj_model.ngeom
        if num_geoms == 0:
            return

        num_worlds = self.mjc_geom_to_source_shape.shape[0]

        mujoco_attrs = getattr(self.model, "mujoco", None)
        shape_geom_solimp = getattr(mujoco_attrs, "geom_solimp", None) if mujoco_attrs is not None else None
        shape_geom_solmix = getattr(mujoco_attrs, "geom_solmix", None) if mujoco_attrs is not None else None
        shape_geom_gap = getattr(mujoco_attrs, "geom_gap", None) if mujoco_attrs is not None else None

        wp.launch(
            update_geom_properties_kernel,
            dim=(num_worlds, num_geoms),
            inputs=[
                self.model.shape_collision_radius,
                self.model.shape_material_mu,
                self.model.shape_material_ke,
                self.model.shape_material_kd,
                self.model.shape_scale,
                self.model.shape_transform,
                self.mjc_geom_to_source_shape,
                self.mjw_model.geom_type,
                self._mujoco.mjtGeom.mjGEOM_MESH,
                self.mjw_model.geom_dataid,
                self.mjw_model.mesh_pos,
                self.mjw_model.mesh_quat,
                self.model.shape_material_mu_torsional,
                self.model.shape_material_mu_rolling,
                shape_geom_solimp,
                shape_geom_solmix,
                shape_geom_gap,
            ],
            outputs=[
                self.mjw_model.geom_rbound,
                self.mjw_model.geom_friction,
                self.mjw_model.geom_solref,
                self.mjw_model.geom_size,
                self.mjw_model.geom_pos,
                self.mjw_model.geom_quat,
                self.mjw_model.geom_solimp,
                self.mjw_model.geom_solmix,
                self.mjw_model.geom_gap,
            ],
            device=self.model.device,
        )


    def update_model_properties(self):
        """Refresh global simulation options such as gravity."""
        if not hasattr(self, "mjw_data") or self.mjw_data is None:
            return

        wp.launch(
            kernel=update_model_properties_kernel,
            dim=self.mjw_data.nworld,
            inputs=[
                self.model.gravity,
            ],
            outputs=[
                self.mjw_model.opt.gravity,
            ],
            device=self.model.device,
        )


    def update_eq_properties(self):
        """Refresh equality-constraint solver parameters."""
        if self.model.equality_constraint_count == 0:
            return

        neq = self.mj_model.neq
        if neq == 0:
            return

        num_worlds = self.mjc_eq_to_source_eq.shape[0]

        mujoco_attrs = getattr(self.model, "mujoco", None)
        eq_solref = getattr(mujoco_attrs, "eq_solref", None) if mujoco_attrs is not None else None

        if eq_solref is not None:
            wp.launch(
                update_eq_properties_kernel,
                dim=(num_worlds, neq),
                inputs=[
                    self.mjc_eq_to_source_eq,
                    eq_solref,
                ],
                outputs=[
                    self.mjw_model.eq_solref,
                ],
                device=self.model.device,
            )
