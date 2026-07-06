from __future__ import annotations


class MujocoMappingsMixin:
    def _init_mappings(self) -> None:
        # Primary runtime mapping vocabulary:
        # MuJoCo[world, entity] -> source-model entity
        self.mjc_body_to_source_body = None
        self.mjc_geom_to_source_shape = None
        self.mjc_jnt_to_source_joint = None
        self.mjc_jnt_to_source_dof = None
        self.mjc_dof_to_source_dof = None
        self.mjc_actuator_to_source_axis = None
        self.mjc_mocap_to_source_joint = None
        self.mjc_eq_to_source_eq = None
        self.mjc_eq_to_source_joint = None

        # Legacy compatibility aliases kept until every bridge/kernel call site
        # has been migrated to the neutral "source_*" naming.
        self.mjc_body_to_newton = None
        self.mjc_geom_to_newton_shape = None
        self.mjc_jnt_to_newton_jnt = None
        self.mjc_jnt_to_newton_dof = None
        self.mjc_dof_to_newton_dof = None
        self.mjc_actuator_to_newton_axis = None
        self.mjc_mocap_to_newton_jnt = None
        self.mjc_eq_to_newton_eq = None
        self.mjc_eq_to_newton_jnt = None

        # Inverse lookup used when externally supplied contacts need to find the
        # MuJoCo geom associated with a source shape index.
        self.source_shape_to_mjc_geom = None
        self.newton_shape_to_mjc_geom = None

        # helper metadata
        self._shapes_per_world = 0
        self._first_env_shape_base = 0
