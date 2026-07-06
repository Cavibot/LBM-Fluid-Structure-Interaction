from __future__ import annotations

import warnings

import warp as wp

from .compiler import MujocoModelCompiler
from .config import MujocoConfig


class MujocoBackendMixin:
    _mujoco = None
    _mujoco_warp = None

    @classmethod
    def _import_mujoco(cls):
        if cls._mujoco is None or cls._mujoco_warp is None:
            with warnings.catch_warnings():
                warnings.simplefilter("always", category=ImportWarning)
                import mujoco
                import mujoco_warp

            cls._mujoco = mujoco
            cls._mujoco_warp = mujoco_warp

        return cls._mujoco, cls._mujoco_warp

    def _init_backend(self, config: MujocoConfig):
        self.config = config
        mujoco, mujoco_warp = self._import_mujoco()
        self._mujoco = mujoco
        self._mujoco_warp = mujoco_warp

        if self.config.separate_worlds is None:
            self.config.separate_worlds = self.model.world_count > 1

        # compiled backend objects
        self.mj_model = None
        self.mjw_model = None
        self.mjw_data = None

        self._build_backend_model(
            source_model=self._source_model,
            separate_worlds=self.config.separate_worlds,
        )

        if self.mjw_model is not None:
            self.mjw_model.opt.run_collision_detection = self.config.use_mujoco_contacts

    def _build_backend_model(self, source_model, separate_worlds: bool):
        """Build the MuJoCo runtime through the local WanPhys compiler."""
        compile_config = self.config
        compile_config.separate_worlds = separate_worlds
        artifacts = MujocoModelCompiler.compile(source_model, compile_config)
        self._adopt_compile_artifacts(artifacts)

    def _adopt_compile_artifacts(self, artifacts) -> None:
        """Attach compiled MuJoCo runtime objects and index mappings."""
        self.mj_model = artifacts.mj_model
        self.mjw_model = artifacts.mjw_model
        self.mjw_data = artifacts.mjw_data

        alias_pairs = (
            ("mjc_body_to_source_body", "mjc_body_to_newton"),
            ("mjc_geom_to_source_shape", "mjc_geom_to_newton_shape"),
            ("mjc_jnt_to_source_joint", "mjc_jnt_to_newton_jnt"),
            ("mjc_jnt_to_source_dof", "mjc_jnt_to_newton_dof"),
            ("mjc_dof_to_source_dof", "mjc_dof_to_newton_dof"),
            ("mjc_actuator_to_source_axis", "mjc_actuator_to_newton_axis"),
            ("mjc_mocap_to_source_joint", "mjc_mocap_to_newton_jnt"),
            ("mjc_eq_to_source_eq", "mjc_eq_to_newton_eq"),
            ("mjc_eq_to_source_joint", "mjc_eq_to_newton_jnt"),
            ("source_shape_to_mjc_geom", "newton_shape_to_mjc_geom"),
        )
        for source_attr, legacy_attr in alias_pairs:
            value = getattr(artifacts, legacy_attr, None)
            setattr(self, source_attr, value)
            setattr(self, legacy_attr, value)

        for attr in (
            "mjc_body_to_newton",
            "mjc_geom_to_newton_shape",
            "mjc_jnt_to_newton_jnt",
            "mjc_jnt_to_newton_dof",
            "mjc_dof_to_newton_dof",
            "mjc_actuator_to_newton_axis",
            "mjc_mocap_to_newton_jnt",
            "mjc_eq_to_newton_eq",
            "mjc_eq_to_newton_jnt",
            "newton_shape_to_mjc_geom",
        ):
            setattr(self, attr, getattr(artifacts, attr, None))

        self._shapes_per_world = artifacts.shapes_per_world
        self._first_env_shape_base = artifacts.first_env_shape_base

        # Keep the compiler backend alive because several warp-side arrays are
        # still owned by the helper object that produced them.
        self._compiler_backend = artifacts.compiler_backend

    def _step_backend(self):
        with wp.ScopedDevice(self.device):
            self._mujoco_warp.step(self.mjw_model, self.mjw_data)

    def expand_model_fields(self, mjw_model, nworld: int):
        """Expand per-world model fields through the adopted compiler backend."""
        if getattr(self, "_compiler_backend", None) is None:
            raise RuntimeError("MuJoCo backend has not been initialized yet.")
        self._compiler_backend.expand_model_fields(mjw_model, nworld)
