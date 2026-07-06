from __future__ import annotations

"""Compilation helpers for the WanPhys MuJoCo solver.

This module builds MuJoCo and mujoco_warp runtime objects directly from the
WanPhys source-model view. It intentionally owns its compile path locally
instead of reusing a legacy solver module.
"""

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.geometry import GeoType, ShapeFlags
from newton._src.sim import EqType, JointType
from newton._src.sim.graph_coloring import color_graph, plot_graph
from newton._src.utils import topological_sort

from .config import MujocoConfig
from .kernels import convert_solref, convert_warp_coords_to_mj_kernel, repeat_array_kernel, update_shape_mappings_kernel


@dataclass
class MujocoCompileArtifacts:
    mj_model: Any
    mjw_model: Any
    mjw_data: Any
    mjc_body_to_source_body: Any = None
    mjc_geom_to_source_shape: Any = None
    mjc_jnt_to_source_joint: Any = None
    mjc_jnt_to_source_dof: Any = None
    mjc_dof_to_source_dof: Any = None
    mjc_actuator_to_source_axis: Any = None
    mjc_mocap_to_source_joint: Any = None
    mjc_eq_to_source_eq: Any = None
    mjc_eq_to_source_joint: Any = None
    source_shape_to_mjc_geom: Any = None
    mjc_body_to_newton: Any = None
    mjc_geom_to_newton_shape: Any = None
    mjc_jnt_to_newton_jnt: Any = None
    mjc_jnt_to_newton_dof: Any = None
    mjc_dof_to_newton_dof: Any = None
    mjc_actuator_to_newton_axis: Any = None
    mjc_mocap_to_newton_jnt: Any = None
    mjc_eq_to_newton_eq: Any = None
    mjc_eq_to_newton_jnt: Any = None
    newton_shape_to_mjc_geom: Any = None
    shapes_per_world: int = 0
    first_env_shape_base: int = 0
    compiler_backend: Any = None


class MujocoModelCompiler:
    """Build MuJoCo/mujoco_warp backend objects without solver-module reuse."""

    def __init__(self, model, config: MujocoConfig):
        self.model = model
        self.config = config
        self.mj_model = None
        self.mj_data = None
        self.mjw_model = None
        self.mjw_data = None

        self.mjc_body_to_source_body = None
        self.mjc_geom_to_source_shape = None
        self.mjc_jnt_to_source_joint = None
        self.mjc_jnt_to_source_dof = None
        self.mjc_dof_to_source_dof = None
        self.mjc_actuator_to_source_axis = None
        self.mjc_mocap_to_source_joint = None
        self.mjc_eq_to_source_eq = None
        self.mjc_eq_to_source_joint = None
        self.source_shape_to_mjc_geom = None

        self.mjc_body_to_newton = None
        self.mjc_geom_to_newton_shape = None
        self.mjc_jnt_to_newton_jnt = None
        self.mjc_jnt_to_newton_dof = None
        self.mjc_dof_to_newton_dof = None
        self.mjc_actuator_to_newton_axis = None
        self.mjc_mocap_to_newton_jnt = None
        self.mjc_eq_to_newton_eq = None
        self.mjc_eq_to_newton_jnt = None
        self.newton_shape_to_mjc_geom = None

        self._shapes_per_world = 0
        self._first_env_shape_base = 0
        self._mujoco, self._mujoco_warp = self._import_mujoco()

    @staticmethod
    def _import_mujoco():
        with warnings.catch_warnings():
            warnings.simplefilter("always", category=ImportWarning)
            import mujoco
            import mujoco_warp

        return mujoco, mujoco_warp

    @staticmethod
    def compile(model, config: MujocoConfig) -> MujocoCompileArtifacts:
        compiler = MujocoModelCompiler(model, config)
        compiler._convert_to_mjc()
        return MujocoCompileArtifacts(
            mj_model=compiler.mj_model,
            mjw_model=compiler.mjw_model,
            mjw_data=compiler.mjw_data,
            mjc_body_to_source_body=compiler.mjc_body_to_source_body,
            mjc_geom_to_source_shape=compiler.mjc_geom_to_source_shape,
            mjc_jnt_to_source_joint=compiler.mjc_jnt_to_source_joint,
            mjc_jnt_to_source_dof=compiler.mjc_jnt_to_source_dof,
            mjc_dof_to_source_dof=compiler.mjc_dof_to_source_dof,
            mjc_actuator_to_source_axis=compiler.mjc_actuator_to_source_axis,
            mjc_mocap_to_source_joint=compiler.mjc_mocap_to_source_joint,
            mjc_eq_to_source_eq=compiler.mjc_eq_to_source_eq,
            mjc_eq_to_source_joint=compiler.mjc_eq_to_source_joint,
            source_shape_to_mjc_geom=compiler.source_shape_to_mjc_geom,
            mjc_body_to_newton=compiler.mjc_body_to_newton,
            mjc_geom_to_newton_shape=compiler.mjc_geom_to_newton_shape,
            mjc_jnt_to_newton_jnt=compiler.mjc_jnt_to_newton_jnt,
            mjc_jnt_to_newton_dof=compiler.mjc_jnt_to_newton_dof,
            mjc_dof_to_newton_dof=compiler.mjc_dof_to_newton_dof,
            mjc_actuator_to_newton_axis=compiler.mjc_actuator_to_newton_axis,
            mjc_mocap_to_newton_jnt=compiler.mjc_mocap_to_newton_jnt,
            mjc_eq_to_newton_eq=compiler.mjc_eq_to_newton_eq,
            mjc_eq_to_newton_jnt=compiler.mjc_eq_to_newton_jnt,
            newton_shape_to_mjc_geom=compiler.newton_shape_to_mjc_geom,
            shapes_per_world=compiler._shapes_per_world,
            first_env_shape_base=compiler._first_env_shape_base,
            compiler_backend=compiler,
        )

    def _resolve_mj_opt(self, val, opts: dict[str, int], kind: str):
        if isinstance(val, str):
            key = val.strip().lower()
            try:
                return opts[key]
            except KeyError as e:
                options = "', '".join(sorted(opts))
                raise ValueError(f"Unknown {kind} '{val}'. Valid options: '{options}'.") from e
        return val

    def _init_pairs(self, model, spec, shape_mapping: dict[int, str], template_world: int) -> None:
        pair_count = model.custom_frequency_counts.get("mujoco:pair", 0)
        if pair_count == 0:
            return

        mujoco_attrs = model.mujoco

        def get_numpy(name):
            attr = getattr(mujoco_attrs, name, None)
            return attr.numpy() if attr is not None else None

        pair_world = get_numpy("pair_world")
        pair_geom1 = get_numpy("pair_geom1")
        pair_geom2 = get_numpy("pair_geom2")
        if pair_world is None or pair_geom1 is None or pair_geom2 is None:
            return

        pair_condim = get_numpy("pair_condim")
        pair_solref = get_numpy("pair_solref")
        pair_solreffriction = get_numpy("pair_solreffriction")
        pair_solimp = get_numpy("pair_solimp")
        pair_margin = get_numpy("pair_margin")
        pair_gap = get_numpy("pair_gap")
        pair_friction = get_numpy("pair_friction")

        for i in range(pair_count):
            if int(pair_world[i]) != template_world:
                continue

            shape1 = int(pair_geom1[i])
            shape2 = int(pair_geom2[i])
            if shape1 < 0 or shape2 < 0:
                continue

            geom_name1 = shape_mapping.get(shape1)
            geom_name2 = shape_mapping.get(shape2)
            if geom_name1 is None or geom_name2 is None:
                warnings.warn(
                    f"Skipping pair {i}: source shapes ({shape1}, {shape2}) not found in MuJoCo mapping.",
                    stacklevel=2,
                )
                continue

            pair_kwargs: dict[str, Any] = {
                "geomname1": geom_name1,
                "geomname2": geom_name2,
            }

            if pair_condim is not None:
                pair_kwargs["condim"] = int(pair_condim[i])
            if pair_solref is not None:
                pair_kwargs["solref"] = pair_solref[i].tolist()
            if pair_solreffriction is not None:
                pair_kwargs["solreffriction"] = pair_solreffriction[i].tolist()
            if pair_solimp is not None:
                pair_kwargs["solimp"] = pair_solimp[i].tolist()
            if pair_margin is not None:
                pair_kwargs["margin"] = float(pair_margin[i])
            if pair_gap is not None:
                pair_kwargs["gap"] = float(pair_gap[i])
            if pair_friction is not None:
                pair_kwargs["friction"] = pair_friction[i].tolist()

            spec.add_pair(**pair_kwargs)

    def find_body_collision_filter_pairs(self, model, selected_bodies, colliding_shapes):
        body_exclude_pairs = []
        shape_set = set(colliding_shapes)

        body_shapes = {}
        for body in selected_bodies:
            shapes = model.body_shapes[body]
            shapes = [s for s in shapes if s in shape_set]
            body_shapes[body] = shapes

        bodies_a, bodies_b = np.triu_indices(len(selected_bodies), k=1)
        for body_a, body_b in zip(bodies_a, bodies_b, strict=True):
            b1, b2 = selected_bodies[body_a], selected_bodies[body_b]
            excluded = True
            for shape_1 in body_shapes[b1]:
                for shape_2 in body_shapes[b2]:
                    s1, s2 = (shape_2, shape_1) if shape_1 > shape_2 else (shape_1, shape_2)
                    if (s1, s2) not in model.shape_collision_filter_pairs:
                        excluded = False
                        break
            if excluded:
                body_exclude_pairs.append((b1, b2))
        return body_exclude_pairs

    def color_collision_shapes(self, model, selected_shapes, visualize_graph: bool = False, shape_keys=None):
        num_shapes = len(selected_shapes)
        shape_a, shape_b = np.triu_indices(num_shapes, k=1)
        shape_collision_group_np = model.shape_collision_group.numpy()
        cgroup = [shape_collision_group_np[i] for i in selected_shapes]
        graph_edges = [
            (i, j)
            for i, j in zip(shape_a, shape_b, strict=True)
            if (
                (selected_shapes[i], selected_shapes[j]) not in model.shape_collision_filter_pairs
                and (cgroup[i] == cgroup[j] or cgroup[i] == -1 or cgroup[j] == -1)
            )
        ]
        shape_color = np.zeros(model.shape_count, dtype=np.int32)
        if len(graph_edges) > 0:
            color_groups = color_graph(
                num_nodes=num_shapes,
                graph_edge_indices=wp.array(graph_edges, dtype=wp.int32),
                balance_colors=False,
            )
            num_colors = 0
            for group in color_groups:
                num_colors += 1
                shape_color[selected_shapes[group]] = num_colors
            if visualize_graph:
                plot_graph(
                    vertices=np.arange(num_shapes),
                    edges=graph_edges,
                    node_labels=[shape_keys[i] for i in selected_shapes] if shape_keys is not None else None,
                    node_colors=[shape_color[i] for i in selected_shapes],
                )
        return shape_color

    def _validate_model_for_separate_worlds(self, model) -> None:
        num_worlds = model.world_count
        if num_worlds == 0:
            raise ValueError("separate_worlds=True requires at least one non-global world.")

        body_world = model.body_world.numpy()
        joint_world = model.joint_world.numpy()
        shape_world = model.shape_world.numpy()
        eq_constraint_world = model.equality_constraint_world.numpy()

        global_bodies = np.where(body_world == -1)[0]
        if len(global_bodies) > 0:
            raise ValueError("Global world (-1) cannot contain bodies.")
        global_joints = np.where(joint_world == -1)[0]
        if len(global_joints) > 0:
            raise ValueError("Global world (-1) cannot contain joints.")
        global_constraints = np.where(eq_constraint_world == -1)[0]
        if len(global_constraints) > 0:
            raise ValueError("Global world (-1) cannot contain equality constraints.")

        if num_worlds <= 1:
            return

        non_global_shapes = shape_world[shape_world >= 0]
        for entity_name, world_arr in [
            ("bodies", body_world),
            ("joints", joint_world),
            ("shapes", non_global_shapes),
            ("equality constraints", eq_constraint_world),
        ]:
            counts = [np.sum(world_arr == w) for w in range(num_worlds)]
            expected = counts[0]
            for w in range(1, num_worlds):
                if counts[w] != expected:
                    raise ValueError(
                        f"separate_worlds requires homogeneous worlds: world 0 has {expected} {entity_name}, "
                        f"world {w} has {counts[w]}."
                    )

        joint_type = model.joint_type.numpy()
        shape_type = model.shape_type.numpy()
        eq_constraint_type = model.equality_constraint_type.numpy()

        joints_per_world = model.joint_count // num_worlds
        if joints_per_world > 0:
            joint_types_2d = joint_type.reshape(num_worlds, joints_per_world)
            if not np.all(joint_types_2d == joint_types_2d[0]):
                raise ValueError("separate_worlds requires matching joint types across worlds.")

        shapes_per_world = len(non_global_shapes) // num_worlds if num_worlds > 0 else 0
        if shapes_per_world > 0:
            non_global_shape_types = shape_type[shape_world >= 0]
            shape_types_2d = non_global_shape_types.reshape(num_worlds, shapes_per_world)
            if not np.all(shape_types_2d == shape_types_2d[0]):
                raise ValueError("separate_worlds requires matching shape types across worlds.")

        constraints_per_world = model.equality_constraint_count // num_worlds if num_worlds > 0 else 0
        if constraints_per_world > 0:
            constraint_types_2d = eq_constraint_type.reshape(num_worlds, constraints_per_world)
            if not np.all(constraint_types_2d == constraint_types_2d[0]):
                raise ValueError("separate_worlds requires matching equality-constraint types across worlds.")

    def update_mjc_data(self, mj_data, model, state=None):
        qpos = wp.empty((1, model.joint_coord_count), dtype=wp.float32, device=model.device)
        qvel = wp.empty((1, model.joint_dof_count), dtype=wp.float32, device=model.device)

        if state is None:
            joint_q = model.joint_q
            joint_qd = model.joint_qd
        else:
            joint_q = state.joint_q
            joint_qd = state.joint_qd

        wp.launch(
            convert_warp_coords_to_mj_kernel,
            dim=(1, model.joint_count),
            inputs=[
                joint_q,
                joint_qd,
                model.joint_count,
                model.up_axis,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
            ],
            outputs=[qpos, qvel],
            device=model.device,
        )
        mj_data.qpos[:] = qpos.numpy().flatten()[: len(mj_data.qpos)]
        mj_data.qvel[:] = qvel.numpy().flatten()[: len(mj_data.qvel)]

    def expand_model_fields(self, mj_model, nworld: int):
        if nworld == 1:
            return

        model_fields_to_expand = {
            "body_pos",
            "body_quat",
            "body_ipos",
            "body_iquat",
            "body_mass",
            "body_inertia",
            "body_gravcomp",
            "jnt_solref",
            "jnt_solimp",
            "jnt_pos",
            "jnt_axis",
            "jnt_stiffness",
            "jnt_range",
            "jnt_actfrcrange",
            "jnt_margin",
            "dof_armature",
            "dof_damping",
            "dof_frictionloss",
            "dof_solimp",
            "dof_solref",
            "geom_solmix",
            "geom_solref",
            "geom_solimp",
            "geom_size",
            "geom_rbound",
            "geom_pos",
            "geom_quat",
            "geom_friction",
            "geom_gap",
            "eq_solref",
            "actuator_gainprm",
            "actuator_biasprm",
        }

        def tile(x: wp.array):
            new_shape = list(x.shape)
            new_shape[0] = nworld
            wp_array = {1: wp.array, 2: wp.array2d, 3: wp.array3d, 4: wp.array4d}[len(new_shape)]
            dst = wp_array(shape=new_shape, dtype=x.dtype, device=x.device)
            src_flat = x.flatten()
            dst_flat = dst.flatten()
            n_elems_per_world = dst_flat.shape[0] // nworld
            wp.launch(
                repeat_array_kernel,
                dim=dst_flat.shape[0],
                inputs=[src_flat, n_elems_per_world],
                outputs=[dst_flat],
                device=x.device,
            )
            return dst

        for field in mj_model.__dataclass_fields__:
            if field in model_fields_to_expand:
                array = getattr(mj_model, field)
                setattr(mj_model, field, tile(array))

    def _convert_to_mjc(self) -> None:
        model = self.model
        config = self.config
        mujoco = self._mujoco
        mujoco_warp = self._mujoco_warp

        # MuJoCo needs at least one articulated entry point to build qpos/qvel state.
        if not model.joint_count:
            raise ValueError("The model must have at least one joint to convert it to MuJoCo.")

        # Multi-world packing is only implemented through the template-world path below.
        if not config.separate_worlds and model.world_count > 1:
            raise ValueError("separate_worlds=False is only supported for single-world models.")
        if config.separate_worlds:
            self._validate_model_for_separate_worlds(model)

        # Every controllable axis is represented by MuJoCo actuators; start from a shared default
        # parameter block and specialize per-axis later.
        actuator_args = {
            "gear": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "trntype": mujoco.mjtTrn.mjTRN_JOINT,
            "gainprm": [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "biasprm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "dyntype": mujoco.mjtDyn.mjDYN_NONE,
            "gaintype": mujoco.mjtGain.mjGAIN_FIXED,
            "biastype": mujoco.mjtBias.mjBIAS_AFFINE,
        }
        if config.default_actuator_gear is not None:
            actuator_args["gear"][0] = config.default_actuator_gear
        actuator_gears = config.actuator_gears or {}

        solver = self._resolve_mj_opt(
            config.solver, {"cg": mujoco.mjtSolver.mjSOL_CG, "newton": mujoco.mjtSolver.mjSOL_NEWTON}, "solver"
        )
        integrator = self._resolve_mj_opt(
            config.integrator,
            {
                "euler": mujoco.mjtIntegrator.mjINT_EULER,
                "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            },
            "integrator",
        )
        cone = self._resolve_mj_opt(
            config.cone,
            {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC},
            "cone",
        )

        def quat_to_mjc(q):
            return [q[3], q[0], q[1], q[2]]

        disableflags = 0
        if config.disable_contacts:
            disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        # Build the canonical single-world MuJoCo spec first. Repeated worlds are expanded after compile.
        spec = mujoco.MjSpec()
        spec.option.disableflags = disableflags
        spec.option.gravity = np.array([*model.gravity.numpy()[0]])
        spec.option.solver = solver
        spec.option.integrator = integrator
        spec.option.iterations = config.iterations
        spec.option.ls_iterations = config.ls_iterations
        spec.option.cone = cone
        spec.option.impratio = config.impratio
        spec.option.tolerance = config.tolerance
        spec.option.ls_tolerance = config.ls_tolerance
        spec.option.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
        spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO

        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_articulation = model.joint_articulation.numpy()
        joint_parent_xform = model.joint_X_p.numpy()
        joint_child_xform = model.joint_X_c.numpy()
        joint_limit_lower = model.joint_limit_lower.numpy()
        joint_limit_upper = model.joint_limit_upper.numpy()
        joint_limit_ke = model.joint_limit_ke.numpy()
        joint_limit_kd = model.joint_limit_kd.numpy()
        joint_type = model.joint_type.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        joint_target_kd = model.joint_target_kd.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_armature = model.joint_armature.numpy()
        joint_effort_limit = model.joint_effort_limit.numpy()
        joint_friction = model.joint_friction.numpy()
        joint_world = model.joint_world.numpy()

        body_mass = model.body_mass.numpy()
        body_inertia = model.body_inertia.numpy()
        body_com = model.body_com.numpy()
        body_world = model.body_world.numpy()

        shape_transform = model.shape_transform.numpy()
        shape_type = model.shape_type.numpy()
        shape_size = model.shape_scale.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_world = model.shape_world.numpy()
        shape_mu = model.shape_material_mu.numpy()
        shape_ke = model.shape_material_ke.numpy()
        shape_kd = model.shape_material_kd.numpy()
        shape_torsional_friction = model.shape_material_mu_torsional.numpy()
        shape_rolling_friction = model.shape_material_mu_rolling.numpy()

        mujoco_attrs = getattr(model, "mujoco", None)

        def get_custom_attribute(name: str):
            if mujoco_attrs is None:
                return None
            attr = getattr(mujoco_attrs, name, None)
            return attr.numpy() if attr is not None else None

        shape_condim = get_custom_attribute("condim")
        shape_priority = get_custom_attribute("geom_priority")
        shape_geom_solimp = get_custom_attribute("geom_solimp")
        shape_geom_solmix = get_custom_attribute("geom_solmix")
        shape_geom_gap = get_custom_attribute("geom_gap")
        joint_dof_limit_margin = get_custom_attribute("limit_margin")
        joint_solimp_limit = get_custom_attribute("solimplimit")
        joint_dof_solref = get_custom_attribute("solreffriction")
        joint_dof_solimp = get_custom_attribute("solimpfriction")
        joint_stiffness = get_custom_attribute("dof_passive_stiffness")
        joint_damping = get_custom_attribute("dof_passive_damping")
        joint_actgravcomp = get_custom_attribute("jnt_actgravcomp")

        eq_constraint_type = model.equality_constraint_type.numpy()
        eq_constraint_body1 = model.equality_constraint_body1.numpy()
        eq_constraint_body2 = model.equality_constraint_body2.numpy()
        eq_constraint_anchor = model.equality_constraint_anchor.numpy()
        eq_constraint_torquescale = model.equality_constraint_torquescale.numpy()
        eq_constraint_relpose = model.equality_constraint_relpose.numpy()
        eq_constraint_joint1 = model.equality_constraint_joint1.numpy()
        eq_constraint_joint2 = model.equality_constraint_joint2.numpy()
        eq_constraint_polycoef = model.equality_constraint_polycoef.numpy()
        eq_constraint_enabled = model.equality_constraint_enabled.numpy()
        eq_constraint_world = model.equality_constraint_world.numpy()
        eq_constraint_solref = get_custom_attribute("eq_solref")

        axis_to_actuator = np.zeros((model.joint_dof_count, 2), dtype=np.int32) - 1
        actuator_count = 0

        supported_joint_types = {
            JointType.FREE,
            JointType.BALL,
            JointType.PRISMATIC,
            JointType.REVOLUTE,
            JointType.D6,
        }
        geom_type_mapping = {
            GeoType.SPHERE: mujoco.mjtGeom.mjGEOM_SPHERE,
            GeoType.PLANE: mujoco.mjtGeom.mjGEOM_PLANE,
            GeoType.CAPSULE: mujoco.mjtGeom.mjGEOM_CAPSULE,
            GeoType.CYLINDER: mujoco.mjtGeom.mjGEOM_CYLINDER,
            GeoType.BOX: mujoco.mjtGeom.mjGEOM_BOX,
            GeoType.ELLIPSOID: mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            GeoType.MESH: mujoco.mjtGeom.mjGEOM_MESH,
            GeoType.CONVEX_MESH: mujoco.mjtGeom.mjGEOM_MESH,
        }

        mj_bodies = [spec.worldbody]
        body_mapping = {-1: 0}
        shape_mapping = {}
        next_mocap_index = 0
        body_name_counts = {}
        joint_names = {}

        # For separate worlds we compile one template world and later tile its buffers across worlds.
        if config.separate_worlds:
            non_negatives = body_world[body_world >= 0]
            first_world = np.min(non_negatives) if len(non_negatives) > 0 else -1
            selected_shapes = np.where((shape_world == first_world) | (shape_world < 0))[0].astype(np.int32)
            selected_bodies = np.where((body_world == first_world) | (body_world < 0))[0].astype(np.int32)
            selected_joints = np.where((joint_world == first_world) | (joint_world < 0))[0].astype(np.int32)
            selected_constraints = np.where((eq_constraint_world == first_world) | (eq_constraint_world < 0))[0].astype(
                np.int32
            )
        else:
            first_world = 0
            selected_shapes = np.arange(model.shape_count, dtype=np.int32)
            selected_bodies = np.arange(model.body_count, dtype=np.int32)
            selected_joints = np.arange(model.joint_count, dtype=np.int32)
            selected_constraints = np.arange(model.equality_constraint_count, dtype=np.int32)

        first_env_shapes = np.where(shape_world == first_world)[0]
        joints_loop = selected_joints[joint_articulation[selected_joints] == -1]
        joints_non_loop = selected_joints[joint_articulation[selected_joints] >= 0]
        joints_simple = [(joint_parent[i], joint_child[i]) for i in joints_non_loop]
        joint_order = topological_sort(joints_simple, use_dfs=True, custom_indices=joints_non_loop)

        colliding_shapes = selected_shapes[shape_flags[selected_shapes] & ShapeFlags.COLLIDE_SHAPES != 0]
        colliding_shapes_per_world = len(colliding_shapes)
        body_filters = self.find_body_collision_filter_pairs(model, selected_bodies, colliding_shapes)
        shape_color = self.color_collision_shapes(
            model,
            colliding_shapes,
            visualize_graph=False,
            shape_keys=model.shape_label,
        )
        selected_shapes_set = set(selected_shapes)

        def add_geoms(source_body_id: int):
            # Emit sites/geoms for one source body and remember the source-shape -> geom name mapping
            # so compiled ids can be recovered after MuJoCo finalizes the spec.
            body = mj_bodies[body_mapping[source_body_id]]
            shapes = model.body_shapes.get(source_body_id)
            if not shapes:
                return
            for shape in shapes:
                if shape not in selected_shapes_set:
                    continue
                is_site = shape_flags[shape] & ShapeFlags.SITE
                if config.include_sites is False and is_site:
                    continue
                if is_site == 0 and not (shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES):
                    continue
                stype = shape_type[shape]
                name = f"{model.shape_label[shape]}_{shape}"

                if is_site:
                    supported_site_types = {GeoType.SPHERE, GeoType.CAPSULE, GeoType.CYLINDER, GeoType.BOX}
                    site_geom_type = stype if stype in supported_site_types else GeoType.SPHERE
                    tf = wp.transform(*shape_transform[shape])
                    site_params = {"type": geom_type_mapping[site_geom_type], "name": name, "pos": tf.p, "quat": quat_to_mjc(tf.q)}
                    size = shape_size[shape]
                    if np.any(size > 0.0):
                        nonzero = size[size > 0.0][0]
                        size[size == 0.0] = nonzero
                        site_params["size"] = size
                    else:
                        site_params["size"] = [0.01, 0.01, 0.01]
                    site_params["rgba"] = [0.0, 1.0, 0.0, 0.5] if shape_flags[shape] & ShapeFlags.VISIBLE else [0.0, 1.0, 0.0, 0.0]
                    body.add_site(**site_params)
                    continue

                if stype == GeoType.PLANE and source_body_id != -1:
                    raise ValueError("Planes can only be attached to static bodies")

                geom_params = {"type": geom_type_mapping[stype], "name": name}
                tf = wp.transform(*shape_transform[shape])
                if stype == GeoType.MESH or stype == GeoType.CONVEX_MESH:
                    mesh_src = model.shape_source[shape]
                    maxhullvert = getattr(mesh_src, "maxhullvert", newton.Mesh.MAX_HULL_VERTICES)
                    size = shape_size[shape]
                    vertices = mesh_src.vertices * size
                    spec.add_mesh(
                        name=name,
                        uservert=vertices.flatten(),
                        userface=mesh_src.indices.flatten(),
                        maxhullvert=maxhullvert,
                    )
                    geom_params["meshname"] = name
                geom_params["pos"] = tf.p
                geom_params["quat"] = quat_to_mjc(tf.q)
                size = shape_size[shape]
                if np.any(size > 0.0):
                    nonzero = size[size > 0.0][0]
                    size[size == 0.0] = nonzero
                    geom_params["size"] = size
                else:
                    geom_params["size"] = [5.0, 5.0, 5.0]
                    geom_params["rgba"] = [0.0, 0.3, 0.6, 1.0]

                if not (shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES):
                    geom_params["contype"] = 0
                    geom_params["conaffinity"] = 0
                else:
                    color = shape_color[shape]
                    if color < 32:
                        contype = 1 << color
                        geom_params["contype"] = contype
                        geom_params["conaffinity"] = np.iinfo(np.int32).max & ~contype

                geom_params["friction"] = [shape_mu[shape], shape_torsional_friction[shape], shape_rolling_friction[shape]]
                geom_params["solref"] = convert_solref(float(shape_ke[shape]), float(shape_kd[shape]), 1.0, 1.0)
                if shape_condim is not None:
                    geom_params["condim"] = shape_condim[shape]
                if shape_priority is not None:
                    geom_params["priority"] = shape_priority[shape]
                if shape_geom_solimp is not None:
                    geom_params["solimp"] = shape_geom_solimp[shape]
                if shape_geom_solmix is not None:
                    geom_params["solmix"] = shape_geom_solmix[shape]
                if shape_geom_gap is not None:
                    geom_params["gap"] = shape_geom_gap[shape]

                body.add_geom(**geom_params)
                shape_mapping[shape] = name

        add_geoms(-1)

        joint_mjc_dof_start = np.full(len(selected_joints), -1, dtype=np.int32)
        dof_to_mjc_joint = np.full(model.joint_dof_count // model.world_count, -1, dtype=np.int32)
        num_dofs = 0
        num_mjc_joints = 0

        # Build bodies in topological order so MuJoCo sees parents before children.
        for j in joint_order:
            parent, child = int(joint_parent[j]), int(joint_child[j])
            body_mapping[child] = len(mj_bodies)
            fixed_base = parent == -1 and joint_type[j] == JointType.FIXED

            tf = wp.transform(*joint_parent_xform[j]) * wp.transform_inverse(wp.transform(*joint_child_xform[j]))
            child_xform = wp.transform(*joint_child_xform[j])
            joint_pos = child_xform.p
            joint_rot = child_xform.q

            name = model.body_label[child]
            if name not in body_name_counts:
                body_name_counts[name] = 1
            else:
                while name in body_name_counts:
                    body_name_counts[name] += 1
                    name = f"{name}_{body_name_counts[name]}"

            inertia = body_inertia[child]
            body = mj_bodies[body_mapping[parent]].add_body(
                name=name,
                pos=tf.p,
                quat=quat_to_mjc(tf.q),
                mass=body_mass[child],
                ipos=body_com[child, :],
                fullinertia=[inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]],
                explicitinertial=True,
                mocap=fixed_base,
            )
            mj_bodies.append(body)
            if fixed_base:
                next_mocap_index += 1

            j_type = joint_type[j]
            qd_start = joint_qd_start[j]
            joint_name = model.joint_label[j]
            if joint_name not in joint_names:
                joint_names[joint_name] = 1
            else:
                while joint_name in joint_names:
                    joint_names[joint_name] += 1
                    joint_name = f"{joint_name}_{joint_names[joint_name]}"

            joint_mjc_dof_start[j] = num_dofs

            if j_type == JointType.FREE:
                body.add_joint(name=joint_name, type=mujoco.mjtJoint.mjJNT_FREE, damping=0.0, limited=False)
                for i in range(6):
                    dof_to_mjc_joint[qd_start + i] = num_mjc_joints
                num_dofs += 6
                num_mjc_joints += 1
            elif j_type == JointType.BALL:
                body.add_joint(
                    name=joint_name,
                    type=mujoco.mjtJoint.mjJNT_BALL,
                    axis=wp.quat_rotate(joint_rot, wp.vec3(1.0, 0.0, 0.0)),
                    pos=joint_pos,
                    damping=0.0,
                    limited=False,
                    armature=joint_armature[qd_start],
                    frictionloss=joint_friction[qd_start],
                )
                for i in range(3):
                    dof_to_mjc_joint[qd_start + i] = num_mjc_joints
                num_dofs += 3
                num_mjc_joints += 1
                for i in range(3):
                    ai = qd_start + i
                    kp = joint_target_ke[ai]
                    kd = joint_target_kd[ai]
                    effort_limit = joint_effort_limit[ai]
                    gear = actuator_gears.get(joint_name)
                    args = dict(actuator_args)
                    args["gear"] = [0.0] * 6
                    args["gear"][i] = gear if gear is not None else 1.0
                    args["forcerange"] = [-effort_limit, effort_limit]
                    args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                    spec.add_actuator(target=joint_name, **args)
                    axis_to_actuator[ai, 0] = actuator_count
                    actuator_count += 1
                    args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                    spec.add_actuator(target=joint_name, **args)
                    axis_to_actuator[ai, 1] = actuator_count
                    actuator_count += 1
            elif j_type in supported_joint_types:
                lin_axis_count, ang_axis_count = joint_dof_dim[j]
                num_dofs += lin_axis_count + ang_axis_count

                for i in range(lin_axis_count + ang_axis_count):
                    ai = qd_start + i
                    axis = wp.quat_rotate(joint_rot, wp.vec3(*joint_axis[ai]))
                    joint_params = {"armature": joint_armature[ai], "pos": joint_pos, "frictionloss": joint_friction[ai]}
                    if joint_dof_limit_margin is not None:
                        joint_params["margin"] = joint_dof_limit_margin[ai]
                    if joint_stiffness is not None:
                        joint_params["stiffness"] = joint_stiffness[ai]
                    if joint_damping is not None:
                        joint_params["damping"] = joint_damping[ai]
                    if joint_actgravcomp is not None:
                        joint_params["actgravcomp"] = joint_actgravcomp[ai]
                    lower, upper = joint_limit_lower[ai], joint_limit_upper[ai]
                    joint_params["limited"] = not (lower <= -newton.MAXVAL and upper >= newton.MAXVAL)
                    joint_params["range"] = (lower, upper) if i < lin_axis_count else (np.rad2deg(lower), np.rad2deg(upper))
                    if joint_limit_ke[ai] > 0:
                        joint_params["solref_limit"] = (-joint_limit_ke[ai], -joint_limit_kd[ai])
                    if joint_solimp_limit is not None:
                        joint_params["solimp_limit"] = joint_solimp_limit[ai]
                    if joint_dof_solref is not None:
                        joint_params["solref_friction"] = joint_dof_solref[ai]
                    if joint_dof_solimp is not None:
                        joint_params["solimp_friction"] = joint_dof_solimp[ai]
                    effort_limit = joint_effort_limit[ai]
                    joint_params["actfrclimited"] = True
                    joint_params["actfrcrange"] = (-effort_limit, effort_limit)

                    axname = joint_name
                    if lin_axis_count > 1 or ang_axis_count > 1:
                        axname += "_lin" if i < lin_axis_count else "_ang"
                    if (i < lin_axis_count and lin_axis_count > 1) or (i >= lin_axis_count and ang_axis_count > 1):
                        axname += str(i if i < lin_axis_count else i - lin_axis_count)

                    body.add_joint(
                        name=axname,
                        type=mujoco.mjtJoint.mjJNT_SLIDE if i < lin_axis_count else mujoco.mjtJoint.mjJNT_HINGE,
                        axis=axis,
                        **joint_params,
                    )
                    dof_to_mjc_joint[ai] = num_mjc_joints
                    num_mjc_joints += 1

                    kp = joint_target_ke[ai]
                    kd = joint_target_kd[ai]
                    gear = actuator_gears.get(axname)
                    args = dict(actuator_args)
                    if gear is not None:
                        args["gear"] = [gear, 0.0, 0.0, 0.0, 0.0, 0.0]
                    args["gainprm"] = [kp, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    args["biasprm"] = [0, -kp, 0, 0, 0, 0, 0, 0, 0, 0]
                    spec.add_actuator(target=axname, **args)
                    axis_to_actuator[ai, 0] = actuator_count
                    actuator_count += 1
                    args["gainprm"] = [kd, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    args["biasprm"] = [0, 0, -kd, 0, 0, 0, 0, 0, 0, 0]
                    spec.add_actuator(target=axname, **args)
                    axis_to_actuator[ai, 1] = actuator_count
                    actuator_count += 1
            elif j_type != JointType.FIXED:
                raise NotImplementedError(f"Joint type {j_type} is not supported yet")

            add_geoms(child)

        # Equality constraints are authored after bodies/joints so all referenced names already exist.
        for i in selected_constraints:
            constraint_type = eq_constraint_type[i]
            if constraint_type == EqType.CONNECT:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_CONNECT
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.body_label[eq_constraint_body1[i]]
                eq.name2 = model.body_label[eq_constraint_body2[i]]
                eq.data[0:3] = eq_constraint_anchor[i]
                if eq_constraint_solref is not None:
                    eq.solref = eq_constraint_solref[i]
            elif constraint_type == EqType.JOINT:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_JOINT)
                eq.type = mujoco.mjtEq.mjEQ_JOINT
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.joint_label[eq_constraint_joint1[i]]
                eq.name2 = model.joint_label[eq_constraint_joint2[i]]
                eq.data[0:5] = eq_constraint_polycoef[i]
                if eq_constraint_solref is not None:
                    eq.solref = eq_constraint_solref[i]
            elif constraint_type == EqType.WELD:
                eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.active = eq_constraint_enabled[i]
                eq.name1 = model.body_label[eq_constraint_body1[i]]
                eq.name2 = model.body_label[eq_constraint_body2[i]]
                cns_relpose = wp.transform(*eq_constraint_relpose[i])
                eq.data[0:3] = eq_constraint_anchor[i]
                eq.data[3:6] = wp.transform_get_translation(cns_relpose)
                eq.data[6:10] = wp.transform_get_rotation(cns_relpose)
                eq.data[10] = eq_constraint_torquescale[i]
                if eq_constraint_solref is not None:
                    eq.solref = eq_constraint_solref[i]

        mjc_eq_to_newton_jnt = {}
        for j in joints_loop:
            eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
            eq.type = mujoco.mjtEq.mjEQ_CONNECT
            eq.active = True
            eq.name1 = model.body_label[joint_parent[j]]
            eq.name2 = model.body_label[joint_child[j]]
            eq.data[0:3] = joint_parent_xform[j][:3]
            mjc_eq_to_newton_jnt[eq.id] = j

            eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_BODY)
            eq.type = mujoco.mjtEq.mjEQ_CONNECT
            eq.active = True
            eq.name1 = model.body_label[joint_child[j]]
            eq.name2 = model.body_label[joint_parent[j]]
            eq.data[0:3] = joint_child_xform[j][:3]
            mjc_eq_to_newton_jnt[eq.id] = j

        if len(spec.geoms) != colliding_shapes_per_world:
            raise ValueError("The number of MuJoCo geoms does not match the number of colliding shapes.")
        if len(spec.bodies) != len(selected_bodies) + 1:
            raise ValueError("The number of MuJoCo bodies does not match the number of selected source bodies.")

        for b1, b2 in body_filters:
            mb1, mb2 = body_mapping[b1], body_mapping[b2]
            spec.add_exclude(bodyname1=spec.bodies[mb1].name, bodyname2=spec.bodies[mb2].name)

        self._init_pairs(model, spec, shape_mapping, first_world)

        # Compile the CPU-side MuJoCo model once, seed state, then upload to mujoco_warp.
        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)
        self.update_mjc_data(self.mj_data, model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        geom_to_shape_idx = {}
        for shape, geom_name in shape_mapping.items():
            geom_idx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_idx >= 0:
                geom_to_shape_idx[geom_idx] = shape

        with wp.ScopedDevice(model.device):
            self.mjw_model = mujoco_warp.put_model(self.mj_model)
            if not hasattr(self.mjw_model, "mesh_pos"):
                self.mjw_model.mesh_pos = wp.array(self.mj_model.mesh_pos, dtype=wp.vec3)

            nworld = model.world_count if config.separate_worlds else 1
            # Geom ids are stable after compile; use them to recover per-world shape mappings for runtime bridges.
            geom_to_shape_idx_np = np.full((self.mj_model.ngeom,), -1, dtype=np.int32)
            first_env_shape_base = int(np.min(first_env_shapes)) if len(first_env_shapes) > 0 else 0
            self._shapes_per_world = len(first_env_shapes)
            self._first_env_shape_base = first_env_shape_base
            geom_is_static_np = np.zeros((self.mj_model.ngeom,), dtype=bool)
            for geom_idx, abs_shape_idx in geom_to_shape_idx.items():
                if shape_world[abs_shape_idx] < 0:
                    geom_to_shape_idx_np[geom_idx] = abs_shape_idx
                    geom_is_static_np[geom_idx] = True
                else:
                    geom_to_shape_idx_np[geom_idx] = abs_shape_idx - first_env_shape_base

            geom_to_shape_idx_wp = wp.array(geom_to_shape_idx_np, dtype=wp.int32)
            geom_is_static_wp = wp.array(geom_is_static_np, dtype=bool)

            self.mjc_geom_to_source_shape = wp.full((nworld, self.mj_model.ngeom), -1, dtype=wp.int32)
            if self.mjw_model.geom_pos.size:
                wp.launch(
                    update_shape_mappings_kernel,
                    dim=(nworld, self.mj_model.ngeom),
                    inputs=[geom_to_shape_idx_wp, geom_is_static_wp, self._shapes_per_world, first_env_shape_base],
                    outputs=[self.mjc_geom_to_source_shape],
                    device=model.device,
                )
            self.mjc_geom_to_newton_shape = self.mjc_geom_to_source_shape

            nbody = self.mj_model.nbody
            bodies_per_world = model.body_count // model.world_count
            self.mjc_body_to_source_body = np.full((nworld, nbody), -1, dtype=np.int32)
            for source_body, mjc_body in body_mapping.items():
                if source_body >= 0:
                    source_body_in_world = source_body % bodies_per_world
                    for w in range(nworld):
                        self.mjc_body_to_source_body[w, mjc_body] = w * bodies_per_world + source_body_in_world
            self.mjc_body_to_source_body = wp.array(self.mjc_body_to_source_body, dtype=wp.int32)
            self.mjc_body_to_newton = self.mjc_body_to_source_body

            njnt = self.mj_model.njnt
            joints_per_world = model.joint_count // model.world_count
            dofs_per_world = model.joint_dof_count // model.world_count

            nmocap = self.mj_model.nmocap
            if nmocap > 0:
                mjc_mocap_to_source_joint_np = np.full((nworld, nmocap), -1, dtype=np.int32)
                body_mocapid = self.mj_model.body_mocapid
                mjc_body_to_source_np = self.mjc_body_to_source_body.numpy()
                for mjc_body in range(nbody):
                    mocap_idx = body_mocapid[mjc_body]
                    if mocap_idx >= 0:
                        source_body = mjc_body_to_source_np[0, mjc_body]
                        if source_body >= 0:
                            source_body_template = source_body % bodies_per_world
                            for j in range(joints_per_world):
                                if joint_child[j] == source_body_template:
                                    for w in range(nworld):
                                        mjc_mocap_to_source_joint_np[w, mocap_idx] = w * joints_per_world + j
                                    break
                self.mjc_mocap_to_source_joint = wp.array(mjc_mocap_to_source_joint_np, dtype=wp.int32)
                self.mjc_mocap_to_newton_jnt = self.mjc_mocap_to_source_joint

            mjc_jnt_to_source_joint_np = np.full((nworld, njnt), -1, dtype=np.int32)
            for template_dof, mjc_jnt in enumerate(dof_to_mjc_joint):
                if mjc_jnt >= 0:
                    for j in selected_joints:
                        j_dof_start = joint_qd_start[j] % dofs_per_world
                        j_lin_count, j_ang_count = joint_dof_dim[j]
                        if j_dof_start <= template_dof < j_dof_start + j_lin_count + j_ang_count:
                            for w in range(nworld):
                                mjc_jnt_to_source_joint_np[w, mjc_jnt] = w * joints_per_world + j
                            break
            self.mjc_jnt_to_source_joint = wp.array(mjc_jnt_to_source_joint_np, dtype=wp.int32)
            self.mjc_jnt_to_newton_jnt = self.mjc_jnt_to_source_joint

            mjc_jnt_to_source_dof_np = np.full((nworld, njnt), -1, dtype=np.int32)
            for template_dof, mjc_jnt in enumerate(dof_to_mjc_joint):
                if mjc_jnt >= 0:
                    for w in range(nworld):
                        mjc_jnt_to_source_dof_np[w, mjc_jnt] = w * dofs_per_world + template_dof
            self.mjc_jnt_to_source_dof = wp.array(mjc_jnt_to_source_dof_np, dtype=wp.int32)
            self.mjc_jnt_to_newton_dof = self.mjc_jnt_to_source_dof

            nv = self.mj_model.nv
            mjc_dof_to_source_dof_np = np.full((nworld, nv), -1, dtype=np.int32)
            for j, mjc_dof_start in enumerate(joint_mjc_dof_start):
                if mjc_dof_start >= 0:
                    source_dof_start = joint_qd_start[j]
                    lin_count, ang_count = joint_dof_dim[j]
                    for d in range(lin_count + ang_count):
                        mjc_dof = mjc_dof_start + d
                        template_source_dof = (source_dof_start % dofs_per_world) + d
                        for w in range(nworld):
                            mjc_dof_to_source_dof_np[w, mjc_dof] = w * dofs_per_world + template_source_dof
            self.mjc_dof_to_source_dof = wp.array(mjc_dof_to_source_dof_np, dtype=wp.int32)
            self.mjc_dof_to_newton_dof = self.mjc_dof_to_source_dof

            nu = self.mj_model.nu
            mjc_actuator_to_source_axis_np = np.full((nworld, nu), -1, dtype=np.int32)
            for source_axis in range(axis_to_actuator.shape[0]):
                template_axis = source_axis % dofs_per_world
                for act_type in range(2):
                    mjc_act = axis_to_actuator[source_axis, act_type]
                    if mjc_act >= 0:
                        for w in range(nworld):
                            world_axis = w * dofs_per_world + template_axis
                            mjc_actuator_to_source_axis_np[w, mjc_act] = world_axis if act_type == 0 else -(world_axis + 2)
            self.mjc_actuator_to_source_axis = wp.array(mjc_actuator_to_source_axis_np, dtype=wp.int32)
            self.mjc_actuator_to_newton_axis = self.mjc_actuator_to_source_axis

            neq = self.mj_model.neq
            eq_constraints_per_world = model.equality_constraint_count // model.world_count
            mjc_eq_to_source_eq_np = np.full((nworld, neq), -1, dtype=np.int32)
            mjc_eq_to_source_joint_np = np.full((nworld, neq), -1, dtype=np.int32)
            for mjc_eq, source_eq in enumerate(selected_constraints):
                template_eq = source_eq % eq_constraints_per_world if eq_constraints_per_world > 0 else source_eq
                for w in range(nworld):
                    mjc_eq_to_source_eq_np[w, mjc_eq] = w * eq_constraints_per_world + template_eq
            for mjc_eq, source_jnt in mjc_eq_to_newton_jnt.items():
                for w in range(nworld):
                    mjc_eq_to_source_joint_np[w, mjc_eq] = w * joints_per_world + source_jnt
            self.mjc_eq_to_source_eq = wp.array(mjc_eq_to_source_eq_np, dtype=wp.int32)
            self.mjc_eq_to_source_joint = wp.array(mjc_eq_to_source_joint_np, dtype=wp.int32)
            self.mjc_eq_to_newton_eq = self.mjc_eq_to_source_eq
            self.mjc_eq_to_newton_jnt = self.mjc_eq_to_source_joint

            self.mjw_model.opt.ls_parallel = False

            # Size warp-side buffers from the compiled model/data and honor any
            # user-provided minimum capacities. Large articulated scenes such as
            # G1 can exceed the initial constraint count at runtime if these
            # buffers are undersized.
            if config.disable_contacts:
                nconmax = 0
            elif config.nconmax is None:
                nconmax = self.mj_data.ncon
            elif config.nconmax < self.mj_data.ncon:
                warnings.warn(
                    f"[WARNING] Value for nconmax is changed from {config.nconmax} to {self.mj_data.ncon} following an MjWarp requirement.",
                    stacklevel=2,
                )
                nconmax = self.mj_data.ncon
            else:
                nconmax = config.nconmax

            if config.njmax is None:
                njmax = self.mj_data.nefc
            elif config.njmax < self.mj_data.nefc:
                warnings.warn(
                    f"[WARNING] Value for njmax is changed from {config.njmax} to {self.mj_data.nefc} following an MjWarp requirement.",
                    stacklevel=2,
                )
                njmax = self.mj_data.nefc
            else:
                njmax = config.njmax

            self.mjw_data = mujoco_warp.put_data(
                self.mj_model,
                self.mj_data,
                nworld=nworld,
                nconmax=nconmax,
                njmax=njmax,
            )
            self.expand_model_fields(self.mjw_model, nworld)
