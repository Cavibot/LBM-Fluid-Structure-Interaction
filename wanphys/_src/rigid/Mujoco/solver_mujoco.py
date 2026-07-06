from __future__ import annotations

from typing import Any

from wanphys._src.rigid.model import RigidModel
from wanphys._src.rigid.solver import RigidSolver
from wanphys._src.rigid.state import RigidState

from .mappings import MujocoMappingsMixin
from .backend import MujocoBackendMixin
from .bridge_state import MujocoStateBridgeMixin
from .bridge_control import MujocoControlBridgeMixin
from .bridge_contact import MujocoContactBridgeMixin
from .config import MujocoConfig
from .sync import MujocoSyncMixin


class _MujocoRuntime(
    MujocoMappingsMixin,
    MujocoBackendMixin,
    MujocoStateBridgeMixin,
    MujocoControlBridgeMixin,
    MujocoContactBridgeMixin,
    MujocoSyncMixin,
):
    """Internal MuJoCo runtime owned by the WanPhys solver facade."""

    def __init__(
        self,
        model: RigidModel,
        config: MujocoConfig,
    ):
        self.model = model
        self.device = model.device

        # Transitional escape hatch: the MuJoCo bridge still targets the
        # compatibility model view exposed by WanPhys.
        self._source_model = model.as_newton_model()
        # Keep the legacy attribute alive for any remaining compatibility paths.
        self._newton_model = self._source_model

        self._init_mappings()
        self._init_backend(config)


class WanPhysMujocoSolver(RigidSolver):
    """WanPhys-facing MuJoCo solver facade.

    The runtime is split into small bridge modules:
    - backend: MuJoCo / mujoco_warp initialization and stepping
    - bridge_state: state synchronization
    - bridge_control: control and force application
    - bridge_contact: contact import/export
    - sync: live model-property updates

    This keeps the external behaviour stable while making the Python
    structure easier to evolve inside WanPhys.
    """

    def __init__(
        self,
        model: RigidModel,
        *,
        iterations: int = 20,
        ls_iterations: int = 10,
        solver: int | str = "cg",
        integrator: int | str = "implicitfast",
        cone: int | str = "pyramidal",
        impratio: float = 1.0,
        separate_worlds: bool | None = None,
        disable_contacts: bool = False,
        use_mujoco_contacts: bool = True,
        update_data_interval: int = 1,
        default_actuator_gear: float | None = None,
        actuator_gears: dict[str, float] | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        tolerance: float = 1e-6,
        ls_tolerance: float = 0.01,
        include_sites: bool = True,
        use_mujoco_cpu: bool = False,
        **newton_solver_options: Any,
    ):
        self.model = model
        self.device = model.device
        self._step = 0
        self._cpu_adapter: RigidSolver | None = None
        self.config = MujocoConfig(
            iterations=iterations,
            ls_iterations=ls_iterations,
            solver=solver,
            integrator=integrator,
            cone=cone,
            impratio=impratio,
            separate_worlds=separate_worlds,
            disable_contacts=disable_contacts,
            use_mujoco_contacts=use_mujoco_contacts,
            update_data_interval=update_data_interval,
            default_actuator_gear=default_actuator_gear,
            actuator_gears=dict(actuator_gears or {}),
            nconmax=nconmax,
            njmax=njmax,
            tolerance=tolerance,
            ls_tolerance=ls_tolerance,
            include_sites=include_sites,
        )

        if use_mujoco_cpu:
            from newton.solvers import SolverMuJoCo
            from wanphys._src.rigid.solver import _NewtonSolverAdapter

            self._cpu_adapter = _NewtonSolverAdapter(
                model,
                SolverMuJoCo,
                iterations=iterations,
                ls_iterations=ls_iterations,
                solver=solver,
                integrator=integrator,
                cone=cone,
                impratio=impratio,
                separate_worlds=separate_worlds,
                disable_contacts=disable_contacts,
                use_mujoco_contacts=use_mujoco_contacts,
                update_data_interval=update_data_interval,
                nconmax=nconmax,
                njmax=njmax,
                tolerance=tolerance,
                ls_tolerance=ls_tolerance,
                include_sites=include_sites,
                use_mujoco_cpu=True,
                **newton_solver_options,
            )
            self._runtime = None
            return

        if newton_solver_options:
            options = ", ".join(sorted(newton_solver_options))
            raise TypeError(f"Unsupported WanPhys MuJoCo bridge options: {options}")

        self._runtime = _MujocoRuntime(
            model,
            self.config,
        )

    @property
    def mj_model(self):
        if self._cpu_adapter is not None:
            return getattr(self._cpu_adapter._newton_backend, "mj_model", None)
        return self._runtime.mj_model

    @property
    def mjw_model(self):
        if self._cpu_adapter is not None:
            return getattr(self._cpu_adapter._newton_backend, "mjw_model", None)
        return self._runtime.mjw_model

    @property
    def mjw_data(self):
        if self._cpu_adapter is not None:
            return getattr(self._cpu_adapter._newton_backend, "mjw_data", None)
        return self._runtime.mjw_data

    def step(
        self,
        state_in: RigidState,
        state_out: RigidState,
        control,
        contacts,
        dt: float,
    ) -> None:
        if self._cpu_adapter is not None:
            self._cpu_adapter.step(state_in, state_out, control, contacts, dt)
            return

        ns_in = state_in.as_newton_state()
        ns_out = state_out.as_newton_state()
        runtime = self._runtime

        runtime._apply_mjc_control(ns_in, control)

        if runtime.config.update_data_interval > 0 and self._step % runtime.config.update_data_interval == 0:
            runtime._update_mjc_data(ns_in)

        runtime.mjw_model.opt.timestep.fill_(dt)

        if runtime.config.use_mujoco_contacts:
            runtime._step_backend()
        else:
            runtime._convert_contacts_to_mjwarp(ns_in, contacts)
            runtime._step_backend()
        # After stepping, we need to sync the new mujoco state back to the Newton-compatible view.
        runtime._update_newton_state(ns_out)
        self._step += 1

    def notify_model_changed(self, flags: int):
        if self._cpu_adapter is not None:
            backend = self._cpu_adapter._newton_backend
            notify = getattr(backend, "notify_model_changed", None)
            if notify is not None:
                notify(flags)
            return
        self._runtime.notify_model_changed(flags)

    def update_contacts(self, contacts, state: RigidState | None = None):
        if self._cpu_adapter is not None:
            self._cpu_adapter.update_contacts(contacts, state)
            return
        self._runtime.update_contacts(contacts)

    def expand_model_fields(self, mjw_model, nworld: int):
        if self._cpu_adapter is not None:
            backend = self._cpu_adapter._newton_backend
            expand = getattr(backend, "expand_model_fields", None)
            if expand is None:
                raise RuntimeError("MuJoCo CPU backend does not expose expand_model_fields.")
            expand(mjw_model, nworld)
            return
        self._runtime.expand_model_fields(mjw_model, nworld)

    def get_max_contact_count(self, default: int | None = None) -> int | None:
        if self._cpu_adapter is not None:
            return self._cpu_adapter.get_max_contact_count(default)
        if self.config.nconmax is not None:
            return int(self.config.nconmax)
        return default
