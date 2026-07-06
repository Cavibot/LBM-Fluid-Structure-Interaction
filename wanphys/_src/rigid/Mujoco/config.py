from __future__ import annotations

"""Configuration objects for the WanPhys MuJoCo solver."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class MujocoConfig:
    """Immutable-style runtime/build settings shared across MuJoCo components."""

    iterations: int = 20
    ls_iterations: int = 10
    solver: int | str = "cg"
    integrator: int | str = "implicitfast"
    cone: int | str = "pyramidal"
    impratio: float = 1.0
    separate_worlds: bool | None = None
    disable_contacts: bool = False
    use_mujoco_contacts: bool = True
    update_data_interval: int = 1
    default_actuator_gear: float | None = None
    actuator_gears: dict[str, float] = field(default_factory=dict)
    nconmax: int | None = None
    njmax: int | None = None
    tolerance: float = 1e-6
    ls_tolerance: float = 0.01
    include_sites: bool = True
