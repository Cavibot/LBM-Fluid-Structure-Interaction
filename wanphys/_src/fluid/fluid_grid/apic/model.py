from dataclasses import dataclass

from ..base import FluidGridModelBase


@dataclass
class FluidGridApicModel(FluidGridModelBase):
    # APIC liquid config
    particle_radius: float = 0.12
    extrap_iterations: int = 16
    density: float = 1000.0
    particle_count: int = 10000
    sort_particles_by_cell: bool = True
    sort_particles_key_mode: str = "linear"

