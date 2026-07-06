from dataclasses import dataclass

from ..base import FluidGridModelBase


@dataclass
class FluidGridLiquidModel(FluidGridModelBase):
    # Liquid-specific config
    particle_radius: float = 0.12
    extrap_iterations: int = 16
    density: float = 1000.0
    particle_count: int = 10000
    flip_pic_blend: float = 0.05
    sort_particles_by_cell: bool = True
    sort_particles_every_n_steps: int = 20
    sort_particles_key_mode: str = "linear"
