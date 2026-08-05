"""Host-facing population encoding helpers."""

from .kernels.encoding import (
    direction_weight,
    direction_x,
    direction_y,
    direction_z,
    enforce_population_admissibility_kernel,
    equilibrium_population,
    home_to_populations_kernel,
    initialize_home_kernel,
    opposite_direction,
    populations_to_home_kernel,
)

__all__ = [
    "direction_weight",
    "direction_x",
    "direction_y",
    "direction_z",
    "enforce_population_admissibility_kernel",
    "equilibrium_population",
    "home_to_populations_kernel",
    "initialize_home_kernel",
    "opposite_direction",
    "populations_to_home_kernel",
]
