"""Grouped LBM device implementation entry point.

The stage modules are the ownership boundaries.  The legacy aggregate is kept
only during Phase 1 while the individual kernels are moved into those files.
"""

from . import (
    admissibility,
    boundary,
    collision,
    common,
    encoding,
    forcing,
    initialization,
    legacy,
    macroscopic,
    moments,
    moving_wall,
    regularization,
    shan_chen,
    streaming,
)
from .legacy import (
    apply_boundary_conditions_kernel,
    apply_moving_wall_transport_kernel,
    compute_shan_chen_force_kernel,
    initialize_equilibrium_kernel,
    moments_to_mac_u_kernel,
    moments_to_mac_v_kernel,
    moments_to_mac_w_kernel,
    reg_trt_kernel,
)

__all__ = [
    "admissibility",
    "boundary",
    "collision",
    "common",
    "encoding",
    "forcing",
    "initialization",
    "legacy",
    "macroscopic",
    "moments",
    "moving_wall",
    "regularization",
    "shan_chen",
    "streaming",
    "apply_boundary_conditions_kernel",
    "apply_moving_wall_transport_kernel",
    "compute_shan_chen_force_kernel",
    "initialize_equilibrium_kernel",
    "moments_to_mac_u_kernel",
    "moments_to_mac_v_kernel",
    "moments_to_mac_w_kernel",
    "reg_trt_kernel",
]
