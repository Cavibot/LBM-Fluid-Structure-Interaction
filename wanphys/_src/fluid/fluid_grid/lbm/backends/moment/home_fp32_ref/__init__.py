# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""fp32 HOME-LBM reference backend (shared base + free-surface branch).

``lbm_backend='home_fp32'`` via ``HomeFp32Bridge``:
``phase_mode='none'`` | ``'vof_sharp'``. See ``NEXT_INCREMENT`` for roadmap tip.
"""

from __future__ import annotations

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bridge import (
    HomeFp32Bridge,
    HomeFp32VofBridge,
    home_domain_bc_from_model,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.generic import (
    generic_home_flags,
    generic_home_vof_flags,
    make_home_model,
    make_home_vof_model,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.surface_policy import (
    HomeVofLatePoolController,
    LatePoolEvent,
    LatePoolFrameStats,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bc import (
    HomeDomainBC,
    HomeFaceBC,
    HomeFaceKind,
    apply_home_domain_bc_to_model,
    face_normal_inward,
    reconstruct_solid_f_i_numpy,
    solid_moments_eq24,
    zou_he_velocity_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.ic import (
    mark_liquid_interfaces,
    seed_dam_break_column,
    seed_droplet,
    seed_full_liquid,
    seed_pool,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.step import (
    HomeMomentArrays,
    make_uniform_equilibrium,
    step_domain_numpy,
    step_periodic_numpy,
    step_periodic_warp,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.vof_step import (
    CELL_GAS,
    CELL_INTERFACE,
    CELL_LIQUID,
    HomeVofState,
    step_home_vof_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.core.hermite import (
    HomeMoments,
    equilibrium_s_from_u,
    home_reconstruct_f_i,
    moments_from_f_numpy,
    reconstruct_f_i_numpy,
    reconstruct_f_numpy,
)
from wanphys._src.fluid.fluid_grid.lbm.core.moments import (
    collide_moments_numpy,
    home_collide_moments,
)

__all__ = [
    "CELL_GAS",
    "CELL_INTERFACE",
    "CELL_LIQUID",
    "HomeDomainBC",
    "HomeFaceBC",
    "HomeFaceKind",
    "HomeFp32Bridge",
    "HomeFp32VofBridge",
    "HomeMomentArrays",
    "HomeMoments",
    "HomeVofLatePoolController",
    "HomeVofState",
    "LatePoolEvent",
    "LatePoolFrameStats",
    "NEXT_INCREMENT",
    "apply_home_domain_bc_to_model",
    "collide_moments_numpy",
    "equilibrium_s_from_u",
    "face_normal_inward",
    "generic_home_flags",
    "generic_home_vof_flags",
    "home_collide_moments",
    "home_domain_bc_from_model",
    "home_reconstruct_f_i",
    "make_home_model",
    "make_home_vof_model",
    "make_uniform_equilibrium",
    "mark_liquid_interfaces",
    "moments_from_f_numpy",
    "reconstruct_f_i_numpy",
    "reconstruct_f_numpy",
    "reconstruct_solid_f_i_numpy",
    "seed_dam_break_column",
    "seed_droplet",
    "seed_full_liquid",
    "seed_pool",
    "solid_moments_eq24",
    "step_domain_numpy",
    "step_home_vof_numpy",
    "step_periodic_numpy",
    "step_periodic_warp",
    "zou_he_velocity_numpy",
]

NEXT_INCREMENT = "H8: foam / dissolved-gas §4.4 (quant deferred)"
