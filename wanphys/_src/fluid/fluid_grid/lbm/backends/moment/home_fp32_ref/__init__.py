# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""fp32 HOME-LBM reference backend (roadmap V2-ref / development path).

Increment status
----------------
**H0–H2:** Hermite, collide, domain BC (``step.py``, ``bc.py``).
**H3:** Distribution VOF FS uses filtered ``\\bar f`` (``vof_home_fs_filter``).
**H4:** Moment-encoded HOME-FREE VOF stepper (``vof_step.py``).
**H5:** Optional ``LbmModel.lbm_backend='home_fp32'`` via ``bridge.py``.
**H6 (done):** Warp GPU HOME-FREE VOF (``vof_warp.py``). Quant deferred.
**H6.1:** Fused stream+mass+FS+collide; massex + IF/IG/GI surface (Home-FSLBM order).
**H6.2:** PLIC κ → Eq.12 Laplace on home FS (mild γ); bubble CCL still deferred.

**Next:** bubble CCL / trapped-gas pressure; quant still deferred.
"""

from __future__ import annotations

from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bridge import (
    HomeFp32VofBridge,
    home_domain_bc_from_model,
)
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref.bc import (
    HomeDomainBC,
    HomeFaceBC,
    HomeFaceKind,
    face_normal_inward,
    reconstruct_solid_f_i_numpy,
    solid_moments_eq24,
    zou_he_velocity_numpy,
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
    seed_dam_break_column,
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
    "HomeFp32VofBridge",
    "HomeMomentArrays",
    "HomeMoments",
    "HomeVofState",
    "NEXT_INCREMENT",
    "collide_moments_numpy",
    "equilibrium_s_from_u",
    "face_normal_inward",
    "home_collide_moments",
    "home_domain_bc_from_model",
    "home_reconstruct_f_i",
    "make_uniform_equilibrium",
    "moments_from_f_numpy",
    "reconstruct_f_i_numpy",
    "reconstruct_f_numpy",
    "reconstruct_solid_f_i_numpy",
    "seed_dam_break_column",
    "solid_moments_eq24",
    "step_domain_numpy",
    "step_home_vof_numpy",
    "step_periodic_numpy",
    "step_periodic_warp",
    "zou_he_velocity_numpy",
]

NEXT_INCREMENT = "H7: bubble CCL / trapped-gas pressure (quant deferred)"
