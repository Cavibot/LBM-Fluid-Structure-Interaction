# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Safe import preamble for home_fslbm tests.

Workaround for Warp 1.12.0 + Python 3.11 inspect/tokenize bug that
crashes when ``@wp.kernel`` decorators in collision modules are
evaluated during import.

Import this module first in any test that needs home_fslbm:
    import wanphys._src.fluid.fluid_grid.home_fslbm.tests._safe_import

The strategy:
    1. Pre-register wanphys and its parent packages as dummy modules
       with correct ``__path__`` so that sub-package discovery still works.
    2. Pre-register crash-triggering leaf packages (liquid, coupling,
       collision, geometry) as empty dummies so that ``fluid_grid/__init__.py``
       can ``from .liquid import ...`` without cascading through the chain
       that reaches ``broad_phase_hash.py`` (which has a ``@wp.kernel``
       decorator that hits the Python 3.11 tokenizer bug).
"""

from __future__ import annotations

import sys
import types


def _make_pkg(name: str, path: str | None = None) -> types.ModuleType:
    m = types.ModuleType(name)
    if path is not None:
        m.__path__ = [path]
    sys.modules[name] = m
    return m


# Top-level + parent chain: set __path__ so real sub-packages can be found.
_make_pkg("wanphys", "wanphys")
_make_pkg("wanphys._src", "wanphys/_src")
_make_pkg("wanphys._src.fluid", "wanphys/_src/fluid")
_make_pkg("wanphys._src.fluid.fluid_grid", "wanphys/_src/fluid/fluid_grid")

# Crash-triggering leaf packages: pre-register as dummies WITHOUT __path__
# so their __init__.py is never executed.
_make_pkg("wanphys._src.fluid.fluid_grid.lbm")
_make_pkg("wanphys._src.fluid.fluid_grid.liquid")
_make_pkg("wanphys._src.fluid.fluid_grid.coupling")
_make_pkg("wanphys._src.collision")
_make_pkg("wanphys._src.collision.rigid")
_make_pkg("wanphys._src.geometry")
_make_pkg("wanphys.collision")
_make_pkg("wanphys.geometry")
_make_pkg("wanphys.utils")

# ---- Populate dummy modules with attributes that various __init__.py
#      files try to import.  Without these, ``from .foo import Bar``
#      raises ImportError when .foo is our empty dummy. -------------------

# wanphys/__init__.py imports
_col = sys.modules["wanphys.collision"]
_col.CollisionPipeline = None
_geo = sys.modules["wanphys.geometry"]
_geo.CollisionTriMeshStyle3D = None
_geo.CollisionTriMeshVBD = None
_util = sys.modules["wanphys.utils"]
_util.load_mesh_file = None
_util.load_point_cloud = None

# wanphys/_src/fluid/fluid_grid/__init__.py imports from .liquid / .lbm
_liquid = sys.modules["wanphys._src.fluid.fluid_grid.liquid"]
_liquid.FluidGridLiquidDomain = None
_liquid.FluidGridLiquidModel = None
_liquid.FluidGridLiquidSolver = None
_liquid.FluidGridLiquidState = None

_lbm = sys.modules["wanphys._src.fluid.fluid_grid.lbm"]
_lbm.LbmDomain = None
_lbm.LbmModel = None
_lbm.LbmSolver = None
_lbm.LbmState = None
