# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicCurvatureState,
    PlicDomain,
    PlicGeometryState,
)
from wanphys.examples.lbm.home_rebuild.home_free_plic.curvature_diagnostic_viewer import (
    PlicCurvatureRenderField,
)


def test_curvature_overlay_maps_sign_to_color_and_y_up_to_z_up() -> None:
    domain = PlicDomain.periodic((3, 3, 3), device="cpu")
    geometry = PlicGeometryState(domain)
    curvature = PlicCurvatureState(domain)
    normal = np.zeros(domain.res + (3,), dtype=np.float32)
    valid = np.zeros(domain.res, dtype=np.int32)
    values = np.zeros(domain.res, dtype=np.float32)
    normal[0, 0, 0] = (0.0, 1.0, 0.0)
    normal[1, 0, 0] = (0.0, 1.0, 0.0)
    valid[0, 0, 0] = 1
    valid[1, 0, 0] = 1
    values[0, 0, 0] = -0.5
    values[1, 0, 0] = 0.5
    geometry.normal.assign(normal)
    curvature.valid.assign(valid)
    curvature.curvature.assign(values)
    overlay = PlicCurvatureRenderField(
        domain.res,
        curvature,
        device="cpu",
        stride=1,
        cell_size=0.1,
        curvature_scale=0.5,
    )

    starts, ends = overlay.update(geometry)
    starts_host = starts.numpy()
    ends_host = ends.numpy()
    colors = overlay.colors.numpy()

    np.testing.assert_allclose(ends_host[0] - starts_host[0], (0.0, 0.0, 0.24))
    np.testing.assert_allclose(colors[0], (0.1, 0.55, 1.0))
    np.testing.assert_allclose(colors[9], (0.95, 0.28, 0.12))
