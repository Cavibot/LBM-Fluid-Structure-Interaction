# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from wanphys.examples.lbm.home_rebuild.home_free_plic.surface_tension_viewer import (
    SurfaceTensionGravityColumnViewer,
)


def test_surface_tension_viewer_requires_positive_lattice_coefficient() -> None:
    try:
        SurfaceTensionGravityColumnViewer(
            object(),
            lattice_surface_tension=0.0,
            config=None,
            substeps_per_frame=1,
            print_every=0,
        )
    except ValueError as error:
        assert "must be positive" in str(error)
    else:
        raise AssertionError("zero surface tension should be rejected")
