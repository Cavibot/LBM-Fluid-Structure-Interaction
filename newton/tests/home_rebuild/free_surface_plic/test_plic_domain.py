# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.home_rebuild.free_surface_plic import (
    PlicDomain,
    PlicGeometricTransport,
)


def test_standalone_domain_owns_closed_walls_without_home_state() -> None:
    domain = PlicDomain.closed_box((8, 7, 3), device="cpu")

    assert domain.res == (8, 7, 3)
    assert domain.model is domain
    assert domain.solid_cell_count == int(np.count_nonzero(domain.host))
    assert np.all(domain.host[[0, -1], :, :])
    assert np.all(domain.host[:, [0, -1], :])
    assert np.all(domain.host[:, :, [0, -1]])


def test_standalone_transport_accepts_analytic_courant() -> None:
    domain = PlicDomain.periodic((12, 8, 3), device="cpu")
    fill = np.zeros(domain.res, dtype=np.float32)
    fill[3:6, 2:5, 1] = 1.0
    cx = np.full((13, 8, 3), 0.2, dtype=np.float32)
    cy = np.zeros((12, 9, 3), dtype=np.float32)
    cz = np.zeros((12, 8, 4), dtype=np.float32)
    transport = PlicGeometricTransport(domain)

    result, diagnostics = transport.transport(
        wp.array(fill, dtype=float, device="cpu"),
        tuple(
            wp.array(field, dtype=float, device="cpu")
            for field in (cx, cy, cz)
        ),
        split_order=(0, 1),
    )

    assert np.isfinite(result.numpy()).all()
    assert abs(diagnostics.relative_volume_drift) < 4.0e-6
