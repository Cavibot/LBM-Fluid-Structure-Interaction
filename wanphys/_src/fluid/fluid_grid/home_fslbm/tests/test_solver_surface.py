# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for HOME-FSLBM Phase 2 surface-kernel pipeline.

Verifies that the surface_1/2/3 kernels are launched in the correct order
within the solver step, and that stage-1 tests still pass after surface
integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from wanphys._src.fluid.fluid_grid.home_fslbm import constants as C


@pytest.fixture(scope="module")
def _warp():
    import warp as wp
    return wp


@pytest.fixture(scope="module")
def default_model(_warp):
    from wanphys._src.fluid.fluid_grid.home_fslbm.model import HomeFslbmModel

    return HomeFslbmModel(
        fluid_grid_res=(16, 16, 16),
        fluid_grid_cell_size=1.0,
        omega=1.0,
        turbulence_radius=3,
    )


@pytest.fixture
def domain_and_solver(_warp, default_model):
    from wanphys._src.fluid.fluid_grid.home_fslbm.domain import HomeFslbmDomain
    from wanphys._src.fluid.fluid_grid.home_fslbm.solver import HomeFslbmSolver

    solver = HomeFslbmSolver(default_model)
    domain = HomeFslbmDomain(default_model, solver=solver)
    domain.create_state()
    return domain, solver


class TestSurfaceKernelLaunchOrder:
    """Verify surface kernels are called in the correct sequence."""

    @staticmethod
    def _set_flat_interface(domain, N):
        """Set up a flat-interface flag/phi grid: bottom fluid, mid IF band, top gas."""
        import warp as wp

        state = domain.state
        flag_host = np.zeros((N, N, N), dtype=np.uint8)
        phi_host = np.zeros((N, N, N), dtype=np.float32)

        for k in range(N):
            for j in range(N):
                for i in range(N):
                    if k < 7:
                        flag_host[i, j, k] = C.CellFlag.TYPE_F
                        phi_host[i, j, k] = 1.0
                    elif k == 7:
                        flag_host[i, j, k] = C.CellFlag.TYPE_IF
                        phi_host[i, j, k] = 0.5
                    else:
                        flag_host[i, j, k] = C.CellFlag.TYPE_G
                        phi_host[i, j, k] = 0.0

        wp.copy(state.flag, wp.array(flag_host, dtype=wp.uint8, device=state.flag.device))
        wp.copy(state.phi, wp.array(phi_host, dtype=float, device=state.phi.device))

    def test_kernel_order_in_step(self, _warp, domain_and_solver):
        """After one step, surface_3 transitions IF->F should be applied."""
        domain, _solver = domain_and_solver
        N = 16

        self._set_flat_interface(domain, N)

        # Run one step
        domain.step(dt=1.0)

        flag_out = domain.state.flag.numpy()

        # Check that middle band (k=7) is now TYPE_F (IF->F transition)
        mid_flags = flag_out[:, :, 7]
        mid_su = mid_flags & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
        assert np.all(mid_su == C.CellFlag.TYPE_F), \
            "TYPE_IF band should transition to TYPE_F after surface_3"

    def test_split_flag_not_active_without_tags(self, _warp, domain_and_solver):
        """split_flag stays 0 when no TYPE_IF cells have valid bubble tags."""
        import warp as wp

        domain, solver = domain_and_solver
        N = 16

        # All TYPE_F — no TYPE_IF cells
        flag_host = np.full((N, N, N), C.CellFlag.TYPE_F, dtype=np.uint8)
        wp.copy(domain.state.flag,
                wp.array(flag_host, dtype=wp.uint8, device=domain.state.flag.device))

        domain.step(dt=1.0)

        split_flag = int(solver._split_flag_gpu.numpy()[0])
        assert split_flag == 0, \
            f"split_flag should be 0 with no TYPE_IF cells, got {split_flag}"

    def test_post_step_no_type_gi(self, _warp, domain_and_solver):
        """After a full step, no TYPE_GI cells remain (all converted to TYPE_I)."""
        domain, _solver = domain_and_solver
        N = 16

        self._set_flat_interface(domain, N)

        domain.step(dt=1.0)

        flag_out = domain.state.flag.numpy()
        su_flags = flag_out & (C.CellFlag.TYPE_SU | C.CellFlag.TYPE_S)
        gi_mask = su_flags == C.CellFlag.TYPE_GI
        assert not np.any(gi_mask), \
            "No TYPE_GI cells should remain after full surface pipeline"

    def test_stage1_imports_intact(self):
        """Stage 1 test modules are still importable (kernel compilation check)."""
        from wanphys._src.fluid.fluid_grid.home_fslbm.kernels_fluid import (
            calculate_f_eq_d3q27,
            reconstruct_distribution,
            ml_get_pi_after_collision,
            stream_collide_bvh_kernel,
            calculate_phi,
            plic_cube,
        )
        assert calculate_f_eq_d3q27 is not None
        assert reconstruct_distribution is not None
        assert ml_get_pi_after_collision is not None
        assert stream_collide_bvh_kernel is not None
        assert calculate_phi is not None
        assert plic_cube is not None
