# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HomeFslbmState — allocation shapes, lifecycle, protocol compliance.

Reference
---------
- ``mrFlow3D.h:31-91`` — mrFlow3D member layout
- ``mrFlow3D.h:17-27`` — mlBubble3D (double-precision bubble arrays)
- Audit items B2, B6, B9
"""

from __future__ import annotations

import pytest


class TestAllocationShapes:
    """All warp arrays must have the correct shapes and dtypes."""

    def test_f_mom_shape(self, default_state, default_model):
        state = default_state
        N = default_model.nx * default_model.ny * default_model.nz  # 32^3 = 32768
        assert state.f_mom.shape == (10 * N,)
        assert state.f_mom_post.shape == (10 * N,)

    def test_3d_field_shapes(self, default_state, default_model):
        state = default_state
        nx, ny, nz = 32, 32, 32
        assert state.flag.shape == (nx, ny, nz)
        assert state.mass.shape == (nx, ny, nz)
        assert state.massex.shape == (nx, ny, nz)
        assert state.phi.shape == (nx, ny, nz)
        assert state.force_x.shape == (nx, ny, nz)
        assert state.force_y.shape == (nx, ny, nz)
        assert state.force_z.shape == (nx, ny, nz)

    def test_gas_field_shapes(self, default_state, default_model):
        state = default_state
        N = 32 * 32 * 32
        assert state.g_mom.shape == (7 * N,)
        assert state.g_mom_post.shape == (7 * N,)
        assert state.delta_g.shape == (32, 32, 32)
        assert state.c_value.shape == (32, 32, 32)
        assert state.src.shape == (32, 32, 32)

    def test_bubble_field_shapes(self, default_state, default_model):
        state = default_state
        assert state.tag_matrix.shape == (32, 32, 32)
        assert state.previous_tag.shape == (32, 32, 32)
        assert state.previous_merge_tag.shape == (32, 32, 32)
        assert state.input_matrix.shape == (32, 32, 32)
        assert state.label_matrix.shape == (32, 32, 32)
        assert state.merge_detector.shape == (32, 32, 32)
        assert state.islet.shape == (32, 32, 32)
        assert state.disjoin_force.shape == (32, 32, 32)

    def test_bubble_property_arrays(self, default_state, default_model):
        state = default_state
        max_b = default_model.max_bubbles  # 65536
        assert state.bubble_volume.shape == (max_b,)
        assert state.bubble_init_volume.shape == (max_b,)
        assert state.bubble_rho.shape == (max_b,)
        assert state.bubble_label_init_volume.shape == (max_b,)
        assert state.bubble_label_volume.shape == (max_b,)

    def test_solid_coupling_shapes(self, default_state):
        state = default_state
        assert state.solid_phi.shape == (32, 32, 32)
        assert state.solid_body_id.shape == (32, 32, 32)
        assert state.vel_solid_u.shape == (33, 32, 32)  # nx+1
        assert state.vel_solid_v.shape == (32, 33, 32)
        assert state.vel_solid_w.shape == (32, 32, 33)


class TestDoublePrecisionBubbleArrays:
    """Audit item B2: bubble volume/rho arrays must be wp.float64."""

    def test_bubble_volume_dtype(self, default_state, _warp):
        assert default_state.bubble_volume.dtype == _warp.float64, \
            "B2: bubble_volume must be float64"

    def test_bubble_init_volume_dtype(self, default_state, _warp):
        assert default_state.bubble_init_volume.dtype == _warp.float64

    def test_bubble_rho_dtype(self, default_state, _warp):
        assert default_state.bubble_rho.dtype == _warp.float64

    def test_bubble_label_arrays_dtype(self, default_state, _warp):
        assert default_state.bubble_label_init_volume.dtype == _warp.float64
        assert default_state.bubble_label_volume.dtype == _warp.float64


class TestDefaultInitialValues:
    """Key fields must start with well-defined sentinel values."""

    def test_solid_phi_initial(self, default_state):
        """solid_phi must be 1000.0 (far away from any geometry)."""
        import warp as wp
        arr = default_state.solid_phi.numpy()
        assert (arr == 1000.0).all(), "solid_phi must init to 1000.0"

    def test_solid_body_id_initial(self, default_state):
        import warp as wp
        arr = default_state.solid_body_id.numpy()
        assert (arr == -1).all(), "solid_body_id must init to -1"

    def test_bubble_count_initial(self, default_state):
        assert default_state.bubble_count == 0

    def test_label_num_initial(self, default_state):
        assert default_state.label_num == -1

    def test_merge_split_flags_initial(self, default_state):
        assert default_state.merge_flag == 0
        assert default_state.split_flag == 0


class TestClearResetsAll:
    """clear() must zero all dynamic arrays (solid_phi = 1000, solid_body_id = -1 excepted)."""

    def test_clear_zeros_moments(self, default_state):
        import warp as wp
        # fill with ones first
        default_state.f_mom.fill_(1.0)
        default_state.f_mom_post.fill_(1.0)
        default_state.clear()
        assert (default_state.f_mom.numpy() == 0.0).all()
        assert (default_state.f_mom_post.numpy() == 0.0).all()

    def test_clear_zeros_phi(self, default_state):
        default_state.phi.fill_(0.8)
        default_state.clear()
        assert (default_state.phi.numpy() == 0.0).all()

    def test_clear_preserves_solid_sentinels(self, default_state):
        """solid_phi → 1000.0, solid_body_id → -1 after clear."""
        default_state.clear()
        assert (default_state.solid_phi.numpy() == 1000.0).all()
        assert (default_state.solid_body_id.numpy() == -1).all()


class TestCloneDeepCopy:
    """clone() must produce an independent deep copy."""

    def test_clone_creates_new_state(self, default_state):
        clone = default_state.clone()
        assert clone is not default_state
        assert clone.f_mom is not default_state.f_mom

    def test_clone_preserves_values(self, default_state):
        import warp as wp
        default_state.phi.fill_(0.42)
        clone = default_state.clone()
        assert (clone.phi.numpy() == 0.42).all()

    def test_clone_is_independent(self, default_state):
        import warp as wp
        default_state.phi.fill_(0.0)
        clone = default_state.clone()
        # mutate original
        default_state.phi.fill_(0.99)
        # clone must still be 0.0
        assert (clone.phi.numpy() == 0.0).all()


class TestDomainStateProtocol:
    """HomeFslbmState must implement DomainState protocol."""

    def test_isinstance_domain_state(self, default_state):
        from wanphys._src.core.domain import DomainState
        assert isinstance(default_state, DomainState), \
            "B9: HomeFslbmState must be instanceof DomainState"

    def test_clear_forces_is_callable(self, default_state):
        """clear_forces() must be a no-op (HOME-FSLBM handles forces in solver)."""
        default_state.clear_forces()  # must not raise
