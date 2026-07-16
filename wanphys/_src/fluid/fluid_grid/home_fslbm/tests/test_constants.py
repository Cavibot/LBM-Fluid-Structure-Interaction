# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HOME-FSLBM lattice constants and cell-flag enumeration.

Reference
---------
- ``mrConstantParamsGpu3D.h`` — D3Q27 velocities, weights, index3dInv
- ``mlLbmCommon.h``         — MLLATTICENODE_SURFACE_FLAG bitfield
"""

from __future__ import annotations


class TestD3Q27Directions:
    """Verify the 27 discrete velocities match the reference exactly."""

    def test_cx_cy_cz_bit_exact(self, _constants):
        """CX/CY/CZ must be bit-exact with ex3d_gpu/ey3d_gpu/ez3d_gpu."""
        C = _constants

        # Reference: mrConstantParamsGpu3D.h
        ref_cx = [
            0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0,
            1, -1, 1, -1, 1, -1, -1, 1,
        ]
        ref_cy = [
            0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1,
            1, -1, 1, -1, -1, 1, 1, -1,
        ]
        ref_cz = [
            0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1,
            1, -1, -1, 1, 1, -1, 1, -1,
        ]

        assert C.CX == ref_cx, "CX mismatch"
        assert C.CY == ref_cy, "CY mismatch"
        assert C.CZ == ref_cz, "CZ mismatch"
        assert len(C.CX) == C.NUM_DIRS == 27

    def test_d3q27_weights(self, _constants):
        """D3Q27 weights must match the reference exactly.

        W0 = 8/27  (rest)
        W1 = 2/27  (6 face neighbours)
        W2 = 1/54  (12 edge neighbours)
        W3 = 1/216 (8 corner neighbours)
        """
        C = _constants

        assert C.W0 == pytest.approx(8.0 / 27.0)
        assert C.WS == pytest.approx(2.0 / 27.0)
        assert C.WE == pytest.approx(1.0 / 54.0)
        assert C.WC == pytest.approx(1.0 / 216.0)

        # Weights must sum to 1.0
        total = C.W0 + 6 * C.WS + 12 * C.WE + 8 * C.WC
        assert total == pytest.approx(1.0)

        # Per-direction weight list must match
        assert len(C.W) == 27
        assert C.W[0] == pytest.approx(C.W0)
        for di in range(1, 7):
            assert C.W[di] == pytest.approx(C.WS)
        for di in range(7, 19):
            assert C.W[di] == pytest.approx(C.WE)
        for di in range(19, 27):
            assert C.W[di] == pytest.approx(C.WC)

    def test_opposite_direction_symmetry(self, _constants):
        """OPPOSITE[i] must always point to the reverse direction.

        For all i: CX[i] + CX[OPPOSITE[i]] == 0, and similarly for Y, Z.
        """
        C = _constants
        for di in range(27):
            opp = C.OPPOSITE[di]
            assert C.CX[di] + C.CX[opp] == 0, f"CX symmetry broken at di={di}"
            assert C.CY[di] + C.CY[opp] == 0, f"CY symmetry broken at di={di}"
            assert C.CZ[di] + C.CZ[opp] == 0, f"CZ symmetry broken at di={di}"

        # Involution: OPPOSITE[OPPOSITE[di]] == di
        for di in range(27):
            assert C.OPPOSITE[C.OPPOSITE[di]] == di, f"OPPOSITE involution broken at di={di}"

        # OPPOSITE must match index3dInv_gpu bit-exact
        ref_index3d_inv = [
            0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
            18, 17, 20, 19, 22, 21, 24, 23, 26, 25,
        ]
        assert C.OPPOSITE == ref_index3d_inv, "OPPOSITE != index3dInv_gpu"


class TestCellFlag:
    """Verify the MLLATTICENODE_SURFACE_FLAG bitfield values."""

    def test_flag_primitive_values(self, _constants):
        """Primitive flags must match mlLbmCommon.h exactly."""
        C = _constants
        assert C.CellFlag.TYPE_S == 0x01
        assert C.CellFlag.TYPE_E == 0x02
        assert C.CellFlag.TYPE_T == 0x04
        assert C.CellFlag.TYPE_F == 0x08
        assert C.CellFlag.TYPE_I == 0x10
        assert C.CellFlag.TYPE_G == 0x20

    def test_flag_composite_masks(self, _constants):
        """Composite flags must equal the bitwise OR of their constituents."""
        C = _constants
        assert C.TYPE_IF == (C.TYPE_I | C.TYPE_F), \
            f"TYPE_IF={C.TYPE_IF:#x} != I|F={C.TYPE_I | C.TYPE_F:#x}"
        assert C.TYPE_IG == (C.TYPE_I | C.TYPE_G)
        assert C.TYPE_SU_MASK == (C.TYPE_G | C.TYPE_I | C.TYPE_F), \
            f"TYPE_SU_MASK={C.TYPE_SU_MASK:#x} != G|I|F={C.TYPE_G | C.TYPE_I | C.TYPE_F:#x}"
        assert C.TYPE_BO_MASK == (C.TYPE_S | C.TYPE_E)

    def test_flag_helper_methods(self, _constants):
        """CellFlag.is_solid / is_gas / is_fluid / is_interface must be correct."""
        C = _constants
        assert C.CellFlag.is_solid(C.TYPE_S) == 1
        assert C.CellFlag.is_solid(C.TYPE_F) == 0
        assert C.CellFlag.is_gas(C.TYPE_G) == 1
        assert C.CellFlag.is_gas(C.TYPE_I) == 0
        assert C.CellFlag.is_fluid(C.TYPE_F) == 1
        assert C.CellFlag.is_fluid(C.TYPE_S) == 0
        assert C.CellFlag.is_interface(C.TYPE_I) == 1
        assert C.CellFlag.is_interface(C.TYPE_G) == 0


class TestD3Q7GasConstants:
    """D3Q7 dissolved-gas lattice constants."""

    def test_d3q7_gas_directions(self, _constants):
        """D3Q7 directions are the first 7 of D3Q27."""
        C = _constants
        assert C.NUM_DIRS_GAS == 7
        # D3Q7 uses the same CX/CY/CZ for the first 7 entries
        for di in range(7):
            assert C.CX[di] == [0, 1, -1, 0, 0, 0, 0][di]
            assert C.CY[di] == [0, 0, 0, 1, -1, 0, 0][di]
            assert C.CZ[di] == [0, 0, 0, 0, 0, 1, -1][di]

    def test_gas_weights_sum_to_one(self, _constants):
        """D3Q7 weights must sum to 1.0."""
        C = _constants
        total = sum(C.W_GAS)
        assert total == pytest.approx(1.0)

    def test_gas_cmr_s_rates(self, _constants):
        """CMR-MRT relaxation-rate vector must match reference hard-coded values.

        s[0]   = 1.0
        s[1-3] = 1 / 0.9  ≈ 1.111...
        s[4-6] = 1.5
        """
        C = _constants
        assert len(C.GAS_CMR_S) == 7
        assert C.GAS_CMR_S[0] == pytest.approx(1.0)
        for i in (1, 2, 3):
            assert C.GAS_CMR_S[i] == pytest.approx(1.0 / 0.9), \
                f"GAS_CMR_S[{i}] mismatch"
        for i in (4, 5, 6):
            assert C.GAS_CMR_S[i] == pytest.approx(1.5), \
                f"GAS_CMR_S[{i}] mismatch"


class TestPhysicalConstants:
    """Physical and solver constants."""

    def test_speed_of_sound(self, _constants):
        C = _constants
        assert C.CS2 == pytest.approx(1.0 / 3.0)
        assert C.CS == pytest.approx(0.57735027)
        assert C.INV_CS2 == pytest.approx(3.0)

    def test_surface_tension(self, _constants):
        """def_6_sigma = 6 * 4e-3."""
        C = _constants
        assert C.SURFACE_TENSION == pytest.approx(6.0 * 4e-3)

    def test_henry_disjoin(self, _constants):
        C = _constants
        assert C.HENRY_CONSTANT == pytest.approx(1e-3)
        assert C.DISJOINT_FACTOR == pytest.approx(0.032)

    def test_num_moments(self, _constants):
        """HOME-LBM stores exactly 10 moments per node."""
        C = _constants
        assert C.NUM_MOMENTS == 10

    def test_bc_enum_values(self, _constants):
        """Boundary-condition enum must be disjoint."""
        C = _constants
        assert C.BC_BOUNCE_BACK == 0
        assert C.BC_VELOCITY_INLET == 1
        assert C.BC_OUTFLOW == 2
        assert C.BC_PERIODIC == 3
        # All four values distinct
        assert len({C.BC_BOUNCE_BACK, C.BC_VELOCITY_INLET, C.BC_OUTFLOW, C.BC_PERIODIC}) == 4


import pytest
