# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from wanphys.examples.lbm.home_rebuild.home_free_plic.curvature_diagnostic_audit import (
    run_audit,
)


def test_read_only_curvature_audit_preserves_state() -> None:
    result = run_audit(steps=2, observer_interval=1, device="cpu")

    assert result["state_unchanged_by_curvature"] is True
    assert result["observation_count"] == 2
    assert result["maximum_invalid_curvature_count"] == 0
    assert result["maximum_required_cell_count"] > 0
