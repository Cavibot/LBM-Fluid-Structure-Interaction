# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

from wanphys.examples.lbm.home_rebuild.home_free_plic.plic_diagnostic_audit import (
    run_audit,
)


def test_read_only_plic_audit_preserves_state() -> None:
    result = run_audit(steps=2, observer_interval=1, device="cpu")

    assert result["state_unchanged_by_plic"] is True
    assert result["observation_count"] == 2
    assert result["invalid_interface_count"] == 0
    assert result["maximum_interface_cell_count"] > 0
