"""Headless verification: run smoke for both feedback modes to confirm the
--feedback-mode switch in fluid_grid_lbm_twoway_fsi_visual.py is wired through.

Run from repo root:
    uv run --extra examples python scripts/diag/_verify_feedback_mode_switch.py
"""

from __future__ import annotations

import sys
import traceback

from wanphys.examples.lbm._lbm_twoway_scene import LbmTwoWaySceneConfig
from wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi_visual import FORCE_COLOR_BY_MODE, run_smoke

MODES: tuple[str, ...] = ("approx", "momentum_exchange")

def main() -> int:
    for mode in MODES:
        print(f"\n===== feedback_mode={mode!r} =====")
        if mode not in FORCE_COLOR_BY_MODE:
            print(f"  ERROR: mode missing from FORCE_COLOR_BY_MODE: {FORCE_COLOR_BY_MODE}")
            return 1
        try:
            run_smoke(
                num_frames=3,
                device=None,
                force_scale=1.0,
                feedback_mode=mode,
                advance_rigid=False,
            )
            print(f"  OK: {mode} smoke completed without exception")
        except Exception as exc:  # noqa: BLE001 - want to surface any failure
            print(f"  FAIL: {mode} raised: {exc}")
            traceback.print_exc()
            return 1
    print("\nAll feedback-mode smoke runs OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())