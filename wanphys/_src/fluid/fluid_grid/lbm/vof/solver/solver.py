"""VOF lifecycle façade.

The concrete transaction implementation is introduced in Phase 2.  Keeping
the entry point here from Phase 1 fixes the package boundary without exposing
the stage implementations from ``vof.__init__``.
"""

from __future__ import annotations


class VofSolver:
    """VOF lifecycle owner reserved for the Phase 2 transaction API."""

    def initialize(self, state, phi0) -> None:
        del state, phi0
        raise NotImplementedError("VofSolver.initialize is implemented in Phase 2")

    def complete_streaming(
        self,
        state_in,
        state_out,
        logical_f_post,
        streamed_populations,
    ) -> None:
        del state_in, state_out, logical_f_post, streamed_populations
        raise NotImplementedError(
            "VofSolver.complete_streaming is implemented in Phase 2"
        )

    def finish_step(self, state_in, state_out) -> None:
        del state_in, state_out
        raise NotImplementedError("VofSolver.finish_step is implemented in Phase 2")


__all__ = ["VofSolver"]
