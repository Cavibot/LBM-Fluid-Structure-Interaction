# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""LBM benchmark metrics and variant registry."""

from .metrics import (
    InterfaceRoughnessMetrics,
    PerfMetrics,
    ValidationMetrics,
    bytes_per_cell,
    collect_interface_roughness,
    collect_validation_metrics,
    perf_metrics_from_step_stats,
)
from .martin_moyce import (
    MARTIN_MOYCE_N2_1,
    MARTIN_MOYCE_N2_2,
    MartinMoyceScale,
    align_T_to_anchor,
    compare_front_to_martin,
    reference_arrays,
)
from .registry import VariantSpec, get_variant, list_variants, register_variant

__all__ = [
    "InterfaceRoughnessMetrics",
    "MARTIN_MOYCE_N2_1",
    "MARTIN_MOYCE_N2_2",
    "MartinMoyceScale",
    "PerfMetrics",
    "ValidationMetrics",
    "VariantSpec",
    "align_T_to_anchor",
    "bytes_per_cell",
    "collect_interface_roughness",
    "collect_validation_metrics",
    "compare_front_to_martin",
    "get_variant",
    "list_variants",
    "perf_metrics_from_step_stats",
    "reference_arrays",
    "register_variant",
]
