# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from dataclasses import dataclass, field
from typing import Optional, Union, Any
from ..base import FluidGridModelBase

@dataclass
class FluidGridModel(FluidGridModelBase):
    # Smoke-specific Config
    buoyancy: float = 0.1
    vorticity_scale: float = 6.0
    damping: float = 0.5
    dissipation_rate: float = 0.8
