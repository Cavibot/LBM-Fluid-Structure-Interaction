# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XPBD constraint group implementations."""

from .deformable import BendingConstraint, SpringConstraint, TetrahedraConstraint
from .joint import BodyJointConstraint
from .particle_contact import ParticleParticleContact, ParticleShapeContact
from .rigid_contact import RigidContactConstraint

__all__ = [
    "BendingConstraint",
    "BodyJointConstraint",
    "ParticleParticleContact",
    "ParticleShapeContact",
    "RigidContactConstraint",
    "SpringConstraint",
    "TetrahedraConstraint",
]
