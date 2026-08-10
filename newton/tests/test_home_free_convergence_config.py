# SPDX-FileCopyrightText: Copyright (c) 2026 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from wanphys.examples.lbm.home_free_offline.acceptance import _physical_speed
from wanphys.examples.lbm.home_free_offline.config import (
    OfflineSceneLevel,
    OfflineSceneName,
)
from wanphys.examples.lbm.home_free_offline.convergence import (
    make_convergence_config,
)
from wanphys.examples.lbm.home_free_offline.factory import (
    build_rectangular_liquid_fill,
)


class TestHomeFreeConvergenceConfig(unittest.TestCase):
    def test_levels_preserve_each_physical_problem(self) -> None:
        for scene in OfflineSceneName:
            configs = [
                make_convergence_config(scene, level)
                for level in OfflineSceneLevel
            ]
            reference = configs[0]
            for config in configs[1:]:
                with self.subTest(scene=scene.value, level=config.level.value):
                    self.assertEqual(
                        config.derived_parameters["physical_extent"],
                        reference.derived_parameters["physical_extent"],
                    )
                    self.assertEqual(
                        config.initial_liquid_height,
                        reference.initial_liquid_height,
                    )
                    self.assertEqual(
                        config.initial_liquid_x_range,
                        reference.initial_liquid_x_range,
                    )
                    self.assertEqual(
                        config.fluid.kinematic_viscosity,
                        reference.fluid.kinematic_viscosity,
                    )
                    self.assertEqual(
                        config.fluid.body_acceleration,
                        reference.fluid.body_acceleration,
                    )
                    self.assertEqual(
                        config.steps * config.fluid.time_step,
                        reference.steps * reference.fluid.time_step,
                    )
                    if scene is OfflineSceneName.DAMBREAK:
                        self.assertAlmostEqual(
                            config.derived_parameters["lattice_viscosity"],
                            reference.derived_parameters["lattice_viscosity"],
                        )
                    else:
                        self.assertAlmostEqual(
                            config.fluid.time_step / config.fluid.cell_size,
                            reference.fluid.time_step / reference.fluid.cell_size,
                        )
                    self.assertAlmostEqual(
                        config.derived_parameters["reynolds"],
                        reference.derived_parameters["reynolds"],
                    )
                    self.assertEqual(config.sphere, reference.sphere)

    def test_dambreak_uses_periodic_spanwise_reference(self) -> None:
        expected_dt = {
            OfflineSceneLevel.L0: 0.5,
            OfflineSceneLevel.L1: 0.125,
            OfflineSceneLevel.L2: 0.03125,
        }
        for level in OfflineSceneLevel:
            config = make_convergence_config(OfflineSceneName.DAMBREAK, level)
            self.assertEqual(config.fluid.periodic, (False, True, False))
            self.assertEqual(config.derived_parameters["physical_extent"], (24.0, 4.0, 16.0))
            self.assertEqual(config.initial_liquid_height, 8.125)
            self.assertEqual(config.initial_liquid_x_range, (0.0, 8.125))
            self.assertEqual(config.fluid.time_step, expected_dt[level])
            self.assertEqual(config.fluid.max_lattice_speed, 0.12)
            self.assertEqual(config.fluid.kinematic_viscosity, 0.10)
            self.assertEqual(
                config.fluid.body_acceleration, (0.0, 0.0, -5.625e-4)
            )
            self.assertEqual(
                config.derived_parameters["lattice_viscosity"], 0.05
            )
            self.assertEqual(
                config.steps * config.fluid.time_step, 400.0
            )

    def test_dambreak_initial_interface_is_non_degenerate_at_every_level(self) -> None:
        physical_volumes = []
        for level in OfflineSceneLevel:
            config = make_convergence_config(OfflineSceneName.DAMBREAK, level)
            fill = build_rectangular_liquid_fill(
                config.fluid.resolution,
                cell_size=config.fluid.cell_size,
                height=config.initial_liquid_height,
                x_range=config.initial_liquid_x_range,
            )
            self.assertTrue(np.any((fill > 0.0) & (fill < 1.0)))
            physical_volumes.append(
                float(np.sum(fill, dtype=np.float64)) * config.fluid.cell_size**3
            )
        for volume in physical_volumes[1:]:
            self.assertAlmostEqual(volume, physical_volumes[0], places=10)

    def test_physical_speed_gate_respects_diffusive_scaling(self) -> None:
        physical_speed = 0.025
        for level in OfflineSceneLevel:
            config = make_convergence_config(OfflineSceneName.DAMBREAK, level)
            lattice_speed = (
                physical_speed
                * config.fluid.time_step
                / config.fluid.cell_size
            )
            self.assertAlmostEqual(
                _physical_speed(config, lattice_speed), physical_speed
            )


if __name__ == "__main__":
    unittest.main()
