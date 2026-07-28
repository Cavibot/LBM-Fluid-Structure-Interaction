# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Momentum / kinetic-energy diagnostics for the demo small-sphere FSI setup.

Matches geometry (``r=0.08``, ``dh=0.02``) used by
``fluid_grid_lbm_dambreak_vof_single_sphere`` / two-spheres.

Default research path: ``empirical=False`` + reconstructed-link ME + Eq.24 walls.
Empirical buoyancy is showcase-only and is not an energy-conservation check.

Inventory (fluid, solid_phi ≥ 0)::

    water_mass = Σ mass
    momentum   = Σ mass · u          (lattice units)
    KE_fluid   = ½ Σ mass |u|²       (lattice units)

Rigid (world units)::

    KE_rigid   = ½ m |v|² + ½ ω·I·ω

Caveats (same as fluid-only suite): domain walls + gravity + XPBD contacts
break strict total-momentum / total-energy conservation. These tests check
(1) water-mass inventory with a moving sphere, (2) rest-state quiescence,
(3) finite bounded KE / |P| under dam-break + ME (approx retained as legacy),
(4) light vs heavy density trend under ME, and (5) no NaN blow-up.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import (
    GridLbmRigidCoupling,
    LbmFeedbackMode,
    recommended_me_force_scale,
)
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain
from wanphys._src.fluid.fluid_grid.lbm.backends.moment.home_fp32_ref import (
    make_home_vof_model,
    seed_dam_break_column,
    seed_pool,
)
from wanphys.examples.lbm._home_vof_empirical_sphere_fsi import (
    EmpiricalSphereFsiConfig,
    EmpiricalSphereFsiPlugin,
)
from wanphys.rigid import RigidDomain, RigidModelBuilder, ShapeConfig

# Showcase small-sphere constants (aligned with single/two-sphere demos).
DH: float = 0.02
SPHERE_RADIUS: float = 0.08
SPHERE_DENSITY: float = 0.7
RHO_LIQUID: float = 1.0
FRAME_DT: float = 1.0 / 60.0
SUBSTEPS: int = 8


@dataclass(frozen=True)
class _FluidInventory:
    water_mass: float
    px: float
    py: float
    pz: float
    ke: float
    u_max: float


@dataclass(frozen=True)
class _RigidInventory:
    mass: float
    ke: float
    speed: float
    px: float
    py: float
    pz: float
    z: float


def _require_cuda() -> None:
    try:
        wp.init()
        if wp.get_cuda_device_count() <= 0:
            raise RuntimeError("no CUDA device")
    except Exception as exc:  # noqa: BLE001
        raise unittest.SkipTest(f"CUDA unavailable: {exc}") from exc


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1.0e-12)


def _fluid_inventory(domain: LbmDomain) -> _FluidInventory:
    home = domain.solver._home_fp32
    assert home is not None and home._gpu is not None
    buf = home._gpu
    wp.synchronize_device(buf.device)
    mass = buf.mass.numpy().astype(np.float64)
    ux = buf.ux.numpy().astype(np.float64)
    uy = buf.uy.numpy().astype(np.float64)
    uz = buf.uz.numpy().astype(np.float64)
    solid = buf.solid_phi.numpy()
    cell = buf.cell_type.numpy()
    # Wet fluid inventory (exclude gas + solid-masked cells).
    mask = (cell != 0) & (solid >= 0.0)
    m = mass[mask]
    wmass = float(m.sum())
    px = float((m * ux[mask]).sum())
    py = float((m * uy[mask]).sum())
    pz = float((m * uz[mask]).sum())
    ke = 0.5 * float((m * (ux[mask] ** 2 + uy[mask] ** 2 + uz[mask] ** 2)).sum())
    if m.size:
        u_max = float(
            np.sqrt(ux[mask] ** 2 + uy[mask] ** 2 + uz[mask] ** 2).max()
        )
    else:
        u_max = 0.0
    return _FluidInventory(wmass, px, py, pz, ke, u_max)


def _rigid_inventory(
    rigid: RigidDomain,
    body_id: int,
) -> _RigidInventory:
    model = rigid.model
    state = rigid.state
    wp.synchronize_device(model.device)
    mass = float(model.get_body_mass(body_id))
    qd = np.asarray(state.body_qd.numpy()[body_id], dtype=np.float64)
    # spatial_vector layout: (vx, vy, vz, ωx, ωy, ωz)
    v = qd[:3]
    w = qd[3:6]
    speed = float(np.linalg.norm(v))
    ke_lin = 0.5 * mass * float(np.dot(v, v))
    inertia = np.asarray(model.body_inertia.numpy()[body_id], dtype=np.float64)
    ke_ang = 0.5 * float(w @ (inertia @ w))
    pos = np.asarray(state.body_q.numpy()[body_id], dtype=np.float64)
    return _RigidInventory(
        mass=mass,
        ke=ke_lin + ke_ang,
        speed=speed,
        px=mass * float(v[0]),
        py=mass * float(v[1]),
        pz=mass * float(v[2]),
        z=float(pos[2]),
    )


class _SmallSphereScene:
    """Minimal single-sphere coupling scene (no viewer). Default = ME core."""

    def __init__(
        self,
        *,
        n: int = 24,
        feedback: LbmFeedbackMode = LbmFeedbackMode.MOMENTUM_EXCHANGE,
        empirical: bool = False,
        gravity_lbm: float = -0.002,
        gravity_rigid: float = -1.0,
        seed: str = "dambreak",
        advance_rigid: bool = True,
        sphere_density: float = SPHERE_DENSITY,
        wall_eq: bool = False,
    ) -> None:
        _require_cuda()
        self.n = int(n)
        self.dh = DH
        self.sim_dt = FRAME_DT / float(SUBSTEPS)
        self.radius = SPHERE_RADIUS
        self.volume = (4.0 / 3.0) * math.pi * self.radius**3
        self.empirical_plugin: EmpiricalSphereFsiPlugin | None = None

        self.model = make_home_vof_model(
            fluid_grid_res=(self.n, self.n, self.n),
            fluid_grid_cell_size=self.dh,
            tau=0.55,
            gravity_z=float(gravity_lbm),
            vof_gamma=0.0,
            initial_density=RHO_LIQUID,
            vof_home_wall_eq=bool(wall_eq),
        )
        self.fluid = LbmDomain(self.model)
        self.fluid.create_state()
        home = self.fluid.solver._home_fp32
        assert home is not None
        if seed == "pool":
            host = seed_pool((self.n, self.n, self.n), fill_z=self.n // 2)
        else:
            host = seed_dam_break_column(
                (self.n, self.n, self.n),
                dam_x=max(2, self.n // 4),
                fill_z=self.n // 2,
            )
        home.seed_host_state(self.fluid.state, host)

        world = float(self.n) * self.dh
        wall_t = 2.0 * self.dh
        z_floor = self.radius + 1.5 * self.dh
        builder = RigidModelBuilder(gravity=float(gravity_rigid))
        wall_cfg = ShapeConfig(
            density=0.0, is_visible=False, is_solid=True, has_shape_collision=True
        )

        def add_wall(
            label: str,
            center: tuple[float, float, float],
            he: tuple[float, float, float],
        ) -> None:
            bid = builder.add_body(position=center, label=label)
            builder.add_shape_box(bid, hx=he[0], hy=he[1], hz=he[2], cfg=wall_cfg)

        add_wall(
            "floor",
            (world * 0.5, world * 0.5, -wall_t * 0.5),
            (world * 0.5, world * 0.5, wall_t * 0.5),
        )
        add_wall(
            "xmin",
            (-wall_t * 0.5, world * 0.5, world * 0.5),
            (wall_t * 0.5, world * 0.5, world * 0.5),
        )
        add_wall(
            "xmax",
            (world + wall_t * 0.5, world * 0.5, world * 0.5),
            (wall_t * 0.5, world * 0.5, world * 0.5),
        )
        add_wall(
            "ymin",
            (world * 0.5, -wall_t * 0.5, world * 0.5),
            (world * 0.5, wall_t * 0.5, world * 0.5),
        )
        add_wall(
            "ymax",
            (world * 0.5, world + wall_t * 0.5, world * 0.5),
            (world * 0.5, wall_t * 0.5, world * 0.5),
        )

        cfg = ShapeConfig(density=float(sphere_density), is_visible=True, is_solid=True)
        center = (world * 0.32, world * 0.5, z_floor)
        self.sphere_id = builder.add_body(position=center, label="sphere")
        builder.add_shape_sphere(self.sphere_id, radius=self.radius, cfg=cfg)
        self.rigid = RigidDomain(builder.finalize(device=self.model._device))
        self.rigid.create_state()
        self.z0 = float(center[2])

        me_scale = recommended_me_force_scale(self.dh, self.sim_dt)
        self.coupling = GridLbmRigidCoupling(self.fluid, self.rigid)
        self.coupling.add_body_sphere(self.sphere_id, radius=self.radius)
        self.coupling.set_rigid_dynamics_enabled(False)
        self.coupling.set_two_way_feedback_enabled(True, force_scale=me_scale)
        self.coupling.set_feedback_mode(feedback)
        self._advance_rigid = bool(advance_rigid)

        if empirical:
            self.empirical_plugin = EmpiricalSphereFsiPlugin(
                device=str(self.model._device),
                body_ids=(self.sphere_id,),
                densities=(float(sphere_density),),
                radius=self.radius,
                volume=self.volume,
                rho_liquid=RHO_LIQUID,
                gravity_abs=abs(float(gravity_rigid)),
                dh=self.dh,
                nx=self.n,
                ny=self.n,
                nz=self.n,
                config=EmpiricalSphereFsiConfig(
                    push_rate=0.0, drag_xy=0.0, drag_z=0.0
                ),
            )

    def step_once(self) -> None:
        self.coupling.step(self.sim_dt)
        if self.empirical_plugin is not None:
            st = self.fluid.state
            rs = self.rigid.state
            self.empirical_plugin.apply(
                phi=st.phi,
                cell=st.cell_type,
                solid=st.solid_phi,
                ux=st.velocity_x,
                uy=st.velocity_y,
                uz=st.velocity_z,
                body_q=rs.body_q,
                body_qd=rs.body_qd,
                body_f_apply=rs.apply_body_forces,
                vel_scale=self.dh / max(self.sim_dt, 1.0e-12),
            )
        if self._advance_rigid:
            self.rigid.step(self.sim_dt)

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step_once()


class TestSmallSphereMassWithFsi(unittest.TestCase):
    """Water mass with r=0.08 sphere raster / two-way ME feedback."""

    def test_static_sphere_pool_mass(self) -> None:
        scene = _SmallSphereScene(
            n=24,
            seed="pool",
            gravity_lbm=0.0,
            gravity_rigid=0.0,
            feedback=LbmFeedbackMode.NONE,
            advance_rigid=False,
        )
        # Coupling re-rasters the fixed sphere each step (rigid not advanced).
        scene.run(2)
        inv0 = _fluid_inventory(scene.fluid)
        self.assertGreater(inv0.water_mass, 10.0)
        scene.run(40)
        inv1 = _fluid_inventory(scene.fluid)
        self.assertLess(
            _rel_err(inv1.water_mass, inv0.water_mass),
            0.04,
            msg=f"mass {inv0.water_mass:.6g} → {inv1.water_mass:.6g}",
        )

    def test_moving_sphere_approx_mass(self) -> None:
        scene = _SmallSphereScene(
            n=24,
            seed="dambreak",
            feedback=LbmFeedbackMode.APPROX,
            empirical=False,
        )
        inv0 = _fluid_inventory(scene.fluid)
        scene.run(60)
        inv1 = _fluid_inventory(scene.fluid)
        self.assertTrue(np.isfinite(inv1.water_mass))
        self.assertLess(
            _rel_err(inv1.water_mass, inv0.water_mass),
            0.08,
            msg=f"mass {inv0.water_mass:.6g} → {inv1.water_mass:.6g}",
        )

    def test_moving_sphere_me_mass(self) -> None:
        scene = _SmallSphereScene(
            n=24,
            seed="dambreak",
            feedback=LbmFeedbackMode.MOMENTUM_EXCHANGE,
            empirical=False,
        )
        inv0 = _fluid_inventory(scene.fluid)
        scene.run(60)
        inv1 = _fluid_inventory(scene.fluid)
        self.assertLess(
            _rel_err(inv1.water_mass, inv0.water_mass),
            0.08,
            msg=f"ME mass {inv0.water_mass:.6g} → {inv1.water_mass:.6g}",
        )


class TestSmallSphereMomentumEnergy(unittest.TestCase):
    """Momentum / KE quiescence and bounded dam-break diagnostics."""

    def test_rest_pool_quiescence(self) -> None:
        """No gravity: fluid |P|/M and KE stay tiny; sphere stays nearly still."""
        scene = _SmallSphereScene(
            n=24,
            seed="pool",
            gravity_lbm=0.0,
            gravity_rigid=0.0,
            feedback=LbmFeedbackMode.MOMENTUM_EXCHANGE,
            empirical=False,
        )
        scene.run(50)
        fl = _fluid_inventory(scene.fluid)
        rg = _rigid_inventory(scene.rigid, scene.sphere_id)
        scale = max(fl.water_mass, 1.0)
        self.assertLess(abs(fl.px) / scale, 5.0e-3)
        self.assertLess(abs(fl.py) / scale, 5.0e-3)
        self.assertLess(abs(fl.pz) / scale, 5.0e-3)
        self.assertLess(fl.ke / scale, 5.0e-5)
        self.assertLess(fl.u_max, 0.05)
        self.assertLess(rg.speed, 0.15)
        self.assertLess(rg.ke, 0.05)

    def test_dambreak_momentum_energy_bounded_approx(self) -> None:
        scene = _SmallSphereScene(
            n=24,
            seed="dambreak",
            feedback=LbmFeedbackMode.APPROX,
            empirical=False,
        )
        fl0 = _fluid_inventory(scene.fluid)
        ke_peak = fl0.ke
        p_peak = abs(fl0.px) + abs(fl0.py) + abs(fl0.pz)
        for _ in range(80):
            scene.step_once()
            fl = _fluid_inventory(scene.fluid)
            rg = _rigid_inventory(scene.rigid, scene.sphere_id)
            self.assertTrue(np.isfinite(fl.ke))
            self.assertTrue(np.isfinite(rg.ke))
            self.assertLess(fl.u_max, 0.45, msg=f"u_max={fl.u_max}")
            ke_peak = max(ke_peak, fl.ke)
            p_peak = max(p_peak, abs(fl.px) + abs(fl.py) + abs(fl.pz))
        fl1 = _fluid_inventory(scene.fluid)
        rg1 = _rigid_inventory(scene.rigid, scene.sphere_id)
        # Mass still roughly held while KE / momentum stayed finite.
        self.assertLess(_rel_err(fl1.water_mass, fl0.water_mass), 0.1)
        self.assertGreater(ke_peak, 0.0)
        self.assertLess(ke_peak, 5.0e4)
        self.assertLess(p_peak, 5.0e3)
        self.assertLess(rg1.ke, 50.0)
        self.assertLess(rg1.speed, 5.0)

    def test_dambreak_momentum_energy_bounded_me(self) -> None:
        scene = _SmallSphereScene(
            n=24,
            seed="dambreak",
            feedback=LbmFeedbackMode.MOMENTUM_EXCHANGE,
            empirical=False,
        )
        fl0 = _fluid_inventory(scene.fluid)
        ke_peak = 0.0
        for _ in range(80):
            scene.step_once()
            fl = _fluid_inventory(scene.fluid)
            rg = _rigid_inventory(scene.rigid, scene.sphere_id)
            self.assertTrue(np.isfinite(fl.ke) and np.isfinite(rg.ke))
            self.assertLess(fl.u_max, 0.45)
            self.assertTrue(np.isfinite(rg.speed))
            self.assertLess(rg.speed, 8.0)
            ke_peak = max(ke_peak, fl.ke + rg.ke)  # diagnostic only (mixed units)
        fl1 = _fluid_inventory(scene.fluid)
        self.assertLess(_rel_err(fl1.water_mass, fl0.water_mass), 0.1)
        self.assertLess(ke_peak, 5.0e4)

    def test_me_density_trend_light_vs_heavy(self) -> None:
        """ME-only: heavy sphere sinks more (lower z) than light; no empirical push."""
        light = _SmallSphereScene(
            n=24,
            seed="pool",
            feedback=LbmFeedbackMode.MOMENTUM_EXCHANGE,
            empirical=False,
            sphere_density=0.45,
            gravity_lbm=-0.002,
            gravity_rigid=-1.0,
        )
        heavy = _SmallSphereScene(
            n=24,
            seed="pool",
            feedback=LbmFeedbackMode.MOMENTUM_EXCHANGE,
            empirical=False,
            sphere_density=1.35,
            gravity_lbm=-0.002,
            gravity_rigid=-1.0,
        )
        light.run(120)
        heavy.run(120)
        zl = _rigid_inventory(light.rigid, light.sphere_id)
        zh = _rigid_inventory(heavy.rigid, heavy.sphere_id)
        self.assertTrue(np.isfinite(zl.z) and np.isfinite(zh.z))
        self.assertTrue(np.isfinite(zl.speed) and np.isfinite(zh.speed))
        # Heavy should not float above light; allow small numerical slack.
        self.assertLessEqual(
            zh.z,
            zl.z + 0.02,
            msg=f"heavy z={zh.z:.4f} should be ≤ light z={zl.z:.4f}",
        )
        # Light sphere must not shoot upward unboundedly without empirical push.
        self.assertLess(zl.z, light.z0 + 0.25, msg=f"light rose too far: z={zl.z}")
        self.assertLess(zl.speed, 5.0)
        self.assertLess(zh.speed, 5.0)

    def test_empirical_buoyancy_mass_and_finite_energy(self) -> None:
        """Showcase buoyancy-only plugin: mass + finite KE (not energy-conserving)."""
        scene = _SmallSphereScene(
            n=24,
            seed="dambreak",
            feedback=LbmFeedbackMode.MOMENTUM_EXCHANGE,
            empirical=True,
            wall_eq=True,
        )
        fl0 = _fluid_inventory(scene.fluid)
        scene.run(60)
        fl1 = _fluid_inventory(scene.fluid)
        rg1 = _rigid_inventory(scene.rigid, scene.sphere_id)
        self.assertLess(_rel_err(fl1.water_mass, fl0.water_mass), 0.1)
        self.assertTrue(np.isfinite(fl1.ke) and np.isfinite(rg1.ke))
        self.assertLess(fl1.u_max, 0.45)
        self.assertLess(rg1.speed, 8.0)


if __name__ == "__main__":
    unittest.main()
