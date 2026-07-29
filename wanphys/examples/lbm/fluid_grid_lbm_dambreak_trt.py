# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0

"""Dam-break demo with orthogonal encoding, collision, interface, and gravity axes.

Default path matches the historical Reg-TRT + Shan-Chen dam-break:

    -e f -c t -i sc --gravity

Short CLI flags (aliases accepted):

    -e / --enc     f|fullf | h|home
    -c / --col     s|srt | t|trt | r|raw|raw_mrt | n|nocm|nocm_mrt
    -i / --interface   0|off | sc|shan_chen
    --gravity / --no-gravity

Controls: [Space] pause/resume  [R] reset  [mouse] orbit  [scroll] zoom
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import (
    DebugVofView,
    FullFLbmState,
    HomeLbmState,
    LbmDomain,
    LbmModel,
    LbmStateBase,
    VofCellType,
    VofInterfaceVisualizer,
)
from wanphys._src.fluid.fluid_viewer import FluidViewerGL, ScreenSpaceFluidRenderer

# ---------------------------------------------------------------------------
N: int = 128
DH: float = 0.02  # domain = 1.536m (20% smaller → fluid traverses faster)

TAU: float = 0.55  # ν = (0.55-0.5)/3 = 0.0167
LAMBDA_TRT: float = 0.03  # τ₊ = 0.03/0.05 + 0.5 = 1.1 (mild ghost damping)
G_SC: float = -5.0
SC_BOUNDARY_PSI: float = -1.0  # gas-like wall ψ: no mirror feedback → no droplets
PSI_TYPE: int = 1
PSI_REF: float = 1.0

GRAVITY: float = -0.005
OMEGA_REG: float = 0.5  # reg-TRT: odd part preserved, even part regularized

DAM_X_FRAC: float = 0.25
RHO_WATER: float = 1.8
RHO_AIR: float = 0.1

SSFR_THRESHOLD: float = 0.3
RAY_MARCH_STEPS: int = 1600  # 128³ diagonal: ~√(3)×128 ≈ 222 → ×7 samples/lu

FRAME_DT: float = 1.0 / 60.0
SIM_SUBSTEPS: int = 5  # lattice steps/frame


# ---------------------------------------------------------------------------
# CLI aliases (short strings)
# ---------------------------------------------------------------------------
_ENC_ALIASES: dict[str, str] = {
    "f": "fullf",
    "fullf": "fullf",
    "h": "home",
    "home": "home",
}
_COL_ALIASES: dict[str, str] = {
    "s": "srt",
    "srt": "srt",
    "t": "trt",
    "trt": "trt",
    "r": "raw_mrt",
    "raw": "raw_mrt",
    "raw_mrt": "raw_mrt",
    "n": "nocm_mrt",
    "nocm": "nocm_mrt",
    "nocm_mrt": "nocm_mrt",
}
_INTERFACE_ALIASES: dict[str, str] = {
    "0": "off",
    "off": "off",
    "sc": "shan_chen",
    "shan_chen": "shan_chen",
}


def _resolve_alias(value: str, table: dict[str, str], axis: str) -> str:
    key = str(value).strip().lower()
    try:
        return table[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(table.keys())))
        raise argparse.ArgumentTypeError(
            f"Unknown {axis} {value!r}; expected one of: {allowed}"
        ) from exc


def _parse_enc(value: str) -> str:
    return _resolve_alias(value, _ENC_ALIASES, "encoding")


def _parse_col(value: str) -> str:
    return _resolve_alias(value, _COL_ALIASES, "collision")


def _parse_interface(value: str) -> str:
    return _resolve_alias(value, _INTERFACE_ALIASES, "interface")


def create_parser() -> argparse.ArgumentParser:
    """CLI with independent physical and observation selectors."""
    import newton.examples

    parser = newton.examples.create_parser()
    parser.add_argument(
        "-e",
        "--enc",
        type=_parse_enc,
        default="fullf",
        metavar="E",
        help="encoding: f|fullf | h|home (default: f)",
    )
    parser.add_argument(
        "-c",
        "--col",
        type=_parse_col,
        default="trt",
        metavar="C",
        help="collision: s|srt | t|trt | r|raw | n|nocm (default: t)",
    )
    parser.add_argument(
        "-i",
        "--interface",
        type=_parse_interface,
        default="shan_chen",
        metavar="I",
        help="interface model: 0|off | sc|shan_chen (default: sc)",
    )
    parser.add_argument(
        "--gravity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable gravity independently (default: enabled)",
    )
    parser.add_argument(
        "-n",
        "--res",
        type=int,
        default=N,
        metavar="N",
        help=f"grid resolution N³ (default: {N})",
    )
    parser.add_argument(
        "--debug-vof-observation",
        action="store_true",
        help="derive a read-only debug VOF view from Shan-Chen density",
    )
    parser.add_argument(
        "--vof-debug-no-normals",
        action="store_true",
        help="show interface points without liquid-to-gas normal lines",
    )
    return parser


# ---------------------------------------------------------------------------
# Init kernels / helpers
# ---------------------------------------------------------------------------
@wp.kernel
def _init_fullf_dam(
    f_post: wp.array(dtype=float),
    density: wp.array3d(dtype=float),
    dam_x: int,
    rho_w: float,
    rho_a: float,
    seed: int,
    nx: int,
    ny: int,
    nz: int,
    stride: int,
) -> None:
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    rho = rho_w if i < dam_x else rho_a
    n = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001) * wp.cos(
        float(i * 419 + j * 233 + k * 577 + seed) * 0.0013
    )
    rho = wp.max(rho + n * 0.005 * rho, 0.01)
    density[i, j, k] = rho
    f_post[0 * stride + idx] = (1.0 / 3.0) * rho
    f_post[1 * stride + idx] = (1.0 / 18.0) * rho
    f_post[2 * stride + idx] = (1.0 / 18.0) * rho
    f_post[3 * stride + idx] = (1.0 / 18.0) * rho
    f_post[4 * stride + idx] = (1.0 / 18.0) * rho
    f_post[5 * stride + idx] = (1.0 / 18.0) * rho
    f_post[6 * stride + idx] = (1.0 / 18.0) * rho
    f_post[7 * stride + idx] = (1.0 / 36.0) * rho
    f_post[8 * stride + idx] = (1.0 / 36.0) * rho
    f_post[9 * stride + idx] = (1.0 / 36.0) * rho
    f_post[10 * stride + idx] = (1.0 / 36.0) * rho
    f_post[11 * stride + idx] = (1.0 / 36.0) * rho
    f_post[12 * stride + idx] = (1.0 / 36.0) * rho
    f_post[13 * stride + idx] = (1.0 / 36.0) * rho
    f_post[14 * stride + idx] = (1.0 / 36.0) * rho
    f_post[15 * stride + idx] = (1.0 / 36.0) * rho
    f_post[16 * stride + idx] = (1.0 / 36.0) * rho
    f_post[17 * stride + idx] = (1.0 / 36.0) * rho
    f_post[18 * stride + idx] = (1.0 / 36.0) * rho


@wp.kernel
def _init_home_dam(
    rho_field: wp.array3d(dtype=float),
    density: wp.array3d(dtype=float),
    dam_x: int,
    rho_w: float,
    rho_a: float,
    seed: int,
) -> None:
    """Rest-fluid HOME dam: persist rho only; momentum/stress left at 0."""
    i, j, k = wp.tid()
    rho = rho_w if i < dam_x else rho_a
    n = wp.sin(float(i * 127 + j * 311 + k * 541 + seed) * 0.001) * wp.cos(
        float(i * 419 + j * 233 + k * 577 + seed) * 0.0013
    )
    rho = wp.max(rho + n * 0.005 * rho, 0.01)
    rho_field[i, j, k] = rho
    density[i, j, k] = rho


def _mirror_state(domain: LbmDomain) -> None:
    """Copy state_in kinetic + observable fields into the double-buffer out state."""
    src: LbmStateBase = domain.state
    dst: LbmStateBase = domain._state_out
    for name in (
        "density",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "force_x",
        "force_y",
        "force_z",
        "solid_phi",
        "solid_body_id",
        "vel_solid_u",
        "vel_solid_v",
        "vel_solid_w",
    ):
        wp.copy(getattr(dst, name), getattr(src, name))
    if isinstance(src, FullFLbmState):
        assert isinstance(dst, FullFLbmState)
        wp.copy(dst.f_post, src.f_post)
    else:
        assert isinstance(src, HomeLbmState) and isinstance(dst, HomeLbmState)
        for a, b in zip(dst.kinetic_fields, src.kinetic_fields):
            wp.copy(a, b)


def build_model(args: argparse.Namespace) -> LbmModel:
    encoding: str = args.enc
    collision: str = args.col
    interface_model: str = args.interface
    n: int = int(args.res)

    use_reg = collision == "trt"
    lambda_trt = LAMBDA_TRT if collision == "trt" else 0.0
    g_sc = G_SC if interface_model == "shan_chen" else 0.0
    gz = GRAVITY if bool(args.gravity) else 0.0

    return LbmModel(
        fluid_grid_res=(n, n, n),
        fluid_grid_cell_size=DH,
        encoding=encoding,
        collision=collision,
        interface_model=interface_model,
        tau=TAU,
        G=g_sc,
        sc_boundary_psi=SC_BOUNDARY_PSI,
        psi_type=PSI_TYPE,
        psi_ref=PSI_REF,
        lambda_trt=lambda_trt,
        use_regularization=use_reg,
        omega_reg=OMEGA_REG if use_reg else 0.0,
        gravity_x=0.0,
        gravity_y=0.0,
        gravity_z=gz,
        debug_vof_observation=bool(args.debug_vof_observation),
        vof_debug_rho_gas=RHO_AIR,
        vof_debug_rho_liquid=RHO_WATER,
        vof_debug_epsilon=0.35,
        vof_debug_show_normals=not bool(args.vof_debug_no_normals),
    )


class DamBreakExample:
    """Dam-break visual demo with selectable encoding / collision / force."""

    def __init__(self, viewer: FluidViewerGL, args: argparse.Namespace):
        self.viewer = viewer
        viewer._paused = True
        self.args = args

        self.model = build_model(args)
        n = int(self.model.nx)
        print(
            f"Dam-Break: {n}^3  "
            f"enc={self.model.encoding}  col={self.model.resolved_collision}  "
            f"force={self.model.force_model}"
        )
        print(f"  tau={TAU}, G={self.model.G}, gz={self.model.gravity_z}")
        if self.model.resolved_collision == "trt":
            print(
                f"  TRT lambda={LAMBDA_TRT}, tau-/+="
                f"{self.model.tau_plus:.3f}/{self.model.tau_minus:.3f}, "
                f"omega_reg={self.model.omega_reg}"
            )
        print(f"  sc_boundary_psi={SC_BOUNDARY_PSI}, dam at x<{DAM_X_FRAC * n:.0f}")
        if self.model.force_model in ("shan_chen", "gravity+shan_chen") and not (
            self.model.encoding == "fullf"
            and self.model.resolved_collision in ("srt", "trt")
        ):
            print(
                "  [warn] SC on non-FullF-SRT/TRT path is research-grade "
                "(may be unstable)",
                file=sys.stderr,
                flush=True,
            )

        self.domain = LbmDomain(self.model)
        self.domain.create_state()
        self.sim_dt = FRAME_DT / SIM_SUBSTEPS
        self.sim_time = 0.0

        state = self.domain.state
        dam_x = int(n * DAM_X_FRAC)
        if isinstance(state, FullFLbmState):
            wp.launch(
                _init_fullf_dam,
                dim=(n, n, n),
                inputs=[
                    state.f_post,
                    state.density,
                    dam_x,
                    RHO_WATER,
                    RHO_AIR,
                    42,
                    n,
                    n,
                    n,
                    n * n * n,
                ],
                device=self.model._device,
            )
        else:
            assert isinstance(state, HomeLbmState)
            for field in state.kinetic_fields[1:]:
                field.zero_()
            wp.launch(
                _init_home_dam,
                dim=(n, n, n),
                inputs=[state.rho, state.density, dam_x, RHO_WATER, RHO_AIR, 42],
                device=self.model._device,
            )
        _mirror_state(self.domain)
        if self.model.debug_vof_observation:
            self.domain.solver.update_debug_mock_sc_to_vof(
                self.domain.state,
                self.domain._state_out,
            )
        wp.synchronize_device(self.model._device)
        print(f"  Water cells: {(state.density.numpy() > SSFR_THRESHOLD).sum()}")

        # Gentle gravity ramp to avoid shocking the interface
        if self.model.force_model in ("gravity", "gravity+shan_chen"):
            target_gz = float(self.model.gravity_z)
            self.model.gravity_z = 0.0
            ramp = 60
            for s in range(ramp):
                self.model.gravity_z = target_gz * float(s + 1) / float(ramp)
                self.domain.step(self.sim_dt)
            self.model.gravity_z = target_gz
            wp.synchronize_device(self.model._device)

        self.ssfr = ScreenSpaceFluidRenderer(
            viewer=viewer,
            max_particles=1,
            particle_radius=0.01,
            device=self.model._device,
        )
        viewer.register_post_render_callback(lambda v: self.ssfr.render(v))
        self.vof_visualizer: VofInterfaceVisualizer | None = None
        if self.model.debug_vof_observation:
            self.vof_visualizer = VofInterfaceVisualizer(
                (n, n, n),
                self.model._device,
                DH,
                point_radius_scale=float(self.model.vof_debug_point_radius_scale),
                normal_length_scale=float(self.model.vof_debug_normal_length_scale),
            )
            print("  VOF observation overlay: enabled (SC physics unchanged)")
        self.frame_count = 0
        self._last_ms = 0.0
        print("Controls: [Space] unpause  [R] reset  [mouse] orbit")

    def step(self) -> None:
        t0 = time.perf_counter()
        for _ in range(SIM_SUBSTEPS):
            self.domain.step(self.sim_dt)
        wp.synchronize_device(self.model._device)
        self._last_ms = (time.perf_counter() - t0) * 1000
        self.sim_time += FRAME_DT
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            r = self.domain.state.density.numpy()
            w = r > SSFR_THRESHOLD
            if w.any():
                c = np.argwhere(w).mean(axis=0)
                vx = float(self.domain.state.velocity_x.numpy()[w].mean())
                vy = float(self.domain.state.velocity_y.numpy()[w].mean())
                vz = float(self.domain.state.velocity_z.numpy()[w].mean())
                print(
                    f"[t={self.sim_time:.1f}s] water={w.sum()} "
                    f"COM=({c[0]:.0f},{c[1]:.0f},{c[2]:.0f}) "
                    f"v=({vx:+.3f},{vy:+.3f},{vz:+.3f}) sim={self._last_ms:.0f}ms",
                    file=sys.stderr,
                    flush=True,
                )
            r_np = self.domain.state.density.numpy()
            x0_slab: np.ndarray = r_np[0, :, :]
            x1_slab: np.ndarray = r_np[-1, :, :]
            water_left: int = int((x0_slab > SSFR_THRESHOLD).sum())
            water_right: int = int((x1_slab > SSFR_THRESHOLD).sum())
            print(
                f"  boundary rho: x=0 [{x0_slab.min():.3f}, {x0_slab.max():.3f}] "
                f"water_cells={water_left}"
                f"  |  x=-1 [{x1_slab.min():.3f}, {x1_slab.max():.3f}] "
                f"water_cells={water_right}",
                file=sys.stderr,
                flush=True,
            )
            debug_mock = self.domain.state.debug_mock_sc_to_vof
            if debug_mock is not None:
                types = debug_mock.cell_type.numpy()
                gas = int((types == int(VofCellType.GAS)).sum())
                interface = int((types == int(VofCellType.INTERFACE)).sum())
                liquid = int((types == int(VofCellType.LIQUID)).sum())
                print(
                    f"  VOF observe: gas={gas} interface={interface} liquid={liquid} "
                    f"epoch={debug_mock.epoch}",
                    file=sys.stderr,
                    flush=True,
                )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self.ssfr.available:
            self.ssfr.set_density_field(
                density=self.domain.state.density,
                grid_origin=(0, 0, 0),
                cell_size=DH,
                threshold=SSFR_THRESHOLD,
                max_steps=RAY_MARCH_STEPS,
            )
        if self.vof_visualizer is not None:
            debug_mock = self.domain.state.debug_mock_sc_to_vof
            assert debug_mock is not None
            self.vof_visualizer.render(
                self.viewer,
                DebugVofView.from_shan_chen_mock(debug_mock),
                self.domain.state.solid_phi,
                show_normals=bool(self.model.vof_debug_show_normals),
            )
        self.viewer.end_frame()


def main() -> None:
    import newton.examples
    from wanphys._src.fluid.fluid_viewer import init as init_fluid_viewer

    parser = create_parser()
    viewer: Any
    args: argparse.Namespace
    viewer, args = init_fluid_viewer(parser)
    newton.examples.run(DamBreakExample(viewer, args), args)


if __name__ == "__main__":
    main()
