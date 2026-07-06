"""Utilities for adaptive timestep estimation in SPH_diff."""

from __future__ import annotations

import numpy as np
import warp as wp

from .kernels import kick


@wp.kernel
def _reduce_max_speed_sq(v: wp.array(dtype=wp.vec3), max_speed_sq: wp.array(dtype=wp.float32)):
	tid = wp.tid()
	vi = v[tid]
	speed_sq = wp.dot(vi, vi)
	wp.atomic_max(max_speed_sq, 0, wp.float32(speed_sq))


@wp.kernel
def _reduce_max_speed_sq_fluid_only(
	v: wp.array(dtype=wp.vec3),
	material: wp.array(dtype=wp.int32),
	fluid_material_id: wp.int32,
	max_speed_sq: wp.array(dtype=wp.float32),
):
	tid = wp.tid()
	if material[tid] != fluid_material_id:
		return

	vi = v[tid]
	speed_sq = wp.dot(vi, vi)
	wp.atomic_max(max_speed_sq, 0, wp.float32(speed_sq))


@wp.kernel
def _reduce_max_body_linear_speed_sq(body_qd: wp.array(dtype=wp.spatial_vector), max_speed_sq: wp.array(dtype=wp.float32)):
	tid = wp.tid()
	qd = body_qd[tid]
	v = wp.spatial_top(qd)
	speed_sq = wp.dot(v, v)
	wp.atomic_max(max_speed_sq, 0, wp.float32(speed_sq))


def estimate_cfl_dt(
	state,
	dt_hint: float,
	diameter: float,
	cfl_factor: float,
	cfl_min_dt: float,
	cfl_max_dt: float,
	velocity_floor: float,
	fluid_material_id: int | None = None,
) -> float:
	"""Estimate CFL timestep from particle/rigid velocities."""
	max_speed_sq = wp.array([float(velocity_floor)], dtype=wp.float32, device=state.particle_qd.device)

	v_src = state.particle_qd
	v_eval = v_src
	if hasattr(state, "a") and state.a is not None:
		# Predict effective velocity by v + a * dt using warp kernel.
		v_tmp = wp.empty_like(v_src)
		wp.launch(
			kernel=kick,
			dim=state.particle_count,
			inputs=[state.a, float(dt_hint), v_src],
			outputs=[v_tmp],
			device=v_src.device,
		)
		v_eval = v_tmp

	if state.particle_count > 0:
		if fluid_material_id is not None and getattr(state, "material_marks", None) is not None:
			wp.launch(
				kernel=_reduce_max_speed_sq_fluid_only,
				dim=state.particle_count,
				inputs=[
					v_eval,
					state.material_marks.material,
					wp.int32(fluid_material_id),
					max_speed_sq,
				],
				device=v_eval.device,
			)
		else:
			wp.launch(
				kernel=_reduce_max_speed_sq,
				dim=state.particle_count,
				inputs=[v_eval, max_speed_sq],
				device=v_eval.device,
			)

	if getattr(state, "body_qd", None) is not None:
		body_count = int(getattr(state, "body_count", 0))
		if body_count > 0:
			wp.launch(
				kernel=_reduce_max_body_linear_speed_sq,
				dim=body_count,
				inputs=[state.body_qd, max_speed_sq],
				device=state.body_qd.device,
			)

	max_vel_sq = float(max_speed_sq.numpy()[0])

	cfl_dt = float(cfl_factor) * 0.4 * (float(diameter) / max(np.sqrt(max_vel_sq), 1.0e-12))
	return float(np.clip(cfl_dt, float(cfl_min_dt), float(cfl_max_dt)))

