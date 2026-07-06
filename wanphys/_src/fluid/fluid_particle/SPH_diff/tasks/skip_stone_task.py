import warp as wp
import warp.optim
import numpy as np
from .base_task import Task


@wp.kernel
def assign_rigid_init_state_kernel(
    opt_rigid_v: wp.array(dtype=wp.vec3),
    opt_rigid_omega: wp.array(dtype=wp.vec3),
    rigid_id: int,
    rigid_v0: wp.array(dtype=wp.vec3),
    rigid_v: wp.array(dtype=wp.vec3),
    rigid_omega0: wp.array(dtype=wp.vec3),
    rigid_omega: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if tid == 0:
        rigid_v0[rigid_id] = opt_rigid_v[0]
        rigid_v[rigid_id] = opt_rigid_v[0]
        rigid_omega0[rigid_id] = opt_rigid_omega[0]
        rigid_omega[rigid_id] = opt_rigid_omega[0]

@wp.kernel
def compute_rigid_loss(
    rigid_x: wp.array(dtype=wp.vec3),
    target_rigid_x: wp.array(dtype=wp.vec3),
    rigid_vel: wp.array(dtype=wp.vec3),
    opt_rigid_vel: wp.array(dtype=wp.vec3),
    stone_rigid_id: int,
    rigid_q: wp.array(dtype=wp.quat),
    target_rigid_q: wp.array(dtype=wp.quat),
    loss: wp.array(dtype=float)
):
    tid = wp.tid()
    upperbound = -5.0
    if tid == 0:
        return
    # Position loss per rigid body
    diff_pos = rigid_x[tid] - target_rigid_x[tid]
    l_pos = wp.dot(diff_pos, diff_pos)
    
    w_penalty = 1.0
    loss_penalty = 0.0
    if tid == stone_rigid_id:
        # Coordinates are stored as xzy here, so the world y component is index 2.
        loss_penalty = w_penalty * wp.max(opt_rigid_vel[0][2] - upperbound, 0.0)
    # Rotation loss (0.5 * ||q - tq||^2)
    # q = rigid_q[tid]
    # tq = target_rigid_q[tid]
    w_pos = 1.

    # Combine losses (add weights here if needed)
    total_loss = w_pos * l_pos + loss_penalty

    wp.atomic_add(loss, 0, total_loss)



class SkipStoneTask(Task):
    def __init__(self, state):
        super().__init__(state)
        self.stone_rigid_id = self._get_stone_rigid_id()
        # Optimization variables for rigid body initial velocities
        initial_v = self.state.rbs.rigid_v.numpy()[self.stone_rigid_id]
        initial_omega = self.state.rbs.rigid_omega.numpy()[self.stone_rigid_id]
        device = self.state.rbs.rigid_v.device
        self.opt_rigid_v = wp.array([initial_v], dtype=wp.vec3, device=device, requires_grad=True)
        self.opt_rigid_omega = wp.array([initial_omega], dtype=wp.vec3, device=device, requires_grad=True)

        self.target_x = None
        self.target_q = None

        self.init_targets()
        self.init_optimizer()

    def _get_stone_rigid_id(self):
        rigid_bodies = self.state._model.cfg.get("RigidBodies", []) if hasattr(self.state._model, "cfg") else []
        if len(rigid_bodies) > 0 and "objectId" in rigid_bodies[0]:
            return int(rigid_bodies[0]["objectId"])
        return 1

    def _get_stone_cfg(self):
        rigid_bodies = self.state._model.cfg.get("RigidBodies", []) if hasattr(self.state._model, "cfg") else []
        for rb in rigid_bodies:
            if int(rb.get("objectId", -1)) == self.stone_rigid_id:
                return rb
        return None

    def init_targets(self):
        if self.state.num_objects > 0:
            self.target_rigid_x = wp.zeros_like(self.state.rbs.rigid_x)
            self.target_rigid_q = wp.zeros_like(self.state.rbs.rigid_quaternion)

            stone_cfg = self._get_stone_cfg()

            target_pos_np = self.state.rbs.rigid_x.numpy().copy()
            if "targetX" in stone_cfg and self.stone_rigid_id < len(target_pos_np):
                target_pos_np[self.stone_rigid_id] = np.array(stone_cfg["targetX"], dtype=np.float32)

            wp.copy(self.target_rigid_x, wp.array(target_pos_np, dtype=wp.vec3))

            target_q_np = self.state.rbs.rigid_quaternion.numpy().copy()
            if self.stone_rigid_id < len(target_q_np):
                if "targetQ" in stone_cfg:
                    target_q_np[self.stone_rigid_id] = np.array(stone_cfg["targetQ"], dtype=np.float32)

            wp.copy(self.target_rigid_q, wp.array(target_q_np, dtype=wp.quat))
            print(f"Initialized targets for stone rigid body {self.stone_rigid_id}: target_x = {self.target_rigid_x.numpy()[self.stone_rigid_id]}, target_q = {self.target_rigid_q.numpy()[self.stone_rigid_id]}")

    def init_optimizer(self):
        # Optimize the initial velocity and angular velocity of rigid body 1.
        # Use dedicated optimization variables.
        self.opt_var = [self.opt_rigid_v, self.opt_rigid_omega]

        # Optimizer
        self.optimizer = wp.optim.Adam(self.opt_var, lr=getattr(self.state._model, 'train_rate', 0.01))

    def compute_loss(self, sim_out_states, loss):
        # 使用 DiffState 中的数据和 kernel 计算 loss
        # This assumes sim_out_states has the required buffers for loss computation.
        if self.state.num_objects > 0:
            wp.launch(
                kernel=compute_rigid_loss,
                dim=self.state.num_objects,
                inputs=[
                    sim_out_states.rbs.rigid_x,
                    self.target_rigid_x,
                    sim_out_states.rbs.rigid_v,
                    self.opt_rigid_v,
                    self.stone_rigid_id,
                    sim_out_states.rbs.rigid_quaternion,
                    self.target_rigid_q,
                ],
                outputs=[
                    loss
                ]
            )

    def get_loss_state_info(self):
        info = {}
        if self.state.num_objects > self.stone_rigid_id:
            target_idx = self.stone_rigid_id
            final_pos = self.state.rbs.rigid_x.numpy()[target_idx]
            target_pos = self.target_rigid_x.numpy()[target_idx]
            final_vel_y = self.state.rbs.rigid_v.numpy()[target_idx][2]
            info[f"stone_final_pos"] = final_pos
            info[f"stone_target_pos"] = target_pos
            info[f"stone_final_vy"] = final_vel_y
        return info

    def init_simulation_state(self, tape=None):
        if self.state.num_objects > 0:
            with tape:
                wp.launch(
                    kernel=assign_rigid_init_state_kernel,
                    dim=1,
                    inputs=[
                        self.opt_rigid_v,
                        self.opt_rigid_omega,
                        self.stone_rigid_id,
                    ],
                    outputs=[
                        self.state.rbs.rigid_v0,
                        self.state.rbs.rigid_v,
                        self.state.rbs.rigid_omega0,
                        self.state.rbs.rigid_omega,
                    ],
                )

    def clear_grad(self):
        if self.opt_rigid_v.grad:
            self.opt_rigid_v.grad.zero_()
        if self.opt_rigid_omega.grad:
            self.opt_rigid_omega.grad.zero_()

    def norm_final_grad(self, v_grad, materialMarks):
        # Normalize rigid body gradients if needed
        if self.opt_rigid_v.grad:
            grad_np = self.opt_rigid_v.grad.numpy()
            norm = np.linalg.norm(grad_np)
            if norm > 1e-10:
                self.opt_rigid_v.grad = wp.array(grad_np / norm, dtype=wp.vec3, device=self.opt_rigid_v.device)
        if self.opt_rigid_omega.grad:
            grad_np = self.opt_rigid_omega.grad.numpy()
            norm = np.linalg.norm(grad_np)
            if norm > 1e-10:
                self.opt_rigid_omega.grad = wp.array(grad_np / norm, dtype=wp.vec3, device=self.opt_rigid_omega.device)
