# 速通 WanPhys LBM 刚体耦合实现

如果你想快速定位代码，可以先看这几个入口：

运行示例：
- `wanphys/examples/lbm/fluid_grid_lbm_dambreak_two_spheres.py`

LBM 求解算法：
- `wanphys/_src/fluid/fluid_grid/lbm/state.py`
- `wanphys/_src/fluid/fluid_grid/lbm/solver.py`
- `wanphys/_src/fluid/fluid_grid/lbm/kernels.py`

流固耦合：
- `wanphys/_src/fluid/fluid_grid/coupling/grid_lbm_rigid_coupling.py`
- `wanphys/_src/fluid/fluid_grid/coupling/coupling_kernels.py`

## 1. 项目框架

这份文档只解释 LBM 刚体耦合这条工程链路。更完整的 LBM 理论、Shan-Chen 多相模型、TRT 碰撞和性能分析见同目录下的 `project_report.md`。

### 1.1 WanPhys 

WanPhys 沿用 Newton 的 Model-State-Solver 组织方式：

- Domain：一个物理域的运行入口，持有 model、state 和 solver，并暴露 `step(dt)`。
- Model：静态配置，例如网格大小、cell size、LBM 参数、刚体几何。
- State：每一帧会变的状态，例如 LBM 分布函数 `f`、密度、速度、刚体位姿和刚体受力。
- Solver：读取 `state_in`，把下一步结果写入 `state_out`。

这里有一个非常关键的双缓冲概念：

- `_state_in` 是当前状态。
- `_state_out` 是下一步状态。
- `Domain.step()` 调用 solver 后交换两个 state。

所以耦合层如果在 LBM step 前写入边界字段，就必须保证这些字段在双缓冲交换后仍然被保留下来。后面的 `solid_phi`、`solid_body_id` 和 `vel_solid_u/v/w` 都和这个机制有关。

### 1.2 耦合类

`GridLbmRigidCoupling` 是组合层，不是新的流体 solver。它内部持有两个已有物理域：

```text
GridLbmRigidCoupling
  |_ LbmDomain    -> LbmSolver
  |_ RigidDomain  -> rigid solver
```

一个典型示例的运行关系可以理解成：

```text
example script
  -> GridLbmRigidCoupling.step(dt)
       -> 准备 LBM 刚体边界字段
       -> LbmDomain.step(dt)
       -> 累计流体反馈到 rigid_state.body_f
       -> RigidDomain.step(dt)
  -> Viewer / SSFR
```

关键点：LBM solver 只认识规则网格上的几个字段：

- `solid_phi`：是否为固体。
- `solid_body_id`：这个固体 cell 属于哪个刚体。
- `vel_solid_u/v/w`：固体壁面在 MAC 面上的速度。

### 1.3 LBM

LBM 是 Lattice Boltzmann Method，中文通常叫格子玻尔兹曼方法。它在规则网格上维护一组离散速度方向的分布函数。

当前 WanPhys LBM 使用 D3Q19：

- D3 表示三维。
- Q19 表示每个格点有 19 个离散速度方向：1 个静止方向、6 个轴向方向和 12 个对角方向。
- 每个格点保存 19 个分布函数 `f_i`。
- 宏观密度 `rho` 和速度 `u` 从这 19 个 `f_i` 求和得到。

可以把 LBM 的一步想成：

1. 从 `f` 算出密度和速度。
2. 处理外力、碰撞和传播。
3. 遇到固体时把分布函数反弹回来，也就是 bounce-back。
4. 再把新的宏观量写回状态。

多相流部分当前使用 Shan-Chen 模型。对这份耦合 guide 来说，只需注意`solid_phi` 不只用于 bounce-back，也会影响 Shan-Chen 计算固壁邻居时使用的虚拟伪势。因此刚体边界在多相场景里既是不可穿透障碍，也会影响固壁润湿和界面吸附。

### 1.4 静态固体边界与动态流固耦合

静态固体边界只需要 LBM 自己处理：

- LBM 发现某个方向的邻居是固体。
- 该方向的分布函数被 bounce-back。
- 固体不会因为流体冲击而运动。

运动刚体耦合分成两个方向：

- 刚体影响流体：耦合层把刚体栅格化到 `solid_phi / solid_body_id`，再把刚体表面速度写入 `vel_solid_u/v/w`。LBM 在 bounce-back 时读取这些速度，形成 moving-wall 修正。
- 流体反馈刚体：LBM step 之后，耦合层扫描 fluid-solid links，用 MEM 动量交换法把边界动量变化累计到 `rigid_state.body_f`。刚体 solver 随后读取 `body_f` 推进刚体。

### 1.5 一次物理模拟时间步概览

当前 `GridLbmRigidCoupling` 的一个 step ：

1. 确保 LBM 和刚体状态已创建，并在必要时上传刚体形状参数。
2. 如果需要新暴露流体单元修复，先保存上一时刻 `solid_phi`。
3. 清空当前 LBM 状态中的 `solid_phi`、`solid_body_id` 和 `vel_solid_u/v/w`。
4. 将刚体几何光栅化到 `solid_phi` 和 `solid_body_id`。
5. 根据上一时刻和当前 `solid_phi`，修复刚体离开后新暴露的流体单元。
6. 将刚体表面速度转换为 LBM 壁面速度，并写入 MAC 面数组。
7. 调用 `fluid_domain.step(dt)` 推进 LBM。
8. 如果 two-way feedback 开启，清空刚体受力，并调用反馈 kernel 累计流体对刚体的力和力矩。
9. 如果刚体推进开启，调用 `rigid_domain.step(dt)` 推进刚体。

详情见第 6 节。

## 2. LBM 状态中如何表示刚体边界

耦合层会把刚体转换成规则网格字段，然后 LBM 仅读取这些字段。

| 字段                | 位置          | 谁写入                     | 谁读取                     | 用途         |
| ----------------- | ----------- | ----------------------- | ----------------------- | ---------- |
| `solid_phi`       | cell center | coupling rasterize      | LBM kernel / SC force   | 判断邻居是不是固体  |
| `solid_body_id`   | cell center | coupling rasterize      | feedback kernel         | 找到力要写回哪个刚体 |
| `vel_solid_u/v/w` | MAC face    | coupling embed velocity | moving-wall bounce-back | 给移动壁面提供速度  |

刚体侧对应的关键状态：

- `body_q`：刚体世界位姿，用于把几何 rasterize 到 LBM 网格。
- `body_qd`：刚体线速度和角速度，用于计算壁面速度。
- `body_com`：刚体局部质心，用于计算表面点速度和力矩臂。
- `body_f`：刚体受力和力矩，流体反馈最终写到这里。

### 2.1 `solid_phi` 和 `solid_body_id`

`solid_phi` 是固体 SDF 风格字段：

- `solid_phi < 0` 表示这个 cell 在固体内部。
- `solid_phi >= 0` 表示这个 cell 是流体或远离固体。

`solid_body_id` 表示这个固体 cell 属于哪个刚体：

- `-1` 表示没有刚体。
- `0, 1, 2...` 表示刚体编号。

静态障碍只需要 `solid_phi` 就能触发 bounce-back；双向反馈还需要 `solid_body_id`，因为反馈 kernel 必须知道每条 fluid-solid link 的力要累加到哪个刚体。

### 2.2 `vel_solid_u/v/w`

文件：`wanphys/_src/fluid/fluid_grid/lbm/state.py`

刚体表面速度存成 MAC face velocity：

```python
self.vel_solid_u: wp.array3d = wp.zeros((nx + 1, ny, nz), ...)
self.vel_solid_v: wp.array3d = wp.zeros((nx, ny + 1, nz), ...)
self.vel_solid_w: wp.array3d = wp.zeros((nx, ny, nz + 1), ...)
```

LBM bounce-back 发生在 fluid-solid link 上，轴向 link 穿过的是 cell face：

- `u` 是 x 方向速度，放在 x 面上，所以有 `nx + 1` 个面。
- `v` 是 y 方向速度，放在 y 面上，所以有 `ny + 1` 个面。
- `w` 是 z 方向速度，放在 z 面上，所以有 `nz + 1` 个面。

这和 `velocity_x/y/z` 不同。`velocity_x/y/z` 是 cell center 的流体宏观速度；`vel_solid_u/v/w` 是固体壁面在 MAC 面上的速度。

`clear()` 会清零这些字段，`clone()` 会深拷贝这些字段，保证双缓冲、重置和复制状态时不会漏掉刚体边界速度。

## 3. 刚体影响流体

### 3.1 静态 bounce-back 的直觉

如果流体分布函数要从一个流体 cell 流向固体 cell，静态 bounce-back 会把它沿反方向弹回来。

```text
fluid cell -> solid wall
           <- reflected distribution
```

这样可以近似 no-slip wall，也就是墙面速度为 0。

### 3.2 moving-wall 修正

如果墙面拥有速度，反弹分布函数时需要加入墙面速度项：

```text
reflected = static_reflected + correction
```

当前 D3Q19 修正项采用：

```text
2 * w_i * rho * (c_i dot u_wall) / c_s^2
```

其中：

- `w_i` 是 D3Q19 该方向的权重。
- `rho` 是局部密度。
- `c_i` 是该 lattice direction。
- `u_wall` 是固体壁面速度。
- `c_s^2 = 1/3`。

如果 `u_wall` 全是 0，这个 correction 就是 0，因此移动壁面路径自然退化为静态墙。

### 3.3 固体速度从哪里来

文件：`wanphys/_src/fluid/fluid_grid/coupling/grid_lbm_rigid_coupling.py`

`GridLbmRigidCoupling.step()` 会在 LBM 推进前准备刚体边界：先写入 `solid_phi / solid_body_id`，再把刚体表面速度写入 `vel_solid_u/v/w`。

嵌入速度时会读取：

- `rigid_state.body_q`：刚体位姿。
- `rigid_state.body_qd`：刚体线速度和角速度。
- `body_com`：刚体局部质心。

刚体速度来自世界单位，LBM kernel 需要格子速度，因此耦合层会做单位转换：

```text
u_wall_lbm = dt / dh * u_wall_world
```

如果估计出的壁面格子速度过大，耦合层会发出 warning。通常这意味着需要减小时间步、增大网格间距，或降低刚体速度。

### 3.4 LBM kernel 如何读取移动墙速度

文件：`wanphys/_src/fluid/fluid_grid/lbm/kernels.py`

`collide_stream_bounceback_kernel` 会读取：

```python
vel_solid_u
vel_solid_v
vel_solid_w
```

相关 helper 包括：

- `_sample_solid_u/v/w()`：按 MAC face layout 采样。
- `_solid_wall_velocity_dot()`：计算 `c_i dot u_wall`。
- `_moving_wall_correction()`：返回 moving-wall 修正项。

轴向方向比较简单：穿过哪个 face，就采样哪个 face 的速度。对角方向会对相邻 MAC face 做平均。这不是最高精度的 SDF 交点插值，但足够作为局部、稳定、可测试的 moving-wall 起点。

## 4. 流体反馈刚体

流体到刚体的反馈发生在 LBM step 之后、刚体 step 之前。核心思路是：

```text
fluid-solid boundary momentum change
    -> force / torque
    -> rigid_state.body_f
    -> rigid solver advances body
```

### 4.1 反馈模式

文件：`wanphys/_src/fluid/fluid_grid/coupling/coupling_kernels.py`

当前保留两条反馈路径：

- `approx`：legacy 宏观速度近似反馈，用于对照和轻量诊断。
- `momentum_exchange`：基于 D3Q19 分布函数的 MEM，是当前推荐路径。

启用双向反馈和 MEM：

```python
coupling.set_two_way_feedback_enabled(True, force_scale=1.0)
coupling.set_feedback_mode("momentum_exchange")
```

默认仍是 one-way，避免旧示例突然让刚体被流体力推动。

### 4.2 legacy approx 模式（已弃用）

`approx` 在 LBM step 之后扫描轴向 fluid-solid face，用宏观密度和速度估算动量通量，再把力和力矩写入 `body_f`。

它的局限是：

- 基于宏观速度场，不是分布函数级别的 MEM。
- 只扫描 6 个轴向邻接方向。
- 输出仍依赖 `feedback_force_scale`。

因此它主要作为 legacy 对照路径保留；正常阅读时可以先跳过。

### 4.3 momentum_exchange 模式：MEM

基于 Ladd 1994。`momentum_exchange` 使用 LBM 分布函数本身估计流体传给固体的动量变化。核心 kernel 是：

```python
accumulate_lbm_momentum_exchange_all_bodies
```

kernel 在 LBM step 之后扫描每个 fluid cell 周围的 D3Q19 非零方向邻居。如果某个方向的邻居是 solid cell，就形成一条 fluid-solid link，并用 `solid_body_id` 找到对应刚体。

对每条 link，kernel 读取：

- `f_post_stream[d]`：LBM step 之后，该方向的分布函数。
- `f_pre_stream[opp[d]]`：LBM step 之前，相反方向的分布函数。

当前实现中，传给刚体的 link 动量近似为：

```text
delta_p = -c_d * (f_post[d] + f_pre[opp[d]]) * dh^3 * force_scale
```

再用 link midpoint 估计力矩：

```text
delta_tau = cross(link_midpoint - com_world, delta_p)
```

最后把 `delta_p / delta_tau` 原子加到 `rigid_state.body_f`。

MEM 当前仍是基础反馈通路：虽然方向和数据路径已经打通，但力的尺度还依赖 `force_scale`，没有完成严格物理单位标定（后续优化点）。

### 4.4 时序约束

two-way feedback 必须在 fluid step 之后、rigid step 之前发生。否则刚体 solver 看不到这一帧流体产生的 `body_f`；完整 step 流程见第 6 节。

## 5. 文件级导览

### 5.1 `state.py`

职责：定义 LBM 每帧状态。

重点看：

- `f`、`density`、`velocity_x/y/z` 这些基础 LBM 状态。
- `solid_phi`、`solid_body_id` 这些固体占据字段。
- `vel_solid_u/v/w` 这些 moving-wall 速度字段。
- clear、clone 是否一起维护这些边界字段。

### 5.2 `solver.py`

职责：推进一次 LBM。

重点看 `step()` 的顺序：

1. copy persistent boundary fields。
2. compute moments。
3. force / Shan-Chen / Guo。
4. collide-stream + bounce-back。
5. boundary condition。
6. macro fields。
7. MAC velocity。

### 5.3 `kernels.py`

职责：LBM 的 Warp kernel。

重点看：

- `collide_stream_bounceback_kernel`
- moving-wall helper
- `solid_phi` 如何触发 bounce-back
- `vel_solid_u/v/w` 如何进入 moving-wall correction

### 5.4 `grid_lbm_rigid_coupling.py`

职责：把 LBM domain 和 rigid domain 组合起来，是整个耦合的 orchestration 层。

重点看：

- `GridLbmRigidCoupling.step()`
- `add_body_sphere(...)` 等刚体注册入口
- `set_two_way_feedback_enabled(...)`
- `set_feedback_mode(...)`
- `set_rigid_dynamics_enabled(...)`

### 5.5 `coupling_kernels.py`

职责：流体-刚体耦合相关 Warp kernel。

重点看：

- `rasterize_all_body_sdf_warp`：把刚体几何写入 `solid_phi / solid_body_id`。
- `embed_all_solid_velocity_u/v/w`：把刚体表面速度写入 MAC 面。
- `accumulate_lbm_boundary_feedback_all_bodies`：legacy approx 反馈。
- `accumulate_lbm_momentum_exchange_all_bodies`：MEM 反馈。

### 5.6 `test_lbm_rigid_coupling.py`

职责：focused regression tests。

基础耦合测试覆盖：

- 静态球 rasterize 到 `solid_phi / solid_body_id`。
- 移动球 stamp `vel_solid_u` 并扰动流体速度。
- world velocity 到 LBM wall velocity 的单位缩放。
- 过大 wall velocity warning。
- advancing sphere 会移动 solid rasterization。

MEM 测试覆盖：

- 静止球和静止流体下合力近零。
- moving sphere 在静止流体中产生反作用力。
- 偏心 link 会产生 torque。
- force/torque 全部有限。

### 5.7 `fluid_grid_lbm_dambreak_two_spheres.py`

职责：当前最重要的 LBM 刚体耦合示例。

它构建一个双球溃坝场景：两个动态刚体球通过 `GridLbmRigidCoupling` 写入 LBM solid field，two-way feedback 打开，并使用 `momentum_exchange` 模式。

这个示例的刚体受力有两条来源：

- MEM feedback：`GridLbmRigidCoupling.step()` 在 fluid step 后读取 D3Q19 fluid-solid links，把动量交换贡献写入 `rigid_state.body_f`。
- 经验辅助项：示例脚本通过 `_estimate_submerged_fraction()` 和 `_apply_sphere_buoyancy_and_drag()` 额外施加浮力和阻尼。

所以轻球浮起不能完全理解成 MEM 自动给出的物理浮力。

典型运行命令：

```bash
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_two_spheres --viewer gl
```

## 6. 一次完整仿真步怎么流动


- `LbmSolver.step(...)`： LBM 自己基于 `state_in` 到 `state_out` 的一次运算。
- `GridLbmRigidCoupling.step(dt)`：讲刚体耦合层完整的一次物理推进，除`LbmSolver.step(...)`外还包括在 LBM step 前后准备数据、读取反馈、推进刚体。

### 6.1 LBM solver 的一步

调用入口是：

```python
fluid_domain.step(dt)
```

内部会调用：

```python
LbmSolver.step(state_in, state_out, dt)
```

这一层只读取已经栅格化好的规则网格字段：

- `solid_phi`
- `solid_body_id`
- `vel_solid_u/v/w`

LBM solver 主流程：

1. **复制 persistent boundary fields**  
   把 `solid_phi / solid_body_id / vel_solid_u/v/w` 从 `state_in` 复制到 `state_out`，避免双缓冲 swap 后边界字段丢失。

2. **从分布函数计算宏观量**  
   从 D3Q19 的 `f` 计算 density 和 velocity。

3. **计算外力**  
   如果启用 Shan-Chen，多相力会读取 density，也会用 `solid_phi` 判断固体邻居；纯外力路径则走 gravity / Guo force。

4. **碰撞、传播和 bounce-back**  
   `collide_stream_bounceback_kernel` 完成 collide-stream。遇到 `solid_phi < 0` 的邻居时，走 bounce-back。

5. **加入 moving-wall 修正**  
   如果边界有运动刚体，kernel 会读取 `vel_solid_u/v/w`，计算 `c_i dot u_wall`，把 moving-wall correction 加到反弹分布函数上。

6. **处理域边界条件**  
   根据模型设置处理周期边界、Zou-He 入口、对流出口等。

7. **写回输出场**  
   更新 density、cell-centered velocity 和 MAC velocity。

8. **交换双缓冲**  
   `LbmDomain.step()` 完成后交换 `_state_in` 和 `_state_out`。

### 6.2 Coupling 的完整一步

调用入口：

```python
coupling.step(dt)
```

同时管理刚体域和 LBM 域。完整顺序如下：

1. **拿到两边 state**  
   获取 `fluid_state`、`rigid_state`、LBM grid size 和 cell size `dh`。

2. **上传刚体 shape 参数**  
   如果刚体注册信息还没有上传到设备端，就把球、盒、胶囊、mesh 等形状参数准备好。

3. **备份上一帧固体场**  
   如果启用新暴露流体单元修复，先保存上一时刻的 `solid_phi`，用于判断哪些 cell 是刚体离开后刚变回流体的。

4. **清空当前 LBM 边界字段**  
   重置：

   ```python
   fluid_state.solid_phi.fill_(1000.0)
   fluid_state.solid_body_id.fill_(-1)
   fluid_state.vel_solid_u.zero_()
   fluid_state.vel_solid_v.zero_()
   fluid_state.vel_solid_w.zero_()
   ```

5. **rasterize 刚体几何**  
   调用 coupling kernel，把刚体写入：

   - `solid_phi`
   - `solid_body_id`

   结果是固体内部 cell 满足 `solid_phi < 0`，并带有对应刚体编号。

6. **修复新暴露流体单元**  
   如果某个 cell 上一帧是固体、这一帧变回流体，耦合层可以从邻近稳定流体单元估计 density / velocity，并重建平衡分布函数。  
   注意：双向 MEM 模式下这类修复被关闭，以避免破坏 MEM 所需的 pre/post 分布函数历史。意味着动态刚体离开后的新暴露流体单元没有额外重初始化，可能留下稳定性限制。（后续优化点）

7. **embed 刚体表面速度**  
   分别对 u/v/w face 写入：

   ```python
   ck.embed_all_solid_velocity_u
   ck.embed_all_solid_velocity_v
   ck.embed_all_solid_velocity_w
   ```

   结果是 `vel_solid_u/v/w` 保存刚体表面在 MAC face 上的 LBM 格子速度。

8. **推进 LBM**  
   调用：

   ```python
   self._fluid_domain.step(dt)
   ```

   这里进入第 6.1 节的 LBM solver 流程。

9. **累计流体反馈到刚体**  
   如果 two-way feedback 打开，先清空刚体受力：

   ```python
   rigid_state.clear_forces()
   ```

   然后根据反馈模式选择：

   ```python
   if feedback_mode == "momentum_exchange":
       ck.accumulate_lbm_momentum_exchange_all_bodies(...)
   else:
       ck.accumulate_lbm_boundary_feedback_all_bodies(...)
   ```

   这些 kernel 会把 force / torque 累加到 `rigid_state.body_f`。`body_f` 是 spatial vector，前 3 个分量是力，后 3 个分量是力矩。

10. **推进刚体**  
    如果 `_advance_rigid` 为 true，调用：

    ```python
    self._rigid_domain.step(dt)
    ```

    刚体 solver 会读取 `body_f` 并推进刚体。

11. **更新时间**  
    更新 coupling 自己维护的时间计数。

关键时序是：

```text
prepare rigid boundary
  -> fluid step
  -> fluid-to-rigid feedback
  -> rigid step
```

two-way feedback 必须在 fluid step 之后、rigid step 之前发生，否则刚体 solver 看不到这一帧流体产生的 `body_f`。

测试里经常关掉 rigid advancement，这样可以稳定观察流体边界和反馈，不让刚体位置变化引入额外变量。


## 7. 实现约束：Python 标注和 Warp kernel

这一节不是理解算法必需的内容，但读代码或继续改实现时会遇到。

仓库规则要求生产代码和示例代码中新增或触碰的 Python 赋值尽量带显式类型标注，例如：

- `self._two_way_feedback_enabled: bool`
- `self._feedback_force_scale: float`
- `fluid_state: LbmState`
- `rigid_state: RigidState`
- `model: LbmModel`
- `nx: int`
- `velocity_launches: list[...]`

但 Warp `@wp.func` 和 `@wp.kernel` 内部不要强行给局部变量加普通 Python 注解。Warp 编译器不支持普通 Python 的 `AnnAssign` 节点，这类写法可能导致 kernel 编译失败。

实际规则可以记成：

- Python orchestration 层：尽量加类型标注。
- Warp kernel / func 参数：使用 Warp 支持的类型标注。
- Warp kernel / func 内部局部变量：保持 Warp 可编译写法。

## 8. 目前工程限制

- MEM feedback 已经打通数据路径，但 `force_scale` 仍是经验缩放，不是严格物理标定。
- two-way feedback 开启时，新暴露流体单元修复会关闭，以保留 feedback kernel 所需的 pre/post 分布函数历史。
- 双球溃坝示例里的轻球浮起不是纯 MEM 结果，还叠加了经验浮力和阻尼。
- `solid_phi` 不只用于 bounce-back，在 Shan-Chen 多相路径里也影响固壁虚拟伪势。
- moving-wall 速度使用 `dt / dh` 从世界速度转成 LBM 格子速度，过大时会导致稳定性问题。
- 当前是显式弱耦合，没有 fluid-rigid 子迭代；高速刚体、轻质量刚体、强界面冲击下稳定性会更敏感。

## 9. 推荐阅读顺序

### 9.1 快速理解路线

目标是先知道“这套东西怎么连起来”，不追每个 kernel 细节。

1. 读 `project_report.md` 的“项目基本架构”和“流体-刚体耦合方法”。
2. 读本文第 1 节，建立 `GridLbmRigidCoupling -> LbmDomain / RigidDomain` 的心智模型。
3. 读本文第 2 节，理解 `solid_phi / solid_body_id / vel_solid_u/v/w` 这些数据契约。
4. 读本文第 6 节，理解一次完整 step 的内外两层流程。
5. 读 `grid_lbm_rigid_coupling.py` 的 `step()`，把第 6 节和真实代码对应起来。

### 9.2 深入实现路线

目标是能继续改 moving-wall、MEM 或示例行为。

1. 读 `state.py`，理解 `f`、density、cell-centered velocity、MAC velocity、solid fields。
2. 读 `solver.py` 的 `step()`，理解 LBM 如何从 `state_in` 写到 `state_out`。
3. 读 `kernels.py` 的 moving-wall helper 和 `collide_stream_bounceback_kernel`。
4. 读 `grid_lbm_rigid_coupling.py` 的 `step()`，理解刚体边界如何在 LBM step 前准备好。
5. 读 `coupling_kernels.py` 的 rasterize、embed solid velocity、legacy approx 和 MEM feedback kernel。
6. 读 `test_lbm_rigid_coupling.py`，把每个测试和上面的阶段对应起来。
7. 如果目标是双球溃坝，读 `fluid_grid_lbm_dambreak_two_spheres.py`，特别注意 MEM feedback 和经验浮力/阻尼是两条力来源。
