# 从零读懂 WanPhys LBM 刚体耦合实现

本文是一份面向新读者的中文导览，目标是让你从没有读过这块代码开始，逐步理解这次 LBM 刚体耦合工作做了什么、为什么这样做、每个文件负责什么、一次仿真步如何流动，以及当前实现还没有解决什么。

如果你只想快速定位代码，可以先看这几个入口：

- `wanphys/_src/fluid/fluid_grid/lbm/state.py`
- `wanphys/_src/fluid/fluid_grid/lbm/solver.py`
- `wanphys/_src/fluid/fluid_grid/lbm/kernels.py`
- `wanphys/_src/fluid/fluid_grid/coupling/grid_lbm_rigid_coupling.py`
- `wanphys/_src/fluid/fluid_grid/coupling/coupling_kernels.py`
- `newton/tests/test_lbm_rigid_coupling.py`
- `wanphys/examples/fluid_grid_lbm_twoway_fsi.py`
- `docs/wanphys/lbm_rigid_visual_acceptance_zh.md`

## 1. 先建立三个基本概念

### 1.1 什么是 LBM

LBM 是 Lattice Boltzmann Method，中文通常叫格子玻尔兹曼方法。它不是直接求 Navier-Stokes 方程里的速度和压力，而是在规则网格上维护一组离散速度方向的分布函数。

当前 WanPhys LBM 使用 D3Q19：

- D3 表示三维。
- Q19 表示每个格点有 19 个离散速度方向。
- 每个格点保存 19 个分布函数 `f_i`。
- 宏观密度 `rho` 和速度 `u` 从这 19 个 `f_i` 求和得到。

你可以把 LBM 的一步想成：

1. 从 `f` 算出密度和速度。
2. 处理外力、碰撞、流动传播。
3. 遇到固体时把分布函数反弹回来，这就是 bounce-back。
4. 再把新的宏观量写回状态。

### 1.2 什么是刚体耦合

刚体耦合的意思是：流体网格和刚体系统互相感知。

最简单的一向耦合是：

- 刚体告诉流体：哪里是固体边界。
- 流体在这些边界处 bounce-back，不穿墙。
- 刚体不受流体反作用力影响。

更进一步的 moving-wall 耦合是：

- 刚体还告诉流体：固体表面正在以什么速度运动。
- 流体在 bounce-back 时加入移动壁面修正。
- 运动的球或盒子可以拖动附近流体。

最后是 two-way coupling：

- 流体边界交互产生力和力矩。
- 这些力和力矩写回刚体状态 `body_f`。
- 刚体动力学再根据这些力推进。

本次实现的目标就是把这三层都接起来。

### 1.3 WanPhys 的 Model-State-Solver

WanPhys 沿用并隔离 Newton 的一个核心模式：

- Model：静态配置，例如网格大小、cell size、LBM 参数、刚体几何。
- State：每一帧会变的状态，例如 `f`、密度、速度、刚体位姿。
- Solver：拿 `state_in` 算 `state_out`。

这里有一个非常关键的双缓冲概念：

- `_state_in` 是当前状态。
- `_state_out` 是下一步状态。
- `Domain.step()` 内部会调用 solver，然后交换两个 state。

所以你看到 `solver.step(state_in, state_out, dt)` 时，要记住：它不是原地改一个 state，而是在两个 state 之间搬运和计算。

## 2. 本次工作解决的断点

开始前，代码里已经有一个 `GridLbmRigidCoupling`，但它像一个还没接上电的插头。它的意图很清楚：

- 读取 `RigidDomain` 的刚体位姿 `body_q` 和速度 `body_qd`。
- 把刚体几何 rasterize 到 LBM 的 `solid_phi` 和 `solid_body_id`。
- 把刚体表面速度写到 `vel_solid_u/v/w`。
- 让 LBM solver 用这些字段做移动壁面 bounce-back。

实际断点是：

- `coupling/__init__.py` 没导出 `GridLbmRigidCoupling`。
- `LbmState` 没有 `vel_solid_u/v/w` 字段。
- `collide_stream_bounceback_kernel` 只看 `solid_phi`，没有用固体表面速度。
- 没有 LBM 到刚体的力反馈路径。
- 没有 focused tests 证明这些路径能跑。

本次工作就是按阶段把这些断点补上。

## 3. Phase 1：让静态刚体边界先跑起来

### 3.1 导出耦合类

文件：`wanphys/_src/fluid/fluid_grid/coupling/__init__.py`

现在可以这样导入：

```python
from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling
```

这件事很小，但很重要。示例和用户代码通常不会直接 import 深层文件，而是从 package export 入口拿类。

### 3.2 给 LbmState 加刚体表面速度场

文件：`wanphys/_src/fluid/fluid_grid/lbm/state.py`

新增字段：

```python
self.vel_solid_u: wp.array3d = wp.zeros((nx + 1, ny, nz), ...)
self.vel_solid_v: wp.array3d = wp.zeros((nx, ny + 1, nz), ...)
self.vel_solid_w: wp.array3d = wp.zeros((nx, ny, nz + 1), ...)
```

为什么形状不是都 `(nx, ny, nz)`？

因为这是 MAC staggered grid 的速度布局：

- `u` 是 x 方向速度，放在 x 面上，所以有 `nx + 1` 个面。
- `v` 是 y 方向速度，放在 y 面上，所以有 `ny + 1` 个面。
- `w` 是 z 方向速度，放在 z 面上，所以有 `nz + 1` 个面。

这和普通 cell-centered velocity 不一样。cell-centered 的 `velocity_x/y/z` 是每个 cell 中心一个值；MAC face velocity 是每个面一个值。

`clear()` 里会清零这些字段：

```python
self.vel_solid_u.zero_()
self.vel_solid_v.zero_()
self.vel_solid_w.zero_()
```

`clone()` 里会深拷贝这些字段：

```python
wp.copy(new_state.vel_solid_u, self.vel_solid_u)
wp.copy(new_state.vel_solid_v, self.vel_solid_v)
wp.copy(new_state.vel_solid_w, self.vel_solid_w)
```

这保证双缓冲、重置和复制状态时不会漏掉刚体表面速度。

### 3.3 在 solver 双缓冲中保留这些字段

文件：`wanphys/_src/fluid/fluid_grid/lbm/solver.py`

LBM solver 开头有一段 persistent field copy：

```python
wp.copy(state_out.solid_phi, state_in.solid_phi)
wp.copy(state_out.solid_body_id, state_in.solid_body_id)
wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)
```

这段非常关键。原因是 `state_out` 是下一帧缓冲，如果不把这些边界字段拷过去，solver 后半段读到的可能是默认空字段，刚体边界信息就会丢。

### 3.4 solid_phi 和 solid_body_id 是什么

`solid_phi` 是固体 SDF 风格字段：

- `solid_phi < 0` 表示这个 cell 在固体内部。
- `solid_phi >= 0` 表示这个 cell 是流体或远离固体。

`solid_body_id` 表示这个固体 cell 属于哪个刚体：

- `-1` 表示没有刚体。
- `0, 1, 2...` 表示刚体编号。

静态障碍只需要 `solid_phi` 就够了：LBM streaming 时发现邻居是 solid，就 bounce-back。

`solid_body_id` 在 two-way feedback 里才变得更重要，因为流体需要知道力要加到哪个刚体上。

## 4. Phase 2：移动刚体边界

### 4.1 静态 bounce-back 的直觉

如果流体分布函数要从一个流体 cell 流向固体 cell，静态 bounce-back 会把它沿反方向弹回来。

直觉上就是：

```text
fluid cell -> solid wall
           <- reflected distribution
```

这样可以近似 no-slip wall，也就是墙面速度为 0。

### 4.2 移动壁面为什么要修正

如果墙面本身在动，普通静态 bounce-back 会错，因为它假设墙速度是 0。

移动壁面需要在反弹分布函数时加入墙面速度项：

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

如果 `u_wall` 全是 0，这个 correction 就是 0，因此静态墙行为保持不变。

### 4.3 固体速度从哪里来

文件：`wanphys/_src/fluid/fluid_grid/coupling/grid_lbm_rigid_coupling.py`

`GridLbmRigidCoupling.step()` 里会做三件事：

1. 清空 `solid_phi / solid_body_id / vel_solid_*`。
2. rasterize 刚体几何，写入 `solid_phi / solid_body_id`。
3. embed 刚体表面速度，写入 `vel_solid_u/v/w`。

嵌入速度时会读取：

- `rigid_state.body_q`：刚体位姿。
- `rigid_state.body_qd`：刚体速度。
- `body_com`：刚体局部质心，用于算表面点速度。

这里还修了一个容易忽略的 axis extent bug：

- u face kernel 用 `nx`。
- v face kernel 用 `ny`。
- w face kernel 用 `nz`。

如果全部传 `nx`，非立方网格上 v/w 方向会潜在越界或采样错误。

### 4.4 LBM kernel 如何读取移动墙速度

文件：`wanphys/_src/fluid/fluid_grid/lbm/kernels.py`

`collide_stream_bounceback_kernel` 现在接受：

```python
vel_solid_u
vel_solid_v
vel_solid_w
```

内部新增几个 helper：

- `_clamp_int()`：采样 MAC face 时防越界。
- `_sample_solid_u/v/w()`：按 face layout 采样。
- `_solid_wall_velocity_dot()`：计算 `c_i dot u_wall`。
- `_moving_wall_correction()`：返回移动墙修正项。

轴向方向比较简单：穿过哪个 face，就采样哪个 face 的速度。

对角方向更麻烦。当前规则是：

- 对每个 active component，采样它跨过的 MAC face。
- 再平均与 diagonal half-link midpoint 相邻的另一张 MAC face。
- 越界时 clamp 到最近有效 face。

这不是最高精度的曲面 SDF 交点插值，但足够作为局部、稳定、可测试的 moving-wall 起点。

## 5. Phase 3：two-way feedback

### 5.1 two-way 要解决什么

一向耦合里，刚体可以影响流体，但流体不会反推刚体。

two-way coupling 要做的是：

```text
刚体边界影响流体
流体边界动量变化产生力
力和力矩写回 rigid_state.body_f
刚体 solver 使用 body_f 推进刚体
```

### 5.2 当前实现选择：后处理边界扫描

文件：`wanphys/_src/fluid/fluid_grid/coupling/coupling_kernels.py`

当前没有把力反馈塞进已经很大的 collide-stream kernel，而是新增一个独立 kernel：

```python
accumulate_lbm_boundary_feedback_all_bodies
```

它在 LBM step 之后扫描所有 fluid cell，看它的六个轴向邻居是否是 solid cell。

如果发现某个方向是固体边界，就：

1. 通过 `solid_body_id` 找到刚体编号。
2. 计算流体相对墙面的法向速度。
3. 如果流体正在冲向墙面，就估算一份动量通量。
4. 把力和力矩 atomically add 到 `body_f`。

力的近似形式是：

```text
df = rho * rel_un * face_area * normal
```

力矩是：

```text
torque = cross(face_pos - com_world, df)
```

这里 `com_world` 从 `body_q` 和 `body_com` 得到。

### 5.3 这是近似，不是最终严格 MEM

这一点非常重要。

当前 two-way feedback 是早期可运行近似：

- 它基于宏观速度场。
- 它只扫描轴向 fluid-solid face。
- 它的输出在 lattice units 下，再乘 `feedback_force_scale`。

它不是最终严格的 D3Q19 population-level momentum exchange method。严格 MEM 通常会在每个 bounce-back link 上使用分布函数动量差来累计力。

为什么先做近似？

- 可以保持 collide-stream kernel 不继续膨胀。
- 可以先验证接口顺序、`body_f` 写回、force/torque 累加和刚体 step 时序。
- 后续要升级到严格 MEM 时，可以替换这个独立反馈 kernel，而不重写整个 coupling adapter。

### 5.4 two-way 开关

文件：`wanphys/_src/fluid/fluid_grid/coupling/grid_lbm_rigid_coupling.py`

新增 API：

```python
coupling.set_two_way_feedback_enabled(True, force_scale=1.0)
```

默认仍是 one-way：

```python
self._two_way_feedback_enabled: bool = False
```

这样已有 Phase 1/2 示例仍保持确定性，不会突然让刚体被流体力推动。

### 5.5 step 顺序

`GridLbmRigidCoupling.step()` 的顺序可以理解成：

```text
1. 确保 fluid/rigid state 已创建
2. 上传刚体 shape 参数
3. 清空 LBM solid fields 和 solid velocity fields
4. rasterize 刚体几何到 solid_phi / solid_body_id
5. embed 刚体表面速度到 vel_solid_u/v/w
6. fluid_domain.step(dt)
7. 如果 two-way enabled:
   7.1 clear rigid forces
   7.2 accumulate LBM boundary feedback into body_f
8. 如果 advance_rigid:
   8.1 rigid_domain.step(dt)
9. 更新 coupling time
```

这里的关键是：two-way feedback 必须在 fluid step 之后、rigid step 之前发生。否则刚体 solver 看不到这一帧流体产生的 `body_f`。

## 6. 文件级导览

### 6.1 `state.py`

职责：定义 LBM 每帧状态。

本次相关新增：

- `vel_solid_u`
- `vel_solid_v`
- `vel_solid_w`
- clear 时清零
- clone 时复制

你读这个文件时重点看：

- cell-centered fields 和 MAC face fields 的区别。
- `solid_phi` 初始填 `1000.0`。
- `solid_body_id` 初始填 `-1`。

### 6.2 `solver.py`

职责：推进一次 LBM。

本次相关改动：

- persistent copy 里保留 `vel_solid_u/v/w`。
- launch `collide_stream_bounceback_kernel` 时传入 solid velocity fields。

你读这个文件时重点看 `step()` 的顺序：

1. copy persistent fields。
2. compute moments。
3. force / Shan-Chen / Guo。
4. collide-stream + bounce-back。
5. boundary condition。
6. macro fields。
7. MAC velocity。

### 6.3 `kernels.py`

职责：LBM 的 Warp kernel。

本次相关改动：

- 新增 solid velocity sampling helper。
- 在 18 个非 rest 方向的 bounce-back 分支加入 moving-wall correction。
- 文档化 diagonal MAC face averaging rule。

为什么是 18 个？

D3Q19 有 19 个方向，其中一个是 rest direction，速度为 0。只有非 rest direction 会穿过边界，所以 moving-wall bounce-back 只影响 18 个方向。

### 6.4 `grid_lbm_rigid_coupling.py`

职责：把 LBM domain 和 rigid domain 组合起来。

这是整个耦合的 orchestration 层。它不继承 Newton 类，而是组合：

- `LbmDomain`
- `RigidDomain`

这符合 WanPhys 的长期目标：逐步替换 Newton，而不是把业务逻辑绑死在 Newton 基类上。

重点 API：

```python
coupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
coupling.add_body_sphere(body_idx=0, radius=...)
coupling.set_rigid_dynamics_enabled(False)
coupling.set_two_way_feedback_enabled(True, force_scale=1.0)
coupling.step(dt)
```

### 6.5 `coupling_kernels.py`

职责：流体-刚体耦合相关 Warp kernel。

已有功能包括：

- rasterize 刚体 SDF。
- embed solid velocity。
- FLIP/MAC 体系的压力力反馈。

本次新增：

- `_accumulate_lbm_face_feedback`
- `accumulate_lbm_boundary_feedback_all_bodies`

读这块时要记住：它是 LBM 的早期 two-way feedback，不要和 FLIP/MAC 压力投影体系的公式混用。

### 6.6 `test_lbm_rigid_coupling.py`

职责：focused regression tests。

现在有四个测试：

1. 静态球会 rasterize 到 `solid_phi / solid_body_id`。
2. 移动球会 stamp `vel_solid_u`，并扰动流体速度。
3. two-way enabled 时 `body_f` 产生非零 force，disabled 时为 0。
4. off-center boundary impulse 会产生 torque。

第四个测试特别重要，因为它不是只看力，还证明 `cross(face_pos - com_world, df)` 这条力矩路径真的工作。

### 6.7 `fluid_grid_lbm_twoway_fsi.py`

职责：headless two-way smoke example。

它构建一个小 LBM 场景：

- 均匀 x 向流。
- 一个刚体球。
- 开启 two-way feedback。
- 跑几步后打印 `body_f` 和 force norm。

典型运行命令：

```bash
uv run --python 3.11 python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi --steps 2
```

看到非零 `body_f`，说明 two-way feedback 路径通了。

### 6.8 `lbm_rigid_visual_acceptance_zh.md`

职责：展示层验收清单。

它记录 one-way GL、two-way GL、two-way null smoke 和可选 USD 导出的运行命令与预期信号。读这份清单时要特别注意：当前 `two-way force magnitude` 只是 diagnostic early feedback，只能证明路径接通和数值有限，不能当作物理正确性阈值。

two-way GL 示例里的“液体”显示层实际是刚体附近的速度扰动体渲染：它只在刚体 SDF band 内的流体 cell 中显示 `|u - u0|`，而不是自由液面或整域绝对速度 `|u|`。原因是 two-way 场景需要一个均匀来流来产生刚体反馈；如果直接显示 `|u|`，整个计算域都会像一整块液体一样被 SSFR 渲染出来，反而看不到球附近的边界扰动。

## 7. 一次完整仿真步怎么流动

下面从调用者视角追踪一次：

```python
coupling.step(dt)
```

### 7.1 耦合层准备边界

`GridLbmRigidCoupling.step()` 先拿到：

- `fluid_state`
- `rigid_state`
- LBM grid size `nx, ny, nz`
- cell size `dh`

然后清空边界字段：

```python
fluid_state.solid_phi.fill_(1000.0)
fluid_state.solid_body_id.fill_(-1)
fluid_state.vel_solid_u.zero_()
fluid_state.vel_solid_v.zero_()
fluid_state.vel_solid_w.zero_()
```

### 7.2 rasterize 刚体

如果 coupling 里有 body，就 launch：

```python
ck.rasterize_all_body_sdf_warp
```

结果：

- 固体内部 cell 的 `solid_phi < 0`
- 对应 cell 的 `solid_body_id = body_id`

### 7.3 embed 刚体表面速度

接着分别对 u/v/w face launch：

```python
ck.embed_all_solid_velocity_u
ck.embed_all_solid_velocity_v
ck.embed_all_solid_velocity_w
```

结果：

- `vel_solid_u/v/w` 保存刚体表面在 MAC face 上的速度。

### 7.4 LBM solver 推进一步

调用：

```python
self._fluid_domain.step(dt)
```

内部会调用 `LbmSolver.step(state_in, state_out, dt)`。

LBM solver 会：

- 复制 persistent boundary fields。
- 从 `f` 算 moments。
- 做 force 和 collide-stream。
- 在 bounce-back 时用 `solid_phi` 判断固体，用 `vel_solid_*` 做移动墙修正。
- 输出新的 density/velocity/MAC velocity。

### 7.5 two-way feedback

如果打开 two-way：

```python
rigid_state.clear_forces()
ck.accumulate_lbm_boundary_feedback_all_bodies(...)
```

这个 kernel 会把流体边界反馈累加到：

```python
rigid_state.body_f
```

`body_f` 是 spatial vector，前 3 个分量是力，后 3 个分量是力矩。

### 7.6 刚体推进

如果 `_advance_rigid` 为 true：

```python
self._rigid_domain.step(dt)
```

刚体 solver 会读取 `body_f` 并推进刚体。

测试里经常关掉 rigid advancement：

```python
coupling.set_rigid_dynamics_enabled(False)
```

这样可以稳定观察流体边界和反馈，不让刚体位置变化引入额外变量。

## 8. 怎么读测试

### 8.1 静态 rasterization 测试

这个测试构建：

- 12x12x12 LBM domain
- 一个半径 0.26 的球
- 关闭刚体推进

跑一步后断言：

- `solid_phi` 有负值。
- `solid_body_id` 有 body id。
- `vel_solid_u/v/w` 字段存在。

它证明 Phase 1 的断点修好了。

### 8.2 moving-wall 测试

这个测试给刚体球一个 x 向速度：

```python
sphere_speed = 0.04
```

跑三步后断言：

- `vel_solid_u` 非零。
- `fluid_ux` 非零。

它证明刚体表面速度确实被 stamp，且 LBM moving-wall correction 能影响流体。

### 8.3 two-way force 测试

这个测试先不开 two-way：

- 跑一步。
- `body_f` 应为 0。

然后打开 two-way：

- 再跑一步。
- `body_f[:3]` 的 norm 应大于 0。

它证明 one-way 默认行为和 two-way enabled 行为是可区分的。

### 8.4 torque 测试

这个测试不是用完整球体运动，而是手动 seed 一个最小边界场：

- 一个 fluid cell 有 x 向速度。
- 它旁边一个 solid cell 属于 body 0。
- 接触 face 相对刚体 COM 有偏移。

然后直接 launch：

```python
ck.accumulate_lbm_boundary_feedback_all_bodies
```

断言：

- x 向力非零。
- z 向 torque 非零。

它证明 force-to-torque 的 `cross` 路径工作。

## 9. 当前验证命令

这次完成时使用过的关键验证命令：

```bash
uv run --python 3.11 --extra dev -m newton.tests -p 'test_lbm*.py'
```

```bash
uv run --python 3.11 --extra dev python -m unittest newton.tests.test_lbm_rigid_coupling -v
```

```bash
uv run --python 3.11 python -m compileall wanphys/_src/fluid/fluid_grid/lbm wanphys/_src/fluid/fluid_grid/coupling wanphys/examples/fluid_grid_lbm_fsi.py wanphys/examples/fluid_grid_lbm_twoway_fsi.py newton/tests/test_lbm_rigid_coupling.py
```

```bash
uv run --python 3.11 python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi --steps 2
```

```bash
uv run --python 3.11 python -c "import wanphys.examples.lbm.fluid_grid_lbm_fsi as f; print(f.GridLbmRigidCoupling.__name__)"
```

```bash
git diff --check
```

`git diff --check` 在 Windows 上可能提示 LF/CRLF warning；这不是 whitespace failure。

## 10. 类型标注和 Warp 限制

仓库规则要求生产代码和示例代码中新增或触碰的赋值尽量带显式类型标注。

这次 Python 层新增或触碰的关键赋值已经加了类型标注，例如：

- `self._two_way_feedback_enabled: bool`
- `self._feedback_force_scale: float`
- `fluid_state: LbmState`
- `rigid_state: RigidState`
- `model: LbmModel`
- `nx: int`
- `velocity_launches: list[...]`

但 Warp `@wp.func` 和 `@wp.kernel` 内部的局部变量没有强行加 Python 注解。原因是 Warp 编译器不支持普通 Python 的 `AnnAssign` 节点，之前验证中已经遇到过类似错误。

所以这里的规则是：

- Python orchestration 层要标注。
- Warp kernel/func 内部局部变量保持 Warp 可编译写法。
- 函数参数仍然使用 Warp 类型标注。

## 11. 当前实现的边界和后续升级方向

### 11.1 当前已经做到

- LBM coupling adapter 可从 package 导入。
- LBM state 有刚体表面速度字段。
- 双缓冲保留刚体边界字段。
- 静态刚体能作为 LBM solid obstacle。
- moving rigid 能通过 moving-wall bounce-back 拖动流体。
- two-way mode 能把近似 force/torque 写入 `body_f`。
- focused tests 覆盖静态、移动、force、torque。
- 有一个 headless two-way smoke example。

### 11.2 当前还不是最终物理模型

two-way feedback 还是早期近似。

主要限制：

- 只扫描轴向 6 邻接 fluid-solid face。
- 用宏观速度估算动量通量。
- 没有直接基于 D3Q19 每条 bounce-back link 的 population momentum exchange。
- diagonal link 的 moving-wall 速度采样是 MAC face 平均，不是精确 SDF 交点插值。
- `force_scale` 仍是显式 scale factor，没有完整物理单位换算体系。

### 11.3 后续最自然的升级路线

下一步如果要提高物理准确性，可以这样做：

1. 在 collide-stream 或独立 MEM kernel 中记录每条 bounce-back link 的动量交换。
2. 用 `solid_body_id` 把每条 link 的贡献映射到刚体。
3. 对 D3Q19 18 个非 rest 方向都计算 force contribution。
4. 用 link midpoint 或 SDF 交点估计 contact position。
5. 累加 force 和 torque 到 `body_f`。
6. 对比当前宏观速度近似，建立回归测试。

如果要提高几何准确性，可以：

- 用 SDF gradient 估计更准确的边界法向。
- 对 moving-wall velocity 做更精细的插值。
- 支持 mesh SDF 的更小误差 band。

如果要做更复杂的物理，不建议马上混在这次路径里：

- free-surface LBM rigid coupling
- Shan-Chen multiphase rigid force
- complex mesh hydrodynamic force calibration

这些最好在当前 one-way、moving-wall、two-way force 接口稳定后再做。

## 12. 推荐阅读顺序

如果你是第一次读这块代码，建议按这个顺序：

1. 读 `state.py`，理解 `f`、`density`、`velocity_x/y/z`、`vel_u/v/w`、`vel_solid_u/v/w`、`solid_phi`。
2. 读 `solver.py` 的 `step()`，理解 LBM 一步从 `state_in` 到 `state_out`。
3. 读 `grid_lbm_rigid_coupling.py` 的 `step()`，理解刚体边界如何在 LBM step 前准备好。
4. 读 `kernels.py` 的 moving-wall helper 和 `collide_stream_bounceback_kernel`。
5. 读 `coupling_kernels.py` 的 `accumulate_lbm_boundary_feedback_all_bodies`。
6. 读 `test_lbm_rigid_coupling.py`，把每个测试和上面的阶段对应起来。
7. 跑 `fluid_grid_lbm_twoway_fsi.py`，看 `body_f` 非零输出。

读完以后，你应该能回答这些问题：

- 为什么 `solid_phi < 0` 表示固体？
- 为什么 `vel_solid_u` 的 shape 是 `(nx + 1, ny, nz)`？
- 为什么 solver 要 copy persistent fields？
- moving-wall correction 加在哪里？
- two-way feedback 为什么在 fluid step 后、rigid step 前？
- 当前 two-way 为什么只是早期近似？
- torque test 为什么能证明力矩路径？

如果这些问题都能回答，你就已经真正读懂了这次 LBM 刚体耦合工作。


