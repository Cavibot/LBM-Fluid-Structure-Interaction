# P3 零表面张力自由面边界工程计划

## 1. 阶段定义

| 项目 | 内容 |
|---|---|
| 阶段 | P3 |
| 前置阶段 | P0-P2 `CPU_ACCEPTED` |
| 当前实施状态 | `CPU_ACCEPTED` |
| 目标状态 | `CPU_ACCEPTED` |
| 物理范围 | FullF、固定拓扑、固定大气压、`gamma=0` 的完整 VOF-LBM step |
| 明确不包含 | 类型转换、重分配、新界面初始化、PLIC、曲率、表面张力、HOME、开口 VOF domain boundary、CUDA |

P3 只回答：

> gas 不推进 LBM 时，目标 INTERFACE 从 gas 方向缺失的 streamed population 如何补全？

P3 完成后可以在 `cell_type` 固定且 `phi` 不越界的前提下执行完整
`LbmDomain.step()`。一旦质量推进要求类型变化，P3 必须 fail-fast，等待 P4/P5。

## 2. 依据与冻结决策

### 2.1 论文 Eq. (11)

目标论文 §3.1：

\[
f_i^*(\mathbf{x},t)
=f_i^{eq}(\rho_g,\mathbf{u}(\mathbf{x},t))
+f_{\bar i}^{eq}(\rho_g,\mathbf{u}(\mathbf{x},t))
-f_{\bar i}(\mathbf{x},t).
\]

对 pull streaming：

```text
target = x
direction = q
source = x - c_q
source type = GAS
write f_star[q, x]
read f_post[opposite(q), x]
```

Home-FSLBM 2D/3D reference 分别在：

```text
mrLbmSolverGpu2D.cu:594-595
mrLbmSolverGpu3D.cu:932-933
```

采用同一方向关系。

### 2.2 大气压力与 gas density

P3 新增：

```text
vof_atmosphere_pressure = c_s^2 = 1/3
vof_surface_tension = 0
```

Eq. (12) 在 P3 退化为：

\[
\rho_g = p_{atmos}/c_s^2.
\]

默认值因此得到：

```text
rho_g = 1
```

配置要求：

- pressure finite 且 `>0`；
- surface tension 必须精确为 `0`，非零值留给 P6；
- 不允许 bubble/disjoining pressure。

### 2.3 equilibrium 层级

P3 FullF 使用当前 D3Q19 solver 的统一二阶：

```text
encoding.equilibrium_population()
```

这使 Eq. (11) 与现有 SRT/TRT FullF collision 使用同一个 equilibrium 定义。

该实现可以声明：

> D3Q19 FullF 与论文自由面压力边界公式兼容。

不能声明：

> 精确复现论文 D3Q27 高阶 HOME-FREE equilibrium/collision。

### 2.4 速度时间层与半力

Eq. (11) 固定读取：

```text
state_in.velocity_x/y/z
```

这些字段已经是项目统一 hydrodynamic closure 写出的物理速度。P3 不执行：

```text
u_pred = u + F/(2 rho)
```

否则会重复添加半力。

### 2.5 调度位置

固定调度：

```text
P2 mass transport from state n
→ ordinary FullF pull streaming
→ Eq.11 overwrite gas-to-interface links
→ ordinary domain boundary completion
→ admissibility
→ collector / force / collision
→ write observables
→ restore semantically invalid GAS kinetic/macros
→ commit fixed-topology mass/phi/type
→ validate fixed-topology output
```

surface completion 必须发生在 collector 前。普通 domain boundary 与 gas surface
使用不同 source 分类，不允许同一 link 有两个最终写入者。

### 2.6 GAS kinetic 语义

P3 仍不为 GAS 推进物理。为了双缓冲稳定和测试隔离：

```text
state_out GAS kinetic/macros = state_in GAS kinetic/macros
```

这些值仍被视为无效；保留只用于：

- 避免无意义的 gas collision 漂移；
- 保证 gas 垃圾值不会反馈 active cells；
- 为 P5 的显式 GAS→INTERFACE 初始化保留清晰边界。

P3 surface 和 P2 mass 均禁止读取 gas kinetic。

### 2.7 domain boundary

P3 支持：

```text
periodic
static bounce-back
```

P3 对 VOF 模式拒绝：

```text
zou_he
pressure
convective
moving_wall
cut_link
```

moving/cut-link 已由 P1 拒绝；普通 VOF 开口边界推迟到 P7 综合扩展。

## 3. 数据与模块

### 3.1 持久状态

不新增字段：

```text
VofGridState.mass
VofGridState.phi
VofGridState.cell_type
```

### 3.2 新增模块

```text
vof/surface.py
    VofSurfaceBoundary
    rho_g host contract
    FullF completion dispatch

vof/surface_kernels.py
    Eq.11 gas-to-interface overwrite
    GAS state restoration

vof/advection_kernels.py
    fixed-topology mass commit

vof/validation.py
    P3 fixed-topology output validation
```

## 4. Solver 调度

```text
LbmSolver.step(state_in, state_out)
  |
  +-- require FullF
  +-- P2.compute_fullf(validate=False)
  +-- copy static boundary fields
  +-- stream_fullf_to_populations
  +-- VofSurfaceBoundary.complete_fullf
  +-- complete ordinary domain boundary
  +-- collect/collide FullF
  +-- write observables
  +-- restore GAS invalid state
  +-- finalize_fixed_topology_vof
  +-- validate_p3_fixed_topology_state
  |
  +-- return; domain swaps buffers
```

若最后验证失败：

- solver 抛出异常；
- domain 不交换 buffer；
- `state_in` 保持当前权威状态；
- 已写坏的 `state_out` 不会成为 current state，下一次成功 step 会完全覆盖。

## 5. P3 不变量

### 5.1 surface link

```text
only target INTERFACE is modified
only in-domain/periodic source GAS triggers Eq.11
all 18 moving directions use correct opposite index
out-of-domain link is left to ordinary boundary
source GAS storage is never read
```

### 5.2 fixed topology

```text
cell_type_out == cell_type_in
GAS mass=0, phi=0
INTERFACE 0<phi<1
LIQUID phi≈1
mass≈density_out*phi on active cells
no direct D3Q19 L-G link
```

### 5.3 equilibrium/static surface

当：

```text
rho=1
u=0
p_atmos=c_s^2
gamma=0
```

Eq. (11) 必须使 equilibrium 成为固定点。

## 6. 测试计划

入口：

```bash
uv run python -m unittest newton.tests.test_lbm_vof_p3 -v
```

### 6.1 配置

```text
[x] 默认 p_atmos=c_s^2 -> rho_g=1
[x] non-finite/non-positive pressure 被拒绝
[x] nonzero surface tension 被拒绝到 P6
[x] HOME step 在写入前 fail-fast
[x] VOF open domain boundary 被拒绝
```

### 6.2 Eq. (11) 手算

```text
[x] 单 gas link 与独立 NumPy equilibrium 一致
[x] 18 个 moving directions 全覆盖
[x] source direction 使用 x-c_q
[x] opposite(q) 正确
[x] rest population 不被 surface overwrite
[x] state velocity 被直接使用
```

### 6.3 隔离

```text
[x] 修改 GAS 的全部 populations 不改变 active f_star
[x] GAS state step 后保持原值
[x] surface stage 不修改 state_in
```

### 6.4 完整 step

```text
[x] gamma=0 平面静止自由面单步固定
[x] 多步无质量漂移和伪速度增长
[x] P2 mass scratch 正确提交到 state_out
[x] type 固定、buffer 正常交换
[x] phi 越界时抛错且不交换 current buffer
```

### 6.5 回归

继续运行：

```text
P2 targeted
P1 targeted
P0 current frozen modules
directional streaming
Ruff F/I
```

## 7. 三份文档与单一提交

P3 单一提交必须包含：

```text
12-p3-engineering-plan.md
13-p3-completion-summary.md
14-p3-change-architecture.md
生产代码
P3 tests
capability/roadmap/README 更新
```

架构说明必须包含改动前后 Mermaid。

## 8. 退出门禁

```text
[x] Eq.11 方向、equilibrium 和时间层已冻结
[x] fixed p_atmos / gamma=0 已冻结
[x] FullF surface completion 在 collector 前执行
[x] gas storage independence 通过
[x] static planar free surface 单步/多步通过
[x] P2 mass 正确提交且 topology 固定
[x] 越界失败不交换 current buffer
[x] P3 targeted 通过
[x] P0-P2 CPU 回归通过
[x] Ruff F/I 与 diff check 通过
[x] summary 与 architecture 文档完成
```
