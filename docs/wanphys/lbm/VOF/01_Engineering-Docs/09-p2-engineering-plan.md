# P2 固定拓扑质量平流工程计划

## 1. 阶段定义

| 项目 | 内容 |
|---|---|
| 阶段 | P2 |
| 前置阶段 | P0/P1 `CPU_ACCEPTED` |
| 当前实施状态 | `CPU_ACCEPTED` |
| 目标状态 | `CPU_ACCEPTED` |
| 物理范围 | FullF、固定 `cell_type`、无类型转换的守恒质量交换 |
| 明确不包含 | 自由面 population 补全、类型转换、质量重分配、新界面初始化、PLIC、曲率、表面张力、HOME 推进、CUDA 验收 |

P2 只回答：

> 在拓扑保持不变时，旧时间层的 D3Q19 populations 使液体质量如何沿格链交换？

P2 不开放 `LbmDomain.step()` 的 authoritative VOF 路径。P3 完成 gas-to-interface
population reconstruction 前，完整 LBM time step 继续 fail-fast；P2 的质量输运通过
独立、无副作用的 transport stage 验收。

## 2. 依据审计与冻结决策

### 2.1 论文印刷公式

目标论文 Wang et al. 2025 §3.1 Eq. (9)-(10) 写为：

$$
\phi(\mathbf{x},t+1)=\phi(\mathbf{x},t)
+\frac{1}{\rho(\mathbf{x},t)}
\sum_i\theta(\mathbf{x})
\left[
f_{\bar i}(\mathbf{x}+\mathbf{c}_i,t)-f_i(\mathbf{x},t)
\right],
$$

其中印刷版 `theta` 由中心格 `x` 的类型决定。论文还明确：

```text
VOF = liquid mass / density
rho 使用 (x,t)，即旧时间层
gas phase dynamics 不推进
```

这与经典 FSLBM 参考代码的邻居类型分段不完全相同，现有
[01-paper-audit.md](01-paper-audit.md) 已标为必须显式决策的差异。

### 2.2 首选参考实现

按用户指定的选择优先级，P2 首选：

```text
/Users/Alexandrite/Workspace/[Code-Ref]Home-FSLBM
```

关键证据：

- 2D `mrLbmSolverGpu2D.cu:461-570`；
- 3D `mrLbmSolverGpu3D.cu:806-929`；
- `AI_freesurface_implements_explanation.md` 的 mass advection 小节。

参考实现对中心 INTERFACE 格采用：

```text
neighbor LIQUID:
    weight = 1
neighbor INTERFACE:
    weight = 0.5 * (phi_center + phi_neighbor)
neighbor GAS:
    weight = 0
```

该方案来自经典 FSLBM/Körner/Lehmann 路径，不能标成目标论文印刷 Eq. (10) 的逐字实现。

### 2.3 冻结决策 D1：质量交换 scheme

P2 固定：

```text
vof_mass_scheme = "fslbm_neighbor"
```

对每个中心 active cell `x` 和 pull 方向 `q`，来源格：

$$
\mathbf{y}=\mathbf{x}-\mathbf{c}_q.
$$

格链裸通量定义为：

$$
J_q(\mathbf{x})
=f_q^n(\mathbf{y})-f_{\bar q}^n(\mathbf{x}).
$$

权重：

$$
w_q(\mathbf{x},\mathbf{y})=
\begin{cases}
1,& x\in L,\,y\in L\cup I,\\
1,& x\in I,\,y\in L,\\
\frac{\phi_x+\phi_y}{2},& x\in I,\,y\in I,\\
0,& \text{otherwise}.
\end{cases}
$$

质量更新：

$$
m^{tmp}(\mathbf{x})=m^n(\mathbf{x})+\sum_{q=1}^{18}w_qJ_q.
$$

该权重在每条 active-active 无向格链两端一致，配合对向 population 后，周期/封闭域
的全局质量交换反对称。

### 2.4 冻结决策 D2：density 时间层

P2 固定：

```text
mass flux reads populations at n
phi input is the canonical cache at n
provisional phi_tmp = mass_tmp / density^n
```

P2 不读取 streamed collector 的 `rho*`，也不读取未来 `state_out.density`。

理由：

- 论文 Eq. (9) 明确写 `rho(x,t)`；
- Home-FSLBM 在质量交换阶段读取当前 `fMom.rho`；
- P2 是独立 transport stage，没有 collision output。

`phi_tmp` 只是 P2 诊断输出。P3/P4 接入完整 time step 后，最终持久
`phi^{n+1}` 必须在选定的最终 density 时间层上重新闭合，不能把 P2 的 provisional
cache 当成已经提交的最终状态。

### 2.5 冻结决策 D3：interface-gas logical population

P2 固定：

```text
interface-gas mass weight = 0
```

因此：

- P2 不需要 Eq. (11)；
- 不读取 gas 格的 kinetic storage；
- 修改 gas populations 不得改变质量输运结果；
- Eq. (11) 只在 P3 为 LBM streaming 补齐缺失 population。

### 2.6 非周期 domain 边界

P2 将 non-periodic out-of-domain link 视为：

```text
impermeable for VOF mass transport
weight = 0
boundary mass flux = 0
```

普通开口边界的 VOF 注入/流出不在 P2 中定义，P7 综合扩展前继续 fail-fast 或保持不启用。

## 3. 数据所有权

P2 不增加持久 VOF 字段。`VofGridState` 仍只有：

```text
mass
phi
cell_type
```

新增 solver-owned scratch：

```text
mass_tmp
phi_tmp
mass_delta
```

scratch 不进入：

- state clone/copy/clear；
- `state_in/state_out` 持久契约；
- checkpoint；
- authoritative visualization。

P2 transport 只读：

```text
state_in.f_post
state_in.density
state_in.vof.mass
state_in.vof.phi
state_in.vof.cell_type
model.bc_periodic
```

P2 transport 只写自身 scratch，不修改 `state_in` 或 `state_out`。

## 4. 模块边界

新增：

```text
lbm/vof/advection.py
    VofMassTransport
    FullF dispatch
    scratch ownership
    no-mutation contract

lbm/vof/advection_kernels.py
    D3Q19 neighbor resolution
    fslbm_neighbor link weight
    fixed-topology mass update
```

修改：

```text
lbm/model.py
    显式只读配置 vof_mass_scheme="fslbm_neighbor"
    非 VOF 模式禁止更改该专用配置

lbm/vof/__init__.py
lbm/__init__.py
    导出 transport 和 scheme 契约

lbm/solver.py
    构造 P2 transport
    提供独立 compute_vof_mass_transport() 调度
    完整 VOF step 仍 fail-fast
```

## 5. 调用链

```text
LbmSolver.compute_vof_mass_transport(state_in)
  |
  +-- require interface_model == "vof"
  +-- require FullFLbmState
  +-- require initialized state.vof
  +-- VofMassTransport.compute_fullf(state_in)
        |
        +-- fixed_topology_mass_advection_fullf_kernel
        |     reads state_in only
        |     writes mass_tmp / mass_delta
        |
        +-- provisional_phi_from_density_n_kernel
              writes phi_tmp
  |
  +-- return read-only-by-contract scratch view
```

## 6. 运行时不变量

输入：

```text
density finite and > 0
mass/phi finite
cell_type legal
topology has no direct D3Q19 L-G link
FullF populations finite
```

输出：

```text
mass_tmp finite
phi_tmp finite
mass_delta finite
mass_tmp = mass_n + mass_delta
phi_tmp = mass_tmp / density_n
cell_type unchanged
state_in byte-for-byte unchanged
```

P2 不 clamp `mass_tmp/phi_tmp`。越过 `[0,1]` 是未来 P4 的转换请求来源；P2 只报告。

## 7. 测试计划

建议入口：

```bash
uv run python -m unittest newton.tests.test_lbm_vof_p2 -v
```

### 7.1 配置与调度

```text
[x] 默认 scheme 固定为 fslbm_neighbor
[x] 未知 scheme 在配置规范化时被拒绝
[x] HOME 调用 P2 明确 fail-fast，留给 P7
[x] authoritative domain.step 仍在写入前 fail-fast
```

### 7.2 单格链手算

逐一覆盖：

```text
L-L weight 1
L-I weight 1
I-L weight 1
I-I weight average(phi)
I-G weight 0
G-I weight 0
rest direction no flux
opposite index and pull source direction
```

### 7.3 守恒

```text
[x] 周期域随机场的全局 mass_delta 和为零
[x] active-active 格链两端 delta 等大反号
[x] 多方向随机有限 populations 与 NumPy oracle 一致
[x] non-periodic out-of-domain link 的 mass flux 为零
```

### 7.4 gas storage independence

保持 active cell populations 不变，仅两次随机化 GAS 的 19 个 populations：

```text
mass_tmp run A == mass_tmp run B
```

### 7.5 无副作用

调用前后逐数组比较：

```text
f_post
density
vof.mass
vof.phi
vof.cell_type
```

全部必须完全相等。

### 7.6 独立 NumPy oracle

oracle 必须：

- 独立实现 D3Q19 pull、opposite、periodic 和权重；
- 不调用 production kernel/helper；
- 在 `3^3`/`5^3` 网格逐格比较 `mass_delta/mass_tmp/phi_tmp`。

## 8. P0/P1 回归

P2 必须继续运行：

```text
P1: newton.tests.test_lbm_vof_p1
P0: 07-p0-baseline.md 冻结集合
directional: newton.tests.test_lbm_directional_streaming
```

P2 不得通过修改 P0/P1 测试期望消除回归。

## 9. 提交要求

P2 最终压成一个提交，必须同时包含：

```text
09-p2-engineering-plan.md
10-p2-completion-summary.md
11-p2-change-architecture.md
生产代码
P2 测试
能力状态表与路线图状态更新
```

提交前执行：

```text
P2 targeted tests
P1 tests
P0 frozen regressions
directional streaming
Ruff F/I
git diff --check
```

## 10. 退出门禁

```text
[x] D1/D2/D3 已由代码常量、文档和测试共同冻结
[x] FullF fixed-topology mass transport 已实现
[x] interface-gas flux 为零且不读 gas kinetic
[x] 周期/封闭域全局质量守恒
[x] 独立 NumPy oracle 逐格匹配
[x] transport 无持久状态副作用
[x] 完整 VOF step 仍 fail-fast
[x] P2 targeted tests 全部通过
[x] P0/P1 CPU 回归全部通过
[x] Ruff F/I 通过
[x] 完成总结和新旧架构图已随提交
```
