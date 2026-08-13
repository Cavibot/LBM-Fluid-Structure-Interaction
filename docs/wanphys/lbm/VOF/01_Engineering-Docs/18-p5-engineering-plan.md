# P5 新界面 kinetic 初始化工程计划

## 1. 阶段定义

| 项目 | 内容 |
|---|---|
| 阶段 | P5 |
| 前置阶段 | P0-P4 `CPU_ACCEPTED` |
| 当前实施状态 | `CPU_ACCEPTED` |
| 目标状态 | `CPU_ACCEPTED` |
| 物理范围 | FullF、D3Q19、GAS→INTERFACE kinetic 初始化与完整 gamma=0 moving-interface commit |
| 明确不包含 | PLIC、曲率、表面张力、HOME、开口 VOF boundary、CUDA、bubble/solid/foam |

P5 只回答：

> P4 将旧 GAS 提升为新 INTERFACE 后，怎样在不读取 gas kinetic 垃圾值的前提下，
> 为下一步构造合法的 post-collision FullF 状态？

P5 完成后，P4 的 `new_interface` 不再触发 gate，FullF gamma=0 moving-interface
step 可以事务式提交。

## 2. 证据与冻结决策

### 2.1 Home-FSLBM 参考：`REFERRED`

用户指定参考库：

```text
mrLbmSolverGpu3D.cu:479-568  surface_2 GAS->INTERFACE branch
```

参考实现：

- 遍历邻居；
- 只接受 fluid/interface/interface→fluid donor；
- 对 post-collision `rho/u` 做算术平均；
- 从平均 `rho/u` 构造 equilibrium kinetic；
- 同批 GAS→INTERFACE 不作为 donor；
- zero donor 使用 `rho=1,u=0` fallback。

### 2.2 论文边界：`UNSPECIFIED`

目标论文没有给出一般 VOF GAS→INTERFACE kinetic 初始化公式。其 moving-solid
fresh-node 方法是不同物理事件，P5 不套用。

### 2.3 本项目适配：`DERIVED`

冻结为 D3Q19/FullF 版本：

```text
donors = old active AND final active
neighborhood = D3Q19 18 moving links
weights = uniform arithmetic mean
time layer = provisional state_out post-collision rho/u
kinetic = project D3Q19 second-order equilibrium
zero donor = fail-fast
```

zero donor 理论上不应出现：每个 P4 新界面都由相邻 `I→L` 候选创建，该候选本身是
old active、final active donor。使用 reference fallback 会掩盖 topology/ordering
错误，因此本项目选择显式失败。

## 3. donor 资格

对目标：

```text
new_interface[x] = 1
```

邻格 `y=x-c_q` 必须同时满足：

```text
type_n[y] != GAS
final_type[y] != GAS
```

这等价于：

- unchanged LIQUID/INTERFACE：可用；
- INTERFACE→LIQUID：可用；
- LIQUID→INTERFACE：可用；
- INTERFACE→GAS：不可用；
- GAS→INTERFACE：不可用；
- old GAS：永远不可用。

因此同批新界面不能相互传播未初始化值，donor 读取与 thread 调度无关。

## 4. 时间层与平均公式

P5 在 P3 collision、`_write_observables()` 和 old-GAS restore 之后运行。

对有效 donor 集合 `D(x)`：

\[
\rho_{init}(x)=\frac{1}{|D|}\sum_{y\in D}\rho_{out}(y),
\]

\[
\mathbf{u}_{init}(x)=
\frac{1}{|D|}\sum_{y\in D}\mathbf{u}_{out}(y).
\]

读取：

```text
state_out.density
state_out.velocity_x/y/z
type_n
P4 final_type/new_interface
```

不读取：

```text
old/new GAS populations
other new-interface rho/u scratch
state_in gas macro fields
geometry
```

uniform arithmetic mean 与参考实现一致；P5 不引入 normal weighting。

## 5. FullF 初始化

使用项目统一：

```text
encoding.equilibrium_population(q, rho_init, u_init)
```

对 `q=0..18` 写入：

```text
state_out.f_post[q,x] = feq_q(rho_init,u_init)
```

并写：

```text
state_out.density = rho_init
state_out.velocity = u_init
state_out.force = rho_init * gravity
```

VOF 模式没有 Shan-Chen force；P5 不读取或复制 gas force 垃圾。

P5 不额外施加 half-force correction。该选择与 P3 Eq. (11) 使用直接 state velocity、
现有 uniform equilibrium initialization 以及参考实现一致；重力下初始化动量误差由
P7 benchmark 定量验收。

初始化后重新生成 MAC face velocity，使 cell-centered 与 face observables 属于同一
候选 state。

## 6. mass/phi 闭合

P5 不修改 P4 `mass_final`。新 density 写入后必须更新：

```text
phi_final[new_interface]
    = mass_final[new_interface] / rho_init
```

然后：

```text
P5 validate kinetic and VOF scratch
-> P4 transition.commit(state_out.vof)
-> final integrated validation
-> domain buffer swap
```

因此：

- VOF 总质量不受 kinetic 初始化影响；
- `mass` 仍是唯一守恒权威；
- density 变化只通过 `phi=mass/density` 反映；
- commit 前任何失败都不能改变 current `domain.state`。

## 7. retired GAS 与其他格点

```text
old active -> final GAS
    cell_type 使 kinetic 明确无效；不再读取

old active -> final active
    保留 P3 collision output

old GAS -> final GAS
    保留 P3 gas isolation 结果

old GAS -> final INTERFACE
    只写 P5 equilibrium output
```

P5 不清零 gas storage，也不把存储内容变成第二套有效性标志；唯一有效性权威是
`final_type`。

## 8. 数值与失败门禁

prepare pass 只写：

```text
donor_count
rho_init
ux_init / uy_init / uz_init
```

在修改 kinetic 前 host 验证：

- 每个 new interface `donor_count>0`；
- `rho_init` finite 且 `>0`；
- `u_init` finite；
- speed 不超过 donor 已有的项目 low-Mach 限制；
- 19 个 equilibrium populations finite；
- positivity 开启时均不低于 `population_floor`。

prepare 失败不得写 `state_out.f_post` 的新界面位置。

## 9. 生产代码边界

新增：

```text
vof/kinetic_init.py
vof/kinetic_init_kernels.py
```

对象：

```text
VofKineticInitializer
VofKineticInitializationResult
```

solver-owned scratch：

```text
donor_count
rho_init
ux_init
uy_init
uz_init
```

不新增 persistent VOF 字段。

## 10. 测试计划

入口：

```bash
uv run python -m unittest newton.tests.test_lbm_vof_p5 -v
```

### 10.1 donor 与手算

```text
[x] old active/final active donor 矩阵逐分支覆盖
[x] D3Q19 18 方向与 periodic seam
[x] uniform arithmetic rho/u mean 手算
[x] 同批 new interface 不互相作为 donor
[x] gas kinetic/macro 垃圾不影响结果
```

### 10.2 equilibrium

```text
[x] 19 populations 与独立 NumPy equilibrium 一致
[x] sum(f)=rho_init
[x] momentum(f)=rho_init*u_init
[x] populations finite/admissible
[x] force=rho_init*gravity
[x] mass 不变且 phi=mass/rho_init
```

### 10.3 隔离与失败

```text
[x] non-new active collision output 不变
[x] final GAS storage 不被当作 donor
[x] zero donor 在 kinetic 写入前失败
[x] prepare 和 initialize 输入无副作用
[x] 失败不交换 current buffer
```

### 10.4 集成

```text
[x] P4 positive front 现在可提交
[x] new interface 下一步可继续 stream/collide
[x] 连续创建/删除无 NaN/Inf
[x] static P3 fixed point 不回归
[x] P4 mass/topology/masks 不回归
```

### 10.5 回归

```text
P4 targeted
P3 targeted
P2 targeted
P1 targeted
P0 frozen core
directional streaming
Ruff F/I
git diff --check
```

## 11. 三份文档与单一提交

P5 单一提交必须包含：

```text
18-p5-engineering-plan.md
19-p5-completion-summary.md
20-p5-change-architecture.md
production code
P5 tests
P4 gate supersession updates
capability/roadmap/README updates
```

架构说明必须包含改动前后 Mermaid。

## 12. 退出门禁

```text
[x] donor eligibility/time layer/weights 已冻结
[x] zero donor 策略已冻结
[x] gas storage 从未作为 donor
[x] FullF equilibrium 19 方向正确
[x] new density 后 mass/phi 重新闭合
[x] VOF 总质量不受 kinetic init 影响
[x] non-new kinetic output 保持不变
[x] P4 new_interface gate 被安全移除
[x] moving-interface 连续 step 无 NaN/Inf
[x] P5 targeted 通过
[x] P0-P4 CPU 回归通过
[x] Ruff F/I 与 diff check 通过
[x] summary 与 architecture 文档完成
```
