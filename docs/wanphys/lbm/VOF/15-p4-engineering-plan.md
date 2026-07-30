# P4 类型转换与守恒质量重分配工程计划

## 1. 阶段定义

| 项目 | 内容 |
|---|---|
| 阶段 | P4 |
| 前置阶段 | P0-P3 `CPU_ACCEPTED` |
| 当前实施状态 | `CPU_ACCEPTED` |
| 目标状态 | `CPU_ACCEPTED` |
| 物理范围 | FullF、D3Q19、类型候选、确定性拓扑修复、守恒 excess/deficit 重分配 |
| 明确不包含 | GAS→active kinetic 初始化、PLIC、曲率、表面张力、HOME、开口 VOF boundary、CUDA |

P4 只回答：

> 当 P2/P3 得到的 provisional mass 越过界面阈值时，怎样得到无 LIQUID–GAS
> 直接邻接且质量守恒的下一时刻 VOF 状态？

P4 必须输出准确的 `new_interface` mask，但不得为这些格点伪造 kinetic state。
完整 moving-interface domain step 仍在新界面出现时于 buffer swap 前 fail-fast，直到
P5 初始化 kinetic。

## 2. 证据与来源分级

### 2.1 论文直接要求：`PAPER`

目标论文 §3.1（PDF 第 6 页）明确：

```text
epsilon_phi = 1e-4
phi >= 1 + epsilon_phi -> INTERFACE to LIQUID
phi <= 0 - epsilon_phi -> INTERFACE to GAS
converted phi is clamped to 1 or 0
clamped volume is redistributed to neighbors to preserve mass
```

论文没有给出 GPU 候选数据结构、冲突规则、接收权重或 zero-receiver 策略。

### 2.2 Home-FSLBM 参考：`REFERRED`

用户指定参考库 3D 实现：

```text
mrLbmSolverGpu3D.cu:444-477   surface_1
mrLbmSolverGpu3D.cu:479-602   surface_2
mrLbmSolverGpu3D.cu:604-690   surface_3
mrLbmSolverGpu3D.cu:983-998   candidate generation
```

可复用语义：

- interface→liquid 时，把相邻 gas 提升为 interface；
- interface→gas 时，把相邻 liquid 降为 interface；
- opposing conversion 被 interface halo 取消；
- clamp 后的 excess/deficit 等分给相邻 active/interface；
- 没有接收者时不丢弃质量。

不直接复制：

- 参考实现从多个 CUDA thread 原地写邻居 flag，结果依赖并行写冲突；
- 参考实现包含 bubble/islet、D3Q27、D3Q7 等本项目明确排除能力；
- 参考实现把 excess 保存到下一步独立数组，本项目 P1 已冻结持久 VOF 只有
  `mass/phi/cell_type`。

### 2.3 本项目适配：`DERIVED`

P4 冻结为：

```text
read-only candidate pass
-> deterministic gather topology pass
-> canonicalize and compute sender excess
-> deterministic gather redistribution
-> validate scratch
-> commit only when P5 kinetic is not required
```

所有邻域均使用 D3Q19 的 18 个 moving links，与 P1 topology 和 P2 transport 一致。

## 3. 候选与阈值契约

新增配置：

```text
vof_transition_epsilon = 1e-4
```

要求 finite 且 `0 <= epsilon < 0.5`。

对旧类型 `INTERFACE`：

```text
phi_pre = mass_pre / density_out

phi_pre >= 1 + epsilon -> proposed LIQUID
phi_pre <= 0 - epsilon -> proposed GAS
otherwise              -> proposed INTERFACE
```

对旧 `LIQUID/GAS`，proposal 初始保持旧类型。比较符严格冻结为 `>=` 与 `<=`。

候选 kernel：

- 只读 `mass_pre/density_out/type_n`；
- 写独立 `proposed_type`；
- 不修改 state、mass 或邻格。

## 4. 确定性拓扑修复

对每个格点独立 gather 邻居 proposal，并冻结参考实现 `surface_1` 先于
`surface_2` 的优先级：

```text
old GAS + any neighboring proposed LIQUID
    -> final INTERFACE

old LIQUID + any neighboring proposed GAS
    -> final INTERFACE

old INTERFACE proposed GAS + any neighboring old INTERFACE proposed LIQUID
    -> cancel I->G, final INTERFACE

old INTERFACE proposed LIQUID
    -> keep LIQUID proposal

otherwise
    -> keep proposal
```

该规则同时覆盖：

- `I→L` 周围旧 GAS 被提升为 INTERFACE；
- `I→G` 周围旧 LIQUID 被降为 INTERFACE；
- 相邻 `I→L` 与 `I→G` 冲突时 `I→L` 获得参考实现一致的确定性优先级，
  `I→G` 被取消为 INTERFACE。

因为输入已由 P1 验证不存在 L-G 直接邻接，任何新 L-G 边都必然至少有一个转换端；
上述 gather 一次即可消除它。输出仍必须再次执行完整 D3Q19 topology validator，
证明该等价推理没有实现偏差。

转换 mask：

```text
new_interface  = type_n == GAS and type_final == INTERFACE
retired_active = type_n != GAS and type_final == GAS
changed        = type_n != type_final
```

P5 以 `new_interface` 为唯一 GAS→INTERFACE kinetic 初始化清单。

## 5. clamp 与 excess 定义

使用 `density_out` 和 topology-repaired `type_final`：

```text
LIQUID:
    mass_base = density_out
    phi_base  = 1

GAS:
    mass_base = 0
    phi_base  = 0

INTERFACE:
    mass_base = clamp(mass_pre, 0, density_out)
    phi_base  = mass_base / density_out

excess = mass_pre - mass_base
```

为避免每步把 float32 roundoff 反复变成真实 redistribution，若一个最终 LIQUID
满足 `abs(mass_pre-density_out) <= 2e-6`，则保留该 mass 并由
`phi=mass/density_out` 闭合；validator 仍要求它在同一容差内等于 canonical
`mass=density, phi=1`。真正的转换 excess 至少越过 `epsilon=1e-4`，不受此规则
影响。

因此：

- 正 excess 表示需要向邻居增加质量；
- 负 excess 表示需要从邻居移除质量；
- excess 的总和必须完整进入 receiver contribution；
- 不建立第二份持久质量权威。

## 6. 重分配契约

### 6.1 接收者

接收者严格为：

```text
type_final == INTERFACE
and D3Q19 moving-link neighbor of sender
```

不向 LIQUID/GAS 分配，也不包含 sender 自身。每个 sender 对其接收者等权分配：

```text
share(sender) = excess(sender) / receiver_count(sender)
```

这与 Home-FSLBM 的等分策略一致，并把“active 邻居”收窄为能合法改变填充率的最终
INTERFACE。

### 6.2 并行策略

禁止原地 scatter。每个 receiver 以固定 `q=1..18` 顺序 gather：

```text
mass_final(receiver)
  = mass_base(receiver)
  + sum(share(neighbor_sender))
```

这避免 atomic 加法和线程调度依赖。多个 sender、多个 receiver 的结果由同一方向
顺序唯一确定。

### 6.3 zero receiver

若：

```text
abs(excess) > 2e-6 and receiver_count == 0
```

则 P4 fail-fast，不提交半完成状态。P4 不采用以下不守契约 fallback：

- 丢弃 excess；
- 修改 GAS/LIQUID canonical mass；
- 写入未记账的持久 side buffer；
- 全域非局部重分配。

`2e-6` 只用于忽略 float32 canonicalization 尾差，不改变转换阈值。

### 6.4 receiver 二次越界

接收后 INTERFACE 允许暂时越过 `[0,1]`。这不是 clamp 或质量丢失：

- `mass_final` 和 `phi_final=mass_final/density_out` 保持一致；
- 下一时间步候选 pass 会按同一阈值处理；
- final GAS 严格 canonical，LIQUID 在 `2e-6` 内 canonical；
- 非有限值和无界异常继续 fail-fast。

该选择对应参考实现“本步产生 excess、下一步接收后再分类”的离散时序，同时避免
新增持久 `massex` 权威。

## 7. P4/P5 事务边界

P4 transition 输出先保存在 solver-owned scratch。

```text
no new_interface:
    commit VOF scratch to state_out
    validate
    domain step may swap

any new_interface:
    keep current state_in authoritative
    raise NotImplementedError before domain buffer swap
    P5 will initialize kinetic then authorize commit
```

`state_out` 是候选缓冲区，即使被临时写入也不得在失败时成为 `domain.state`。

独立测试入口可以读取完整 P4 scratch，用于验收 positive-front 情况而不要求 P5。

## 8. 生产代码边界

新增：

```text
vof/transition.py
vof/transition_kernels.py
```

计划对象：

```text
VofTopologyTransition
VofTransitionResult
```

solver-owned scratch：

```text
proposed_type
final_type
mass_base
excess
receiver_count
mass_final
phi_final
new_interface
retired_active
changed
```

P4 不修改 `vof/state.py`，这些数组不属于 checkpoint/clone 的持久状态。

## 9. 测试计划

入口：

```bash
uv run python -m unittest newton.tests.test_lbm_vof_p4 -v
```

### 9.1 配置与阈值

```text
[x] 默认 epsilon=1e-4
[x] non-finite、negative、>=0.5 被拒绝
[x] 阈值等号使用 >= / <=
[x] 阈值内保持 INTERFACE
```

### 9.2 topology

```text
[x] I→L 邻接 GAS 被提升为 INTERFACE
[x] I→G 邻接 LIQUID 被降为 INTERFACE
[x] opposing candidates 按参考优先级确定性解决
[x] periodic seam 使用相同 D3Q19 邻域
[x] 输出无直接 L-G link
[x] new_interface mask 精确
```

### 9.3 redistribution 手算

```text
[x] 单 positive excess 等分
[x] 单 negative deficit 等分
[x] 多 sender / 多 receiver 固定顺序 gather
[x] LIQUID/GAS canonical
[x] mass_final = density_out * phi_final
[x] zero receiver 显式失败
[x] 输入数组无副作用
```

### 9.4 独立 oracle 与集成

```text
[x] 3^3/5^3 NumPy oracle 逐数组匹配
[x] 周期随机合法拓扑质量守恒
[x] 重复运行结果逐位一致
[x] 无 transition 的 P3 固定点不回归
[x] new_interface 出现时 P5 gate 前不交换 current buffer
```

### 9.5 回归

```text
P3 targeted
P2 targeted
P1 targeted
P0 frozen core
directional streaming
Ruff F/I
git diff --check
```

## 10. 三份文档与单一提交

P4 单一提交必须包含：

```text
15-p4-engineering-plan.md
16-p4-completion-summary.md
17-p4-change-architecture.md
production code
P4 tests
P3 supersession test updates
capability/roadmap/README updates
```

架构说明必须包含改动前后 Mermaid。

## 11. 退出门禁

```text
[x] threshold/type proposal 已冻结
[x] topology repair 无原地邻居写入
[x] final topology 无 L-G 直接邻接
[x] excess/deficit 定义与接收者集合已冻结
[x] redistribution 使用 deterministic gather
[x] zero-receiver 明确失败且不提交
[x] 单/多 sender 质量守恒
[x] new_interface mask 准确交接 P5
[x] P4 targeted 通过
[x] P0-P3 CPU 回归通过
[x] Ruff F/I 与 diff check 通过
[x] summary 与 architecture 文档完成
```
