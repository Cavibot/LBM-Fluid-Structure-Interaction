# FIX1 工程计划：有界 VOF 重分配与正式 Runtime Profile

状态：`CPU_ACCEPTED`

FIX1 是 P1–P8 之后的独立修复阶段，不引入固体、气泡或泡沫物理。该阶段必须以
一个提交完成，并同时交付本 Plan、完成总结和包含新旧 Mermaid 架构图的改动说明。

## 1. 触发缺陷

P4 当前在同一步内先 clamp sender，再把所有 sender share gather 到最终 INTERFACE。
receiver 没有容量约束，因此提交后的 `phi=mass/rho` 仍可能越出 `[0,1]`。P4 将该
越界视为下一步转换输入，P7 却以 `3e-6` 容差要求每个 committed phi 有界。

12^3、`gravity_z=-2e-4` dam-break 已复现：

```text
mass relative error = 4.45e-8
max velocity         = 1.152e-2
invalid topology     = 0
non-finite           = 0
phi_min              = -1.00148e-5
```

这不是流场发散，而是 P4/P7 契约冲突。

Home-FSLBM 参考路径在每步 canonicalization 时：

```text
mass -> clamp(mass, 0, rho)
phi  -> clamp(mass/rho, 0, 1)
被 clamp 的 signed mass -> massex side channel
下一步由邻居消费 massex
```

FIX1 保留该“一拍延迟 excess”原则，同时维持本工程的 D3Q19、gather-only、
double-buffer transaction 和 unchanged-LIQUID 守恒语义。

## 2. 冻结状态语义

`VofGridState` 新增：

```text
pending_excess        float32 grid
pending_receiver_count uint8 grid
reference_mass        host scalar
```

冻结定义：

```text
mass
  committed resident liquid mass
  GAS = 0
  INTERFACE in [0, density]
  unchanged LIQUID preserves its conservative mass

phi
  geometry/advection fill cache
  committed value is always in [0, 1]
  GAS = 0, LIQUID = 1

pending_excess
  signed total sender mass removed by the final clamp
  it is not geometric fill and is consumed exactly once on the next step

pending_receiver_count
  number of final D3Q19 INTERFACE receivers for pending_excess

reference_mass
  initialization-time closed-domain mass used by runtime diagnostics
```

正式守恒账本改为：

```text
M = sum(mass) + sum(pending_excess)
```

`pending_excess` 不是第二种流体，也不是第二份长期质量权威；它是 transaction 中
尚未进入 receiver resident mass 的一拍延迟传输量。

## 3. 冻结时序

### 3.1 下一步 residual gather

P2 在普通 population mass flux 之前，由每个当前 INTERFACE receiver 以固定
`q=1..18` 顺序 gather：

```text
received_excess(x)
  = sum(
      pending_excess(sender)
      / pending_receiver_count(sender)
    )

mass_tmp(x)
  = mass_n(x) + received_excess(x) + population_flux(x)
```

sender count 与 receiver gather 使用完全相同的 D3Q19/periodic 映射。因而每份
pending excess 被消费一次且仅一次。

### 3.2 P4 bounded commit

P4 继续使用 provisional `mass_tmp/density_out` 做 proposal 和 topology repair，
然后只做一次 canonicalization：

```text
final GAS:
  mass_final = 0
  phi_final = 0

final INTERFACE:
  mass_final = clamp(mass_tmp, 0, density_out)
  phi_final = mass_final / density_out

unchanged LIQUID:
  mass_final = mass_tmp
  phi_final = 1

INTERFACE -> LIQUID:
  mass_final = density_out
  phi_final = 1

pending_excess_out = mass_tmp - mass_final
```

本步禁止再把 `pending_excess_out` gather 回 `mass_final`。有 material excess 但
没有最终 INTERFACE receiver 时仍 fail-fast；小于 mass tolerance 的零 receiver
roundoff 可以保留在 side channel，且不得静默丢失。

P4 守恒门禁变为：

```text
sum(mass_final) + sum(pending_excess_out) == sum(mass_tmp)
```

## 4. Runtime Profile 冻结

公共 profile：

```text
strict
  每一步执行 P2/P3/P4/P5/P6 完整 host 验收和最终完整 ledger；
  用于单元测试、冻结与 headless acceptance。

sampled
  每 validation_interval 步执行一次与 strict 相同的完整验收；
  其余步骤只执行不可绕过的 transaction/epoch 门禁。
  validation_interval 必须在 [30, 100]，默认 60。

device
  跳过全场 host copy；
  GPU/设备 reduction 检查守恒、phi、topology、finite、density、speed、
  residual routing 和 surface pressure；
  每步只回读一个小型 diagnostics array。

off
  不执行全场 host 验收或设备 reduction；
  只保留 buffer 独立性、source/target epoch 和 geometry epoch 门禁。
```

库级 `LbmModel` 默认 `strict`，避免现有调用者静默降低验收等级。

## 5. P8 运行策略

P8 CLI 新增：

```text
--runtime-profile {auto,strict,sampled,device,off}
--validation-interval N
--steps-per-frame N
```

`auto` 冻结为：

```text
interactive viewer -> device, 4 simulation substeps/frame
null/headless      -> strict, 1 simulation step/frame
```

显式参数始终覆盖 auto。headless strict 继续逐步产生完整 ledger/CSV。interactive
device 使用 solver 的 compact device report，不再对所有场逐步 `.numpy()`。

## 6. Device diagnostics

一个设备 reduction kernel 生成一个固定长度 float64 array，至少包含：

```text
conserved total mass
pending excess total/max
phi min/max
max speed
interface count
invalid LIQUID-GAS links
non-finite count
non-positive active density count
invalid residual routing count
illegal type count
out-of-range phi count
invalid surface-pressure count
epoch / geometry_epoch
```

host 只回读该数组，并结合 host epoch scalar 构造 `VofDiagnostics`。closed-domain
mass 与 `reference_mass` 比较。

## 7. 不可绕过门禁

四个 profile 均必须检查：

```text
state_in is not state_out
VOF persistent arrays do not alias
source epoch >= 0
source geometry_epoch == source epoch
target epoch == source epoch + 1
target geometry_epoch == target epoch
reference_mass is unchanged
```

任何异常发生在 `LbmDomain` buffer swap 之前。

## 8. 验收矩阵

- P1：新增字段初始化为零、clone/copy 独立、reference mass 一致；
- P2：pending excess D3Q19 gather 独立 NumPy oracle、periodic 与守恒；
- P4：final phi 严格有界、pending excess/receiver count、两步消费、零 receiver；
- P5/P6：bounded `phi_final` 与 PLIC closure；
- P7：诊断总质量包含 pending excess，FullF/HOME 仍对齐；
- FIX1：四 profile dispatch、sample interval、device compact readback、off epoch gate；
- P8：auto resolution、interactive substeps、headless strict ledger；
- 长时：12^3、`gravity_z=-2e-4` 至少覆盖原 CPU 失败步，phi 始终有界；
- 回归：P0–P8 全集、Ruff F/I；
- CUDA：有 CUDA 时运行 device/profile smoke；当前无 CUDA则明确保持未验收。

## 9. 非目标

- receiver capacity constrained same-step redistribution；
- 多相气体动力学；
- trapped bubbles、foam、solid wetting/contact angle；
- 开口边界质量通量；
- 改变 PLIC、曲率或 Eq.11/Eq.12 公式；
- 声称未实际执行的 CUDA 验收。

## 10. 提交门禁

FIX1 只能形成一个提交。提交前必须：

1. 完成并更新本 Plan 状态；
2. 新增 FIX1 完成总结；
3. 新增 FIX1 改动文件说明及新旧 Mermaid；
4. 更新 capability status、roadmap 和 VOF README；
5. 运行目标测试、P0–P8 回归与 Ruff F/I；
6. 保留用户在 P8 示例中的既有未提交参数实验，除非用户明确要求纳入。

完成门禁：

```text
[x] bounded resident mass/phi 与 pending excess state
[x] next-step D3Q19 pending gather
[x] strict/sampled/device/off runtime profile
[x] single compact device diagnostics readback
[x] P8 interactive auto=device+4 substeps
[x] P8 headless auto=strict
[x] 750-step strong-gravity regression
[x] 140-test P0-P8+FIX1 regression
[x] Ruff F/I and git diff --check
[x] Plan/summary/change-architecture in one FIX1 commit
```

## 11. Post-freeze zero-receiver trace extension

The `64^3` CUDA dam-break reproduced a material zero-receiver route at epoch
832.  The compact device report must distinguish zero receivers from a stored
count mismatch and, only for the first failing sender, capture:

```text
center old -> proposed -> final type
18-neighbor valid mask
18-neighbor old/proposed/final INTERFACE and LIQUID masks
pending excess, stored count and recomputed final count
```

Each 18-bit mask is exactly representable in the existing float64 compact
buffer.  The buffer may grow from 16 to 33 scalars, but full fields must never
be copied to the host.  This extension is diagnostic only: zero-receiver mass
lifetime and redistribution semantics remain fail-fast and are not changed.

Acceptance target:

```text
[x] first failing cell is deterministic by sender category and linear index
[x] device report separates zero receiver and count mismatch
[x] old/proposed/final D3Q19 neighborhood is decoded in the exception
[x] normal device reports still agree with the host ledger
[x] synthetic zero-receiver regression covers the decoded trace
```
