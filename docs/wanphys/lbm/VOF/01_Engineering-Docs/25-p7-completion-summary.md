# P7 综合验收、HOME 与 CPU/CUDA 扩展完成总结

## 1. 结论

P7 已达到：

```text
CPU_ACCEPTED
CUDA_NOT_ACCEPTED
```

FullF 与 HOME 现在使用同一 authoritative VOF 子系统。HOME 不再在 step 入口
fail-fast；它通过共享逻辑 D3Q19 population 视图进入 P2/P3，并在 P5 写回
equilibrium `rho/rho*u/rho*S`。

当前 Warp 1.12.0 构建明确报告 `CUDA not enabled in this build`。P7 已加入同输入
CPU/CUDA 条件测试，但本环境只能记录一个 CUDA skip，不能宣称 `CUDA_ACCEPTED`。

## 2. 共享 population 契约

```text
FullF -> state.f_post
HOME  -> home_to_populations(10 persistent moments) -> solver scratch
```

两者都形成 post-collision `f_n[q,x]`，随后：

```text
P2 reads f_n for mass exchange
P3/P6 reads local opposite f_n for Eq.11/Eq.12
streaming forms shared f_star
```

HOME scratch 不持久化、不进入 clone/copy/checkpoint，也不能与 streamed `f_star`
混用。

equilibrium 初始条件下，HOME 三阶 Hermite decoder 相对 FullF 二阶 equilibrium 的
最大 population 差约 `4.6e-6`；因此 provider 单步门禁冻结为 `atol=5e-6`，
cross-encoding 综合场景使用计划中更宽但预先冻结的 `2e-4/3e-4` 阈值。

## 3. HOME GAS 与新界面

old GAS collision 后恢复：

```text
10 HOME kinetic moment fields
density / velocity / force
```

所以任意 GAS storage 垃圾不会影响 active surface completion 或 P2 mass。

old GAS→final INTERFACE 使用与 FullF 相同 donor prepare，然后写：

```text
rho
rho*ux, rho*uy, rho*uz
rho*ux^2, rho*uy^2, rho*uz^2
rho*ux*uy, rho*ux*uz, rho*uy*uz
density / velocity / force
phi=mass/rho
```

这些字段由独立 HOME moment oracle 验证。

## 4. P4 长时组合缺陷与修复

60-step 静止平面账本暴露了 P4 的隐藏误判：

```text
old behavior:
  every final LIQUID mass projected to current density_out
  tiny compressibility/float32 difference treated as excess
  deep LIQUID has no interface receiver -> false zero-receiver failure
```

失败量从 `2.0265579e-6` 起并随步数缓慢累积。提高 `mass_tolerance` 只能延迟失败，
不能修复语义，因此 P7 没有放宽 P4 的 `2e-6` 门禁。

修复后：

```text
unchanged LIQUID:
  preserve conservative mass
  phi remains canonical 1

INTERFACE -> LIQUID:
  canonicalize against density_out
  redistribute material excess
```

P2 输入闭合也相应只对 INTERFACE 强制 `mass≈rho*phi`；LIQUID 的 `phi=1` 与有限守恒
mass 分别验证。这与参考 `calculate_phi()` 对 LIQUID 直接返回 1 的语义一致。

## 5. 综合诊断

新增只读 `VofDiagnostics`：

```text
total/initial mass
closed boundary flux
relative mass error
phi range
D3Q19 LIQUID-GAS adjacency count
non-finite count
non-positive active density count
max velocity
interface count
VOF/geometry epochs
```

closed-domain 门禁：

```text
relative mass error <= 5e-6
boundary mass flux = 0
committed phi in [0,1] within 3e-6
D3Q19 L-G adjacency count = 0
non-finite count = 0
active density > 0
max velocity <= configured low-Mach limit
geometry_epoch == epoch
```

## 6. 场景证据

CPU 场景包括：

- equilibrium logical-population FullF/HOME 对照；
- HOME P2 与 Eq.11 FullF oracle；
- HOME GAS storage 隔离；
- advancing front HOME equilibrium moment 初始化；
- 20-step 静止平面逐步 differential；
- 12-step periodic uniform-fill translation；
- gravity 与 nonzero-gamma short runs；
- FullF/HOME 各 60-step closed-domain ledger；
- 15-step headless dam-break quantitative time series；
- diagnostics negative fixtures。

headless dam-break 每步记录质量误差、最大速度、界面数、有限性、拓扑和 epoch；不以
截图作为 P7 验收。

## 7. 验收记录

验收日期：2026-07-30。

基础 HEAD：

```text
13348a2 — feat(lbm): complete P6 VOF interface geometry
```

P7 targeted：

```text
Ran 11 tests
OK (skipped=1 CUDA)
```

组合回归：

```text
P0 core:       39
directional:    6
P1:            18
P2:             8
P3:             9
P4:            10
P5:            10
P6:            11
P7:            11
total:        122

Ran 122 tests
OK (skipped=1 CUDA)
```

代码质量：

```text
Ruff F/I: passed
git diff --check: passed
```

## 8. 设备状态

```text
CPU FullF: accepted
CPU HOME: accepted
CUDA FullF: not accepted
CUDA HOME: not accepted
```

CUDA 测试在可用构建上要求：

```text
cell_type exact
mass/phi/curvature atol=rtol=5e-5
same scenario and step count as CPU
```

当前 skip 原因是运行时无 CUDA device，而不是测试失败。两者在能力表中必须严格区分。

## 9. 已知边界

P7 不能用于声称：

```text
CUDA 已验收
open VOF domain boundary 已支持
HOME 精确复现论文 D3Q27
solid / bubble / foam 已支持
P8 可视化入口已经完成
```

## 10. P8 交接

P8 可直接复用：

```text
authoritative state.vof
VofDiagnostics
headless dam-break initializer/thresholds
FullF/HOME CPU path
conditional device selection
```

P8 只新增用户可启动的可视化与 smoke CLI，不改变 P0–P7 物理。

## 11. Phase 提交

P7 的代码、测试、Plan、完成总结和架构差异说明位于同一个 P7 唯一提交中。提交身份
以包含本文件的 `git log` 记录为准。
