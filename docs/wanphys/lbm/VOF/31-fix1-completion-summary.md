# FIX1 有界重分配与 Runtime Profile 完成总结

## 1. 结论

FIX1 已达到：

```text
CPU_ACCEPTED
CUDA_NOT_ACCEPTED
```

该修复 supersede P4 的 same-step equal-share committed state 和 P8 的逐帧 strict
执行方式，不改变 Eq.9–12、PLIC、曲率符号或 FullF/HOME kinetic 语义，也未加入
固体、气泡或泡沫。

## 2. 根因与修复

旧 P4：

```text
sender clamp
-> equal share
-> receiver same-step gather
-> phi_final = gathered_mass / density
```

receiver 没有容量约束，导致 P4 文档允许 transient overshoot，而 P7 committed-state
validator 又要求 `phi` 位于 `[0,1]`。强重力 dam-break 因而在仍保持质量、finite、
低速和合法 topology 时被验收器终止。

FIX1 采用 Home-FSLBM 的一拍延迟 excess 原则：

```text
sender clamp
-> bounded resident mass / bounded phi
-> pending_excess + pending_receiver_count
-> next P2 step fixed-order D3Q19 gather
```

正式 closed-domain 守恒量为：

```text
M = sum(mass) + sum(pending_excess)
```

`pending_excess` 是 signed sender transfer，不参与几何。下一步每个 final
INTERFACE receiver gather `pending_excess/count`，并由 double buffer 覆盖旧 pending，
所以每份 excess 只消费一次。

## 3. Committed-state 契约

每次成功 swap 前：

```text
GAS:
  mass = 0
  phi = 0

INTERFACE:
  0 <= mass <= density
  phi = mass/density
  0 <= phi <= 1

LIQUID:
  unchanged LIQUID preserves conservative resident mass
  phi = 1

pending:
  finite signed mass
  material pending has exact final-INTERFACE receiver count

geometry_epoch = epoch
```

P4 守恒 validator 已从 `sum(mass_final)==sum(mass_pre)` 改为：

```text
sum(mass_final) + sum(pending_excess) == sum(mass_pre)
```

P1 初始化、clear、clone/copy 和双缓冲验证均覆盖新增字段与 `reference_mass`。

## 4. Runtime Profile

`LbmModel` 新增：

```python
vof_runtime_profile: str = "strict"
vof_validation_interval: int = 60
```

冻结行为：

| Profile | 每步行为 | 用途 |
|---|---|---|
| `strict` | P2–P6 host validator + 最终完整 ledger | 测试、冻结、headless acceptance |
| `sampled` | 每 30–100 步执行一次 strict；其余仅 transaction/epoch | 长运行抽样验收 |
| `device` | 设备 reduction + 单一 16-float64 readback | 交互、CUDA/大网格 runtime |
| `off` | 仅 buffer/transaction/source-target/geometry epoch | 明确选择的最低同步演示 |

四档 profile 均保留：

```text
distinct double buffers
no persistent-array alias
source initialized/current geometry
target epoch = source epoch + 1
target geometry_epoch = target epoch
stable reference_mass
exception before LbmDomain swap
```

## 5. Compact Device Diagnostics

`VofDeviceDiagnostics` 用一个设备 reduction buffer 检查：

```text
resident + pending total mass
pending total / max magnitude / routing
phi min/max/out-of-range cells
D3Q19 LIQUID-GAS adjacency
finite fields
positive active density
max velocity
interface count
legal cell type
positive Eq.12 surface pressure
epoch / geometry_epoch
```

host 每步只执行一次 `metrics.numpy()`，读取 16 个 float64；不复制 mass、phi、
cell_type、density、velocity、normal、curvature 或 PLIC 全场数组。

## 6. P8 运行策略

CLI 新增：

```text
--runtime-profile {auto,strict,sampled,device,off}
--validation-interval N
--steps-per-frame N
```

`auto`：

```text
interactive -> device + 4 simulation substeps/frame
headless    -> strict + 1 simulation step
```

因此默认交互不再运行逐步 Python topology oracle，也不会为绘制每一帧复制所有诊断
场。headless acceptance 继续逐步生成完整 ledger/CSV。

显式示例：

```bash
uv run --frozen python -m wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak \
  --device cuda:0 --runtime-profile device --steps-per-frame 4
```

抽样模式：

```bash
uv run --frozen python -m wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak \
  --runtime-profile sampled --validation-interval 60 \
  --steps-per-frame 4
```

## 7. 原始崩溃回归

CPU Warp、FullF：

```text
grid             = (12,12,12)
gravity_z        = -2e-4
profile          = device
steps            = 750
final mass       = 461.9999867
relative error   = 2.875e-8
phi              = [0,1]
vmax             = 1.1524e-2
interface cells  = 256
pending total    = 1.7165e-4
max pending      = 9.7871e-5
epoch            = geometry_epoch = 750
invalid pending  = 0
non-finite       = 0
```

该运行覆盖原 CPU 第 474 步与 CUDA 第 690 步后的失败区间。

## 8. Runtime 基准

同一 CPU、`12³`、`gravity_z=-2e-4`，warm cache 后 20 步：

| Profile | 秒/步 | steps/s |
|---|---:|---:|
| strict | 0.09437 | 10.6 |
| sampled（未遇 cadence） | 0.00238 | 420.2 |
| device | 0.00274 | 365.0 |
| off | 0.00252 | 396.2 |

该数据只用于证明 host validation bottleneck 已被 profile 隔离，不作为跨设备性能
承诺。device 相对 strict 在此 CPU fixture 上约为 34 倍吞吐。

## 9. 验收记录

验收日期：2026-07-30。

FIX1 + P1–P8 targeted：

```text
Ran 95 tests
OK (skipped=2 CUDA)
```

最终组合回归：

```text
P0 core:       39
directional:    6
P1-P8+FIX1:    95
total:        140

Ran 140 tests
OK (skipped=2 CUDA)
```

代码质量：

```text
Ruff F/I: passed
git diff --check: passed
```

当前本机 Warp 1.12.0 明确报告 `CUDA not enabled in this build`。CUDA smoke 已加入，
但未执行，因此不能标记 `CUDA_ACCEPTED`。用户提供的 RTX 4070 SUPER 日志只作为缺陷
复现证据，不替代修复后 CUDA 验收。

## 10. 已知边界

FIX1 不支持或不声明：

```text
CUDA acceptance
same-step receiver-capacity flow solver
open-boundary VOF mass flux
moving/cut-link solid coupling
bubble pressure/tracking
foam
off profile 的 material acceptance
```

## 11. Phase 提交

FIX1 的代码、测试、Plan、完成总结和改动架构说明位于同一个唯一提交中。提交身份以
包含本文件的 `git log` 记录为准。
