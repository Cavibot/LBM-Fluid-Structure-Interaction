# P8 authoritative VOF dam-break 可视化完成总结

## 1. 结论

P8 已达到：

```text
CPU_ACCEPTED
CUDA_NOT_ACCEPTED
```

仓库现在提供用户可直接启动的 authoritative VOF dam-break。交互渲染、无窗口运行和
CSV 均读取同一个 P7 已验收状态，不引入另一套演示专用 VOF 物理。

## 2. 场景与状态来源

默认场景为 `(48, 12, 32)` closed domain：

```text
liquid column: x < nx/3 and z < 2*nz/3
interface: one legal D3Q19 layer around the column
gravity: -2e-5 along z
surface tension: 0
solid / bubble / foam: absent
```

渲染数据流：

```text
state.vof.phi -> ScreenSpaceFluidRenderer
state.vof.cell_type/normal -> DebugVofView.from_authoritative_vof()
                            -> VofInterfaceVisualizer
```

adapter 为 zero-copy 只读引用，并在渲染前要求
`vof.geometry_epoch == vof.epoch`。它不会从 Shan-Chen density 生成 phi，也不计算
render-only normal/curvature。

## 3. headless 与诊断

`--viewer null` 不创建 viewer 或 GL renderer，只执行有限步数。每步调用 P7 的：

```text
collect_vof_diagnostics()
validate_vof_diagnostics()
```

因此同时检查：

```text
closed-domain mass error <= 5e-6
phi in [0,1] within tolerance
zero D3Q19 LIQUID-GAS links
finite state and positive active density
low-Mach velocity bound
geometry_epoch == epoch
```

`--csv PATH` 将同一 per-step ledger 写成确定列序的 CSV；它不是从控制台文本重新解析。

## 4. 启动方式

CPU 交互：

```bash
uv run --frozen python -m wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak
```

CPU headless + CSV：

```bash
uv run --frozen python -m wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak \
  --viewer null --num-frames 30 --print-every 10 \
  --csv vof-dambreak.csv
```

HOME：

```bash
uv run --frozen python -m wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak \
  --encoding home
```

条件 CUDA：

```bash
uv run --frozen python -m wanphys.examples.lbm.fluid_grid_lbm_vof_dambreak \
  --device cuda:0
```

最后一条只在 `wp.is_cuda_available()` 为真时可用。当前 Warp 1.12.0 明确报告
`CUDA not enabled in this build`，所以 CUDA 仍未验收；显式请求会在 model 分配前
给出错误。

## 5. 验收记录

验收日期：2026-07-30。

基础 HEAD：

```text
ad57cfc — feat(lbm): complete P7 HOME VOF integration
```

P8 targeted：

```text
Ran 9 tests
OK
```

默认 30-step visual/headless 验收暴露并修复了一个旧 P6 near-axis PLIC float32
消减缺陷。另增加 1 个 P6 regression；没有放宽 `2e-5` volume closure tolerance。

组合回归：

```text
P0 core:       39
directional:    6
P1:            18
P2:             8
P3:             9
P4:            10
P5:            10
P6:            12
P7:            11
P8:             9
total:        132

Ran 132 tests
OK (skipped=1 P7 CUDA)
```

实际 CPU CLI smoke：

```text
grid=(48,12,32), steps=30
final mass=4259.9999
final relative mass error=1.731e-8
phi=[0,1]
interface cells=456
final max velocity=6.156e-4
epoch=geometry_epoch=30
```

代码质量：

```text
Ruff F/I: passed
git diff --check: passed
```

## 6. P8 测试覆盖

- D3Q19 legal interface-layer initializer；
- authoritative scene 且没有 SC debug state 或 embedded solid；
- zero-copy authoritative visual adapter 与 stale-geometry 拒绝；
- FullF 3-step bounded headless；
- HOME 3-step bounded headless；
- CSV header、行数、epoch 与最终质量；
- parser defaults 与关键覆盖参数；
- CUDA availability guard；
- fake viewer 证明 volume 与 interface overlay 读取 authoritative arrays。

## 7. P6 near-axis PLIC supersession

默认场景第 8 步首次生成：

```text
normal ~= (1.1247e-4, 1.1244e-4, 1)
phi ~= 0.499472
```

旧 float32 inclusion-exclusion evaluator 在二分中把接近零的正确 offset 算成
`0.09771065`，host 重建得到 `0.59771065`。这不是容差问题。

依据用户指定的第一优先参考
`/Users/Alexandrite/Workspace/[Code-Ref]Home-FSLBM/inc/3D/gpu/mrUtilFuncGpu3D.h`
及其标注的 Scardovelli–Zaleski/Kawano 解，P8 改为：

```text
symmetry-reduced analytical plane-offset inversion
float64 reduced branch on device
cancellation-safe divided-difference forward oracle on host
same centered cube / liquid half-space / 1e-4 component policy
```

near-axis fixture 在不放宽 `volume_tolerance=2e-5` 的条件下通过，默认 30 步命令也完整
结束。

## 8. 已知边界

P8 不支持或不声明：

```text
CUDA acceptance
open VOF domain boundaries
moving/cut-link solids or FSI
bubble pressure or disconnected bubble tracking
foam
validation by screenshots alone
```

## 9. Phase 提交

P8 的代码、测试、Plan、完成总结和改动架构说明位于同一个 P8 唯一提交中。提交身份以
包含本文件的 `git log` 记录为准。
