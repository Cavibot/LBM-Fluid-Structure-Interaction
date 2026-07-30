# P8 authoritative VOF dam-break 可视化工程计划

## 1. 阶段目标

P8 为 P7 已验收的 closed-domain dam-break 增加用户可直接启动的 authoritative VOF
可视化入口、无窗口 smoke 和诊断 CSV。P8 不修改 P0–P7 物理。

本阶段包含：

```text
reusable dam-break phi initializer
FullF/HOME scene wrapper
authoritative phi volume rendering
authoritative interface point/normal overlay
headless bounded runner
per-step VofDiagnostics ledger
optional CSV output
CPU launch command
conditional CUDA launch command
```

明确不包含：

```text
solid / FSI
bubble / foam
open VOF boundary
new mass/topology/geometry algorithms
automatic CUDA acceptance
```

## 2. authoritative 可视化来源

可视化必须直接读取：

```text
state.vof.phi
state.vof.cell_type
state.vof.normal
state.vof.epoch / geometry_epoch
```

禁止读取：

```text
state.debug_mock_sc_to_vof
Shan-Chen density threshold
单独重算的 render-only phi/normal/curvature
```

`DebugVofView` 增加只读 authoritative adapter，但保留 `source` 标签：

```text
source="authoritative_vof"
```

这样现有 `VofInterfaceVisualizer` 可以复用 compact/render kernel，而不会混淆
observation 和 authoritative 状态。

## 3. 场景

默认网格：

```text
(48, 12, 32)
```

液柱：

```text
x < nx/3
z < 2*nz/3
all y
```

液体外包一层 D3Q19 INTERFACE，避免非法 LIQUID-GAS 邻接。边界为 static
bounce-back，gravity 沿 `-z`，无固体、气泡或泡沫。

默认：

```text
encoding=fullf
collision=srt
gravity_z=-2e-5
surface_tension=0
device=cpu unless user explicitly selects an available CUDA device
```

## 4. 运行模式

### 4.1 交互可视化

Newton/WanPhys viewer：

```text
state.vof.phi -> ScreenSpaceFluidRenderer
state.vof.cell_type/normal -> VofInterfaceVisualizer
```

phi threshold 只用于渲染采样，不写回物理状态。

### 4.2 headless

```text
--viewer null --num-frames N
```

不创建 GL renderer，严格执行 N 步并返回/打印最终 diagnostics。`--csv PATH` 时每步写
一行标准库 CSV。

## 5. CLI

```text
--encoding fullf|home
--collision srt|nocm_mrt
--grid-res NX NY NZ
--cell-size
--tau
--gravity-z
--surface-tension
--dam-x-fraction
--dam-z-fraction
--steps-per-frame
--print-every
--show-normals / --no-show-normals
--csv PATH
--device cpu|cuda:0
--viewer gl|null|usd|rerun|viser
--num-frames
```

HOME + unsupported collision 由既有 model contract 拒绝。请求 CUDA 但
`wp.is_cuda_available()==False` 时在分配前给出明确错误。

## 6. 诊断输出

每步保存：

```text
step
total_mass
relative_mass_error
phi_min / phi_max
invalid_liquid_gas_adjacency_count
non_finite_count
nonpositive_active_density_count
max_velocity
interface_cell_count
epoch / geometry_epoch
```

每步调用 P7 `validate_vof_diagnostics()`，所以 visual demo 不能绕过物理门禁。

## 7. 测试计划

- initializer 生成合法 D3Q19 interface layer；
- scene 使用 `interface_model="vof"`，无 SC debug state；
- authoritative adapter 引用原数组且不修改；
- headless FullF smoke 有界结束并通过质量/拓扑/finite/epoch；
- headless HOME smoke；
- CSV header/行数/最终值与 ledger 一致；
- fake viewer render 读取 `state.vof.phi`；
- parser 默认值和关键参数；
- 请求不可用 CUDA 明确失败；
- P0-P7 基线与 P8 targeted 合并为 P0-P8 全量回归持续通过。

### 7.1 实施中基线缺陷处置

按 README 的默认 `(48,12,32)`、30-step headless 命令验收时，旧 P6
inclusion-exclusion 二分在第 8 步出现 near-axis PLIC 闭合失败：

```text
normal ~= (1.1247e-4, 1.1244e-4, 1)
phi = 0.4994719923
wrong offset = 0.09771065
reconstructed volume = 0.59771065
```

这是 float32 小分母与相消导致的真实几何错误，不能缩短 demo 或放宽
`volume_tolerance`。按用户指定的参考优先级，采用
`[Code-Ref]Home-FSLBM/inc/3D/gpu/mrUtilFuncGpu3D.h` 所指向的
Scardovelli–Zaleski/Kawano symmetry-reduced analytical inverse：

```text
keep the frozen centered-cube half-space and 1e-4 component policy
replace device inclusion-exclusion bisection with analytical inversion
evaluate the cancellation-sensitive reduced branch in float64
use a stable host divided-difference forward-volume oracle
add a near-axis regression fixture from the failing state
```

这属于 P8 为保证默认可视化可运行而纳入同一提交的 P6 数值 supersession，不改变
mass transport、topology、curvature 符号或 surface-pressure 语义。

## 8. 退出门禁

```text
[x] headless FullF/HOME smoke 通过
[x] visual source 是 authoritative VOF
[x] 每步 diagnostics 与 P7 阈值一致
[x] CSV 可复现且行数正确
[x] 交互 render 路径可由 fake viewer 验证
[x] CPU 启动方式已验证
[x] CUDA 启动方式有 availability guard
[x] 不引入 solid/bubble/foam
[x] P0-P8 全部 CPU 回归通过
[x] Ruff F/I 与 git diff --check 通过
[x] P8 三份文档与代码进入唯一提交
```

## 9. 阶段状态

```text
CPU_ACCEPTED
CPU: FullF/HOME visual and headless accepted
CUDA: not accepted; Warp 1.12.0 reports CUDA not enabled
```

验收日期：2026-07-30。P8 targeted 为 9/9，新增 P6 near-axis regression 1 项；
P0-P8 合计 132 tests OK，其中 P7
CUDA 同场景测试按设备能力跳过 1 项。
