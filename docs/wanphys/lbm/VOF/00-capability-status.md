# VOF 能力与状态表

本文是当前 LBM/VOF 能力的状态入口，严格区分 Shan-Chen 调试观察与未来的守恒
VOF。公式依据见 [01-paper-audit.md](01-paper-audit.md)，阶段门禁见
[06-development-roadmap.md](06-development-roadmap.md)。P0-P8 当前均为
`CPU_ACCEPTED`；冻结证据分别见 [07-p0-baseline.md](07-p0-baseline.md)、
[08-p1-engineering-plan.md](08-p1-engineering-plan.md) 和
[10-p2-completion-summary.md](10-p2-completion-summary.md)、
[13-p3-completion-summary.md](13-p3-completion-summary.md)、
[16-p4-completion-summary.md](16-p4-completion-summary.md)、
[19-p5-completion-summary.md](19-p5-completion-summary.md)、
[22-p6-completion-summary.md](22-p6-completion-summary.md)、
[25-p7-completion-summary.md](25-p7-completion-summary.md)、
[28-p8-completion-summary.md](28-p8-completion-summary.md)。P4/P7 重分配缝隙与交互
同步开销由 [31-fix1-completion-summary.md](31-fix1-completion-summary.md)
正式 supersede；CUDA 尚未验收。

## 1. 状态含义

| 状态 | 含义 |
|---|---|
| `EXISTING` | 当前 API/契约已经存在 |
| `OBSERVE_CPU_ACCEPTED` | 调试观察路径已有 CPU 针对性测试；不是 VOF 物理验收 |
| `AUTHORITATIVE_CPU_ACCEPTED` | authoritative 状态与初始化通过 CPU 验收；不代表可推进 |
| `TRANSPORT_CPU_ACCEPTED` | 独立 fixed-topology transport 通过 CPU 验收；不代表完整 step 可运行 |
| `SURFACE_CPU_ACCEPTED` | FullF fixed-topology、零表面张力自由面 step 通过 CPU 验收 |
| `TOPOLOGY_CPU_ACCEPTED` | transition/redistribution scratch 与无新界面提交通过 CPU 验收 |
| `KINETIC_CPU_ACCEPTED` | GAS→INTERFACE FullF kinetic 初始化与 moving step 通过 CPU 验收 |
| `GEOMETRY_CPU_ACCEPTED` | authoritative normal/PLIC/curvature 与 Eq.12 通过 CPU 验收 |
| `INTEGRATION_CPU_ACCEPTED` | FullF/HOME 与综合 closed-domain 场景通过 CPU 验收 |
| `VISUAL_CPU_ACCEPTED` | authoritative dam-break 可视化、headless 与 CSV 通过 CPU 验收 |
| `FIX_CPU_ACCEPTED` | 有界 excess side channel 与 runtime profile 通过 CPU 验收 |
| `NOT_STARTED` | 尚未实现 |
| `FAIL_FAST` | 当前只允许显式拒绝 |
| `CUDA_NOT_ACCEPTED` | 尚未完成 CUDA 验收 |

## 2. 当前配置契约

```python
interface_model: Literal["off", "shan_chen", "vof"] = "off"
debug_vof_observation: bool = False
vof_mass_scheme: Literal["fslbm_neighbor"] = "fslbm_neighbor"
vof_atmosphere_pressure: float = 1 / 3
vof_surface_tension: float = 0
vof_transition_epsilon: float = 1e-4
vof_runtime_profile: Literal["strict", "sampled", "device", "off"] = "strict"
vof_validation_interval: int = 60
```

`interface_model` 是界面物理的唯一权威选择。`force_model` 是不可由调用者传入的内部
执行组合，只由 `interface_model` 与 gravity 单向生成：

```text
off        + no gravity -> none
off        + gravity    -> gravity
shan_chen  + no gravity -> shan_chen
shan_chen  + gravity    -> gravity+shan_chen
vof         + no gravity -> none
vof         + gravity    -> gravity
```

必要的拒绝规则：

- 未知 `interface_model`：`ValueError`；
- 非 Shan-Chen 模式配置 `debug_vof_observation=True`：`ValueError`；
- 非 Shan-Chen 模式配置非零 `G`：`ValueError`；
- VOF 配置 moving-wall 或 cut-link：`NotImplementedError`；
- VOF 请求 `requires_grad=True`：`NotImplementedError`；
- VOF 配置非正/非有限大气压力：`ValueError`；
- VOF 配置负数或非有限表面张力：`ValueError`；
- VOF 配置开口 domain boundary：`NotImplementedError`；当前 closed-domain 范围不支持；

旧字段 `vof_debug_labels` 和调用者可配置的 `force_model` 均已删除，没有 deprecated
alias。

## 3. 当前字段、容器与接口

| 字段或接口 | 状态 | 写入者 | 当前语义 |
|---|---|---|---|
| `interface_model` | `EXISTING` | 配置层 | 选择 `off/shan_chen/vof` |
| `debug_vof_observation` | `OBSERVE_CPU_ACCEPTED` | 配置层 | 只控制 SC 调试观察 |
| internal `force_model` | `EXISTING` | `LbmModel.__post_init__` | interface + gravity 的执行组合 |
| `state.vof` | `AUTHORITATIVE_CPU_ACCEPTED` | P1 initializer | VOF 模式分配的正式状态 |
| `state.vof.mass` | `FIX_CPU_ACCEPTED` | initializer/P4 | committed resident liquid mass |
| `state.vof.pending_excess` | `FIX_CPU_ACCEPTED` | P4 clamp | 下一步由 final INTERFACE 邻居消费的 signed 总 excess |
| `state.vof.pending_receiver_count` | `FIX_CPU_ACCEPTED` | P4 topology | pending sender 的 D3Q19 receiver 数 |
| `state.vof.reference_mass` | `FIX_CPU_ACCEPTED` | initializer | closed-domain 初始化守恒参考 |
| `state.vof.phi` | `FIX_CPU_ACCEPTED` | bounded mass/density | 始终位于 `[0,1]` 的几何/平流占据率缓存 |
| `state.vof.cell_type` | `AUTHORITATIVE_CPU_ACCEPTED` | 严格 phi 分类 | GAS/INTERFACE/LIQUID 拓扑 |
| `vof_mass_scheme` | `TRANSPORT_CPU_ACCEPTED` | 配置层 | 固定为已审计的 `fslbm_neighbor` |
| `VofMassTransport` | `INTEGRATION_CPU_ACCEPTED` | solver scratch | 共享 logical-population 固定拓扑质量交换 |
| `mass_tmp/phi_tmp/mass_delta` | `TRANSPORT_CPU_ACCEPTED` | P2 transport | 旧时间层 provisional scratch，不是持久状态 |
| `compute_vof_mass_transport()` | `INTEGRATION_CPU_ACCEPTED` | solver | FullF/HOME 独立 transport 测试入口 |
| `vof_atmosphere_pressure` | `SURFACE_CPU_ACCEPTED` | 配置层 | 固定气相压力，默认 `c_s^2` |
| `vof_surface_tension` | `GEOMETRY_CPU_ACCEPTED` | 配置层 | 非负常数 gamma，Eq.12 唯一消费方 |
| `VofSurfaceBoundary` | `GEOMETRY_CPU_ACCEPTED` | solver stage | Eq.11 + Eq.12 gas-to-interface population 补全 |
| `compute_vof_surface_populations()` | `INTEGRATION_CPU_ACCEPTED` | solver | FullF/HOME pull + Eq.11/Eq.12 测试入口 |
| FullF/HOME `step()` | `INTEGRATION_CPU_ACCEPTED` | solver/domain | closed-domain moving interface 可提交 |
| `validate_p3_fixed_topology_state()` | `SURFACE_CPU_ACCEPTED` | validator | 越界/非法 topology 在 buffer swap 前失败 |
| `vof_transition_epsilon` | `TOPOLOGY_CPU_ACCEPTED` | 配置层 | 论文阈值，默认 `1e-4` |
| `VofTopologyTransition` | `TOPOLOGY_CPU_ACCEPTED` | solver scratch | proposal/topology/clamp/redistribution |
| `VofTransitionResult` | `TOPOLOGY_CPU_ACCEPTED` | solver scratch | final VOF 与 new/retired/changed masks |
| P4 pending excess emit | `FIX_CPU_ACCEPTED` | transition kernels | 本步 bounded commit；下一步固定顺序 gather |
| `VofKineticInitializer` | `INTEGRATION_CPU_ACCEPTED` | solver scratch | donor mean + FullF populations/HOME moments |
| `VofKineticInitializationResult` | `KINETIC_CPU_ACCEPTED` | solver scratch | donor count 与 rho/u 初始化值 |
| GAS→INTERFACE handoff | `KINETIC_CPU_ACCEPTED` | P4→P5 | kinetic 成功后才提交 VOF |
| `state.vof.normal` | `GEOMETRY_CPU_ACCEPTED` | P6 geometry | 液体到气体 Parker–Youngs unit normal |
| `state.vof.plic_offset` | `GEOMETRY_CPU_ACCEPTED` | P6 geometry | centered unit-cube liquid plane offset |
| `state.vof.curvature` | `GEOMETRY_CPU_ACCEPTED` | P6 geometry | `-0.5 div(normal)` mean curvature |
| `state.vof.epoch/geometry_epoch` | `GEOMETRY_CPU_ACCEPTED` | commit/geometry | 过期几何硬门禁 |
| `VofInterfaceGeometry` | `GEOMETRY_CPU_ACCEPTED` | solver stage | final phi/type normal→PLIC→curvature |
| shared logical `f_post` provider | `INTEGRATION_CPU_ACCEPTED` | solver scratch | FullF direct、HOME ten-moment decode |
| `VofDiagnostics` | `INTEGRATION_CPU_ACCEPTED` | read-only host ledger | mass/topology/finite/speed/epoch 门禁 |
| `VofDeviceDiagnostics` | `FIX_CPU_ACCEPTED` | device reduction | 单一 16-float64 compact readback |
| `vof_runtime_profile` | `FIX_CPU_ACCEPTED` | solver policy | strict/sampled/device/off |
| `vof_validation_interval` | `FIX_CPU_ACCEPTED` | sampled policy | `[30,100]`，默认 60 |
| `LbmDomain.initialize_vof()` | `AUTHORITATIVE_CPU_ACCEPTED` | domain | candidate 双缓冲完整初始化 |
| `validate_initialized_vof_state()` | `AUTHORITATIVE_CPU_ACCEPTED` | validator | 只读检查实际数组不变量 |
| `DebugMockScToVofState` | `OBSERVE_CPU_ACCEPTED` | debug observer | 与正式 VOF 分离的调试容器 |
| `state.debug_mock_sc_to_vof.phi` | `OBSERVE_CPU_ACCEPTED` | density mapping | density 派生的有界显示填充率 |
| `.cell_type` | `OBSERVE_CPU_ACCEPTED` | debug classifier | 调试 GAS/INTERFACE/LIQUID 标签 |
| `.normal` | `OBSERVE_CPU_ACCEPTED` | `InterfaceGeometry` | 调试 Parker-Youngs 法向 |
| `.epoch/.normal_valid_epoch` | `OBSERVE_CPU_ACCEPTED` | observer/geometry | 调试状态有效期 |
| `update_debug_mock_sc_to_vof()` | `OBSERVE_CPU_ACCEPTED` | solver | density 写出后刷新调试副本 |
| `DebugVofView` | `VISUAL_CPU_ACCEPTED` | view adapter | SC debug 或 authoritative VOF 的只读渲染输入 |
| `VofInterfaceVisualizer` | `VISUAL_CPU_ACCEPTED` | visualization | 点云压缩、坐标转换与 normal line |
| P8 dam-break scene/headless | `FIX_CPU_ACCEPTED` | example | visual auto=device+4 substeps；headless auto=strict |
| P8 volume/interface render | `VISUAL_CPU_ACCEPTED` | example | 直接读取 `state.vof.phi/cell_type/normal` |
| observation 物理不变性 | `OBSERVE_CPU_ACCEPTED` | 测试约束 | 开关不改变 populations/宏观量/force |
| 单次 hydrodynamic closure | `EXISTING` | force pipeline | positivity 两侧均保持第一步 `u=0.5g` |
| CUDA observation | `CUDA_NOT_ACCEPTED` | — | 当前构建没有 CUDA |

## 4. 写入权限

| 模式 | 权威相态 | 调试观察写入 | Visualizer |
|---|---|---|---|
| `off` | 单相 LBM | 禁止 | 无 VOF view |
| `shan_chen` | density | 只写 `state.debug_mock_sc_to_vof` | 只读 `DebugVofView` |
| `vof`（FIX1） | resident mass + pending excess / bounded phi / type | 不得覆盖正式状态 | FullF/HOME closed-domain + authoritative visual |

P0 数据流：

```text
Shan-Chen density
  -> DebugMockScToVofObserver.update_from_density()
  -> state.debug_mock_sc_to_vof
  -> DebugVofView
  -> VofInterfaceVisualizer
```

Visualizer 可以做点云压缩、坐标转换、颜色和线段等渲染数据计算，但不能计算或修改
`phi/cell_type/normal/curvature`。调试标签也不能决定 streaming、collision、
population 有效性或边界行为。

## 5. Authoritative VOF

| 能力 | 状态 | 未来权威写入者 |
|---|---|---|
| physical VOF construction/initialization | `AUTHORITATIVE_CPU_ACCEPTED` | P1 initializer |
| physical VOF time stepping | `INTEGRATION_CPU_ACCEPTED` | P7 FullF/HOME closed-domain |
| `mass/phi/cell_type` initialization | `AUTHORITATIVE_CPU_ACCEPTED` | P1 initializer |
| FullF fixed-topology mass advection | `TRANSPORT_CPU_ACCEPTED` | P2 mass transport scratch |
| HOME fixed-topology mass advection | `INTEGRATION_CPU_ACCEPTED` | P7 logical population provider |
| transition writes | `TOPOLOGY_CPU_ACCEPTED` | P4 scratch/conditional commit |
| authoritative normal | `GEOMETRY_CPU_ACCEPTED` | P6 geometry step |
| FullF logical population access for mass flux | `TRANSPORT_CPU_ACCEPTED` | P2 direct FullF access |
| shared FullF/HOME logical population provider | `INTEGRATION_CPU_ACCEPTED` | P7 adapter |
| gas-to-interface completion | `SURFACE_CPU_ACCEPTED` | P3 `VofSurfaceBoundary` |
| topology repair / redistribution | `TOPOLOGY_CPU_ACCEPTED` | P4 deterministic gathers |
| new-interface kinetic initialization | `INTEGRATION_CPU_ACCEPTED` | FullF populations / HOME moments |
| PLIC / curvature / surface tension | `GEOMETRY_CPU_ACCEPTED` | P6 geometry/free-surface pressure |
| authoritative dam-break visual/headless | `VISUAL_CPU_ACCEPTED` | P8 example/read-only adapter |

正式 VOF 状态与 `debug_vof_observation` 无关：即使不显示也必须分配。FIX1 后的
closed-domain 守恒量为：

```text
sum(state.vof.mass) + sum(state.vof.pending_excess)
```

`pending_excess` 是一拍延迟的 sender transfer，不参与几何；`phi` 在 commit 后
始终有界。P2 的独立
FullF fixed-topology `mass_tmp/phi_tmp/mass_delta` 仍不修改持久状态；P3 将
`mass_tmp` 接入完整 FullF step，并用最终 `density_out` 反算持久 `phi`。`mass`
仍是守恒权威。P4 获得 `cell_type` 的唯一 transition 写权限，并在 scratch 中先完成
拓扑与质量验证。P5 消费 `new_interface`，完成 kinetic 与 phi 重闭合后才提交
domain 候选状态。

P6 在三个守恒字段之外增加 epoch-scoped 派生几何。每步 P4/P5 commit 后先推进
`vof.epoch`，再从 final `phi/type` 重建 normal/PLIC/curvature，使
`geometry_epoch == epoch` 后才允许下一步 Eq.12 读取。

P2 冻结语义：

```text
scheme = fslbm_neighbor
mass/density/populations/phi read at n
interface-gas mass flux = 0
gas kinetic storage is never read for VOF mass
non-periodic out-of-domain VOF mass flux = 0
```

P3 冻结语义：

```text
rho_g = vof_atmosphere_pressure / c_s^2
surface_tension = 0
Eq.11 reads type/velocity/opposite population at n
surface completion overwrites only GAS -> INTERFACE pull links
gas persistent kinetic storage is not a physical input
phi(n+1) = mass_tmp / density_out
topology remains fixed; required transitions fail before buffer swap
```

P4 冻结语义：

```text
epsilon_phi = 1e-4 with >= / <= comparisons
I-to-L has reference-order priority over adjacent I-to-G
topology and redistribution are gather-only
receivers are final D3Q19 INTERFACE neighbors
positive and negative excess use one signed formula
committed phi remains bounded; excess is persisted for next-step gather
zero receiver with material excess fails before commit
new_interface mask is the P5 kinetic handoff
```

P5 冻结语义：

```text
donor = old active AND final active D3Q19 neighbor
same-batch new interfaces and all old GAS are excluded
rho/u use uniform provisional state_out donor means
new FullF populations are D3Q19 equilibrium
force = rho_init * gravity
mass_final remains read-only; phi_final = mass_final/rho_init
zero donor fails before kinetic write
```

P6 冻结语义：

```text
normal = liquid-to-gas Parker-Youngs normal
PLIC plane: dot(normal,r) <= offset in centered unit cube
kappa = -0.5*div(normal), so a convex liquid sphere has kappa=-1/R
rho_g = (p_atmos-2*gamma*kappa)/cs2
gamma=0 follows the exact P3 density branch
non-finite/non-positive rho_g fails before surface population writes
PLIC is derived geometry and never advects mass
geometry_epoch must equal vof.epoch
```

P7 冻结语义：

```text
FullF uses direct logical post-collision populations
HOME decodes rho/rho*u/rho*S into solver-owned logical population scratch
P2 and P3 consume the same logical population contract
old GAS HOME moments/macros are restored before topology transition
new HOME interfaces receive equilibrium rho/rho*u/rho*S
unchanged LIQUID preserves conserved VOF mass and canonical phi=1
closed-domain diagnostics include pending excess and require mass error<=5e-6
zero D3Q19 L-G links
CUDA test exists but is skipped when wp.is_cuda_available() is false
```

P8 冻结语义：

```text
dam-break initialization creates one legal D3Q19 interface layer
visual volume density is exactly state.vof.phi
interface overlay reads state.vof.cell_type/normal at the current geometry epoch
headless auto mode creates no viewer and uses strict per-step validation
optional CSV serializes the same per-step diagnostics
FullF and HOME share the same scene and acceptance path
explicit unavailable CUDA requests fail before allocation
no solid, bubble or foam physics is introduced
```

FIX1 supersession：

```text
strict  -> every-step full host acceptance
sampled -> full host acceptance every 30-100 steps
device  -> device reduction + one 16-float64 readback per step
off     -> transaction and epoch gates only

interactive CLI auto -> device + 4 simulation substeps/frame
headless CLI auto    -> strict + 1 simulation step

committed phi in [0,1]
pending excess is consumed once by next-step D3Q19 INTERFACE gather
closed-domain mass = resident mass + pending excess
```

P8 对 P6 PLIC 数值实现的 supersession：

```text
unit-cube plane semantics and 1e-4 component policy remain unchanged
float32 inclusion-exclusion bisection is replaced by the
Scardovelli-Zaleski/Kawano symmetry-reduced analytical inverse
the cancellation-sensitive reduced inverse executes in float64
host volume closure uses a cancellation-safe divided-difference form
near-axis normals around 1e-4 have a dedicated regression
```
