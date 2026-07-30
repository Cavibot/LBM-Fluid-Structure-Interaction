# VOF 能力与状态表

本文是当前 LBM/VOF 能力的状态入口，严格区分 Shan-Chen 调试观察与未来的守恒
VOF。公式依据见 [01-paper-audit.md](01-paper-audit.md)，阶段门禁见
[06-development-roadmap.md](06-development-roadmap.md)。P0-P4 当前均为
`CPU_ACCEPTED`；冻结证据分别见 [07-p0-baseline.md](07-p0-baseline.md)、
[08-p1-engineering-plan.md](08-p1-engineering-plan.md) 和
[10-p2-completion-summary.md](10-p2-completion-summary.md)、
[13-p3-completion-summary.md](13-p3-completion-summary.md)、
[16-p4-completion-summary.md](16-p4-completion-summary.md)，CUDA 尚未验收。

## 1. 状态含义

| 状态 | 含义 |
|---|---|
| `EXISTING` | 当前 API/契约已经存在 |
| `OBSERVE_CPU_ACCEPTED` | 调试观察路径已有 CPU 针对性测试；不是 VOF 物理验收 |
| `AUTHORITATIVE_CPU_ACCEPTED` | authoritative 状态与初始化通过 CPU 验收；不代表可推进 |
| `TRANSPORT_CPU_ACCEPTED` | 独立 fixed-topology transport 通过 CPU 验收；不代表完整 step 可运行 |
| `SURFACE_CPU_ACCEPTED` | FullF fixed-topology、零表面张力自由面 step 通过 CPU 验收 |
| `TOPOLOGY_CPU_ACCEPTED` | transition/redistribution scratch 与无新界面提交通过 CPU 验收 |
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
- VOF 配置非零表面张力：`NotImplementedError`，等待 P6；
- VOF 配置开口 domain boundary：`NotImplementedError`，等待 P7；
- HOME authoritative VOF step：`NotImplementedError`，等待 P7；
- P4 FullF step 产生 `new_interface`：`NotImplementedError`，等待 P5 kinetic init。

旧字段 `vof_debug_labels` 和调用者可配置的 `force_model` 均已删除，没有 deprecated
alias。

## 3. 当前字段、容器与接口

| 字段或接口 | 状态 | 写入者 | 当前语义 |
|---|---|---|---|
| `interface_model` | `EXISTING` | 配置层 | 选择 `off/shan_chen/vof` |
| `debug_vof_observation` | `OBSERVE_CPU_ACCEPTED` | 配置层 | 只控制 SC 调试观察 |
| internal `force_model` | `EXISTING` | `LbmModel.__post_init__` | interface + gravity 的执行组合 |
| `state.vof` | `AUTHORITATIVE_CPU_ACCEPTED` | P1 initializer | VOF 模式分配的正式状态 |
| `state.vof.mass` | `AUTHORITATIVE_CPU_ACCEPTED` | P1 initializer | 唯一守恒权威 |
| `state.vof.phi` | `AUTHORITATIVE_CPU_ACCEPTED` | `mass/density` 反算 | 派生占据率缓存 |
| `state.vof.cell_type` | `AUTHORITATIVE_CPU_ACCEPTED` | 严格 phi 分类 | GAS/INTERFACE/LIQUID 拓扑 |
| `vof_mass_scheme` | `TRANSPORT_CPU_ACCEPTED` | 配置层 | 固定为已审计的 `fslbm_neighbor` |
| `VofMassTransport` | `TRANSPORT_CPU_ACCEPTED` | solver scratch | 无副作用 FullF 固定拓扑质量交换 |
| `mass_tmp/phi_tmp/mass_delta` | `TRANSPORT_CPU_ACCEPTED` | P2 transport | 旧时间层 provisional scratch，不是持久状态 |
| `compute_vof_mass_transport()` | `TRANSPORT_CPU_ACCEPTED` | solver | 独立 P2 transport 测试入口 |
| `vof_atmosphere_pressure` | `SURFACE_CPU_ACCEPTED` | 配置层 | 固定气相压力，默认 `c_s^2` |
| `vof_surface_tension` | `SURFACE_CPU_ACCEPTED` | 配置层 | P3 必须为零，非零留给 P6 |
| `VofSurfaceBoundary` | `SURFACE_CPU_ACCEPTED` | solver stage | Eq.11 gas-to-interface population 补全 |
| `compute_vof_surface_populations()` | `SURFACE_CPU_ACCEPTED` | solver | 独立 FullF pull + Eq.11 测试入口 |
| FullF surface `step()` | `TOPOLOGY_CPU_ACCEPTED` | solver/domain | gamma=0；无新界面时可提交 |
| `validate_p3_fixed_topology_state()` | `SURFACE_CPU_ACCEPTED` | validator | 越界/非法 topology 在 buffer swap 前失败 |
| `vof_transition_epsilon` | `TOPOLOGY_CPU_ACCEPTED` | 配置层 | 论文阈值，默认 `1e-4` |
| `VofTopologyTransition` | `TOPOLOGY_CPU_ACCEPTED` | solver scratch | proposal/topology/clamp/redistribution |
| `VofTransitionResult` | `TOPOLOGY_CPU_ACCEPTED` | solver scratch | final VOF 与 new/retired/changed masks |
| P4 deterministic gather | `TOPOLOGY_CPU_ACCEPTED` | transition kernels | 无原地邻居写、无 atomic scatter |
| `LbmDomain.initialize_vof()` | `AUTHORITATIVE_CPU_ACCEPTED` | domain | candidate 双缓冲完整初始化 |
| `validate_initialized_vof_state()` | `AUTHORITATIVE_CPU_ACCEPTED` | validator | 只读检查实际数组不变量 |
| `DebugMockScToVofState` | `OBSERVE_CPU_ACCEPTED` | debug observer | 与正式 VOF 分离的调试容器 |
| `state.debug_mock_sc_to_vof.phi` | `OBSERVE_CPU_ACCEPTED` | density mapping | density 派生的有界显示填充率 |
| `.cell_type` | `OBSERVE_CPU_ACCEPTED` | debug classifier | 调试 GAS/INTERFACE/LIQUID 标签 |
| `.normal` | `OBSERVE_CPU_ACCEPTED` | `InterfaceGeometry` | 调试 Parker-Youngs 法向 |
| `.epoch/.normal_valid_epoch` | `OBSERVE_CPU_ACCEPTED` | observer/geometry | 调试状态有效期 |
| `update_debug_mock_sc_to_vof()` | `OBSERVE_CPU_ACCEPTED` | solver | density 写出后刷新调试副本 |
| `DebugVofView` | `OBSERVE_CPU_ACCEPTED` | view adapter | 只读渲染输入 |
| `VofInterfaceVisualizer` | `OBSERVE_CPU_ACCEPTED` | visualization | 点云压缩、坐标转换与 normal line |
| observation 物理不变性 | `OBSERVE_CPU_ACCEPTED` | 测试约束 | 开关不改变 populations/宏观量/force |
| 单次 hydrodynamic closure | `EXISTING` | force pipeline | positivity 两侧均保持第一步 `u=0.5g` |
| CUDA observation | `CUDA_NOT_ACCEPTED` | — | 当前构建没有 CUDA |

## 4. 写入权限

| 模式 | 权威相态 | 调试观察写入 | Visualizer |
|---|---|---|---|
| `off` | 单相 LBM | 禁止 | 无 VOF view |
| `shan_chen` | density | 只写 `state.debug_mock_sc_to_vof` | 只读 `DebugVofView` |
| `vof`（P4） | `state.vof.mass/phi/cell_type` | 不得覆盖正式状态 | FullF transition；新界面等待 P5 |

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
| physical VOF time stepping | `TOPOLOGY_CPU_ACCEPTED` | P4 FullF；new interface 仍 gate |
| `mass/phi/cell_type` initialization | `AUTHORITATIVE_CPU_ACCEPTED` | P1 initializer |
| FullF fixed-topology mass advection | `TRANSPORT_CPU_ACCEPTED` | P2 mass transport scratch |
| HOME fixed-topology mass advection | `NOT_STARTED` | P7 logical population provider |
| transition writes | `TOPOLOGY_CPU_ACCEPTED` | P4 scratch/conditional commit |
| authoritative normal | `NOT_STARTED` | geometry step |
| FullF logical population access for mass flux | `TRANSPORT_CPU_ACCEPTED` | P2 direct FullF access |
| shared FullF/HOME logical population provider | `NOT_STARTED` | P7 adapter |
| gas-to-interface completion | `SURFACE_CPU_ACCEPTED` | P3 `VofSurfaceBoundary` |
| topology repair / redistribution | `TOPOLOGY_CPU_ACCEPTED` | P4 deterministic gathers |
| new-interface kinetic initialization | `NOT_STARTED` | initialization step |
| PLIC / curvature / surface tension | `NOT_STARTED` | geometry/free-surface pressure |

正式 VOF 状态与 `debug_vof_observation` 无关：即使不显示也必须分配。P2 的独立
FullF fixed-topology `mass_tmp/phi_tmp/mass_delta` 仍不修改持久状态；P3 将
`mass_tmp` 接入完整 FullF step，并用最终 `density_out` 反算持久 `phi`。`mass`
仍是守恒权威。P4 获得 `cell_type` 的唯一 transition 写权限，并在 scratch 中先完成
拓扑与质量验证；产生新界面时仍不提交 domain 当前状态。

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
zero receiver with material excess fails before commit
new_interface mask is the P5 kinetic handoff
```
