# VOF 能力与状态表

本文是当前 LBM/VOF 能力的状态入口，严格区分 Shan-Chen 调试观察与未来的守恒
VOF。公式依据见 [01-paper-audit.md](01-paper-audit.md)，阶段门禁见
[06-development-roadmap.md](06-development-roadmap.md)。

## 1. 状态含义

| 状态 | 含义 |
|---|---|
| `EXISTING` | 当前 API/契约已经存在 |
| `OBSERVE_CPU_ACCEPTED` | 调试观察路径已有 CPU 针对性测试；不是 VOF 物理验收 |
| `NOT_STARTED` | 尚未实现 |
| `FAIL_FAST` | 当前只允许显式拒绝 |
| `CUDA_NOT_ACCEPTED` | 尚未完成 CUDA 验收 |

## 2. 当前配置契约

```python
interface_model: Literal["off", "shan_chen", "vof"] = "off"
debug_vof_observation: bool = False
```

`interface_model` 是界面物理的唯一权威选择。`force_model` 是不可由调用者传入的内部
执行组合，只由 `interface_model` 与 gravity 单向生成：

```text
off        + no gravity -> none
off        + gravity    -> gravity
shan_chen  + no gravity -> shan_chen
shan_chen  + gravity    -> gravity+shan_chen
vof                      -> NotImplementedError (P0)
```

必要的拒绝规则：

- 未知 `interface_model`：`ValueError`；
- `off + debug_vof_observation`：`ValueError`；
- 非 Shan-Chen 模式配置非零 `G`：`ValueError`；
- `interface_model="vof"`：`NotImplementedError`。

旧字段 `vof_debug_labels` 和调用者可配置的 `force_model` 均已删除，没有 deprecated
alias。

## 3. 当前字段、容器与接口

| 字段或接口 | 状态 | 写入者 | 当前语义 |
|---|---|---|---|
| `interface_model` | `EXISTING` | 配置层 | 选择 `off/shan_chen/vof` |
| `debug_vof_observation` | `OBSERVE_CPU_ACCEPTED` | 配置层 | 只控制 SC 调试观察 |
| internal `force_model` | `EXISTING` | `LbmModel.__post_init__` | interface + gravity 的执行组合 |
| `state.vof` | `NOT_STARTED` | P1 VOF stepper | 为 authoritative VOF 保留；P0 恒为 `None` |
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
| `vof`（P1+） | `state.vof.mass/phi/cell_type` | 不得覆盖正式状态 | 只读 authoritative view |

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
| physical VOF mode | `FAIL_FAST` | VOF stepper |
| `mass/phi/cell_type` | `NOT_STARTED` | mass transport / transition |
| authoritative normal | `NOT_STARTED` | geometry step |
| logical population provider | `NOT_STARTED` | FullF/HOME adapter |
| gas-to-interface completion | `FAIL_FAST` | surface boundary step |
| topology repair / redistribution | `NOT_STARTED` | transition steps |
| new-interface kinetic initialization | `NOT_STARTED` | initialization step |
| PLIC / curvature / surface tension | `NOT_STARTED` | geometry/free-surface pressure |

正式 VOF 状态必须与 `debug_vof_observation` 无关：即使不显示，也必须分配并演化。
`mass` 是计划中的守恒权威；`phi` 与 `cell_type` 不得由独立的第二权威写入。
