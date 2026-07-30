# LBM VOF 当前能力与开发入口

当前仓库已经冻结 authoritative VOF 的 P1 状态/初始化、P2 FullF 质量 transport、
P3 自由面边界、P4 类型转换/守恒重分配、P5 新界面 kinetic 初始化和 P6
normal/PLIC/curvature/Eq.12 表面张力，以及 P7 FullF/HOME 综合验收。两种 persistent
encoding 现在都可以提交 closed-domain、常数 gamma 的 moving-interface step；
Shan-Chen 调试观察路径继续与正式 VOF 完全隔离。

## 当前 API

```python
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel

model = LbmModel(
    fluid_grid_res=(64, 64, 64),
    device="cpu",
    interface_model="shan_chen",
    G=-0.1,
    debug_vof_observation=True,
    vof_debug_rho_gas=0.1,
    vof_debug_rho_liquid=1.8,
    vof_debug_epsilon=0.05,
)
domain = LbmDomain(model)
state = domain.create_state()
```

物理与观察是正交轴：

```text
interface_model = off / shan_chen / vof
debug_vof_observation = false / true
gravity = independent
```

`force_model` 仅是内部派生结果，调用者不能配置。`interface_model="vof"` 的 P6
路径支持 FullF/HOME、非负常数 `vof_surface_tension` 以及 periodic/static
bounce-back domain boundary；开口 VOF domain boundary 仍会 fail-fast。

## 状态隔离

`interface_model="vof"` 分配 authoritative：

```text
state.vof.mass
state.vof.phi
state.vof.cell_type
state.vof.normal
state.vof.plic_offset
state.vof.curvature
state.vof.epoch
state.vof.geometry_epoch
```

P2 的 `mass_tmp/phi_tmp/mass_delta` 由 solver-owned `VofMassTransport` 持有，不进入
state clone/copy/checkpoint。

只有 `interface_model="shan_chen"` 且 `debug_vof_observation=True` 时才分配：

```text
state.debug_mock_sc_to_vof
  phi                  density 派生的调试填充率
  cell_type            调试 GAS/INTERFACE/LIQUID
  normal               液体指向气体的调试法向
  epoch
  normal_valid_epoch
```

`DebugMockScToVofObserver.update_from_density()` 在初始化后和每个 step 最终宏观量写出
后更新该副本。它不能反向控制 LBM。开关前后的 populations、density、velocity 和
force 有完全一致性测试。

## 可视化

`DebugVofView` 把调试容器适配成只读渲染输入，`VofInterfaceVisualizer` 可以压缩
点云、转换坐标、生成颜色和 normal line。它不能计算或修改
`phi/cell_type/normal/curvature`。

dam-break 示例的相关参数为：

```text
-i sc | --interface shan_chen
--gravity | --no-gravity
--debug-vof-observation
--vof-debug-no-normals
```

## 当前阶段边界

- P1 authoritative `state.vof`：`CPU_ACCEPTED`；
- P2 FullF fixed-topology mass transport：`CPU_ACCEPTED`；
- P3 FullF fixed-topology gamma=0 surface step：`CPU_ACCEPTED`；
- P4 deterministic topology/redistribution：`CPU_ACCEPTED`；
- P5 FullF new-interface kinetic initialization：`CPU_ACCEPTED`；
- P6 authoritative geometry 与 Eq.12 surface tension：`CPU_ACCEPTED`；
- P7 FullF/HOME 综合 closed-domain 验收：`CPU_ACCEPTED`；
- CUDA：条件测试已建立，当前 Warp 构建不可用，`CUDA_NOT_ACCEPTED`；
- bubble pressure、moving-solid VOF coupling 和 foam。

## 文档导航

- [00-capability-status.md](00-capability-status.md)：当前 API、状态和写入权限；
- [01-paper-audit.md](01-paper-audit.md)：论文依据与公式差异审计；
- [02-framework-and-formulas.md](02-framework-and-formulas.md)：正式 VOF 框架和公式；
- [03-data-model-and-uml.md](03-data-model-and-uml.md)：状态与模块数据模型；
- [03-B-bubble-solid-foam-extensions.md](03-B-bubble-solid-foam-extensions.md)：
  气泡、固体与泡沫扩展；
- [04-sequence-diagrams.md](04-sequence-diagrams.md)：算法时序；
- [05-control-flow-diagrams.md](05-control-flow-diagrams.md)：控制流；
- [06-development-roadmap.md](06-development-roadmap.md)：P0-P8 开发门禁；
- [07-p0-baseline.md](07-p0-baseline.md)：P0 冻结提交、环境与固定回归记录；
- [08-p1-engineering-plan.md](08-p1-engineering-plan.md)：P1 authoritative 状态、初始化、双缓冲、测试与提交拆分；
- [09-p2-engineering-plan.md](09-p2-engineering-plan.md)：P2 决策、实现和验收计划；
- [10-p2-completion-summary.md](10-p2-completion-summary.md)：P2 完成与 CPU 验收总结；
- [11-p2-change-architecture.md](11-p2-change-architecture.md)：P2 改动文件与关键架构新旧对比。
- [12-p3-engineering-plan.md](12-p3-engineering-plan.md)：P3 决策、实现和验收计划；
- [13-p3-completion-summary.md](13-p3-completion-summary.md)：P3 完成与 CPU 验收总结；
- [14-p3-change-architecture.md](14-p3-change-architecture.md)：P3 改动文件与关键架构新旧对比。
- [15-p4-engineering-plan.md](15-p4-engineering-plan.md)：P4 决策、实现和验收计划；
- [16-p4-completion-summary.md](16-p4-completion-summary.md)：P4 完成与 CPU 验收总结；
- [17-p4-change-architecture.md](17-p4-change-architecture.md)：P4 改动文件与关键架构新旧对比。
- [18-p5-engineering-plan.md](18-p5-engineering-plan.md)：P5 决策、实现和验收计划；
- [19-p5-completion-summary.md](19-p5-completion-summary.md)：P5 完成与 CPU 验收总结；
- [20-p5-change-architecture.md](20-p5-change-architecture.md)：P5 改动文件与关键架构新旧对比。
- [21-p6-engineering-plan.md](21-p6-engineering-plan.md)：P6 几何、符号、epoch 与验收计划；
- [22-p6-completion-summary.md](22-p6-completion-summary.md)：P6 完成与 CPU 验收总结；
- [23-p6-change-architecture.md](23-p6-change-architecture.md)：P6 改动文件与关键架构新旧对比。
- [24-p7-engineering-plan.md](24-p7-engineering-plan.md)：P7 HOME、综合场景和设备计划；
- [25-p7-completion-summary.md](25-p7-completion-summary.md)：P7 完成与 CPU 验收总结；
- [26-p7-change-architecture.md](26-p7-change-architecture.md)：P7 改动文件与关键架构新旧对比。
