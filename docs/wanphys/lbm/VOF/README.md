# LBM VOF 当前能力与开发入口

当前仓库实现了 Shan-Chen density 的只读调试观察，但尚未实现守恒自由表面 VOF。
调试路径不会参与 streaming、forcing、collision 或 boundary。

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

`force_model` 仅是内部派生结果，调用者不能配置。`interface_model="vof"` 在 P0 会
`NotImplementedError`，防止把未完成能力误当成正式 VOF。

## 状态隔离

```python
state.vof is None  # P0；为 authoritative VOF 保留
```

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

## 尚未实现

- authoritative `state.vof`；
- `mass` 与守恒质量平流；
- 物理 GAS/INTERFACE/LIQUID 类型转换；
- free-surface population reconstruction；
- excess/deficit mass redistribution；
- new-interface kinetic initialization；
- PLIC、curvature 与表面张力；
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
- [06-development-roadmap.md](06-development-roadmap.md)：P0-P7 开发门禁；
- [07-p0-baseline.md](07-p0-baseline.md)：P0 冻结提交、环境与固定回归记录；
- [08-p1-engineering-plan.md](08-p1-engineering-plan.md)：P1 authoritative 状态、初始化、双缓冲、测试与提交拆分。
