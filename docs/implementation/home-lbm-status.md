# HOME-LBM 重构完成状态

本文是 HOME-LBM 重构的最终状态入口，记录截至 2026-07-16 已交付的功能、验收证据和剩余研究边界。详细算法过程见 [重构实施计划](home-lbm-plan.md)，公共 API 与旧后端迁移方式见 [后端迁移与兼容](home-lbm-migration.md)。

## 总体结论

重构计划 M0-M6 已全部完成。新实现位于 `wanphys/_src/fluid/fluid_grid/home_lbm/`，没有删除或改写旧 `wanphys/_src/fluid/fluid_grid/lbm/` 的 D3Q19 Shan-Chen/TRT 实现。

当前公共默认 `wanphys.fluid.HomeFreeDomain` 指向新的几何 HOME-Free 后端。旧 link-wise HOME-Free 以 `HomeFreeLegacyDomain` 保留，用于结果复现和迁移期回归；旧 D3Q19 LBM 继续使用原模块和原示例，不做静默参数映射。

## 已完成范围

| 阶段 | 状态 | 主要交付 |
|---|---|---|
| M0 数学与数据契约 | 完成 | D3Q27、十矩 SoA、单位和守恒账本契约 |
| M1 单相 HOME | 完成 | 按需 PDF 重构、碰撞迁移、周期/壁面/入口出口及基础验证 |
| M2 曲面边界与双向 FSI | 完成 | cut-link、移动边界、动量交换、切格重映射、润滑及强耦合事务 |
| M3 HOME-Free | 完成 | sharp VOF、PLIC、拓扑转换、气压边界、曲率和表面张力 |
| M4 自由表面 FSI | 完成 | entry/exit、fresh/dead cell、质量与动量队列、润湿、漂浮和冲击场景 |
| M5 性能与量化 | 完成 | CUDA Graph PCG、设备端诊断、布局基准、寄存器审计及 int16 可行性结论 |
| M6 替换与兼容 | 完成 | 几何后端默认入口、legacy 显式回退、示例、迁移说明和最终回归 |

## 当前默认实现

默认几何 HOME-Free 路径综合采用：

- D3Q27 HOME 十矩 FP32 主状态，27 个 PDF 仅在 kernel 内按需重构；
- Weymouth-Yue 方向分裂 PLIC 和质量加权动量输运；
- only-missing 自由表面气压边界、局部曲率和表面张力；
- 可配置接触角，以及液体、界面、气体和固体的确定性拓扑状态机；
- 相感知 cut-link 动量交换、移动固体切格重映射和双向刚体反馈；
- 与面通量离散配对的 Hodge Courant 投影，PCG 使用压力暖启动和 CUDA Graph 迭代块；
- 事务式双缓冲与强耦合回滚，失败时不提交部分更新。

## 最终验收

最终验收在 RTX 5090、Warp 1.12.0 上执行。

- 完整 HOME-Free/HOME-LBM 回归：`330 passed`，另有 `124 subtests passed`，耗时 `456.24 s`。
- `54x54x86` 动态自由表面 FSI，10 步预热、50 步统计：平均 `53.951 ms/step`、`4.648 MLUPS`、p95 `77.125 ms`。
- 同一场景实际产生 52 个 fresh cell、52 个 dead cell；最多 368 条湿 cut-link 和 1620 条干 link。
- Courant 投影平均/最大迭代数为 `387.36/416`，最大投影后散度 `3.26e-9`。
- Warp mempool 峰值 `343.14 MiB`；十个 HOME 矩的候选 int16 编码均无饱和。
- 公共默认液柱示例连续运行 100 步，液体质量相对误差 `2.54e-8`，体积 `288.999999`，最大 fill 为 1。
- 本地与 RTX 5090 远端的 118 个重构代码、测试、基准和公共入口文件已逐文件核对 SHA-256，结果一致。

上述数据是固定硬件、软件版本和场景参数下的回归基线，不应直接解释为其他分辨率或其他 GPU 的性能承诺。

## 性能与精度取舍

生产状态继续使用 FP32。int16 对十矩的范围审计没有发生饱和，但二阶矩单次量化误差已高于部分现有物理误差尺度，因此没有为了显存收益启用低精度状态。这是验收结论，不是待补的默认优化。

纯周期数学对照中，显式 split PDF 布局比当前 fused HOME 核快约 24%，但每格额外需要 256 B scratch，且尚未包含气压、cut-link、外力、非周期壁面和联合诊断。它只构成后续 split/hybrid 研究依据，不能替换当前已验收的生产路径。

PCG 严格投影仍是动态自由表面大网格的主要成本。当前实现没有降低残差或散度门槛，而是通过暖启动和 CUDA Graph 将严格投影基线从 `102.54 ms/step` 降到最终约 `53.95 ms/step` 的完整场景水平。

## 明确未纳入范围

下列事项不属于本轮 M0-M6 完成定义：

- 封闭气泡的连通分量识别、独立气体状态方程和气泡合并/分裂；
- 可压缩、高 Mach、热流、非牛顿或化学反应流；
- 已通过完整物理回归的 FP16/int16 生产状态；
- 与生产边界功能逐链等价的 split/shared-memory HOME kernel；
- 对所有复杂 CAD、极端润湿参数和工业尺度网格的普适验证；
- 删除 legacy HOME-Free 或旧 D3Q19 LBM。

这些内容若继续研发，应作为新里程碑立项，不应通过放宽现有守恒、散度、稳定性或作用反作用门槛并入当前实现。

## 交接入口

- 默认示例：`wanphys/examples/lbm/fluid_grid_home_free_column.py`
- 动态 FSI 基准：`scripts/bench/bench_home_free_fsi.py`
- HOME 布局对照：`scripts/bench/bench_home_lbm_layouts.py`
- 测试集合：`newton/tests/test_home_free_*.py`、`newton/tests/test_home_lbm_*.py`
- 详细阶段记录：[HOME-LBM 重构实施计划](home-lbm-plan.md)
- API 迁移：[HOME-LBM 后端迁移与兼容](home-lbm-migration.md)

建议后续修改先运行相关小规模 CPU/CUDA 规范测试，再运行完整 HOME 回归和固定 5090 基准。任何默认后端、状态布局、力所有权、拓扑提交或精度类型变化，都必须同步更新本页和迁移文档。
