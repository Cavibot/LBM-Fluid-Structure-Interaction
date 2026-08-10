# 当前项目实现说明

本目录记录仓库当前版本的实现架构，供后续开发、调试、算法验证和代码评审使用。内容基于以下三类依据交叉整理：

1. CodeGraph 截至 2026-07-16 对当前工作树中 895 个文件、20,347 个符号和 55,942 条结构关系的索引结果。
2. `docs/guide/`、`docs/concepts/` 和 `docs/api/` 中随 Newton 源码提供的官方文档。
3. 当前工作树中的 Newton、WanPhys、LBM、耦合实现及对应测试。

本文档描述的是**当前仓库实现**，不是对上游未来版本的承诺。Newton 仍处于活跃开发状态；WanPhys 的 LBM、多相流和双向流固耦合也包含研究性实现。阅读旧审计、实验记录和 roadmap 时，应以本目录和当前源码为准。

## 文档入口

- [HOME-LBM / HOME-Free 重构中期实验报告](home-rebuild-midterm-experiment-report.md)：面向中期答辩，按 R0-R7 整理旧实现基线、官方复现、HOME 核心、link-wise HOME-Free、PLIC、投影、拓扑、动量修正和润湿实验，包含关键数据、阶段图片、当前完成度与后续计划。
- [HOME-Free 重构 R8-R12 最终组合审计](home-rebuild-final-validation.md)：记录曲率、表面张力、静态润湿、低黏度、双向 FSI、三档性能取舍、完整回归和 balanced VNC 入口。
- [HOME-LBM 项目中期答辩口述稿](home-rebuild-midterm-speaking-script.md)：用较口语化的方式说明原项目与论文调研、第一次综合实现的困难，以及当前隔离重做过程中按顺序取得的实验结果，并配有对应展示图。
- [HOME 重建、隔离验证与可视化对照计划](home-rebuild-validation-plan.md)：定义暂停旧几何组合后的新阶段路线，先做官方冒烟和纯 HOME 验证，再忠实复现论文 HOME-Free，并以全过程 Dam-break 图片和数值指标逐层比较 PLIC、投影、拓扑、动量输运与 FSI。
- [HOME 重建执行进度](home-rebuild-progress.md)：按日期记录官方冒烟、旧基线冻结、纯 HOME 数值测试和可视化场景的实际结果与下一增量。
- [HOME-Free Viewer Baseline V1](home-free-viewer-baseline-v1.md)：冻结 `128^3` link-wise HOME-Free 三维溃坝阶段成果、参数、5090 验收数据、参考库对照和后续逐项接入顺序。
- [HOME-Free 单变量增量 A/B 验收](home-rebuild-incremental-ab.md)：以 Viewer Baseline V1 为固定对照，记录 PLIC-VOF、投影、表面张力、润湿和 FSI 的逐项数值、性能与 VNC 决策。
- [HOME-LBM 重构完成状态](home-lbm-status.md)：集中说明 M0-M6 的最终完成范围、验收证据、性能取舍、未纳入范围和交接入口。
- [HOME-LBM 最终场景离线演示计划](home-lbm-offline-demo-plan.md)：定义三套最终场景、共享配置、5090 数值预验收、PLIC 表面重建、Blender/OptiX 离线渲染、恢复与回传门槛。
- [Newton 官方库架构与实现](newton.md)：说明 `newton/` 的分层、核心数据模型、构建流程、碰撞、求解器、状态推进、可视化和扩展点。
- [WanPhys 扩展、LBM 与流固耦合实现](wanphys-lbm.md)：说明 `wanphys/` 如何复用和隔离 Newton，Domain/Composite 架构，D3Q19 LBM 的完整推进流程，以及刚体-LBM 耦合。
- [HOME-LBM 重构实施计划](home-lbm-plan.md)：定义并行新实现的不可退让原则、阶段边界和逐阶段验收门槛。
- [HOME-LBM 后端迁移与兼容](home-lbm-migration.md)：说明几何默认、legacy 回退、旧 TRT 边界和迁移验收。

```{toctree}
:maxdepth: 1
:hidden:

HOME-LBM / HOME-Free 重构中期实验报告 <home-rebuild-midterm-experiment-report>
HOME-LBM 项目中期答辩口述稿 <home-rebuild-midterm-speaking-script>
HOME-LBM 重构完成状态 <home-lbm-status>
HOME 重建、隔离验证与可视化对照计划 <home-rebuild-validation-plan>
HOME 重建执行进度 <home-rebuild-progress>
HOME-LBM 最终场景离线演示计划 <home-lbm-offline-demo-plan>
Newton 官方库架构与实现 <newton>
WanPhys 扩展、LBM 与流固耦合实现 <wanphys-lbm>
HOME-LBM 重构实施计划 <home-lbm-plan>
HOME-LBM 后端迁移与兼容 <home-lbm-migration>
```

## 仓库边界

```text
LBM-Fluid-Structure-Interaction/
├── newton/                    # Newton 官方库源码、示例与官方测试
├── wanphys/                   # WanPhys 扩展库与 LBM/耦合实现
├── docs/
│   ├── guide/                 # Newton 官方用户指南
│   ├── concepts/              # Newton 官方概念文档
│   ├── api/                   # Newton 官方 API 文档入口
│   ├── wanphys/               # 既有 WanPhys 笔记、审计和实验材料
│   └── implementation/        # 本套当前实现说明
├── newton/tests/              # Newton 测试，也包含本仓库新增的 LBM 测试
└── scripts/                   # 诊断和辅助脚本
```

需要特别注意：`newton/tests/test_lbm_*.py` 虽位于 Newton 测试目录，但测试对象主要是 `wanphys._src.fluid...lbm` 和耦合代码。因此，“文件在 `newton/` 下”不自动等于“Newton 上游官方功能”。本套文档按被测模块和版权头判断归属，而不是只按路径判断。

## 术语和归属约定

| 名称 | 本文含义 |
|---|---|
| Newton | 仓库中的 `newton/` 官方物理引擎库 |
| WanPhys | 仓库中的实际 Python 包 `wanphys/`；用户口述的 `wanphy/` 指此目录 |
| Warp | NVIDIA Warp；两套库使用的数组、kernel、设备和自动微分基础 |
| Domain | WanPhys 对单一物理子系统的统一生命周期抽象 |
| CompositeSimulation | WanPhys 对多个 Domain 的编排与耦合抽象 |
| LBM | WanPhys 网格流体中的 D3Q19 Lattice Boltzmann Method 实现 |
| 单向耦合 | 刚体几何和速度影响流体，但流体不回写刚体力 |
| 双向耦合 | 在单向路径上再由流体边界动量计算力/矩并写入刚体状态 |

## 依据优先级

当文档、注释和代码不一致时，采用以下优先级：

1. 当前执行路径中的源码和测试。
2. Newton 随库官方文档。
3. 当前模块 docstring 和示例。
4. `docs/wanphys/` 中的审计、实验记录和 roadmap。

例如，`docs/wanphys/lbm_core_audit_zh.md` 曾指出 Guo 力施加顺序和物理速度回写问题；当前 `LbmSolver.step()` 已把 gravity-only Guo 力移到碰撞后，并在 Shan-Chen 路径中恢复物理速度。本目录按当前代码记录为“已修正后的实现”，旧审计仅作为历史背景。

## 快速理解主线

Newton 的典型执行链是：

```text
Importer / application
        -> ModelBuilder
        -> Model (静态设备数据)
        -> State + Control + Contacts
        -> CollisionPipeline
        -> Solver.step(state_in, state_out, ...)
        -> 双缓冲交换
        -> Viewer
```

WanPhys 的 LBM-刚体执行链是：

```text
RigidModelBuilder --委托--> Newton ModelBuilder
        -> RigidModel / RigidState --零拷贝兼容壳--> Newton solver/viewer

LbmModel -> LbmDomain -> LbmState(in/out)
        ^
        |
GridLbmRigidCoupling
  1. 刚体 SDF 栅格化
  2. 世界速度转换为格子壁面速度
  3. LBM 碰撞、迁移、边界和多相力
  4. 可选动量反馈到 rigid_state.body_f
  5. 可选推进刚体
```

## 维护要求

后续修改关键实现时，建议同步更新本目录：

- `newton/` 上游升级导致 `Model/State/Solver/Contacts` 协议变化。
- `RigidModel` 的 Newton 兼容桥被移除或数据所有权改变。
- LBM 的格子、碰撞算子、forcing、边界条件或状态布局改变。
- 刚体-LBM 的单位换算、SDF 栅格化、移动壁面或反馈算法改变。
- 旧的实验性路径成为默认路径，或当前限制被正式解决。
