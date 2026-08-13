# VOF P2–P8 代码实现上手讲解计划

> 目标：已经理解 VOF 原理之后，用最短路径理解本仓库的工程实现。讲解以“数据属于谁、一步怎样流动、什么时候允许提交、怎样证明没有坏掉”为主线，先建立地图，再看 Warp kernel。

## 先给结论：当前实现应该怎样看

整个 VOF 实现可以先记成一句话：

> 两套状态仓库 + 一条事务式流水线 + 一组阶段化工人 + 一套质量审计系统。

这里的“两套状态仓库”不是两套 VOF 物理，而是：

1. `state.vof`：正式的、守恒的 authoritative VOF 状态；
2. `state.debug_mock_sc_to_vof`：只从 Shan–Chen 密度派生的观察副本，不能反过来控制 LBM。

P2–P8 的主线已经实现并通过 CPU 验收：P2 质量输运、P3 自由面边界、P4 类型转换和重分配、P5 新界面 kinetic 初始化、P6 几何和表面张力、P7 FullF/HOME 综合集成、P8 dam-break 可视化与 headless/CSV。当前仍要记住：CUDA 尚未验收，FIX2 的孤立界面 residual 仍标为 `PARTIALLY_RESOLVED`，移动固体、气泡、泡沫和 open boundary 不在本轮范围内。

## 第 0 部分：P0/P1 前置阅读

在进入第一部分前，先阅读 `vof-p2-p8-part-00-p0-p1-prerequisite.md`：

```text
P0：冻结已有 LBM / Shan–Chen，并建立只读 debug observation
P1：建立 authoritative mass/phi/cell_type，并完成事务式初始化
```

这一步用于避免把“从 Shan–Chen density 派生的显示 phi”和“真正守恒的 VOF 状态”混为一谈。

## 分成六部分

### 第一部分：架构地图——先看四个总角色

回答：

- `LbmModel`、`LbmDomain`、`LbmSolver` 分别管什么？
- `state_in`、`state_out` 和 solver scratch 分别是什么？
- 为什么 VOF 不是一堆 kernel，而是一条“可提交的流水线”？

阅读入口：`README.md`、`00-capability-status.md`、`model.py`、`domain.py`、`solver.py`。

### 第二部分：状态地图——谁是账本，谁是缓存，谁只是显示

回答：

- `mass`、`phi`、`cell_type`、`pending_excess`、几何字段的职责是什么？
- FullF 和 HOME 怎样共用同一个 logical D3Q19 population 视图？
- 为什么 `phi` 不是守恒量，真正的守恒量为什么是 `sum(mass) + sum(pending_excess)`？

阅读入口：`state.py`、`vof/state.py`、`vof/contracts.py`、`03-data-model-and-uml.md`。

### 第三部分：单步时序——两条主线怎样汇合

回答：

- LBM 的 streaming → collector → force → collision 在哪里发生？
- VOF 的质量线为什么读旧时间层？
- 事务门禁、epoch 和双缓冲怎样阻止半成品状态被提交？

阅读入口：`04-sequence-diagrams.md`、`05-control-flow-diagrams.md`、`LbmSolver.step()`。

### 第四部分：P2/P3——质量怎么走，气相缺失的 population 怎么补

回答：

- `fslbm_neighbor` 的质量通量怎样变成 `mass_tmp/phi_tmp`？
- 为什么 `INTERFACE–GAS` 的质量通量为零？
- Eq. (11) 和 Eq. (12) 如何只补 `GAS → INTERFACE` pull link？

阅读入口：P2/P3 engineering plan、completion summary、`vof/advection.py`、`vof/surface.py`。

### 第五部分：P4/P5——格子身份怎样变化，新界面怎样获得合法 kinetic 状态

回答：

- 为什么要先 proposal，再修拓扑，再 clamp，再产生 `pending_excess`？
- 为什么 GPU 上采用 gather-only、single-writer？
- GAS→INTERFACE 为什么必须找旧的 active donor，不能直接复用 gas storage？

阅读入口：P4/P5 engineering plan、completion summary、`vof/transition.py`、`vof/kinetic_init.py`。

### 第六部分：P6/P7/P8——几何、验收、可视化和运行策略

回答：

- 为什么几何必须在最终 `phi/type` 后重建？
- `geometry_epoch == epoch` 是什么质量门禁？
- FullF/HOME、strict/sampled/device/off、headless/CSV 如何共享同一正式状态？

阅读入口：P6/P7/P8 文档以及 FIX1/FIX2 的边界说明。

## 推荐阅读顺序

每一部分都采用同一套问法：

1. 这一阶段要解决什么物理/工程问题？
2. 它读取哪些旧状态？
3. 它写哪些 scratch 或候选状态？
4. 谁是唯一写入者？
5. 失败时能不能提交？
6. 哪个测试或诊断证明它没有破坏守恒和拓扑？

这样读，看到一个 kernel 时就不会只问“这几行数学是什么意思”，而会先知道“它在流水线的哪一站、读谁、写谁、写完是否能提交”。
