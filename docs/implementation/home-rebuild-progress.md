# HOME 重建执行进度

本文只记录 [HOME 重建、隔离验证与可视化对照计划](home-rebuild-validation-plan.md) 的实际执行证据，不改写旧 M0-M6 历史结论。

## 2026-07-21：R0、R1 与 R2 首个闭环

| 阶段 | 状态 | 当前证据 |
|---|---|---|
| R0 旧实现冻结 | 完成 | 已登记 `nu=0.10` 稳定但高耗散序列，以及 `nu=0.04` 在 step 4053 因 z-face Courant 约 1.097716 失败的序列 |
| R1 官方 Home-FSLBM 冒烟 | 完成 | 官方提交 `4311560` 未改源码，在 RTX 5090/CUDA 12.8、`sm_120` 上编译；30 秒生成 260 组 mass/velocity PPM，无 NaN 或 CUDA fault |
| R2 纯 HOME 候选基础回归 | 完成 | RTX 5090 上 `test_home_lbm_foundation.py` 为 `10 passed`；新增可视化测试与 periodic 联集为 `11 passed` |
| R2 Taylor-Green 可视化 | 完成 | `96x96x1`、`nu=0.04`、1500 步；六个时刻涡量图、CSV、manifest 和拼图已回传本地 |
| R2 Taylor-Green 低黏度扫描 | 完成 | `nu=0.04/0.01/0.001/0.0001` 均完成 1500 步、六帧输出，无非法单元 |
| R2 周期剪切层 | 完成 | `128^2, nu=0.001` 与 `256^2, nu=0.0001` 均形成非线性卷起；最长 10000 步，无非法单元 |
| R2 最小核心迁移 | 完成 | 新 `home_rebuild/core` 不依赖旧 `home_lbm`；CPU/CUDA 多步十矩对齐，5090 无渲染基准加速 `1.414x` |
| R3.1 FSL 状态与质量交换 oracle | 完成 | 新 `free_surface_fsl` 仅含论文标量状态；静止界面零变化，`u_x=0.05` 平移界面左右质量变化为 `-0.05/+0.05`，总质量差为 0 |
| R3.2 Warp 质量交换 | 完成 | NumPy/Warp CPU/CUDA 逐格对齐；联合回归 `38 passed`；`256^2` HOME+FSL 为 `708.99 MLUPS` |
| R3.3 only-missing 气压边界 | 完成 | 论文 Eq. (11) 的 NumPy/Warp CPU/CUDA 逐格对齐；5090 联合回归 `39 passed`；三种气压的界面脉冲方向、幅值和作用域均通过视觉审计 |
| R3.4 fixed-topology 组合事务 | 完成 | mass exchange、only-missing stream 与 collision 共享同一提交态；CPU/CUDA oracle、100 步静界面和失败回滚通过；5090 联合回归 `47 passed`，百万格三维严格事务 `869.74 MLUPS` |

Taylor-Green 最终指标：

- 相对质量漂移：`+8.7111e-6`；
- 解析衰减振幅误差：`-1.0511e-3`；
- 最大速度：`0.0477162`；
- 密度范围：`[0.9976934, 1.0023115]`；
- 非法单元：`0`；
- 含六次出图和拼图的总耗时：`1.605 s`。

当前 R2 可视化入口：

```bash
PYTHONPATH=. python -m wanphys.examples.lbm.home_rebuild.home_core.taylor_green \
  --output /path/to/run \
  --resolution 96 \
  --viscosity 0.04 \
  --amplitude 0.08 \
  --steps 0 100 250 500 1000 1500 \
  --device cuda:0
```

该入口目前明确标记为 `audit-candidate`：它只读取现有纯 HOME solver，不接 VOF/PLIC/FSI，也尚未宣称新的 `home_rebuild/core` 已经完成。周期剪切层已经补齐非线性卷起证据；下一步迁移最小数学核心，避免把旧边界和耦合代码复制进新包。

同一 `96x96x1` Taylor-Green 场景的 1500 步黏度扫描结果：

| `nu` | 相对质量漂移 | 解析振幅误差 | 最终最大速度 | 最终密度范围 | 非法单元 |
|---:|---:|---:|---:|---:|---:|
| `0.04` | `+8.711e-6` | `-1.051e-3` | `0.047716` | `[0.997693, 1.002311]` | 0 |
| `0.01` | `+8.886e-6` | `-2.030e-3` | `0.070161` | `[0.994492, 1.005514]` | 0 |
| `0.001` | `+9.561e-6` | `-2.492e-3` | `0.078822` | `[0.992915, 1.007116]` | 0 |
| `0.0001` | `+9.609e-6` | `-2.547e-3` | `0.079750` | `[0.992736, 1.007299]` | 0 |

这个结果证明当前纯 HOME 候选在光滑、周期、低 Mach 的 Taylor-Green 模式下可以稳定运行到论文量级低黏度；它不能提前证明自由表面、撞壁或强变形稳定。`nu=0.0001` 中涡量结构没有出现网格噪声或非物理衰减，下一步需要用周期剪切层增加非线性卷起和小尺度结构，再决定是否进入最小核心迁移。

## 2026-07-21：R2 周期双剪切层

新增无窗口入口 `wanphys.examples.lbm.home_rebuild.home_core.shear_layer`，输出六个固定色标涡量帧、速度箭头、CSV、manifest 和拼图。它仍是纯 HOME 审计场景，不含自由表面或液体体积分数。

| 配置 | 最终质量漂移 | 最终动能变化 | 最终最大速度 | 最终密度范围 | 非法单元 | 含六次出图耗时 |
|---|---:|---:|---:|---:|---:|---:|
| `128^2, nu=0.001, 5000` 步 | `+3.855e-5` | `-4.480%` | `0.102267` | `[0.970080, 1.010944]` | 0 | `2.021 s` |
| `256^2, nu=0.0001, 10000` 步 | `+8.106e-5` | `-1.224%` | `0.123142` | `[0.944669, 1.012159]` | 0 | `2.766 s` |

两组序列都从平直的反向剪切带开始，经正弦扰动弯曲后形成一对 Kelvin-Helmholtz 涡。`256^2` 低黏度序列出现更锐利的涡核和多圈卷绕，直至末帧仍无棋盘噪声、断带或局部数值爆炸。低黏度组最低密度约 `0.9447`，表明弱可压缩效应已经不可忽略，但当前仍处于连续、可控状态。

本地证据目录：

- `tmp/home-rebuild/r2-home-core-shear-layer-preview`：`128^2, nu=0.001`；
- `tmp/home-rebuild/r2-home-core-shear-layer-low-nu`：`256^2, nu=0.0001`。

上述周期剪切层完成了旧纯 HOME 候选的迁移前审计。它不能直接运行物理意义上的 Dam-break，因为 Dam-break 还需要液体体积分数、界面单元重构、气相压力边界和拓扑转换。随后执行的最小核心迁移记录如下；第一个可比较 Dam-break 将在 R3 出现。

## 2026-07-21：R2 最小核心迁移完成

新实现位于 `wanphys/_src/fluid/fluid_grid/home_rebuild/core`，使用独立的 `HomeCoreModel`、`HomeCoreState`、`HomeCoreSolver` 和 `HomeCoreDomain`。旧 `home_lbm` 目录没有修改，新核心源码也禁止反向导入旧包。

迁入内容：

- D3Q27 权重、速度与十矩 SoA 状态；
- 三阶 Hermite 分布重构；
- 周期 pull-stream 与 NOCM 二阶矩松弛；
- 格点体力项；
- 非有限值、非正密度、最大速度和非平衡应力诊断；
- 错误时间步、超 Mach 初始化和未实现边界组合的硬失败。

刻意没有迁入的内容：固体 SDF、cut-link/Bouzidi、自由表面 flag、气压边界、VOF/PLIC、局部耦合力、动量账本与 FSI。这些能力必须在后续独立层中组合，不能重新塞回基础推进内核。

迁移验证使用非均匀密度、Taylor-Green 速度场、`nu=0.003` 和微小二维体力，在 CPU 与 CUDA 上分别推进 20 步。新旧十矩满足 `rtol=2e-6, atol=2e-7`，全部核心、基础和周期测试为 `28 passed`。低黏度 Taylor-Green 与双剪切层入口已经切换到 `home_rebuild.core`，不再读取旧 solver。

RTX 5090 无渲染基准采用 `256x256x1`、`nu=0.0001`、200 步预热、10000 步计时、五次交替运行并取中位数：

| 路径 | 中位耗时 | 吞吐 | 估算每格字节 |
|---|---:|---:|---:|
| 旧通用 `HomeLbmSolver` | `0.89685 s` | `730.74 MLUPS` | `124 B` |
| 新周期 `HomeCoreSolver` | `0.63433 s` | `1033.15 MLUPS` | `80 B` |

新核心在该配置下加速 `1.414x`，按双缓冲状态和旧通用路径实际预分配的每格占位缓冲估算，内存减少 `35.5%`。10200 步后新旧结果的最大绝对十矩差为 `1.194e-5`，相对 L2 差为 `1.667e-6`，相对质量差为 `2.181e-7`。基准证据为 `tmp/home-rebuild/r2-isolated-core-benchmark-256.json`。

新核心视觉证据：

- `tmp/home-rebuild/r2-isolated-core-taylor-green-low-nu`；
- `tmp/home-rebuild/r2-isolated-core-shear-layer-low-nu`。

R2 的周期单相最小核心至此可以冻结。它依然不能直接产生真实 Dam-break；下一阶段 R3 将在独立的 `free_surface_fsl` 包中只实现论文 HOME-Free 所需的 link-wise 质量交换、LIQUID/INTERFACE/GAS 拓扑和缺失分布气压边界，并首先从静止平面界面开始。

## 2026-07-22：R3.1 FSL 状态与 link-wise 质量交换 oracle

新包位于 `wanphys/_src/fluid/fluid_grid/home_rebuild/free_surface_fsl`，只依赖 R2 的 `home_rebuild/core`。源码门禁禁止导入旧 `home_lbm`、PLIC、Courant 投影、刚体或 `excess_momentum`。

本增量实现论文与官方 Home-FSLBM 共有的最小状态：

- `FslCellFlag = GAS / INTERFACE / LIQUID`；
- `mass = rho * fill_level`；
- `excess_mass` 标量队列；
- `fill_level` 与提交态 flags；
- 26 邻域液-气直连硬失败；
- 三阶 Hermite 十矩到 D3Q27 临时分布重构；
- 论文 Eq. (9)-(10) 的 active-active link-wise 质量交换。

链路权重严格采用官方路径：液-液和液-界面使用完整通量，界面-界面使用两端 fill 平均值，气体链不参与内部质量交换。拓扑转换、缺失分布气压边界和多余质量重分配尚未加入，因此当前结果明确标记为 `r3.1-cpu-oracle`，不能描述成完整 HOME-Free 时间步。

新旧隔离联合测试在 RTX 5090/Warp 1.12 上为 `34 passed`，其中 R3.1 六项测试覆盖：

- fill 到 L/I/G 分类和 `mass=rho*fill`；
- 非法 liquid-gas 直连拒绝；
- 静止周期平面界面逐格质量变化为 0；
- `u_x=0.05` 平移平面界面内部链总质量严格守恒；
- 全液周期域退化为 HOME streamed density；
- Warp 设备状态只持久化论文已有的四个标量/格。

`96x32x1` 周期液体薄层的单步视觉审计结果：

| 指标 | 结果 |
|---|---:|
| 初始总质量 | `1568.0` |
| 静止界面最大逐格变化 | `0.0` |
| 静止界面总质量变化 | `0.0` |
| 平移界面总质量变化 | `0.0` |
| 左界面平均变化 | `-0.05` |
| 右界面平均变化 | `+0.05` |
| 液-气直连数 | `0` |

证据目录为 `tmp/home-rebuild/r3-fsl-interface-exchange-oracle`，包含静止和平移两张独立图片及 manifest。下一增量是把同一 oracle 实现成独立 Warp CPU/CUDA kernel 并逐格对齐，然后接入 only-missing 气压边界；在气压边界和 HOME collision 形成一个事务之前，不开始多步界面推进或 Dam-break。

## 2026-07-22：R3.2 Warp link-wise 质量交换

新增 `free_surface_fsl/kernels.py` 和 `FslMassAdvector`。设备核直接复用 R2 `HomeCore` 的十矩三阶 Hermite 重构函数，避免 CPU oracle 与 GPU 路径维护两套分布公式。`advect()` 只提交设备工作，不隐式同步；错误计数、液-气直连和质量统计在显式诊断阶段回读。

测试使用非均匀密度、二维速度扰动和周期液体薄层，将 NumPy oracle 与 Warp CPU/CUDA 的逐格 `advected_mass` 对齐，容差为 `rtol=2.5e-6, atol=3e-7`。另有静止/平移解析门禁和设备侧液-气直连失败测试。R2+R3 联合集在 RTX 5090/Warp 1.12 上为 `38 passed`。

Warp/CUDA 的 `96x32x1` 平移界面视觉审计：

| 指标 | 结果 |
|---|---:|
| 静止界面最大逐格变化 | `0.0` |
| 平移界面左侧平均变化 | `-0.05000001` |
| 平移界面右侧平均变化 | `+0.05000013` |
| 平移单步总质量变化 | `-1.793e-4` |
| 相对总质量变化 | `-1.144e-7` |
| 非法单元 / 液-气直连 | `0 / 0` |

FP32 的两侧变化差约 `1.2e-7`，总质量误差与 CPU/CUDA 对齐容差一致，没有使用全局质量回填。图片和 manifest 位于 `tmp/home-rebuild/r3-warp-fsl-interface-exchange`。

`256x256x1`、液体体积分数约 `0.5039`、200 步预热、5000 步计时、五次取中位数的 5090 基准：

| 路径 | 中位耗时 | 吞吐 |
|---|---:|---:|
| FSL 质量交换单核 | `0.16449 s` | `1992.07 MLUPS` |
| HOME core | `0.27896 s` | `1174.66 MLUPS` |
| HOME + FSL 顺序提交 | `0.46218 s` | `708.99 MLUPS` |

当前未融合的质量层使 HOME 时间增加约 `65.7%`。该数字是后续融合和图捕获优化的基线，不能因为性能压力删除论文质量交换或降低界面更新精度。基准证据为 `tmp/home-rebuild/r3-fsl-mass-benchmark-256.json`。

R3.2 仍只计算 active-active 内部质量交换，尚未推进 HOME moments，也不处理 gas-side 缺失分布。下一增量实现 only-missing 气压边界的 NumPy oracle 和 Warp 对齐，再把 HOME collision、气压边界和质量交换组合成一个可回滚事务。

## 2026-07-22：R3.3 only-missing 气压边界

新增 `gas_pressure_boundary_populations()` NumPy oracle、`only_missing_stream_moments()` 参考 pull-stream、Warp `only_missing_stream_kernel` 与主机侧 `FslOnlyMissingStreamer`。实现严格采用论文 Eq. (11)：

```text
f_i* = f_i^eq(rho_g, u_interface)
     + f_opp(i)^eq(rho_g, u_interface)
     - f_opp(i)(interface)
```

这个公式只用于 pull 来源为 GAS 的缺失链。来源为 LIQUID 或 INTERFACE 的已有分布仍从其 HOME 十矩做三阶 Hermite 重构，不会被气压边界覆盖。气压项使用论文的标准二阶平衡分布，最后一项使用 HOME 三阶 filtered population；两者的阶次差异是方法本身的一部分，没有人为改成同一公式。LIQUID 直接拉取 GAS 被视为拓扑错误并硬失败。

设备端保持事务隔离：输入 HOME moments 与 FSL flags 均不改写，输出写入独立 `HomeCoreState`。每次提交同时记录：

- 非法单元数；
- 液-气直连数；
- 实际补齐的 gas link 数；
- 活动区最小/最大密度；
- 活动区最大速度。

验证分三层完成：

1. 单格解析式：静止 `rho_g=1` 精确恢复 D3Q27 权重；`rho_g=1.1` 得到 `1.2 w_i`；
2. 平面 oracle：`16x8x1` 周期薄层静止误差小于 `8e-16`，两侧共识别 `144` 条气相缺失链；
3. 设备对齐：非均匀密度和二维速度扰动下，Warp CPU/CUDA 与 NumPy 十矩逐格满足 `rtol=4e-6, atol=4e-7`，设备 gas-link 图逐格完全相同。

RTX 5090/Warp 1.12 的 R3.3 专项为 `10 passed`；`newton/tests/home_rebuild` 为 `29 passed`；再包含旧 HOME foundation 门禁的联合回归为 `39 passed`。没有 NaN、非正活动区密度或液-气直连。

`96x32x1` 固定平面拓扑的 CUDA 视觉审计结果：

| `rho_g` | 界面密度 | 左界面 `j_x` | 右界面 `j_x` | 最大速度 | 缺失链 / 非法 / 直连 |
|---:|---:|---:|---:|---:|---:|
| `0.98` | `0.99333322` | `-0.00666666` | `+0.00666666` | `0.00671141` | `576 / 0 / 0` |
| `1.00` | `0.99999982` | `0` | `0` | `0` | `576 / 0 / 0` |
| `1.02` | `1.00666666` | `+0.00666666` | `-0.00666666` | `0.00662251` | `576 / 0 / 0` |

高气压在左界面产生向右脉冲、右界面产生向左脉冲，低气压恰好反向，匹配法向压力作用；一步响应只位于界面，液体内部保持静止。图片与 manifest 位于 `tmp/home-rebuild/r3-warp-pressure-boundary`，可复现实验入口为：

```bash
PYTHONPATH=. python -m wanphys.examples.lbm.home_rebuild.home_free_fsl.pressure_boundary \
  --output /path/to/run \
  --device cuda:0
```

R3.3 仍不是完整 HOME-Free 时间步：它只有 fixed-topology 的 pull-stream，没有碰撞提交、质量更新提交、拓扑转换和多余质量重分配，因此不能据此运行 Dam-break。下一增量应把 R2 HOME collision、R3.2 link-wise mass exchange 和 R3.3 only-missing stream 组合为一个可回滚事务；通过静水平面多步门禁后，再实现官方拓扑转换与 excess-mass 队列。

## 2026-07-22：R3.4 fixed-topology 组合事务

新增 `HomeCoreSolver.collide()` 和 `FslFixedTopologyStepper`。碰撞数学没有复制到 FSL 包：原 fused `stream_collide_periodic_kernel` 与新 collision-only kernel 共同调用 `_collide_stress()`，全液周期域的“独立 stream + collide”与原 fused 路径逐格最大差为 `0.0`。

每步从同一个已提交 post-collision 状态并行产生两个候选：

1. R3.2 `FslMassAdvector` 根据源 moments、mass、fill 和 flags 计算 link-wise 候选质量；
2. R3.3 `FslOnlyMissingStreamer` 完成 pull-stream，并只在 GAS 来源链应用气压边界；
3. collision-only kernel 对 streamed moments 执行 NOCM 碰撞和体力；
4. fixed-topology commit kernel 生成候选 mass/fill/excess/flags；
5. 气压、质量、HOME 状态、相分类和总质量诊断全部通过后，才复制到调用方输出态。

当前 fixed-topology 规则是：GAS 保持 `mass=fill=0`；INTERFACE 使用 `fill=advected_mass/rho`；LIQUID 保持规范 `fill=1, mass=rho`。LIQUID 中每步 `advected_mass-rho` 的 float32 差没有丢弃，而是带符号转入该格 `excess_mass`，因此 `mass+excess_mass` 总量不变。该规范化有独立的总量和最大绝对值诊断，并保留 `2e-6` 单步硬门限；超过门限仍要求进入 topology，而不是被静默修正。

事务失败不修改调用方输出。测试故意把界面 fill 置于 topology epsilon 内，确认抛出 `requires a topology transition` 后 fluid moments、mass、fill 和 flags 均逐字节保持原值，`last_diagnostics` 也不伪造成功记录。

RTX 5090/Warp 1.12 验证结果：

- collision-only NumPy/Warp CPU/CUDA 逐格对齐，`rtol=3e-6, atol=3e-7`；
- 非均匀密度、二维速度和体力下，完整单步事务与组合 NumPy oracle 对齐；
- 静止 `32x12x1` 平面界面连续 100 步，无非法单元或 phase crossing；
- 100 步 CPU/CUDA moments 满足 `rtol=3e-6, atol=3e-7`，界面 fill 最大后端差 `3.87e-6`；
- phase-crossing 失败不提交；
- `newton/tests/home_rebuild` 加旧 foundation 门禁共 `47 passed`。

性能诊断最初使用每格 FP64 全局 atomic 累加质量，导致百万线程争用。最终实现改为每格 FP64 诊断缓冲配合 Warp 并行 `array_sum`，健康路径只对真正的错误计数使用原子操作。严格事务包含每步诊断和主机判定，不能与无验证 raw kernel 混为一个数字。

| 配置 | raw HOME | 每步验证 HOME | R3.4 严格事务 | 事务/raw 时间比 |
|---|---:|---:|---:|---:|
| `256x256x1` | `1294.36 MLUPS` | `432.56 MLUPS` | `84.94 MLUPS` | `15.24x` |
| `128x128x64` | `3895.87 MLUPS` | `2641.20 MLUPS` | `869.74 MLUPS` | `4.48x` |

二维小网格主要受每步同步和 Python 事务固定成本限制；百万格三维配置更能反映 kernel 吞吐。R3.4 仍是四次主要网格遍历（质量、stream、collision、commit），后续可以融合 stream/collision 和诊断归约，但不能以删除质量交换、气压边界或失败回滚换性能。基准证据：

- `tmp/home-rebuild/r3-fixed-topology-benchmark-256.json`；
- `tmp/home-rebuild/r3-fixed-topology-benchmark-128x128x64.json`。

多步视觉审计使用 `128x48x1`、`rho_g=1.002`、`nu=0.02`，输出 step `0/1/5/15/30/60/100`。压力脉冲从两侧界面对称向内传播，step 100 的活动区密度为 `[1.002043, 1.003611]`，最大速度 `3.939e-4`，总质量漂移 `1.742e-8`，界面 fill 为 `[0.400016, 0.400020]`，全程非法单元和 phase crossing 均为 0。证据目录为 `tmp/home-rebuild/r3-fixed-topology-pressure-wave`。

```bash
PYTHONPATH=. python -m wanphys.examples.lbm.home_rebuild.home_free_fsl.fixed_topology_pressure_wave \
  --output /path/to/run \
  --device cuda:0 \
  --gas-density 1.002
```

R3.4 已证明基础算法和管线可以多步组合，但仍刻意禁止 topology 变化。图中界面 fill 已从 `0.5` 降到约 `0.4`，继续施压最终必然需要 INTERFACE 转 GAS 并在液侧创建新 INTERFACE。下一阶段 R3.5 应实现论文原始 topology、fresh/dead cell 初始化和 excess-mass 邻域重分配；通过独立平移界面和相变往返后，才开始真正的液柱坍塌。

## 2026-07-22：R3.5 动态 topology 与 excess-mass 事务

新增 `topology_reference.py`、`topology_kernels.py`、`FslTopologyUpdater` 和 `FslDynamicTopologyStepper`，R3.4 固定拓扑入口保持不变。实现依据论文规则与官方 Home-FSLBM 3D `surface_1/2/3` 路径逐段核对，采用 D3Q27 全部 26 个邻居：

1. INTERFACE 在 `mass > rho` 或没有 GAS 邻居时标记为 I→F；在 `mass < 0` 或没有 LIQUID 邻居时标记为 I→G；
2. I→F 取消相邻 I→G，并把相邻 GAS 标记为 fresh G→I；
3. 剩余 I→G 把相邻 LIQUID 或 I→F 降为 INTERFACE，保证最终没有 LIQUID-GAS 直连；
4. fresh interface 从持续活动且最终非 GAS 的邻居平均密度和速度，并写成 HOME 平衡十矩；没有 donor 时硬失败；
5. 最终 LIQUID 规范为 `mass=rho, fill=1`，INTERFACE 把 mass 限制到 `[0,rho]`，GAS 规范为零；截出的带符号质量进入延迟 excess 队列。

官方 CUDA 直接并发改写同一 flag 数组，执行结果可能依赖线程顺序。本实现拆成 candidates、growth resolution、final transitions、fresh initialization 和 commit 五个只读阶段，CPU/CUDA 必须逐格一致。拓扑事务只写内部候选态；invalid moment、fresh cell 无 donor、液气直连或质量漂移超限时，调用方输出不变。

`excess_mass` 有一处有意的等价表示差异：官方数组存 `excess / recipient_count` 的单邻居 share，本实现存 donor 的总待分配量，下一步接收时再除以最终有效邻居数。因此总表示质量始终可直接写成 `sum(mass) + sum(excess_mass)`，不会因邻居数变化而误读账本；分发到每个邻居的数值与官方公式相同。没有有效接收者的 excess 保留在 donor 队列，不写回 GAS mass。

验证覆盖：

- 推进界面生成一格 fresh interface 壳层；
- 退缩界面把相邻液体降为界面；
- 相邻 I→F/I→G 冲突按官方顺序消解；
- fresh HOME moments 与 NumPy 邻居平均 oracle 对齐；
- donor-total excess 只分发一次，孤立 excess 不丢失；
- I→F→I、G→I→G 往返恢复原拓扑并保持质量；
- 匀速液体 slab 在 20 步内跨越格点并实际触发相变；
- 非法液气直连失败后调用方 moments/mass 不变；
- Warp CPU/CUDA 与 NumPy 的 flags、mass、fill、excess、moments 和 recipient counts 逐格对齐。

RTX 5090/Warp 1.12 联合回归为 `60 passed`。动态平移视觉审计使用 `128x48x1`、`u_x=0.05`、`nu=0.02`，输出 step `0/5/10/15/20/30/40`：

| 指标 | 结果 |
|---|---:|
| 液体质心 x | `55.5 -> 57.49596` |
| 累计 I→F / I→G | `96 / 96` |
| 最终 LIQUID / INTERFACE 单元 | `2304 / 96` |
| 40 步最大累计相对质量漂移 | `3.593e-7` |
| 非法单元 / 液气直连 | `0 / 0` |

视觉证据位于 `tmp/home-rebuild/r3-dynamic-topology-translation`，包括七张固定步长帧、总览图和 manifest。可复现实验：

```bash
PYTHONPATH=. python -m wanphys.examples.lbm.home_rebuild.home_free_fsl.dynamic_topology_translation \
  --output /path/to/run \
  --device cuda:0
```

严格动态事务包含质量交换、only-missing stream、HOME collision、excess 接收、四阶段 topology、设备诊断和每步主机判定。5090 预热后性能为：

| 配置 | 严格动态事务吞吐 | 计时内相变 | 最大单步相对质量漂移 |
|---|---:|---:|---:|
| `256x256x1`, 100 步 | `21.74 MLUPS` | `2560` | `1.187e-7` |
| `128x128x64`, 20 步 | `568.93 MLUPS` | `16384` | `1.182e-7` |

二维小网格仍被十余次 kernel launch、设备同步和每步失败判定的固定开销主导；百万格三维吞吐约为 R3.4 严格固定事务的 `65.4%`，代价来自真正执行的 excess/topology/fresh/分离检查，不能用删除拓扑正确性换回。复现入口为 `benchmark_dynamic_topology.py`。

R3.5 完成的是可移动自由表面拓扑，不是 Dam-break。进入液柱坍塌前仍缺少墙面边界组合与重力水静压初始化；随后才是 R3.6 最小 gravity + wall 场景。PLIC、曲率/表面张力、VOF 几何输运和 FSI 仍保持隔离，不应在该门禁通过前重新混入。

## 2026-07-22：R3.6 closed wall、gravity 与静水初始化

R3.6 在 `free_surface_fsl` 内新增独立 `solid_mask` 组合层，没有修改 R2 周期 HOME core，也没有改动 R3.4/R3.5 入口。墙格点不能伪装成 GAS 或 LIQUID：前者会让压力边界和 topology 把墙当自由表面，后者会引入虚假液体质量。因此墙是与 FSL flags 正交的独立状态。

新增组件：

- `FslWallMask`：验证六面闭合的固体壳，阻断底层周期索引跨边界回绕；
- `wall_only_missing_stream_moments()` 与 Warp 对应核：活动单元从墙来源的 pull link 使用 halfway bounce-back，从 GAS 来源仍只应用论文 Eq. (11)；
- `wall_link_mass_exchange()` 与 Warp 对应核：固体链质量通量严格为零；
- `FslWallActiveCollider`：只对非固体 LIQUID/INTERFACE 施加格点重力，GAS 和 SOLID moments 不积累虚假加速度；
- `FslWallTopologyUpdater`：solid 从 gas-neighbor、donor、recipient、fresh initialization 和 liquid-gas separation 检查中排除；
- `FslWallDynamicTopologyStepper`：wall stream、质量交换、active collision、动态 topology 与失败回滚的独立事务；
- `initialize_hydrostatic_fields()`：使用 `c_s^2=1/3` 和 `dp/dy=rho*g` 初始化
  `rho(y)=rho_surface*exp(3*g_lattice*(y-y_surface))`，并以水平自由表面气压为锚点。

墙面和静水门禁覆盖：

- 六个边界平面必须完全闭合；
- 静止均匀液体经墙面 bounce-back 后 moments 不变；
- 墙链质量通量逐格为零；
- Warp CPU/CUDA wall stream、wall-link count 和 advected mass 与 NumPy oracle 对齐；
- GAS/SOLID 不参与重力碰撞；
- 墙邻居不被误判为 GAS，贴墙 INTERFACE 可以按非墙邻居正常转 LIQUID；
- 水平静水层连续 20 步保持低速且不发生 topology 转换；
- 每步提交后 wall mass、fill 和 wall momentum 均严格为零；
- 非法液气直连失败不修改调用方输出。

RTX 5090/Warp 1.12 上，R2-R3.6 联合回归为 `70 passed`。

### 重力液柱预演

最终预演使用 `160x96x8` 网格、六层内部深度、格点黏度 `0.02`、格点重力 `g_y=-2e-4`、初始液柱 `40x60`。初始底部相对自由表面的密度增幅约 `3.67%`，没有用过大的重力或密度比强行加速出图。

最初曾使用 `nz=3`，只有一层活动流体夹在前后 no-slip 墙之间，形成极强的 Hele-Shaw 阻尼：5000 步前缘只移动三格。该结果证明墙面稳定但不是合理的准二维 Dam-break，已改为 `nz=8`，不作为最终效果证据。

`nz=8` 的 3000 步结果：

| 指标 | 结果 |
|---|---:|
| 前缘 x | `41 -> 112` |
| 湿润高度 | `61 -> 45` |
| 液体质心 x | `20.755 -> 42.694` |
| 液体质心 y | `30.572 -> 13.340` |
| 全程最大活动速度 | `0.07684` |
| 活动密度总范围 | `[0.99286, 1.03666]` |
| 3000 步累计相对质量漂移 | `8.921e-6` |
| I→F / I→G | `124254 / 109044` |
| G→I / F→I | `109518 / 124591` |
| 严格事务吞吐 | 约 `129 MLUPS` |

液柱从竖直矩形逐步形成底部前缘并向右铺展，所有 3000 步均通过速度、有限值、质量、wall 与 topology 硬门禁。step 2000 附近出现少量短寿命内部空洞，step 3000 又闭合；它可能来自低分辨率界面 topology churn，必须保留为后续 R3.7 的观察项，不能用渲染平滑掩盖。

证据目录为 `tmp/home-rebuild/r3-wall-gravity-column-preview-final`，包含九张固定步长帧、速度图、总览图和 manifest。复现命令：

```bash
PYTHONPATH=. python -m wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_preview \
  --output /path/to/run \
  --device cuda:0
```

R3.6 证明 closed wall、重力、静水初始化和动态自由表面可以在一个严格事务中共同运行，但仍不是最终 Dam-break 验收：前缘尚未撞击对侧墙，没有反射浪、爬墙和回落阶段；halfway bounce-back 也不是后续 FSI 所需的 cut-link moving boundary。下一阶段 R3.7 应延长水槽冲击场景，逐帧审计短寿命空洞与 excess 队列，再决定是否先改善 topology 还是进入 PLIC 几何输运。

## 2026-07-22：R3.7 对侧墙冲击、周期深度基准与反射验收

R3.7 首先把 R3.6 的六面封闭 `160x96x8` 场景延长到 20000 步。前缘在 step 9000 到达右侧内部最后一列，至 step 20000 始终停在 `x=158`，证明没有穿墙；但液体只缓慢铺成浅层，右壁没有形成清晰的上冲和反射。该结果不是自由表面管线失效，而是边界条件造成的机制变化：六层内部深度同时受前后 halfway bounce-back 无滑移墙约束，是一个强阻尼的狭窄 Hele-Shaw 通道，不是常用二维 Dam-break 截面。

为隔离边界影响，墙面组合层新增了通用 `closed_axes` 契约：

- `axis_aligned_wall_mask()` 只在指定轴生成固体边界面；
- `validate_wall_mask()` 分别验证封闭轴和周期轴的最低尺寸要求；
- `FslWallMask.closed_box()` 保持六面封闭，R3.6 行为不变；
- `FslWallMask.periodic_depth_channel()` 使用 x/y 固壁和 z 周期边界；
- 单层 `nz=1` 在 z 方向沿用 HOME core 的周期寻址，x/y 墙链仍执行零质量通量和 halfway bounce-back；
- 静水初始化显式接收同一 `closed_axes`，不会把周期轴错误要求为固壁。

直接把 R3.6 重力 `g_lattice=-2e-4` 放到无前后壁阻力的周期深度通道，会在 step 1153 达到 `|u|=0.20138` 并触发低 Mach 硬门禁。这说明旧狭槽确实隐藏了过强的驱动力；放开门限继续计算会进入明显可压缩区，不能作为可信结果。依据特征速度 `U~sqrt(gH)`，把重力降为四分之一，即 `g_lattice=-5e-5`，预计速度减半、事件时间尺度加倍。黏度仍保持 `nu_lattice=0.02`，没有用提高黏度换稳定。

最终 R3.7 基准配置：

| 参数 | 数值 |
|---|---:|
| 网格 | `160x96x1` |
| 边界 | x/y halfway bounce-back，z 周期 |
| 初始液柱 | `40x60`，边缘半填充 interface |
| 格点黏度 | `0.02` |
| 格点重力 | `-5e-5` |
| 仿真步数 | `20000` |
| RTX 5090 总耗时 | `18.34 s` |
| 平均严格事务吞吐 | 约 `134.0 MLUPS` |

视觉阶段与量化证据一致：step 4000 的采样帧已撞击右壁并上冲；质心继续由惯性前移，在 step 6000 达到 `x=92.197`；反射后于 step 12500 回退至 `x=74.134`，反射质心位移为 `18.064` 格；后续出现幅度更小的二次往复，并逐渐趋于水平浅水层。全程没有短寿命内部气穴复现，说明 R3.6 的少量单格空洞与狭槽低分辨率 topology churn 有关，而不是周期深度基准中的持续拓扑缺陷。

`manifest.json` 新增机器可读 `impact_audit`，不再仅凭图片主观判断：

| 验收项 | 结果 |
|---|---:|
| 首个采样撞墙步 | `4000` |
| 最大右壁湿润高度 | `31` 格 |
| 反射质心位移 | `18.06374` 格，通过 `>=5` 门限 |
| 全程最大活动速度 | `0.076774`，通过低 Mach 门限 |
| 最大累计相对质量漂移 | `4.590e-5`，通过 `1e-3` 门限 |
| 固体穿透 | 无 |
| 持续内部气穴 | 无 |

证据目录为 `tmp/home-rebuild/r3-periodic-depth-wall-impact-final`，包含 step `0/2000/4000/6000/8000/10000/12500/15000/20000` 的九张帧、总览图和 manifest。远端复现命令：

```bash
PYTHONPATH=. python -m wanphys.examples.lbm.home_rebuild.home_free_fsl.gravity_column_impact \
  --output /path/to/run \
  --device cuda:0
```

R3.7 完成的是 HOME-FSL 基线 Dam-break 的“坍塌、撞墙、上冲、反射、回落”全流程门禁，并把六面闭槽和二维周期深度两个物理问题明确分开。它仍不等于最终综合项目完成：当前界面是 link-wise fill，不含 PLIC 几何重建、曲率/表面张力或润湿；墙仍是静止 halfway bounce-back，不是 FSI 需要的移动 cut-link；严格事务的多 kernel 和逐步同步也尚未进入性能融合阶段。下一阶段应保持本基准不变，单独引入 PLIC/几何 VOF 后做 A/B 对照，再决定低黏度与表面张力稳定性优化。

## 2026-07-24：R3.8 link-wise 真三维基线与挂壁审计

在二维全过程通过后，新增统一体数据工件与固定 45 度相机流程。模拟只保存提交态
`moments/mass/fill/excess/flags`，表面提取和 Blender 预览均为只读操作。旧渲染入口保留，
新公共对照模式不会改变模拟。

正式真三维场景使用 `160x96x64`、`nu=0.02`、`g_y=-5e-5` 和 20000 步。
RTX 5090 结果如下：

| 指标 | 结果 |
|---|---:|
| 总耗时 | `25.94 s` |
| 最大速度 | `0.07827` |
| 首个采样撞墙步 | `4000` |
| 最大前进质心步 | `6000` |
| 反射质心位移 | `10.3186` 格 |
| 最大累计相对质量漂移 | `1.883e-4` |
| 固体穿透 | 无 |

三维基线完成撞墙、爬升、反射和趋稳，数值门禁通过。固定相机也确认了明显挂壁：
最终左壁 `fill>=0.5` 单元最高到 `y=40`，右壁最高到 `y=16`；左/右壁最终
fill 体积约为 `1086/948`。这不是相机或容器绘制错误，而是当前 link-wise 基线
没有接触角、润湿和壁膜排液模型的真实缺口。后续场景的 manifest 已加入
`wall_film_audit`，不能再把挂壁只记录成主观观感。

证据目录：

- `tmp/home-rebuild/r3.9-linkwise-true3d-wide-formal`；
- `tmp/home-rebuild/r3.9-linkwise-true3d-wide-formal/linkwise-true3d-wide-overview.png`。

## 2026-07-24：R4 独立 PLIC/GVOF 复验

R4 新增不持有 HOME/FSL 状态的 `PlicDomain`，并把 PLIC 几何重构、方向分裂输运
和解析参考实现从 HOME 耦合模块中拆开。独立涡旋审计使用解析离散无散速度场，
不读取 HOME moments，不启用投影、拓扑、动量账本或 FSI。

强形变往返测试覆盖分辨率 `32/48/64/96` 和 Courant `0.175/0.35`。中点界面
各向异性约 `1.93`，确保测试确实发生强拉伸而不是近静态小扰动。最终结果：

| Courant | L1 拟合收敛阶 | 最大单步相对体积漂移 |
|---:|---:|---:|
| `0.175` | `1.7477` | 约 `1e-9` |
| `0.35` | `1.7337` | 约 `1e-9` |

所有有限性、边界、体积、强形变和单调收敛门禁通过。由此可知 PLIC 算子在其
成立前提下没有基础错误；这不代表它可以直接接收非离散无散的 HOME 面速度。

证据目录为 `tmp/home-rebuild/r4-plic-standalone-strong-final`。

## 2026-07-24：R5 Dam-break 分阶段消融

R5 首先完成 `FSL+PLIC-geometry` 只读叠加：九个快照前后 SHA 完全一致，
PLIC 法向均有效，红色界面分段改善了显示和曲率输入，但没有改变挂壁或任何物理量。
随后建立四个显式组合类型，禁止用隐藏开关切换语义：

- `GvofFslWallStepper`：原始 HOME Courant + PLIC 体积/质量输运 + 旧 FSL 拓扑；
- `ProjectedGvofFslWallStepper`：在上一项上只增加 Courant 投影；
- `ProjectedGeometricFslStepper`：再只替换为严格几何拓扑，不输运动量；
- `GeometricHomeFslStepper`：最后加入一致面通量动量账本。

同一 `96x56x1`、200 步预演的统一结果：

| 变体 | 最后有效步 | 相对 link-wise 速度 | 相对质量漂移 | excess 比例 | 界面增长 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| link-wise | `200` | `1.000x` | `3.204e-7` | `1.938e-5` | `1.08x` | 候选 |
| GVOF，无投影 | `1` | `0.028x` | `0` | `6.839e-6` | `1.00x` | step 2 有 48 个越界单元 |
| GVOF + 投影 | `200` | `0.017x` | `1.087e-6` | `6.471e-3` | `1.05x` | 旧拓扑账本积压且慢约 57 倍 |
| + 几何拓扑 | `89` | `0.014x` | `1.539e-9` | `0` | `13.90x` | step 90 出现 3 条液气直连 |
| + 动量账本 | `71` | `0.016x` | `2.294e-9` | `0` | `13.64x` | step 72 同类失败 |

未投影 GVOF 的失败原因是原始 HOME 面 Courant 在界面区不满足离散无散条件。
关闭 Weymouth-Yue 压缩项可以恢复全局通量守恒，但会立即产生 fill 越界；放宽
容差不是合法修复。投影把最大散度从 `9.52e-3` 压到 `2.06e-7`，证明数学前提
得到恢复，但 PCG 每步最多打满 320 次，并暴露几何质量输运与旧 topology 的
excess 表示不匹配。

几何 topology 直接提交 PLIC fill/mass 后不再积压 excess，但界面单元从 59
增长到约 800，随后产生直接 LIQUID-GAS 邻接。加入动量账本没有修复拓扑，
反而把存活步数从 89 降到 71。因此当前不能把“全量几何”描述成 link-wise 的
升级版；它保持为失败研究变体，后续必须先独立实现守恒拓扑闭合和邻接修复。

统一对照证据：

- `tmp/home-rebuild/r5-dambreak-ablation-summary/dambreak-ablation-overview.png`；
- `tmp/home-rebuild/r5-dambreak-ablation-summary/summary.json`；
- `tmp/home-rebuild/r5-dambreak-ablation-summary/summary.md`。

R5 当前只选择 link-wise 进入长期和三维场景。该选择不是宣称 link-wise 已满足
最终真实感：它仍有挂壁、无接触角、界面精度有限和静止 halfway bounce-back
等缺口。当前重构专项回归为 `101 passed`。下一研究增量应先在独立小场景解决
几何 topology 的 sliver 闭合，再重复 R5；在此之前不把几何组合接入 FSI。

## 2026-07-24：R6 拓扑闭合、单位校正与速度失败分层

R5 后续首先精确回放了 `ProjectedGeometricFslStepper` 的 step 90 失败。原始
PLIC 输运在 `(25,26,0)` 一带生成近满界面尾巴，旧分类仍把它保留为
INTERFACE，最终与 GAS 形成直接邻接。R6 加入两项不改变守恒账本的拓扑闭合：

- `fill >= 1-endpoint_tolerance` 的近满单元进入 LIQUID 候选；
- 第二个只读邻接 pass 把紧邻 GAS 的 LIQUID 候选退回 INTERFACE。

源状态液气直连和单步 GAS→LIQUID / LIQUID→GAS 跳变仍在修复前检查，修复不能
掩盖非法输入。`endpoint_tolerance=5e-6` 后原 step 90 故障消失，200 步门禁
通过；`2e-6` 在 step 183 仍会遇到不可重构的近满法向，因此 5e-6 是当前最小
通过值，而不是任意放宽。

随后发现 R5 几何预览存在物理量/格子量错配。`HomeCoreModel` 接收物理黏度与
物理加速度，默认 `dx=0.1, dt=1`。旧几何配置传入 `nu=0.02, g=-5e-5`，实际
得到：

| 旧几何预览 | 格子量 |
|---|---:|
| 黏度 | `2.0` |
| 重力 | `-5e-4` |

正式 link-wise 基线实际使用 `nu_lattice=0.02, g_lattice=-5e-5`。因此旧 R5
表格只保留为故障发现历史，不能继续作为算法优劣比较。几何预览、拓扑回放和
动量回放现统一为物理 `nu=2e-4, g=-5e-6`，并增加配置回归测试，明确验证转换后
格子量为 `0.02/-5e-5`。

对齐尺度后的结果：

| 变体 | 结果 | 失败阶段 |
|---|---:|---|
| full geometric momentum | step `625` | collision `0.19098`，动量提交后 `0.20421` |
| projected geometric，无独立动量回写 | step `1306` | collision/commit 均 `0.20007` |

full 变体在 `(35,5,0)` 的回放证明现公式发生双重平流：HOME streaming 已把局部
速度从 `-0.0772` 更新到 `+0.1889`，PLIC 又输运得到 `-0.0638`，随后公式把完整
`candidate-old` 增量再次叠加，得到 `+0.2022`。该实现不能继续作为候选；若要
保留独立动量账本，必须把 HOME 压力/黏性应力冲量与对流通量严格分解。

无独立动量回写的变体稳定通过 1000 步，最大速度 `0.05466`。延长后在 step
1306 被一个 `fill=1.63e-9` 的亚网格尾巴触发低 Mach 门禁；此时所有
`fill>5e-6` 的可解析连续液体最大速度仅 `0.04982`。共有 259 个亚网格尾巴，
总体积 `6.34e-5`。这不是主体液体失稳，但这些格点仍参与 HOME collision，
因此也不能简单忽略速度门禁。下一增量是守恒 endpoint closure：把低于分辨率
的 volume/mass/momentum 从 LBM 活跃相移出并严格回灌到可解析界面；未来可把
同一账本替换为 spray 粒子，而不是放宽连续液体门限。

新增证据：

- `tmp/home-rebuild/r6.10-aligned-momentum-failure-replay`；
- `tmp/home-rebuild/r6.11-aligned-projected-geometric-1000`；
- `tmp/home-rebuild/r6.13-projected-geometric-failure-replay`。

### 守恒 endpoint closure 与首次几何撞墙

R6 随后实现全局守恒 endpoint closure。每步先把
`0 < fill <= endpoint_tolerance` 的不可解析尾巴从活动相中移出，再按所有可解析
INTERFACE 单元的剩余容量比例回灌 volume、mass 和三分量 momentum。回灌前先
验证总容量不小于待回灌体积；容量不足、非有限账本或越界均硬失败。该步骤不删除
质量，也不通过忽略 tiny-fill 速度来绕过低 Mach 门禁。

NumPy oracle、Warp CPU、既有 topology/transport/stepper 联合回归均通过；当前
PLIC 重构目录为 `56 passed`。在原 step 1306 失败场景中：

| 指标 | closure 前 | closure 后 |
|---|---:|---:|
| 可运行步数 | `1305` | `5000`（场景完成） |
| 最大速度 | `0.20007` 后失败 | `0.04890` |
| 最大 sub-tolerance 活跃界面 | `260` | `5` |
| 前缘最终位置 | `x=65` | `x=94`（右壁） |

5000 步中每步最多移除 6 个 endpoint 单元，累计处理 2829 个单元事件，累计回灌
体积 `0.0044975`、质量 `0.0045157`。累计量会重复计入跨多步再次落入阈值的同一
小体积，因此不是净体积改变；最终总质量相对初始量的漂移仍约 `7.6e-7`。

视觉序列在 step 2500 首次完整接触右壁，step 3000–5000 出现右侧上冲和回落。
左侧初始接触墙仍保留明显挂壁，证明 endpoint closure 只解决数值尘埃，不替代
接触角或退润湿模型。当前权威候选是“HOME 持有动量，投影 PLIC 持有
volume/mass”；独立 PLIC momentum 变体继续保留为失败研究分支，除非以后完成
HOME 压力/黏性应力冲量与对流通量的严格分解。

新增证据目录：

- `tmp/home-rebuild/r6.14-endpoint-closure-projected-geometric-2000`；
- `tmp/home-rebuild/r6.15-endpoint-closure-projected-geometric-5000`。

### 近端点无梯度回退与 20000 步闭合

5000 步之后延长到 20000 步时，旧实现于 step 11055 在
`fill=0.9999947` 的主体液体内部格点失败。该格点被近满 fill 判为几何界面，
但其邻域全部近满，因此 Youngs 梯度为零，PLIC 法向在数学上不可定义。这与挂壁
无关，也不是主体速度或质量失稳。

R6 增加范围受限、可计数的端点 donor-cell 回退：

- 只允许落在 `max(4*endpoint_tolerance, 32*float32_eps)` 范围内；
- 中间 fill 的无梯度界面继续硬失败；
- 回退只替换该方向面的几何通量，不修改 fill/mass 账本；
- 每个回退面写入 axis diagnostics 和场景 manifest。

原 step 11054 快照回放到 11064 时只触发 1 个回退面，绝对体积/质量漂移分别为
`1.87e-6/-5.44e-6`。完整 20000 步正式场景只累计触发 3 个回退面，单步峰值为
1，说明它是稀有浮点退化处理，不是大范围降阶。最终结果：

| 指标 | 结果 |
|---|---:|
| 完成步数 | `20000` |
| 耗时 / 速度 | `390.12 s / 51.27 step/s` |
| 最大速度 | `0.0489038` |
| 最大相对质量漂移 | `4.351e-6` |
| 反射质心位移 | `5.0495` 格 |
| 端点回退面总数 | `3` |
| 最终左壁半满高度 / 主体高度 | `18 / 10` 格 |

权威证据为
`tmp/home-rebuild/r6.19-endpoint-fallback-projected-geometric-20000`。PLIC 专项
回归在进入壁面阶段前为 `59 passed`。

## 2026-07-24：R7 静态接触角闭合与挂壁 A/B

`home_lbm/vof` 已有静态接触角法向闭合，R7 将同一约定隔离移植到
`home_rebuild/free_surface_plic`。该功能默认关闭；启用时只在壁面法向和界面
切向均可解析的三相接触线施加
`n = cos(theta) n_wall + sin(theta) n_tangent`。界面与壁面平行的薄膜没有唯一
接触切向，不再伪造方向，而是保留原 PLIC 法向并单独计数。

CPU/CUDA 对 `60/90/120` 度平面壁解析法向、非法角度、平行壁膜和基线不变性均
通过。当前 PLIC 专项回归为 `62 passed, 6 subtests passed`。

相同 `96x56x1`、`nu_lattice=0.02`、`g_y=-5e-5` 的 2500 步筛选结果：

| 壁面设置 | 左壁半满高度 | 主体高度 | 主体上方壁膜体积 | 右壁半满高度 |
|---|---:|---:|---:|---:|
| 关闭接触角 | `31` | `12` | `17.863` | `0` |
| `90` 度 | `31` | `12` | `17.754` | `5` |
| `120` 度 | `31` | `12` | `17.684` | `0` |
| `150` 度 | `31` | `12` | `17.819` | `0` |

`120` 度仅减少约 1% 的壁膜体积，挂壁高度完全不变；`90` 度还增加了冲击墙
附着。由此确认静态接触角是必要边界条件，但单独不足以为平行壁膜提供毛细压力
或退润湿驱动力。不能继续通过调角度宣称问题已解决。下一门是独立实现并验证
二维/三维一致的曲率、Laplace 压力和毛细动量，再重复同一 A/B；若平膜仍无法
退去，需显式加入有物理尺度的薄膜/动态接触角模型。

证据目录：

- `tmp/home-rebuild/r7.2-wetting-90-2500`；
- `tmp/home-rebuild/r7.3-wetting-120-2500`；
- `tmp/home-rebuild/r7.4-wetting-150-2500`。

## 2026-07-31：R7.5 初始左壁 offset 诊断

为区分“初始化时液柱贴壁遗留的水膜”和“动态接触线/毛细模型不足”，溃坝配置
新增 `column_offset_x`，默认值为 0，因此旧场景完全不变。offset 大于 0 时，
该值表示左壁与液柱左侧界面之间保留的干燥活动格数量；初始化显式构造左界面、
液体主体和右界面，不能用数组平移产生 LIQUID-GAS 直接邻接。

offset 初始化使用两侧各 0.25 fill 的界面列，并保持与旧贴壁液柱相同的总初始
体积。`offset=3` 的本地 CPU 门禁结果：

| 指标 | 结果 |
|---|---:|
| 基线 / offset 初始体积 | `845.5 / 845.5` |
| 初始左壁 fill 体积 | `0` |
| 初始干燥间隙 | `3` 格 |
| 10 步最大速度 | `0.003286` |
| 10 步相对质量漂移 | `3.691e-10` |
| 10 步左壁 fill / 半满高度 | `0 / 0` |
| 活动区 LIQUID-GAS 直连 | `0` |

该结果只证明初始化、投影和早期拓扑合法，不回答长期挂壁是否改善。5090 恢复后
应在相同 `96x56x1`、`nu_lattice=0.02`、`g_y=-5e-5` 下运行 offset
`0/2/4` 的 2500 步 A/B；比较初始体积归一化后的左壁膜、首次动态接触左壁的
时刻、主体液面高度和右壁冲击。若 offset 场景始终不再接触左壁，它只能证明
“初始贴壁膜被保留”，不能单独证明壁面边界正确；还需设计液体主动撞击左壁的
对称场景。

## 2026-07-31：R7.6 offset=4 三维诊断

在 RTX 5090 上完成 `96x56x32`、`column_offset_x=4`、`nu_lattice=0.02`、
`g_y=-5e-5` 的 2500 步 projected-geometric 运行，并按
`0/500/1000/1500/2000/2500` 步输出三维对角视角图片。当前 z 方向是周期挤出，
用于检查水槽、界面和壁膜的三维视觉关系，不代表已完成具有深度方向扰动的三维
物理基准。

| 指标 | 结果 |
|---|---:|
| 完成步数 | `2500` |
| 耗时 / 速度 | `88.97 s / 28.10 step/s` |
| 最大速度 | `0.0485203` |
| 最大投影后散度 | `3.162e-8` |
| 最大相对质量漂移 | `1.514e-7` |
| 初始左壁 fill / 半满高度 | `0 / 0` |
| 最终左壁半满高度 | `0` |
| 最终主体高度 | `13` 格 |
| 最终主体上方左壁薄膜体积 | `17.843` |

与贴壁基线 step 2500 的左壁半满高度 `31` 相比，offset=4 的厚竖直挂壁降为
`0`，说明初始液柱贴壁是旧现象的重要来源。液体塌落后会再次接触左壁，最终仍
留下低 fill 薄膜，因此动态润湿/退润湿问题没有被 offset 解决。后续边界改进应
以“主动回触后的薄膜能否合理退去”为验收项，不能继续只比较半满高度。

权威证据目录为
`tmp/home-rebuild/r7.6-offset4-plic-nz32-2500`，其中
`frames-diagnostic-diagonal` 保存完整六帧图片组。

## 2026-07-31：R7.7 短槽与 0.1 物理 offset

为提前观察对侧墙冲击，将水槽从 `96x56x32` 缩短为 `64x56x32`，水柱尺寸保持
`24x34` 不变。模型默认格长为 `0.1`，因此 `column_offset_x=1` 对应物理间隔
`0.1`。场景在 RTX 5090 上完整运行 3000 步，并输出
`0/500/1000/1500/2000/2500/3000` 七帧对角视图。

| 指标 | 结果 |
|---|---:|
| 完成步数 | `3000` |
| 耗时 / 速度 | `138.39 s / 21.68 step/s` |
| 首次采样到右壁接触 | `1500` 步 |
| 最大速度 | `0.0629232` |
| 最大投影后散度 | `6.147e-8` |
| 最大绝对相对质量漂移 | `6.553e-7` |
| 最终左 / 右壁半满高度 | `0 / 14` 格 |
| 固体穿透 | 无 |

短槽使右壁冲击在当前窗口内清晰出现。初始左壁 offset 仍能消除厚挂壁，但右壁
冲击后形成明显爬升；截至 3000 步，质心仍向右移动，尚未进入可量化的整体反射
阶段。该结果不能用来判定右壁附着是否会自行退去，后续若研究撞墙回落，应延长
运行时间并保持同一采样与相机配置。

权威证据目录为
`tmp/home-rebuild/r7.7-short-tank-offset1-plic-nz32-3000`。

## 2026-07-31：R7.8 物理等尺度高分辨率对照

短槽场景从 `64x56x32` 加密到 `128x112x64`。为保持物理尺寸和参数一致，格长
从 `0.1` 降到 `0.05`，时间步从 `1.0` 降到 `0.5`，水柱尺寸同步从 `24x34`
增至 `48x68`，offset 从 1 格增至 2 格，因此物理间隔仍为 `0.1`。配置新增
显式 `cell_size/time_step`，缩放回归为 `4 passed`。

首轮在 step 1127 遇到沿全部 64 个 z 切片重复的近满端点退化
`fill=0.999977`。将界面端点容差从 `5e-6` 调整到 `1e-5` 后，场景完整运行到
3000 步；该容差只分类万分之一量级的近空/近满界面。

| 指标 | 结果 |
|---|---:|
| 完成步数 | `3000` |
| 耗时 / 速度 | `226.65 s / 13.24 step/s` |
| 首次采样到右壁接触 | `3000` 步 |
| 最大速度 | `0.0755690` |
| 最大投影后散度 | `1.329e-7` |
| 最大绝对相对质量漂移 | `2.587e-7` |
| 最终左 / 右壁半满高度 | `0 / 15` 格 |
| 固体穿透 | 无 |

高分辨率图像的前缘和斜坡轮廓更连续，但没有出现新的深度方向细波纹。这是符合
当前实现的：stepper 只沿 `(x,y)` 输运，初始 fill 在 z 方向严格一致，z 方向
只是周期挤出；此外，重构 PLIC 场景尚未接入曲率、Laplace 压力和表面张力。
因此继续单纯增加 `nz` 只会复制更多相同切片，不能产生三维水花或毛细波。下一
个有效实验应隔离开启 z 输运，加入可复现的小幅深度扰动，再决定是否接入已经在
旧 `home_lbm/vof` 中验证过的表面张力模块。

权威证据目录为
`tmp/home-rebuild/r7.8-hires-short-tank-offset01-plic-3000`。

## 2026-08-03：R8-R12 最终组合审计

曲率、Laplace 压力、静态润湿、低黏度扫描和双向 FSI 已按隔离门槛完成。最终不
再把所有机制无条件叠加，而是固定 `fast / balanced / strict` 三档。RTX 5090
同一 `54x54x86` FSI 基准分别为 `6.092 / 18.359 / 60.959 ms/step`；strict
最大投影散度为 `2.794e-9`，但不作为默认 Viewer。完整 HOME/HOME-Free 回归为
`520 passed, 185 subtests passed in 475.59s`，新增三档 Viewer 专项为 `4 passed`。

动态接触角未以经验局部流速公式接入：当前缺少滑移长度、接触线速度历史和网格
收敛标定。自由表面低黏度可信下限暂为 `nu=0.02`；`0.01875` 及以下在约 2100
步后越过速度稳定门槛。详细数据、误差预算和 VNC 命令见
`home-rebuild-final-validation.md`。
