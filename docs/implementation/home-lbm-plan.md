# HOME-LBM 重构实施计划

本文档定义 WanPhys 新一代 LBM 的实现边界。旧目录 `wanphys/_src/fluid/fluid_grid/lbm/` 保持原样，仅作为行为和性能回归参考；新实现位于 `wanphys/_src/fluid/fluid_grid/home_lbm/`，在达到各阶段验收门槛前不替换默认后端。

## 不可退让的原则

1. 单相核心采用 D3Q27 HOME-LBM，持久流体状态为 SoA 布局的 `rho`、`rho*u`、`rho*S` 共十个标量，不以 D3Q19 或完整分布持久化冒充 HOME。
2. 三阶 Hermite filtered reconstruction 与高阶中心矩 collision 必须配套；BGK/TRT 只允许作为对照实现。
3. 物理时间步、格距、参考密度、黏度、速度、加速度、力和力矩通过统一 scaling 显式转换，禁止忽略 `dt` 或依靠经验 `force_scale`。
4. 流固边界使用真实 cut-link 交点；边界分布和反作用动量在同一离散事件中产生，保证链路级作用反作用一致。
5. 自由表面选择 HOME-Free sharp VOF。多流体相场不是当前主线，不在 VOF 之前并行建设。
6. FP32 物理基线通过守恒、收敛和稳定性验收后才允许低精度量化。

## 目标结构

```text
home_lbm/
  constants.py        D3Q27、Hermite 和十矩布局
  scaling.py          物理量与格子量转换
  model.py            静态配置和稳定性约束
  state.py            双缓冲十矩状态
  reference.py        NumPy 可执行数学规范
  kernels.py          Warp 热路径
  collision.py        NOCM-MRT 闭式矩碰撞
  boundary.py         cut-link 与边界矩
  diagnostics.py      守恒、Mach、应力和饱和监控
  rigid_geometry.py   全量初始化与无同步动态区域 SDF 更新
  rigid_coupling.py   中点几何、流体载荷和刚体子步调度
  vof/                fill/mass/flag、PLIC、曲率和气泡

coupling/
  home_lbm_rigid_coupling.py
```

## 阶段与验收门槛

### M0：数学与数据契约

- 固定 D3Q27 顺序、权重和 opposite 映射。
- 实现 `rho/rho*u/rho*S` SoA 状态和显式单位转换。
- 实现论文式 (17) 的三阶 Hermite 参考重构及 Warp 一致性测试。
- 验收：格子二阶各向同性；十矩往返误差满足 FP64 `2e-13`、FP32 `2e-5`；状态不存在持久 `f[27N]`。

### M1：单相 HOME 推进

- 实现无边界 pull-stream、矩提取、外力半步修正和中心矩 MRT 闭式 collision。
- 临时分布仅存在于寄存器/shared memory；先正确实现，再依据 profile 选择 split 或 fused kernel。
- 验收：均匀流不漂移；周期域质量/动量守恒；Poiseuille 恢复目标黏度；Taylor-Green 速度误差和收敛率达标。

### M2：曲面边界和双向 FSI

- 通过 BVH/窄带构造 cut links，保存交点、法线、body ID 和壁面速度。
- 按论文继承流体节点的非平衡应力，在交点重构出射分布。
- 使用 Galilean-invariant momentum exchange，并累加到独立 `fluid_wrench`，不得清除刚体其他力。
- 加入刚体预测姿态、流体/固体子步以及守恒 fresh/dead-cell 重构。
- 验收：静壁零净力、Couette、移动/旋转球、球阻力、自由沉降及全系统动量收支。

### M3：HOME-Free sharp VOF

- 实现 `LIQUID/INTERFACE/GAS/SOLID`、质量与 fill fraction、link-wise mass exchange。
- 实现界面缺失分布、PLIC 法线/曲率、表面张力和守恒拓扑转换。
- 第一阶段气体使用统一压力；随后再做封闭气泡 CCL 和压力演化，泡沫与溶解气体最后评估。
- 验收：平移界面、静水、Laplace 压差、液柱断裂和总体液体质量。

### M4：自由表面 FSI

- 对跨相界面的 cut link 使用相感知动量交换和压力/密度过滤。
- 统一处理固体入水、出水和界面附近 fresh/dead cells。
- 验收：静水浮力、球体入水、轻重球、溃坝冲击刚体及分辨率收敛。

### M5：性能和量化

- 建立每 MLUPS 的带宽、寄存器、kernel launch 和峰值显存基线。
- 比较 fused、split、shared-memory tile 和 on-demand reconstruction。
- 按稳定性分析为各矩设置量化范围、scale 和 saturation 计数。
- 验收：量化误差预算明确；无静默饱和；性能提升不能破坏前述物理门槛。

### M6：替换与兼容

- 提供显式 backend 选择和旧场景迁移工具。
- 新实现通过完整回归后才切换默认 backend；旧 `lbm` 继续保留一个发布周期。
- 默认切换不允许依赖参数补偿来伪造旧视觉结果。

## 当前状态

M0 已完成：独立包、D3Q27 常量、单位体系、十矩状态、参考重构、Warp 投影 kernel 和基础测试已经建立。

M1 已完成：已实现论文式 (20)-(23) 的矩空间 forcing 与中心矩闭式 collision、单-pass pull-stream、逐轴周期/静态 halfway 域壁、双缓冲 `HomeLbmDomain/HomeLbmSolver` 和显式运行时诊断，并通过以下门槛：

- Warp 单步与逐格 NumPy 参考一致；
- 周期域全局质量和动量守恒；
- 均匀流二十步保持；
- 均匀体力每步产生精确的全局动量增量；
- Poiseuille 全速度剖面误差小于 0.3%；
- Taylor-Green 模态恢复配置黏度，误差小于 1%；
- 16² 到 32² Taylor-Green 网格加密呈二阶收敛；
- NaN/Inf、非正密度、最大速度和非平衡应力可显式诊断。

M2 正在实施，当前已完成并通过测试的部分如下：

- `HomeLbmRigidGeometry` 直接复用共享 `RigidShapeQueryData`，从 Newton/WanPhys 刚体形状统一生成物理单位 SDF 和内部 `body_id`，不再要求按球、盒、胶囊手工注册近似几何。
- 几何首次初始化执行精确全场 SDF；后续更新用保守 body AABB 半径覆盖上一/当前姿态区域，每个区域单元仍对全部耦合刚体求最小距离。重叠区域按 body entry 确定唯一写入者，避免 GPU 数据竞争；固定区域容量在初始化时生成，时间步内不做 CPU 同步。当区域 launch 容量不小于全网格时显式选择全量 kernel。测试要求全局 solid mask 与全量重算完全一致，当前表面两格带 SDF 数值一致。
- `CutLinkBuffer` 通过 GPU count、exclusive scan 和 compact write 构造按流体单元分组的稀疏链路，保存方向、SDF 交点比例、真实交点和 body ID，只按表面链路数分配。
- pull-stream 遇到内部固体时采用 Bouzidi 线性插值反弹：`q<0.5` 从远离壁面的第二流体节点插值，`q>=0.5` 从当前节点的入射/出射分布插值，移动壁修正使用真实 SDF 交点速度。缺失 cut link 或 `q<0.5` 时缺少第二流体支撑节点都会被显式计数并使状态验证失败，不允许静默退化。
- 每条链路在同一边界事件中按式 (25) 计算 Galilean-invariant momentum exchange，再按式 (26)-(27) 归并到独立 `FluidWrenchBuffer`。力与力矩分别使用明确的物理单位写入 `body_f`，写回只做增量累加，不清除其他载荷。
- 动态壁速使用 Newton 的世界系 `(linear, angular)` 空间速度、局部质心和真实交点计算 `u_p = v + omega x r`，然后通过 `dt/dx` 转成晶格速度。
- `ConservativeCellRemapper` 处理 moving-solid fresh/dead cells。早期逐 cell 局部分摊方案虽然全局守恒，但在 Ten Cate 沉降中首次 8 个 cell 转换便把密度偏差从 `1.52%` 推到 `23.8%`。现实现采用两阶段算法：fresh cell 先从持续流体邻居重构，dead cell 从有效域移除；随后在 GPU 上计算旧/新有效流体十矩差，将小残差均匀回写到持续流体集合。这样逐分量保持全局十矩守恒，同时避免局部压力脉冲；无可用供体仍会显式失败。
- `RigidMotionPredictor` 按世界系 constant twist 生成半步/整步姿态；偏心局部质心通过推进世界质心后反求 body 原点，避免旋转时的错误平移。统一 shape AABB 同时给出保守表面半径，用于检查 `(|v|+|omega|R)dt/dx` 位移 CFL。
- `HomeLbmRigidInterface` 负责中点姿态增量栅格化、守恒 cell remap、cut-link 安装及流体推进后的 wrench 归并；`HomeLbmRigidCoupling` 保留原显式中点路径，并提供显式选择的强耦合路径。强耦合先固定二阶中点几何与 fresh/dead 拓扑，再从同一 `fluid_n/rigid_n` 快照反复计算壁速、一个 HOME 步、GI wrench 和刚体子步，以最大表面晶格速度作为残差并使用有上限的 Aitken 松弛。收敛后才提交；超过最大迭代会恢复流体矩、SDF、cut links、刚体状态和调用方外力后抛错。固定拓扑是必要的：在非线性迭代内反复切换 cell 类型会形成非光滑映射并导致残差周期跳变。
- `HomeLbmSolver.initialize_hydrostatic_lattice` 构造受力密闭域的离散静水平衡。持久状态是碰撞后矩，因此零物理速度使用 `j=rho*a/2`；D3Q27 沿每个非周期轴的密度比为 `r=(1+1.5a)/(1-1.5a)`。初始化按目标平均密度归一化，强迫周期轴会被拒绝。Ten Cate 场景中，该初始化把预热浮力相对标准差从 `19.5%` 降到 `1.2e-3` 以下。

现有测试覆盖平面精确链接计数、球面交点、刚体平移/旋转壁速、Bouzidi 两分支连续性、插值边界与式 (25) 逐链接参考对照、静壁作用反作用、偏心 wrench 归并、物理单位写回和 moving-solid 十矩守恒。周期域移动球进一步连续推进 24 个流体步和 48 个刚体子步，在确实发生 fresh/dead 转换的条件下检查“有效流体物理动量 + 刚体动量”，相对漂移小于 `2e-5`，并确认球体受到阻力后减速；该场景已在 CPU 与 RTX 4090 CUDA 上通过。周期域旋转球连续推进 10 个流体步，以固定世界原点计算流体角动量、刚体轨道角动量和 `R I_body R^T omega` 自旋角动量；总角动量相对漂移小于 `3e-5`，且流体获得正向角动量、球体角速度下降。该场景同样已在 CPU 与 RTX 4090 CUDA 上通过。当前 halfway 域壁仍只用于 M1 基准，不作为最终曲面 FSI 边界。

定常周期球阻力采用 Hasimoto/Sangani 周期阵列低 Reynolds 数基准。测试以恒定体积流量反馈维持 intrinsic mean velocity，并比较

`F = 6*pi*mu*R*U*(1-phi) / (1 - 1.7601*phi^(1/3) + phi - 1.5593*phi^2)`。

在相同体积分数、松弛时间和 `Re=0.0224` 下，RTX 4090 上的粗网格 `L=24, R=3.2` 相对误差为 `7.91%`，细网格 `L=36, R=4.8` 相对误差降至 `4.02%`；稳态采样力相对标准差分别为 `1.66e-4` 和 `2.12e-4`，流量控制相对误差均小于 `7.3e-7`。该结果证明当前曲面边界随空间加密向周期球阻力解析基准收敛，但还不是自由沉降和时间步无关性的替代品。公式依据 [Hasimoto (1959)](https://doi.org/10.1017/S0022112059000222)、[Sangani and Acrivos (1982)](https://doi.org/10.1016/0301-9322(82)90047-7) 及 [van der Hoef et al. (2005)](https://doi.org/10.1017/S0022112004003295)。

自由沉降采用 [ten Cate et al. (2002)](https://doi.org/10.1063/1.1512918) E1 实验：15 mm、1120 kg/m³ 尼龙球，在 100×100×160 mm 密闭硅油槽中从距底 120 mm 处释放；流体密度 970 kg/m³、动力黏度 0.373 Pa·s，`Re=1.5`、实验无限域速度 0.038 m/s，实验最大速度为 `0.947 U_inf`。实现保持密度比、Galileo 数、Re 和槽体几何比，并将释放点流体密度归一化为参考密度。显式分区在密度比 1.155 下出现 added-mass 反向振荡，成为启用强耦合的直接证据。

RTX 4090 的空间序列并非逐层单调：`D=8` 与 `D=12` 仍有明显格点相位误差，但 `D=16`（半径 8 格点，约 196 万流体 cell）得到最大速度 `0.93144 U_inf`，距离实验值 1.56 个百分点；平台速度 `0.92393 U_inf`、相对标准差 `0.40%`，静水浮力误差 `1.84%`，最大密度偏差 `0.335%`，最大横向速度 `1.80e-4 U_inf`。三倍理论平流时间内球体下沉 2.475 个直径，确实发生 fresh/dead 转换，强耦合残差低于 `2e-6`。D=8 的时间步缩小 25% 后最大速度变化低于一个百分点，时间离散门槛已受控。空间结果应解释为误差包络随 D=16 改善，不能宣称二阶单调收敛。

距离感知边界、动态区域几何更新、显式与强耦合调度、移动球全系统线动量、旋转球角动量、周期球阻力以及 Ten Cate E1 加速/准终速沉降门槛已经关闭。Couette 继续以 `q=0.25/0.5/0.75` 的真实 SDF 交点作为解析无滑移平面；强耦合测试额外验证收敛提交、失败时双域回滚和外力所有权恢复。

底部近壁阶段采用 Ten Cate 式解析差值而不是经验弹簧。当球心法向速度为 `u_n`、球半径为 `R`、动力黏度为 `mu`、间隙为 `h`，且 `h<h_c` 时，增加

```text
F_corr = -6*pi*mu*R^2*(1/h - 1/h_c)*u_n*n,
h_c = 1 cell.
```

差值在 `h=h_c` 连续为零，避免重复计算网格已解析的远场阻力。`h` 由显式配置的粗糙度间隙截断，达到该间隙后执行零法向恢复系数的位置/速度投影；周期方向没有墙。Ten Cate 给出的表面粗糙度约为 0.1–1 mm，`D=8` 时默认 `0.5 cell` 对应 0.94 mm，位于实验范围内并且没有超出当前 cut-link 分辨能力。当前实现只接受每个选定动态刚体恰好一个解析球，复合体、非球体和静态体直接拒绝。构造阶段检查最坏角落三壁阻尼数 `C_max*dt_rigid/m<1`，不稳定时要求增加刚体子步或修改有物理依据的粗糙度，绝不裁剪力。

当 `q<0.5` 的第二 Bouzidi 支撑点落入另一固体时仍是硬错误；只有支撑点越过非周期槽壁且球壁润滑已显式启用时，才允许记录为 `wall_confined_interpolation_count` 的局部反弹闭合。该局部反弹只闭合流体分布，不向刚体重复提交伪冲量。润滑模式还会从 cut-link 载荷中扣除初始化时的解析静水基准压力，再用排开流体体积加入 Archimedes 浮力；动态压力与非平衡黏性载荷仍由 HOME 计算。这样亚格点模型不会掩盖球-球窄隙或一般几何缺陷。RTX 4090 上 `D=8`、16 个理论平流时间的 Ten Cate E1 全轨迹已推进到 `0.5` 格粗糙度接触；最终速度和最大反弹均低于 `0.01 U_inf`，最大密度偏差低于 3%，横向漂移低于 `0.002 U_inf`，全程没有穿透或非物理反弹。

### M2 性能基线

`scripts/bench/bench_home_lbm_fsi.py` 固化了 RTX 4090 性能场景：`54x54x86`、250,776 个格点、半径 4 格球体、两个刚体子步，并分别测量显式分区和强耦合。阶段模式在每个被测调用边界同步，用于定位开销；`--total-only` 只在整步边界同步，用于报告真实生产吞吐。两类数据不可混用。

当前性能路径包含以下不改变离散方程的优化：

- fresh/dead 计数均为零时，remap 直接保留十矩并更新几何历史，不再启动完整的十矩守恒修正链；实际发生 cell 转换时仍执行原两阶段守恒算法和硬错误检查。
- solid sign mask 不变时，cut-link 的 CSR 拓扑、offset 和分配保持不变，只在原位更新 `q`、交点、body ID 与壁速；任一 cell 类型变化仍完整 count/scan/compact 重建。
- post-collision 有限性、密度、速度和非平衡应力归约融合进 `stream_collide_kernel`。每一步仍执行全部诊断；初始化或外部修改状态可用 `force_recompute=True` 走独立全网格扫描，两条路径已有逐字段一致性测试。
- 八个 GPU 诊断标量先在设备端打包为一个整数向量和一个浮点向量，再进行两次主机回读；计数仍使用 `int32` 原子量，不以浮点近似计数。
- 强耦合第一个固定点迭代直接使用刚生成的 prepared/input 状态；后续回滚只重置会变化的流体十矩，冻结的 SDF/body ID 由双缓冲首步传播。静态包围半径只在构造时回读一次。固定点方程、Aitken 松弛、失败回滚和外力所有权未改变。

同一阶段计时方法下，显式路径由最初 `4.364 ms/step`、`229 SPS` 降至 `2.869 ms/step`、`349 SPS`，峰值 Warp mempool 保持 `40.37 MiB`；强耦合在冗余回滚优化前已经由 `7.719 ms/step`、`130 SPS` 降至 `6.109 ms/step`、`164 SPS`。最终 `--total-only` 生产基线为：显式 `2.753 ms/step`、`363 SPS`；强耦合平均 2.07 次固定点迭代时为 `4.889 ms/step`、`205 SPS`，峰值分别为 `40.37 MiB` 和 `63.33 MiB`。这些数字只适用于上述固定场景和 Warp 1.12.0，后续以同脚本、同硬件和同参数做回归。

至此 M2 的物理与性能门槛关闭。默认后端仍不切换，因为 HOME-Free VOF、自由表面 FSI 和最终跨场景性能验收分别属于 M3、M4 和 M5，不能用单相刚体结果提前替代。

### M3 当前进度

M3 已建立独立的 `home_lbm/vof/` 子包并接入 HOME 单相推进器。`HomeFreeState` 最初在十矩流体状态之外持久保存 `mass`、`fill_level`、`excess_mass` 和 `flags` 四个 SoA 标量；M4 为严格处理人工质量重分配的 donor 动量，又加入三分量 SoA `excess_momentum`，当前总计为 7 个标量/格。提交态只允许 `LIQUID/INTERFACE/GAS/SOLID`，拓扑转换标志使用独立 scratch buffer，避免把中间态暴露给下一流体步。初始化强制满足 `mass=rho*fill_level`，固体覆盖为零液体质量和零队列，并沿全部 26 条 D3Q27 链路检查液体与气体之间至少存在一层 interface cell。

NumPy 数学规范已经实现 HOME-FREE 式 (9)-(10)：液体邻居权重为 1，界面-界面链路权重为两端 fill level 的平均值，气体和固体不交换液体质量。质量归一化与论文参考实现保持同一语义：液体提交质量固定为 `rho`，界面质量裁剪到 `[0,rho]`，气体为零；裁剪产生的 excess 除以可接收的液体/界面邻居数后存入 `excess_mass`，因此该数组保存的是每个接收邻居的一份，而不是本格总 excess。没有接收邻居时 excess 留在本格，保证 `committed_mass + queued_share*recipient_count` 严格等于推进前质量。

`HomeFreeMassAdvector` 已在 Warp 中实现同一套 pull-link 质量推进：每条链路直接从两端 HOME 十矩按需重构相向分布，非周期域外与固体为零通量，上一时刻的 per-recipient excess 在相邻活动格点汇入；任何液体-气体直连都计数并硬失败。第二个 kernel 将 advected mass 归一化到独立 scratch 数组并计算 26 邻域接收计数，不会在拓扑转换前覆盖提交态。CPU、CUDA、周期和非周期逐格参考对照均通过。

拓扑更新采用确定性的三 pass scratch 状态机，不复刻参考 CUDA 直接跨线程改邻居 flag 的竞态。第一 pass 按 `mass >= (1+epsilon)rho`、`mass <= -epsilon*rho` 及是否缺少气体/液体邻居生成 `INTERFACE_TO_LIQUID/GAS` 候选，其中 `epsilon=1e-4`；第二 pass 为液体扩张创建 `GAS_TO_INTERFACE` 外层并取消相邻的气体扩张；第三 pass 将仍与气体扩张相邻的旧液体改成 interface。resolved flags 之后重新计算最终 26 邻域接收数，才提交质量、fill 和 excess share，因此 excess 分母不会沿用旧拓扑。新建 interface 允许一个时间步的零 fill endpoint，但不创建液体质量；提交前先验证其旧液体/interface donor，再把 donor 平均 `rho,u` 编成平衡 HOME 十矩。相关 VOF 测试覆盖 CPU/CUDA 参考一致性、扩张冲突、防抖、最终拓扑守恒和 fresh-interface 矩初始化。

统一气压边界已经按论文式 (11) 直接接入 HOME pull-stream 热路径。当 interface 单元沿某方向从 gas 邻居拉取缺失分布时，内核计算

```text
f_i^*(x) = f_i^eq(rho_g,u(x)) + f_opp(i)^eq(rho_g,u(x)) - filtered_f_opp(i)(x),
```

其中 `filtered_f_opp(i)` 从当前 interface 单元的十个 HOME 矩即时重构，第一阶段 `rho_g` 为显式配置的统一气压密度。gas 单元不参与流体 collision；liquid-gas 直连由流体内核和质量输运分别计数，并在验证时硬失败。关闭自由表面参数时不执行额外物理分支；全 liquid flags 与原单相路径已有逐位一致性测试。独立 NumPy 规范 `gas_pressure_boundary_populations` 对 27 个方向执行相同公式，CPU 与 RTX 4090 CUDA 均通过逐格单步参考对照。

`HomeFreeDomain` 现在把 HOME 与 sharp-VOF 组织成事务式双缓冲状态，每一步固定执行：旧时刻十矩上的 link-wise 质量输运；使用旧 flags 和式 (11) 的 HOME stream/collision；使用新密度分类拓扑；向后缓冲提交质量、fill、excess 与 resolved flags，并初始化 fresh interface 十矩；最后以新 flags 重新诊断活动流体单元，全部成功后才交换缓冲。任一直接液-气链接、无 donor fresh interface、非有限状态、速度上限或质量归一化错误都会在交换前终止，前一提交态保持不变。8 步移动平面界面测试按 `mass + excess_share*recipient_count` 检查总液体质量，CPU/CUDA 调度结果与分解执行逐数组一致。

压力边界与双缓冲完成时的中间回归结果为 `82 passed, 3 subtests passed`；47 条 warning 均来自既有 WanPhys/Newton 临时桥接接口。后续 PLIC、曲率、局部气压和场景验收均继续沿用这一事务式状态契约；封闭气泡 CCL 与压力演化仍按计划后置，不混入当前开放自由表面里程碑。

PLIC 几何基础已经开始实施。`vof/plic.py` 以单位立方体截面的 inclusion-exclusion 解析体积作为独立 NumPy 规范，并通过单调反演得到给定 fill 与法线的平面偏移；100 组随机方向的体积反演误差小于 `2e-13`，同时验证法线尺度和液/气补集对称性。GPU 热路径不执行迭代求根，而使用 Scardovelli-Zaleski/Kawano 分支化闭式偏移；512 组一般方向、轴向、演化近轴噪声和退化端点样本已在 CPU/CUDA 上与独立规范对照。截面积计算将相对主分量低于 `1e-4` 的 FP32 法线分量按退化维处理；这一级别的面积影响低于 FP32 分辨率，并避免 inclusion-exclusion 在近轴截面除以极小系数积后发生灾难性消减。

`HomeFreeInterfaceGeometry` 使用 3D Parker-Youngs/Sobel 27 点模板构造液体指向气体的单位法线，并为每个 interface cell 保存 PLIC 偏移、单位格截面积和有效标志。闭合球形界面覆盖斜法线方向，最差径向对齐大于 `0.99`，逐格偏移与 NumPy 反演一致。截面积由 clipped-volume inclusion-exclusion 对平面偏移的解析导数给出；低于主法线分量 `1e-4` 的横向分量按退化维处理，其面积影响低于 FP32 分辨率，同时避免除以极小系数积。`fill=0/1` 的拓扑缓冲 interface 允许合法零面积，严格分数填充仍要求面积有限且为正。当前没有湿润/contact-angle 模型，因此 interface 模板只要越过非周期域边界或接触 solid 就硬失败；零梯度法线和非有限偏移同样分开计数并拒绝，绝不以随机法线或复制邻格来补洞。随机一般/近轴/端点截面及闭合球面的最新 PLIC/几何组在 CPU/CUDA 上为 `17 passed`。

曲率采用以中心 PLIC 法线为局部竖轴的二次 Monge patch：邻接 interface 平面变换成 `z=A*x^2+B*y^2+C*x*y+H*x+I*y` 的样本，至少需要五个独立邻居。NumPy 规范用 SVD 检查矩阵秩和条件数；GPU 使用 5×5 正规方程、部分主元消元、逐列主元阈值和条件数上限。参考仓库在样本不足时缩小求解维数、并用固定随机向量构造切向基的做法没有沿用；当前实现使用与法线最不共线的坐标轴生成确定性正交基，欠定或病态模板直接失败。曲率不裁剪到 `[-1,1]`，非有限值单独计数。闭合球面上 GPU/NumPy 逐格最大差约 `1.9e-7`，平均曲率相对解析 `-1/R` 的误差约 1.7%。

`HomeFreeSurfaceTension` 已按式 (12) 生成逐界面压力密度：`rho_g(x)=rho_ambient-6*gamma_lattice*kappa(x)`。物理到晶格的表面张力单位显式定义为 `rho_ref*dx^3/dt^2`，不接受隐含格子单位。局部密度随后直接进入式 (11)；标量统一气压路径仍保留给 `gamma=0`。任何曲率无效、气体密度非有限或非正的单元都会终止，不会裁剪表面张力。空间变化气压的 CPU/CUDA 单步结果已与逐格 NumPy Eq. (11)-(12) 对齐。

当前几何/表面张力验收包括三类场景。零曲率平面在非零 `gamma` 下推进 20 步，曲率与气压偏移保持零，最大速度低于 `2e-6`。周期双界面以 `u=0.02` 平移 10 步，两侧 fill 分别按 `0.75-u*t` 和 `0.25+u*t` 演化，速度误差低于 `2e-7`，拓扑不变且含 queued excess 的总质量误差低于 `5e-6`；CPU/CUDA 均通过。粗网格球滴 `R=5.25`、`gamma=0.003` 推进 3000 步后，测得压差相对当前离散 PLIC 平均曲率的 Laplace 目标误差低于 16%，最大寄生速度低于 `1e-4`，累计 FP32 相对质量漂移低于 `2e-5`。

带重力的自由表面不能沿用密闭域的平均密度归一化。`HomeLbmSolver.initialize_anchored_hydrostatic_lattice` 按连续格子坐标 `x_ref` 和指定压力密度 `rho_ref` 构造

```text
rho(x) = rho_ref * product_axis(r_axis^(x_axis - x_ref_axis)),
r_axis = (1 + 1.5*a_axis) / (1 - 1.5*a_axis),
x_cell = index + 0.5.
```

`HomeFreeDomain.initialize_planar_hydrostatic_lattice` 再根据 PLIC 平面偏移 `alpha`，把大气压位置放在界面外侧半条离散链路：`x_ref = i+0.5+s*(alpha+0.5)`，其中 `s` 指向气体。对半填充水平界面有 `alpha=0`，因此界面中心密度为 `rho_g*r^(-1/2)`，不是 `rho_g`；后者会在式 (11) 下持续丢失界面质量。`initialize_anchored_hydrostatic_lattice` 同时作为非平面初态的显式 API，调用者必须给出有物理意义的压力锚点，代码不会猜测或平均多个自由表面。

静水验收使用 `6x6x12` 网格、`a_z=-2e-4`、顶部气体和单层 `fill=0.5` 界面推进 200 步。CPU/CUDA 都保持原拓扑；最大 fill 漂移低于 `1e-4`，活动层密度保持离散递推，原始动量保持 `j_z/rho=a_z/2`，代表质量相对漂移低于 `5e-6`。初始化还会拒绝非均匀界面平面、横向加速度、越界气体/液体邻层和强迫周期轴。

液柱坍塌验收使用 `24x4x16` 网格、`a_z=-2e-4`、`nu=0.08`，以 8 格宽高的液柱、半填充 PLIC 外层、周期 spanwise 方向及大气压顶部锚点推进 400 步。水平质心由 `3.76` 移至 `5.21`，竖直质心由 `3.76` 降至 `2.49`，前沿由第 8 格推进至第 13 格，界面单元增加至少 20 个且至少 200 个单元改变拓扑；最大速度保持在 `0.01` 到 `0.03`，无液气直连，代表质量误差低于 `5e-6`。该门槛同时要求显著动力演化和稳定守恒，静止不动或数值爆炸都不能通过。

至此，M3 规定的平移界面、静水、Laplace 压差、液柱坍塌/拓扑演化和总体质量门槛均已在 CPU 与 RTX 4090 CUDA 上关闭。最终远端全部 `test_home*.py` 回归为 `111 passed, 3 subtests passed in 420.25s`；47 条 warning 仍全部来自既有 WanPhys/Newton 临时桥接接口。开放自由表面 M3 的完成不代表封闭气泡模型已经存在，也不代表自由表面 FSI 已完成；下一步进入 M4 的相感知 cut-link、入水/出水 fresh/dead cell 和自由表面刚体场景。

### M4 当前进度

M4 建立独立的 `HomeFreeRigidInterface`，不直接套用单相 `HomeLbmRigidCoupling`。后者只拥有一个 `HomeLbmState`，其 moving-solid remapper 只知道十矩，不知道 `mass/fill/excess/flags`；在自由表面中调用它会让 solid mask 与 VOF 标志脱节。接口构造仍要求自由表面尚未初始化：先栅格化统一刚体 SDF 并安装 cut links，再由 `HomeFreeDomain` 根据该 solid mask 初始化，从源头保证 `SOLID` flag 与 SDF 符号一致。

部分浸没物体还需要明确的压力基准。HOME cut-link 的静止平衡冲量包含绝对压力项 `-2*w_i*rho*c_i`；完整闭合湿表面会自行抵消，但部分湿润表面没有干侧液体链路，若直接归并就会把大气压误报成刚体载荷。当前接口只对 `LIQUID/INTERFACE` 所有的湿链路增加

```text
Delta J_atm = 2*w_i*rho_ambient*c_i,
```

再归并 gauge-pressure wrench；`GAS` 所有的干链路必须保持零冲量，solid/未知 flag 所有的 cut link 是硬错误。该修正应用于全部湿链路：它显式移除环境压力基准，避免完整湿表面依赖数千项 FP32 原子冲量偶然抵消；它不使用液体静水参考剖面，因此不会消除浮力。链路相态必须读取本次 stream/collision 的 source flags，而不是拓扑提交后的 destination flags；移动场景曾由错误时间层产生 16 个“干链路非零冲量”假告警，现已在流体步前保存设备端 load-ownership flags。

冻结球横跨水平自由表面的 CPU/CUDA 测试同时观察到湿链路和干链路；干链路冲量为零，均匀环境压力修正后的净力与力矩绝对值低于 `2e-5`。完全浸没球的静水浮力进一步与 `rho*V*|g|` 比较：半径 4 格时误差约 `13.4%`，半径 6 格时降至 `7.6%`；横向力和力矩均限制为目标浮力的 `1e-4` 以下。粗细分辨率门槛在 CPU 与 RTX 4090 CUDA 上均通过。

专用 `HomeFreeRigidTransitionRemapper` 已实现 NumPy 相分类规范和 Warp 运行时。fresh 单元只允许从旧、新 SDF 中都保持流体的 26 邻域取 donor：全液 donor 得到 `LIQUID`，全气得到 `GAS`，出现 interface donor 或液气混合必须得到 `INTERFACE`；无 donor、旧 mask/flag 不一致或重映射后液气直连均硬失败。运行时先在 scratch 中完成旧代表质量和活动十矩归约、fresh phase/moment 重构、十矩全局残差修正、按新密度提交 mass/fill，再把质量差编码成新界面的 per-recipient excess share。排队量超过界面密度总量的 50%、无界面可承接非零差值、非正密度或代表质量误差超过 `2e-5` 时不提交。SDF 符号没有变化时走逐位 no-op 快路径，不重排 excess。

移动切格和普通自由表面拓扑现在共用明确的代表态契约，而不是未加权的活动格点一阶矩：`P_l = sum[mass*j/rho + recipient_count*excess_momentum]`。`excess_mass` 与 `excess_momentum` 都是 per-recipient 份额；拓扑裁剪把质量差编码为 `e_m=(m_adv-m_commit)/N`，并以裁剪格点速度编码 `e_p=e_m*j/rho`。下一步 advection 分别收集相邻 donor 的 `e_m/e_p`；HOME stream/collision 得到接收格点基线速度后，只施加 `e_p,in-e_m,in*u_base` 对应的 Galilean 修正，因此人工队列动量替换了原本隐含的本地速度动量，而 LBM 链路输运不被重复计算。修正同时更新一阶矩和六个二阶矩；无队列格点逐位不变，非零修正却缺少有限 transported mass 时硬失败。

moving-solid remap 也保存、回滚和重建同一队列。remap 完成质量提交后先按新队列计算物理动量残差，再向所有活动 HOME 态施加同一个 Galilean 速度增量；一阶矩按 `j'=j+rho*du` 更新，六个二阶矩按原始二阶矩平移公式更新，同时队列执行 `e_p'=e_p+e_m*du`，所以代表态整体严格平移。非均匀三维速度场测试同时用 NumPy 契约和 GPU diagnostics 检查旧、最终代表动量，归一误差低于 `2e-6`；unchanged-mask no-op 与失败回滚均包含三分量队列的逐位检查。

水平和竖直合成刚体掩码已逐格对照 NumPy oracle。水平移动产生 6 fresh/6 dead 时代表质量误差约 `4.7e-7`；向上离开液体产生 4 fresh/4 dead 时误差约 `7.4e-8`，活动十矩密度和误差约 `1.7e-6`。失败路径保持 moments/mass/fill/flags 原样。真实球形 SDF 以 0.10 格中点位移横跨自由表面时同时发生 fresh/dead，联合 remap 加一个 HOME-Free 步的质量误差低于 `1e-5`，最大速度约 `0.093`，湿/干 cut links 均存在且没有液气直连。

`HomeFreeRigidCoupling` 已接入显式中点调度：验证表面位移、预测中点姿态、联合 remap、推进 HOME-Free、归并 gauge wrench、以固定总载荷执行刚体子步，最后恢复调用方拥有的 `body_f/particle_f`。完全浸没移动球在自由表面压力信号到达前，以 fill 加权的运动方向液体动量加刚体动量做账，首步相对误差低于 `3e-4`；场景确实发生 solid transition、球体减速、液体质量误差低于 `2e-5`。RTX 4090 上最新静态接口与显式耦合集为 `9 passed`，transition/reference/interface/domain 联合集此前为 `20 passed`；warning 仍来自既有 WanPhys/Newton 临时桥接接口。

自由表面强耦合现已复用单相路径中验收过的固定点思想，但快照和回滚覆盖完整 `HomeFreeDomainState`。算法先保存 HOME 十矩、SDF/body id、7 个 VOF 标量、刚体完整状态和旧 diagnostics，再用初始速度预测并冻结二阶中点 SDF、fresh/dead 拓扑及 prepared source。每次固定点迭代从同一 prepared source 和 rigid input 恢复，只更新 cut-link 壁速，执行一个完整 HOME-Free 事务、gauge wrench 和刚体子步，以最大刚体表面晶格速度为残差并使用有上限 Aitken 松弛。成功只提交最后候选；不收敛或 contact provider 等任意中途异常都会恢复 front state、刚体状态、remapper geometry history、cut links、solver 绑定和调用方外力。CPU/CUDA 收敛、`1e-20` 强制失败逐数组回滚、外部异常回滚及原显式路径合计使全部 `test_home_free_*.py` 达到 `80 passed`。

自由表面 FSI 场景矩阵现已补齐。半浸没 `R=6`、刚体密度 `0.5` 的球在离散静水中，初始浮力/重量误差约 `17.9%`；8 个强耦合步的竖向漂移低于 `2e-3` 格、横向漂移低于 `2e-5` 格，CPU/CUDA 均通过。`R=3,4,5,6,8` 的静态浮力误差依次约为 `37.9%, 30.0%, 24.4%, 17.9%, 15.3%`，证明主要误差随界面分辨率收敛，不是固定漏项。球体入水和出水各推进 20 步，均出现正确的干湿链路转换、fresh/dead 固体单元、同号减速和低于 `2e-5` 的质量误差。密度 `0.5/1.5` 的两个全浸没球在同一静水场中分别上升和下沉，方向、位移大小及速度符号均设置硬门槛。

液柱撞球使用 `28x14x14` 全周期域、初速 `0.06`、`R=3` 等密度球和 40 个强耦合步。球体从全干变为最多 164 条湿链路，最终正向速度约 `0.03969`；液体正向代表动量下降约 `3.85`，同时产生 32 fresh/8 dead 单元，质量误差低于 `2e-5`。该场景把动量账拆成四个可读分量：气压边界的实验室系冲量、湿润切链转为表压时加入的环境压力修正、Galilean-invariant 切链冲量相对实验室系的移动壁参考系修正，以及刚体切格重映射前后的代表动量。气压冲量已在随机非平衡态逐链对照式 (11)，CPU/CUDA 误差低于 `2e-6`；刚体动量变化与积分切链载荷误差约 `4e-7`，重映射单步绝对误差低于约 `1.5e-3`，相对初始总动量低于 `2e-5`。

总账明确暴露一个尚未写回生产状态的项：40 步后，`Delta(P_liquid+P_rigid)` 扣除气压、环境表压和移动壁参考系冲量后的最大残差，CPU/CUDA 分别约为初始总动量的 `0.8764%/0.8752%`。它不能归因于 cut-link 或 moving-solid remap。官方 HOME-FSLBM 代码只以 `mass/phi/massex` 跟踪自由表面质量与拓扑，HOME 的 `rho/u` 和矩不按 fill 加权，也没有 `excess_momentum`；因此本项目采用的 `P_l=sum(mass*j/rho)` 是面向严格 FSI 账本的扩展状态，不能把官方算法本身描述成已经守恒该量。生产路径仍把未关闭残差硬限制在 `0.9%`，禁止用事后全局速度平移掩盖局部误差。

独立 NumPy 规范和 Warp CPU/CUDA 诊断现已给出精确的链路质量-动量事务。对活动链路 `source -> destination`，设反向离散速度为 `opp(q)`、液体链路权重为 `alpha`，则

```text
Delta m_q = alpha * (f_q(source) - f_opp(q)(destination)),
Delta p_q = alpha * c_q * (f_q(source) + f_opp(q)(destination)).
```

第一式与 HOME-FSLBM 标量质量交换一致；第二式由两个相向粒子布居各自携带的离散动量直接得到，同时保留对流、压力和非平衡应力，而不是只拟合宏观速度。interface-interface 内部链路使用对称 fill 权重，因而链路两端的内部质量和动量逐对抵消；全液周期域中它逐格退化为 HOME stream 后的 `rho/j`。接收端 excess mass/momentum 继续只提交 `p_queue-m_queue*u_destination` 的 Galilean 修正，避免重复计算 LBM 链路输运。

先前计划的 `VOF donor candidate - HOME nonlinear convection` 已由实测分解否决：40 步撞击中旧账残差约为 `-0.898`，donor 候选累计约 `+0.071`，HOME 界面非线性反事实约 `+2.040`，两者相减约 `-1.969`，方向和数量级都不匹配。零体力撞击中的精确原始目标为 `weighted_internal_link_momentum + incoming_queue_momentum + (home_destination_momentum - unweighted_home_internal_link_momentum)`；对应累计缺陷加入旧账后仅余约 `8.91e-4`，比旧残差改善约三个数量级。CPU/CUDA 撞击测试逐步要求归一最大残差低于 `2e-5`，且不修改当前生产状态。

把该目标局部写回时还必须去掉绝对环境压力。对于每条 active-active 链路，参考项为 `alpha*2*w_q*rho_atm*c_q`；gas、solid 和非周期域边界链路拥有完整的 `2*w_q*rho_atm*c_q`。NumPy oracle 与 Warp CPU/CUDA 已逐数组一致。含体力时，最终实验目标写成

```text
P_target = P_internal_weighted + P_queue
         + (j_HOME_destination - P_HOME_internal)
         - P_reference_pressure
         + (M_advected - rho_destination) * a.
```

最后一项把 HOME 对满计算格施加的 `rho*a` 改为输运液体质量的 `M*a`；其余动态压力、黏性、Laplace 压差和移动壁冲量仍由 HOME 离散增量所有。联合 topology commit 已作为显式可选事务实现：它按最终 flags 同时规范化 `M/P`，生成 per-recipient excess mass/momentum，并用 Galilean 平移写回 HOME 一阶和六个二阶矩。随机非均匀目标速度在 CPU/CUDA 上逐格满足 `m_commit*u+N*e_p=P_target`，中心二阶矩保持到 `3e-7`；零质量新生界面保留 donor HOME 状态，近零质量携带有限动量和无 recipient 的非零 excess 都硬失败。

该联合事务经长期场景审计后明确不接入默认 `HomeFreeDomain.step()`。静止球试接时，若不减环境参考压力，fill 约 `0.00368` 的界面格会被绝对压力推到约 `96.7` 的非物理晶格速度；逐链 gauge 修正后，`sigma_lattice=0.001` 的部分液体目标仍给出约 `0.317`。加入精确 PLIC 面牵引后，一步目标降为约 `0.02583`，但零表面张力球滴的多步局部写回仍从 `2.1e-6` 指数增长到第 10 步约 `0.119`；`sigma_lattice=0.003` 在第 3 步越过速度上限。这证明问题不只是毛细面积，而是把扩展代表动量逐格除以 `mass=fill*rho` 后写回 HOME 速度，改变了官方算法的状态语义。

官方 HOME-FSLBM 源码在同一个 kernel 中独立推进 `mass/phi/massex`，随后仍由完整 27 个流入分布计算 `rho/u/S`；HOME moments 从不乘 fill，局部惯性始终是满格 `rho`。因此 `0.317` 是扩展部分液体账本的量，不是官方流体速度 CFL。原 Eq. (11)-(12) 在满格语义下的实际一步毛细速度约 `1.3e-3`，并已通过 3000 步 Laplace 门槛。生产路径继续只提交 queue momentum 的 Galilean 替换，精确 `P_target` 和联合 topology commit 保留为 FSI 收支诊断，不再被描述为待直接默认开启的流体更新。

PLIC 面牵引与 gas-link 阶梯冲量之差仍有独立 NumPy/Warp CPU/CUDA 规范；其 3N 缓冲、方向表和额外 kernel 仅在 `enable_plic_capillary_momentum()` 后惰性分配。把该差值作为满格 HOME 点冲量可稳定推进 3000 步，但会产生约 `1.25e-3` 的寄生流，差于官方路径的 `<1e-4`，说明它尚未与体压力梯度形成 balanced-force 离散，不能进入生产域。满格应用内核及中心应力保持测试保留，供后续 balanced-force 方案对照，不在默认 step 中调用。

#### Balanced-force 约束与下一实现增量

经典 VOF balanced-force 的关键不是单独提高界面力的几何精度，而是让表面张力和压力使用兼容的离散梯度。Popinet 的离散平衡写成 `-grad(p) + sigma*kappa*grad(C)=0`；当两个梯度算子相同且离散曲率为常数时，`p=sigma*kappa*C+constant` 才是机器精度的静态解。该结论不能原样移植为 HOME 的点力，因为 HOME 的压力梯度隐含在 PDF/HOME moments 的碰撞、迁移和自由表面反弹中，而不是独立的有限体积压力投影。[Popinet 2009](https://doi.org/10.1016/j.jcp.2009.04.042) 同时指出 CSF balanced-force 本身不保证全局动量守恒，移动液滴仍可能暴露静态液滴看不出的误差；因此本项目不会只凭静止球测试开启它。

当前生产 kernel 只在上游格为 gas 时重构缺失 PDF，不依据法向覆盖已有 PDF，因而对应 Schwarzmeier 与 Rüde 所比较的 `OM`（only missing）变体。该研究证明四类常用 FSLBM 重构在一般运动界面上都不能严格平衡液相和气相压力，但 `OM` 在五类动态场景中最准确，并明确建议不要丢弃已有流场信息。[Schwarzmeier and Rüde 2023](https://arxiv.org/abs/2207.13962) 这支持保留当前 missing-link 所有权，也否决了为了凑局部法向而覆盖额外分布的退路。

在静止、零非平衡应力、恒曲率界面上，Eq. (12) 给出的 `rho_g` 与液体内部相同常密度构成 HOME 的离散固定点：gas link 的反弹和平衡态 active-active 迁移使用同一个 D3Q27 权重，不需要再叠加 PLIC 面力。曲率随格点振荡时，各界面格的 `rho_g` 不再代表一个常压固定点，随后产生的流动不能靠把 staircase 冲量替换成独立 PLIC 面冲量消除。Bogner 等给出的 FSL 方案通过界面位置的链路线性插值以及压力/剪切闭合把自由表面边界提升为二阶，但它依赖边界距离、TRT 参数和应力外推，必须作为独立边界方案验证，不能只摘取一个插值系数。[Bogner, Ammer and Rüde 2015](https://doi.org/10.1016/j.jcp.2015.04.055)

下一代码增量按以下顺序关闭，每一步都先有 NumPy oracle，再有 Warp CPU/CUDA 对照：

1. **已完成。** `only_missing_boundary_residual` 把每个 OM pull-link 的 `Delta(rho,j)` 精确分成 `pressure`、`equilibrium_transport` 和 `non_equilibrium`，并保留逐格 gas-link 数量。匀速、恒定 `rho_g` 的任意合法 slab 拓扑达到约机器精度固定点；纯密度跳跃只进入 pressure 项，人工 HOME 应力只进入 non-equilibrium 项，随机非均匀状态的三项和逐格等于真实 stream/collision 的守恒矩增量。Warp 诊断核与 NumPy 在 CPU/CUDA 上对齐，且 non-periodic、solid 和液气直连所有权不会被静默跳过。压力、表面张力和 3000 步 Laplace 联合集为 `24 passed in 31.33s`。
2. **几何所有权已完成，压力闭合进行中。** `plic_pull_link_intersection` 已按生产 pull 方向定义 `x=-delta_q*c_q` 和 `delta_q=alpha/(-n dot c_q)`，返回参数分数、格内交点和欧氏距离。背向液体、切向平行、交点位于节点后方或落在本格 PLIC 片段之外分别拒绝，绝不回退为 `delta_q=1/2`；平面缩放、方向缩放和节点穿越均有解析测试。Warp 批量核对 512 条随机合法链与 NumPy 对齐，并稳定区分 not-facing、behind、outside 三类状态。

   对 `17^3`、`R=5.25` 球滴的旧 OM gas-link 所有权审计表明，4490 条缺失链中只有 488 条能与目标 interface cell 的 clipped PLIC 片段相交；3906 条交点位于目标节点后方。原因不是几何求交精度，而是当前 HOME-Free 保留官方“所有 interface cell 都是满格流体节点”的语义，fill 小于约 `0.5` 的 interface cell 中心实际位于 PLIC 气侧，因而不满足 Bogner FSL 对 fluid-side boundary node 的前提。FSL 几何层现在保持 VOF flags 不变，另行定义 hydrodynamic-active 节点：liquid 以及中心位于 PLIC 液侧的 interface cell。由 active interface 指向 gas 的链使用目标单元平面；由 liquid 指向中心在气侧的 interface cell 的链使用源单元平面。这样 `17^3` 球滴的 3242 条边界链全部有唯一所有者和 `0<=delta_q<=1` 的交点。

   源单元 clipped PLIC 片段在粗曲面上不能覆盖所有完整中心链。`11^3`、`R=3.25` 球滴中部分链使用相邻 PLIC 平面的显式链路延拓；其 owner-cell 越界距离均约 `0.01206` 格，且原始 `delta_q` 约 `0.48794`，仍位于完整 FSL 链段。实现将其标为独立 `EXTRAPOLATED` 状态并保存延拓距离，绝不并入 `VALID`；测试要求最大距离低于 `0.02` 格。NumPy 与两阶段 Warp 实现分别完成 active-node 分类和逐链 owner/status/delta/extrapolation 构造，闭合球面在 CPU/CUDA 上逐元素一致。

   Bogner Eq. (11) 还要求边界节点液侧的第三个 PDF 支撑点 `x_b-c_out`。将这一条件加入覆盖审计后，`R=3.25/5.25/7.25` 三档球面的 `NO_SUPPORT/total` 分别为 `168/1226`、`192/3242`、`192/6098`，对应可闭合比例随分辨率提高，但任何档位都不是纯 FSL 全覆盖；可用链中的 PLIC 平面延拓数分别为 `48/0/48`。默认 reference 对 `NO_SUPPORT` 硬失败。另有显式 `only_missing` 研究策略，只对这些链使用 Eq. (13) 并逐链计数，不把它们标成二阶 FSL，也不在生产路径自动启用。`11^3` 静止 Laplace 球的一步最大速度由现有 OM 的 `1.6063e-3` 降到 hybrid 的 `5.2354e-4`，当前回归要求 hybrid 小于 OM 的 40%。固定 PLIC/曲率、每步重新线性外推界面速度的 50 步 reference 审计中，峰值由 OM 的 `1.7826e-3` 降到 hybrid 的 `9.3208e-4`，未出现累积爆炸；常规回归使用 20 步并要求 hybrid 低于 OM 的 70%。这些结果仍不能替代移动界面和守恒验收。

   Bogner Table 1 的三点 multi-reflection 系数已按当前正松弛率约定实现：`a0=1/2-delta`、`abar0=1/2`、`a1=delta-1`、`C=omega_plus(3/2-delta)`、`D=-3(1/omega_plus-1/2)w_q`。NumPy 单链 oracle 显式接收边界速度和对称剪切率张量，Warp 批量核在 512 条随机非平衡链上与其 CPU/CUDA 对齐；均匀 HOME 平衡态在全部 26 个方向及 `delta=0..1` 上保持到双精度机器精度。网格 reference 已验证 destination/source 双所有权平面的均匀平移固定点和缺支撑硬错误。逐链速度采用 `u_b=(1+delta)u_local-delta*u_support`，在两类 owner 的任意仿射速度场上都精确命中各自 PLIC 交点；`NO_SUPPORT` 只有显式 hybrid 策略才使用局部 OM 速度。

   实验 Warp 网格路径现在分成速度外推、owner-aware FSL/hybrid pull stream、十矩提取和无体力 HOME collision 三阶段。`11^3` 闭合球面的随机密度、速度及非平衡应力状态在 CPU/CUDA 上逐格对齐 NumPy，168 条 fallback 精确计数且全部非法计数为零。完整剪切模式逐链外推 bulk 对称剪切率，清零法切分量、保留切切分量，并显式设置 `n dot S_b dot n`；NumPy/Warp CPU/CUDA 均通过仿射张量解析对照。简化 `D=0` 和完整 `D c c:S_b` 因而先具备了可执行规范。后续生产接线新增 `HomeFreeFslBulkStrainBuilder`：它直接按论文的动量剪切率定义，在每个 hydrodynamic-active 单元的确定性 `5x5x5` 邻域对三分量动量做加权仿射拟合，线性场在截断边界仍精确；少于四个 donor、秩亏、条件数超限和非法矩都硬失败，常量场也必须先通过满秩检查才返回逐位零梯度。法向目标使用 FSL 链上的液相密度外推 `rho_l,b=(1+delta)rho_local-delta*rho_support`，再按 `S_nn=c_s^2(rho_l,b-rho_g,b)/(2 nu)` 施加包含气压/曲率跳跃的完整法向应力条件。

   动态 active-node 转换现有独立 NumPy 契约和 Warp 实现。界面移动后，fresh 节点只从旧、新 active mask 中都持续活动的 26 邻域取供体，以平均 `rho/u` 初始化平衡 HOME 十矩；dead 节点的存储保持不变。无持续供体时在修改状态前硬失败，previous mask 也不前移；这里不施加全局满格十矩修正，因为真实液体质量仍由 VOF 事务拥有。周期 `8x3x2` 平面界面跨过 hydrodynamic-active 阈值时准确产生 6 fresh/6 dead，随后完成动态 PLIC 覆盖重建、FSL 速度外推、pull stream 和 HOME collision；均匀平移平衡态保持，geometry/stream/collision/fallback 计数均为零，CPU 与 RTX 4090 CUDA 同时通过。

   对官方 Home-FSLBM 参考代码的调度复核发现，`stream_collide_bvh` 对所有 `TYPE_F/TYPE_I` 节点重构和推进 HOME moments，并直接用相邻 `fhn-fon` 及 interface-interface fill 权重更新 `mass`；它没有 Bogner FSL 的“PLIC 中心在液侧才是 hydrodynamic-active”划分。现有 `HomeFreeMassAdvector` 同样读取所有 interface moments。若把 Bogner active mask 直接接到该质量核，中心位于气侧的 interface moments 会停止推进，非均匀流的质量通量随后读取陈旧状态；均匀平移无法暴露这一错误。因此本项目不把官方 link-wise mass advection 与 Bogner active 所有权强行拼接，也不把参考仓库描述成已经实现 Bogner 二阶闭合。

   FSL 研究路径新增几何 PLIC 轴向体积通量。每个面只由迎风 donor 的 PLIC 平面计算一次 swept-slab 液体体积，相邻两格以相反符号共享该通量；周期端面的首尾 Courant 必须逐值一致，非周期封闭端面必须为零，`|C|>1`、非法平面和超出 fill 物理界限均硬失败。NumPy 规范用仿射缩放后的精确 clipped-cube 体积计算斜平面通量；Warp 使用相同的一至三维 inclusion-exclusion，并以独立 face/cell kernels 保持无原子写冲突。解析轴对齐、64 个随机 PLIC 互补薄片、三轴正反流向以及 20 步周期平移均通过；后者跨越 active 阈值、准确产生 6 fresh/6 dead，CPU/CUDA 与 NumPy 对齐，几何液体体积保持到 FP32 允许误差。几何通量专项为 `9 passed in 5.80s`。

   `HomeFreeFslResearchStepper` 首次把源 HOME 态工作副本、动态 PLIC/active/coverage、fresh/dead remap、几何 VOF、FSL pull stream、无体力 HOME collision 和几何类别提交组织成一个失败可回滚的隔离事务。周期 `12x3x2` 平面以 `u=0.02` 推进 20 步，CPU 与 RTX 4090 CUDA 都保持均匀平衡态与液体体积，全部 invalid/fallback 计数为零。active 转换发生前故意破坏周期 Courant 端面，事务抛错后源 moments/fill/flags 逐位不变，修正后重试仍报告 6 fresh/6 dead，证明 remapper history 没有提前提交。后续审计又发现碰撞曾直接写调用方 destination，虽然源态不变，后置健康检查失败仍可能留下半提交输出；现已增加独立 `candidate_fluid/candidate_free_surface`，拓扑与全部诊断成功后才复制到 destination，失败测试逐数组锁定 source、destination、remapper history 和 split parity。

   生产候选调度不再要求调用方手工传入面速度。`HomeFreeFslFaceCourantBuilder` 在 active-active 面取两侧 HOME 法向速度的中点值；只要相邻单元仍含正液体体积而面两侧并非都 active，就在确定性的 `6x5x5` 窄带内对 active 供体做以面中心为原点的加权三维仿射拟合。少于四个供体、秩亏、条件数超限、非有限 moments 或超 CFL 面全部分别计数并失败；相邻均为零 fill 的纯气面严格为零。NumPy 规范用加权设计矩阵 SVD 检查秩和条件数，Warp 用 4x4 正规方程、部分主元消元和平方条件数门槛；严格常量供体走逐位常量保持分支，不引入消元噪声。三轴仿射速度场的外推面值达到解析精度，Warp CPU/CUDA 与 NumPy 的逐面 Courant 和 donor count 对齐。

   builder 已在 active remap 后、几何 VOF 前接入 stepper，因而正常路径积分实际 HOME/FSL 速度，不再强制理想 `u=0.02`。20 步全程 transported-face Courant 相对理想值的最大绝对偏差低于 `4e-7`，累计界面位置偏差低于 `1e-6`，共享通量体积守恒及 6 fresh/6 dead 不变。face-Courant、几何通量、stepper、active transition 与 FSL pressure 联合集当时为 `44 passed in 44.37s`。stepper 后续已移除固定 VOF 类别和 Bogner `D=0` 两项限制；每步 FSL stream 前由 `HomeFreeFslStressClosure` 从时间层起点 active HOME 状态构造 bulk、normal 和 boundary strain，并以 `has_boundary_strain=1` 执行完整闭合。它目前仍明确限制为调用方给定的 gas-density 场、零体力，且不替换 `HomeFreeDomain.step()`。

   多轴 split 保持一个流体步只执行一次 FSL stream/collision，同时依次执行多个几何 VOF sweep。时间层起点的 FSL active/link coverage、各轴面 Courant 和 Weymouth-Yue 中心颜色 `c=Theta(C^n-1/2)` 都冻结给整个分裂步；中间态只提交 fill/类别并重建下一轴需要的 PLIC 法线和截距，不重新采样速度，也不计算没有时间层物理意义的曲率。这满足 Weymouth-Yue 对“全部方向使用同一个离散速度场和同一个中心颜色”的要求。成功步之间按配置顺序及其逆序交替，失败不推进 split parity。周期 `16x16x4` 斜平面双界面使用三分量非零均匀速度连续推进四步，实际顺序为 `XYZ/ZYX/XYZ/ZYX`；每个轴都使用共享面通量，界面发生可测移动，总体积守恒，CPU/CUDA 的最终 fill 与 HOME moments 对齐。周期 `12x12x2` 斜带的 X 子步真实生成 24 个 gas-to-interface 并把 24 个 liquid 降为 interface；修复薄片体积后 Y 子步不再产生旧实现记录的 24 个伪端点吸附，最终体积仍保持 144，CPU/CUDA 对齐。

   几何路径没有复用旧 `HomeFreeTopologyUpdater`：后者只从旧 interface 的质量溢出生成 `INTERFACE_TO_*`，再由邻接扩张建立 `GAS_TO_INTERFACE/LIQUID_TO_INTERFACE`，而几何 VOF 会让旧 gas 单元直接收到 swept volume。新增 `HomeFreeGeometricTopologyResolver` 直接按 transported fill 分类，以 `4e-7` 容差只吸附数值端点；它禁止单步 gas/liquid 直接互换，fresh gas-to-interface 只从源、目标中都持续为液体或界面的 26 邻域平均 `rho/u` 并初始化平衡 HOME 十矩，随后验证所有目标 active 矩有限、密度为正且无直接液气邻接。分类、donor、临时矩和全局验证全部成功后才原子式写回 fluid/free-surface；失败测试证明 fluid 与 destination 均保持逐位不变。碰撞输出先继承已初始化的新界面矩，再只覆盖时间层起点 active 单元；最终以碰撞后密度重建 `mass=rho*fill`。动态拓扑阶段聚焦联合集为 `61 passed in 45.34s`；加入完整 stress 的 NumPy/Warp、CPU/CUDA、单轴/多轴和 stepper 回归后为 `70 passed in 45.21s`。20 步均匀平移中完整法向应力对 FP32 密度噪声产生受控反馈：密度/速度误差分别不超过 `2.4e-7/2.8e-7`，总体积变化不超过 1.5 个 FP32 求和量化单位，全部 stress invalid/NO_SUPPORT 计数为零。

   Weymouth-Yue 动态球审计发现 transformed swept slab 的法线在小 Courant 下会缩到约 `2e-4`，而截距可能已经位于该薄立方体支撑范围之外。原 inclusion-exclusion 多项式仍继续求和，发生灾难性消减并把本应为满薄片的通量算成零；失败样本 `C=0.99987644, u=1.9622757e-4` 因而违反论文附录下界 `F>=max(0,u-(1-C))=7.2667e-5` 并造成过填充。CPU 规范和 Warp kernel 现在都先以 `alpha <= -0.5 sum|m|`/`alpha >= 0.5 sum|m|` 返回严格空/满体积，再进入 inclusion-exclusion；专项测试同时检查论文通量上下界，原失败样本恢复为完整面通量。该修复消除了动态球的越界和内部伪界面，而不是依靠 fill 裁剪或增大端点容差。

   曲率所有权也已与 FSL 链路分离。中间方向分裂态只构造 PLIC 平面；完整时间层新增 `curvature_required`，仅当 interface 中心位于气侧或 26 邻域含 gas、即该平面可能拥有自由表面链路时才计算 Laplace 曲率。液侧且无气体邻居的几何薄片仍参与 VOF 输运，但其曲率和 gas-density 修正保持未启用，并由 `curvature_required/skipped` 诊断显式报告；真正需要曲率的欠定、秩亏或病态拟合仍硬失败。

   动态 Laplace 验收改用球与单元的几何体积分数，而不是 `clip(0.5+(R-r)/2)` 的两格平滑场；边界单元用 `32^3` 确定性子体素积分，并保持严格开区间以免采样把真实微小交叠误判为直接液气链路。`gamma=0.001`、20 步、`R=3.25/5.25/7.25` 的 CPU 审计中，相对体积误差依次为 `4.37e-4/9.23e-5/5.31e-5`，液体质量误差为 `2.81e-4/8.46e-5/3.44e-5`，峰值寄生速度为 `5.55e-4/1.14e-4/4.13e-5`，均随分辨率整体下降。正式 `R=5.25` 测试固定体积、质量、速度、曲率、192 条 `only_missing` 回退和 CPU/CUDA 逐场一致性门槛；本阶段 PLIC/几何/曲率/表面张力/Courant/拓扑/stress/stepper 联合集为 `87 passed in 32.74s`。

   多轴几何输运现已抽成共享 `HomeFreeGeometricTransport`。它在时间层起点一次构造 Weymouth-Yue 中心颜色并冻结各轴 Courant，方向 sweep 之间只重建 PLIC 几何和拓扑，返回逐轴 Courant/拓扑诊断；OM 与 FSL 不再各自维护一份容易漂移的 VOF 调度。新增 `HomeFreeOmResearchStepper` 保留官方所有 `LIQUID/INTERFACE` 节点推进及 only-missing 压力边界，只替换为同一几何 VOF 事务，并与 FSL 一样使用候选流体/自由表面态和可回滚 active history。两个 stepper 及共享事务均从 `home_lbm.vof` 和 `home_lbm` 公共入口导出。

   同初态后端比较固定 `17^3`、`R=5.25`、`gamma=0.001`、`nu=0.5`、相同 `XYZ/ZYX` 交替顺序和 20 步。RTX 4090 CUDA 上，OM 的相对体积误差、`rho*C` 质量误差和峰值寄生速度分别为 `5.9209e-5/6.3186e-5/1.0087e-4`；FSL/hybrid 分别为 `9.2338e-5/8.4587e-5/1.1423e-4`。三项均是 OM 更低，CPU 得到同一排序，且两后端 CPU/CUDA 最终 fill、HOME moments 和指标均满足逐场一致性门槛。该结果与固定界面一步/20 步审计中 hybrid 较低的寄生流并不矛盾：动态几何球仍有 192 条 `NO_SUPPORT` 链退回 OM，且 FSL 的 active 切换和完整应力闭合引入了额外离散误差。正式测试把这个场景限定为“当前证据支持 OM”，禁止据此宣称 OM 在所有移动界面上占优。

   移动 Laplace 验收使用同一 `17^3`、`R=5.25` 几何球，初始均匀速度为 `(0.008,-0.005,0.003)`，`gamma=0.001`，周期边界下推进 20 步；目标同时约束体积、`rho*C` 质量、质心位移、相对目标球的 fill L1 误差、速度保持、拓扑闭合修正和 CPU/CUDA 一致性。未投影的弱可压缩面 Courant 在 x86 CPU 第 19 步使斜法向近满单元达到 `C=1.0000021458`；初始面散度并不大，但 Weymouth-Yue 的机器精度全局守恒前提是离散不可压缩速度，因此不能以增大 fill 容差解决。共享 transport 新增显式 `project_courant=True` 研究选项：它在源 `LIQUID/INTERFACE` 单元上解离散 Hodge 投影，气体为零压 Dirichlet，使用与面梯度严格配对的七点压力算子。NumPy dense oracle 和 Warp matrix-free Jacobi-PCG 独立对照；投影仅允许完整 XYZ 输运，不能与调用方给定的单轴 Courant 混用。

   PCG 最初用 `atomic_add` 归约五个标量内积，RTX 4090 上相同 CUDA 输入重复 20 步后 OM/FSL fill 分别漂移到 `8.2e-6/2.2e-5`，FSL 的拓扑闭合计数也会变化。现实现改为固定连续块、固定树序的 FP64 内积归约，再显式窄化 `alpha/beta` 到 FP32；压力、Courant 和 HOME 状态仍为 FP32。修复后同设备两次 OM/FSL 的 fill、moments 和闭合计数逐位相同。跨 CPU/CUDA 的 PLIC 三角函数和 FP32 代码生成仍不要求逐位相同，正式契约改为：各设备独立通过物理门槛、CUDA 重复运行逐位确定，并同时约束跨设备 fill/moments 的 L-infinity 与平均 L1 误差，而不是用单一逐点 `allclose` 混淆确定性和跨架构舍入。

   第 19 步逐轴审计还发现独立于散度的 FP32 薄片积分问题。失败单元 `C=0.99999940395`、斜法向 `(-0.935174,0.016533,0.353803)`、`u_z=0.0030099968`；旧 Warp inclusion-exclusion 返回 `F=0.0030073416`，低于任何单位体积交集必须满足的下界 `max(0,|u|-(1-C))=0.0030094008`。NumPy 双精度值为 `0.0030095326`，证明 PLIC 平面本身正确，误差来自近满体积的大数消减。CPU 规范和 Warp 现在都用半空间互补对称只计算不超过 `1/2` 的体积，再把最终 swept volume 限定到严格几何交集区间 `[max(0,|u|-(1-C)), min(C,|u|)]`。这不是 fill 后裁剪：同一个受限共享面通量仍以相反符号进入相邻单元，保持守恒；每轴诊断记录触发面数和最大修正，超过 `4e-6` 的修正被视为不一致 PLIC 几何并在写状态前硬失败。最终移动验收只触发 `5..8` 个面，最大修正为 `9.36e-9..1.42e-8`。

   最终 x86 CPU / RTX 4090 CUDA 指标如下。OM 的相对体积误差为 `8.54e-8 / 8.74e-8`，质量误差 `1.013e-5 / 1.006e-5`，质心误差 `1.287e-3 / 1.282e-3`，形状 L1 `1.701e-3 / 1.700e-3`，峰值速度误差 `1.828e-4 / 1.831e-4`。FSL/hybrid 分别为体积 `8.15e-8 / 8.69e-8`、质量 `9.592e-6 / 9.550e-6`、质心 `1.281e-3 / 1.280e-3`、形状 L1 `1.730e-3 / 1.730e-3`、峰值速度误差 `1.420e-4 / 1.419e-4`。投影前最大散度约 `7.23e-5..9.73e-5`，投影后统一不超过 `1.40e-9`，最大 32 次 PCG 迭代。该场景体现明确 trade-off：FSL 的质量、质心和速度保持更好，OM 的形状误差更低；没有证据允许把任一后端宣称为全面优胜者。

   4090 上同一小场景的 10 步、5 次计时中位数显示，显式投影使 OM 从 `1.0224 s` 增至 `1.0854 s`，FSL 从 `1.0312 s` 增至 `1.0949 s`，开销均约 `6.2%`。此数字包含完整 stepper 而非孤立 PCG kernel，且 `17^3` 小网格受 launch latency 影响，不能外推为大网格吞吐率。投影仍保持显式关闭的研究选项；移动自由表面验收明确启用它，默认后端切换则必须等待更大分辨率、旋转/撞击和动态 FSI 场景共同证明收益。

   `HomeFreeGeometricDomain` 现已把共享 HOME solver、OM 边界、投影 Courant 和几何 PLIC 组织为可供刚体接口调用的双缓冲域；旧 `HomeFreeDomain` 的 link-wise mass advection 不变。动态刚体接入时发现旧 moving-solid remapper 的守恒态不是单纯的 `sum(mass)`，而是 `sum(mass+N*excess_mass)`。一个 `20x12x18` 的浸没球横向冲量步中，remap 的旧/新代表质量为 `3204.0/3203.9999962`，自身相对误差仅 `1.19e-9`，但 240 个界面源留下 `0.2890673` 的代表 excess。几何 topology 原先直接重建 `mass=rho*C` 并清零队列，因而连同弱可压密度变化形成 `1.0759e-4` 的总质量缺口；该数据流已被测试锁定，不能以放宽最终质量门槛处理。

   新增 `HomeFreeGeometricQueueMaterializer` 作为 moving-solid remap 与 PLIC sweep 之间的独立桥。旧队列的一份 share 会被全部活动邻居计入代表质量，但已满液体格没有正质量容量，直接汇入会在上述场景生成 `C=1.000637`。桥因此先把每个源的代表总量 `N e_m, N e_p` 按符号感知的局部容量重新分配：正质量使用 `w_r=max(rho_r-m_r,0)/sum_s max(rho_s-m_s,0)`，负质量使用 `w_r=max(m_r,0)/sum_s max(m_s,0)`；质量和三分量动量使用同一权重。汇入质量转成 `Delta C=Delta m/rho`，随后由几何 topology 原子提交类别和质量，再以 `Delta p-Delta m*u_HOME` 的 Galilean 修正同时更新一阶矩和六个二阶矩。零质量携带有限动量、源无活动接收者、局部容量为零、物化 fill 越界或前后质量/动量账本误差超过 `4e-6` 均在修改调用方提交态前硬失败；正负队列、动量和容量不足均有独立 CPU/CUDA 测试。

   浸没球验收现真实产生 dead cells 和非零 queue，桥报告 240 个源/240 个物化格，桥接相对质量误差 `1.19e-9`、动量误差 0；完整步继续要求液体质量误差 `<2e-5`、流体与刚体总动量误差 `<3e-4`、dry link 与 missing cut-link 均为零、投影后最大散度 `<=2e-8`。强耦合固定点还新增显式 `HomeFreeOmTransactionHistory`：每次候选重试同时恢复 `XYZ/ZYX` split parity、FSL active remapper 的 `previous_active` 和上一份几何诊断。测试强制多次迭代时，同一物理步的所有候选都使用 `XYZ`，成功后 parity 只推进一次；`1e-20` 强制不收敛后 parity、remapper 存在性、诊断、HOME/VOF/刚体数组、cut links 和外力所有权全部恢复。

   RTX 4090 的无队列小网格计时以 `16x16x4` 完整几何步为对象，并用保留相同状态复制、只省略 queue kernel 的路径对照；两次 5 组中位数测得桥增量 `0.201..0.217 ms`，约 `2.74..2.92%`。真实 240 源刚体转换的第一版中，HOME/VOF copy 为 `0.0525 ms`，copy 加物化为 `2.1361 ms`，桥增量 `2.0836 ms`。审计确认第一版会在每个目标格为相邻队列源重新计算 26 邻域容量，最坏局部工作为 `26x26`；现已增加 source-preparation kernel，每个 queue source 只计算一次 `N`、代表质量/动量和容量和，destination gather 只遍历 26 邻域。优化后 copy 为 `0.0521 ms`，copy 加物化为 `2.0363 ms`，桥增量 `1.9842 ms`，约快 `5.0%`，质量 `1.19e-9` 和动量零误差不变。理论邻域复杂度已降为 O(26)，但实测表明 host 同步、几何 topology 与前后账本验证已占主要固定成本；M5 不能继续靠删验证换速度，应研究设备端失败标志延迟回读、账本固定树归约与 topology 调度合并。
3. 对非均匀平移平面、旋转 Couette/膜流、静止及平移 Laplace 球做阶数和寄生流对比。FSL 只有在目标动态场景中同时改善误差、守恒和稳定性，且 `NO_SUPPORT` 所有权有明确闭合时，才允许成为可选 Warp backend；在此之前 OM 是共享几何 VOF 上的基线边界，不等于整个新 HOME-Free 已替换旧生产路径。
4. 流体更新只拥有 free-surface gas-pressure boundary；刚体只拥有 wet cut-link momentum exchange。三相接触或切格转换通过独立 ledger 验证二者互斥，PLIC 面牵引仅用于解析对照，禁止重复施力。

RTX 4090 的独立 `bench_home_free_fsi.py` 使用 `54x54x86`（250,776 格）、`R=4` 球、50 个计时步和 10 个预热步；测量期间确实发生 52 fresh/52 dead、最多 308 条湿链路和 1640 条干链路。重映射原先在提交后把 17 个标量/格的 HOME-Free 状态全量回读到 NumPy 验证，造成明显的切格尖峰。现实现以 Warp 设备核逐格检查有限性、相标志、`mass=rho*fill`、队列质量/动量契约和 26 邻域锐界面约束，只回读错误计数；测试会故意制造“零队列质量携带有限动量”的非法状态，确认该优化没有放宽验收。

优化后的显式路径整体为 `7.862 ms/step`、`127.20 SPS`，37 个无切格步平均 `6.556 ms`，13 个切格步平均 `11.578 ms`，Warp mempool 峰值 `102.55 MiB`；旧切格均值为 `46.28 ms`。强耦合平均 4.02 次固定点迭代，最大残差 `1.92e-6`；整体为 `19.784 ms/step`、`50.55 SPS`，无切格/切格步分别为 `18.625/23.083 ms`，峰值 `138.90 MiB`；旧切格均值为 `43.42 ms`。精确动量的 12N scratch、PLIC 毛细 3N scratch 和额外累计均为显式惰性启用，默认路径不分配；此前同参数复测为 `122.17/124.61/128.02 SPS`，本轮为 `125.01 SPS`（`7.999 ms/step`），稳态/切格步分别为 `6.699/11.702 ms`，峰值继续保持 `102.55 MiB`。这些数字来自同一 RTX 4090、Warp 1.12.0 和同一基准参数，动态尖峰已关闭，仍不能替代后续更大分辨率和真实场景的 M5 验收。

当前远端 RTX 4090 的最新自由表面完整回归为 `test_home_free_*.py: 234 passed in 134.38s`；1719 条 warning 均来自既有 WanPhys/Newton 临时桥接接口，没有新增失败或数值告警。几何队列、几何域、Courant 投影和 OM stepper 的 CPU/CUDA 聚焦集合为 `22 passed in 96.25s`，其中包含首次 CUDA kernel 编译；加严后的几何域单集复跑为 `8 passed in 6.47s`。此前未受本轮 VOF/投影改动影响的 HOME-LBM 基线仍为 `test_home_lbm_*.py: 59 passed, 3 subtests passed in 382.10s`。

M4 的旧 HOME-Free 场景验收、队列动量契约、切格性能优化、精确链路规范和联合 topology 诊断事务已经关闭；共享几何域现已完成第一轮动态刚体、强耦合接入和 queue source 预计算，但整体替换工作尚未完成。论文及官方 HOME-FSLBM 代码的 `massex` 只有标量；本项目新增的 `excess_momentum` 与几何 queue materializer 都是为严格 FSI 账本作出的守恒扩展，不能写成论文原算法已有的能力。下一增量应扩展到多步 entry/exit、漂浮和柱体冲击的几何域场景，同时用大分辨率动态切格基准定位 host 同步、topology 和账本归约成本；随后研究与 HOME 压力梯度配对的 balanced-force 表面张力，及如何把精确部分液体账本用于 FSI 载荷所有权而不改变官方满格速度场。通过这些场景门槛后再进入图捕获、量化评估和默认后端切换前的 M5 综合验收。

### M5 当前结果

M5 在 RTX 5090、Warp 1.12.0、`54x54x86`（250,776 格）、动态球体切格场景上建立了统一基线。未优化几何 OM 且关闭 Courant 投影时为 `17.593 ms/step`、`14.254 MLUPS`、峰值 `320.99 MiB`。动量守恒投影原先每步把 10 个 FP64 ledger 标量回读 NumPy；现由设备核计算速度 Lagrange 修正，并与非法计数合并同步。PLIC 每轴体积、质量、动量的三次诊断回读也合并为一次 sweep 提交检查。两项完成后同口径为 `16.867 ms/step`、`14.868 MLUPS`，约快 `4.1%`，峰值不变；柱体冲击、移动界面和 OM 回归保持原门槛。

大网格暴露出小场景没有覆盖的投影扩展问题：默认 160 次 Jacobi-PCG 在相对残差 `1.65e-4` 时失败，严格收敛实际约需 400 次。实现保留固定树 FP64 归约和 FP32 压力/Courant，加入已收敛压力暖启动、强耦合事务快照以及 8 次迭代 CUDA Graph 块；每块后仍回读真实残差，不改变停止条件。无 graph 的严格投影基线为 `102.54 ms/step`、`2.446 MLUPS`；最终 20 步默认配置平均 391.6、最多 416 次迭代，最大投影散度 `2.79e-9`，达到 `56.51 ms/step`、`4.438 MLUPS`、峰值 `342.19 MiB`，相对基线约快 `44.9%`。默认最大迭代数改为 `max(160,8*max(resolution))`，只扩大失败上限，不放宽残差或散度。

sm_120 PTX 经 CUDA 12.8 `ptxas` 资源审计：生产 `stream_collide` 前向核使用 108 registers，PLIC 面通量 42，几何构造 78，曲率拟合 112，topology 最高 72，PCG 算子 48；上述前向核均为零 spill stores/loads。生产 HOME 是 fused stream/collision、按需重构 27 个 PDF、持久化 10 个矩。独立 `bench_home_lbm_layouts.py` 另实现数学等价的纯周期 split 对照：显式重构 27 PDF、periodic pull、提取十矩、无力碰撞；一步最大矩差 `1.82e-12`。RTX 5090 的 2000 步长批次中，fused 为 `0.1341 ms`、1870 MLUPS，split 为 `0.1020 ms`、2460 MLUPS，split 快约 24%。代价是每格额外 216 B 双 PDF scratch 和 40 B 中间矩，而且对照没有生产路径的气压、cut-link、力、非周期壁和联合诊断。该结果否定“fused 必然最快”，但不能用功能较少的 split 核替换已验收 FSI。shared-memory halo 方案没有 spill 驱动证据，也尚无包含全部边界所有权的等价实现，因此不进入默认；后续若做 split/hybrid，必须先达到功能逐链等价再比较。十矩主状态的理论最低读写量为 `80 B/LUP`；完整 FSI 的 launch 计数约为无投影 105.3 个普通 kernel/step，严格投影 345.3 个普通 kernel 加 50 个 graph launch/step。

量化采用独立审计而不改写 FP32 主路径。`rho` 以参考密度为 offset，`rho*u` 和六个 Hermite 二阶矩以零为中心；范围由允许密度变化、最大格子速度和 D3Q27 二阶矩可实现界给出。设备审计逐分量统计非有限值和 int16 饱和，默认饱和即硬失败。20 步动态 FSI 十分量均为零饱和；但候选 int16 的单次最大舍入误差为：密度 `3.81e-6`、动量 `2.86e-6`、对角二阶矩 `1.27e-5`、非对角二阶矩 `1.91e-5`。后两组已高于若干现有物理误差尺度，因此 M5 的结论是“范围可编码但累计物理误差尚未获准”，FP32 继续作为唯一生产状态，不以显存收益交换未证明的稳定性。

### M6 当前结果

`wanphys.fluid.HomeFreeDomain` 已切换为几何 OM 公共默认；`HomeFreeBackend` 和 `create_home_free_domain()` 提供显式 `geometric/legacy` 选择，未知值硬失败。旧 link-wise HOME-Free 以 `HomeFreeLegacyDomain` 保留一个兼容发布周期，原测试已改为显式 legacy，不借默认别名继续运行。最小公共示例 `fluid_grid_home_free_column` 在 RTX 5090 上通过，2 步质量相对误差 `6.99e-9`。

旧 `wanphys._src.fluid.fluid_grid.lbm` 的 D3Q19 Shan-Chen/TRT 实现和示例保持原样。它与 D3Q27 HOME-Free 的状态及多相模型不同，不能做静默类名替换或参数补偿迁移。完整迁移契约、初始化要求、回退方式和基准命令见 `home-lbm-migration.md`。M6 是否最终关闭以默认/legacy API、全部 HOME-Free/HOME-LBM 回归及最终场景验收共同通过为准。

最终 RTX 5090 验收已关闭上述门槛。`test_home_free_*.py test_home_lbm_*.py` 合计 `330 passed`、`124 subtests passed in 456.24s`；3566 条 warning 全部来自既有 WanPhys/Newton 临时桥接弃用接口，没有新增数值或实现告警。最终 `54x54x86`、10 步预热、50 步动态 FSI 严格投影并启用量化审计：`53.951 ms/step`、`4.648 MLUPS`、p95 `77.125 ms`，稳态/切格步分别为 `50.047/65.063 ms`；实际产生 52 fresh/52 dead、最多 368 条湿链和 1620 条干链，平均/最大投影迭代为 `387.36/416`，最大投影散度 `3.26e-9`，峰值 `343.14 MiB`，十矩全部零饱和。公共默认柱体示例连续 100 步的质量相对误差为 `2.54e-8`，液体体积 `288.999999`，最大 fill 为 1。至此 M5 性能与量化评估、M6 默认替换与兼容验收完成；int16 存储量化明确保持未启用状态，这是误差门槛结论，不是遗留的静默降级。
