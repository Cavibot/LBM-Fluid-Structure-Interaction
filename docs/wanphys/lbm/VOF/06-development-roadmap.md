# VOF 综合开发路线图

## 1. 文档目的

本文把开放大气自由表面 VOF 拆成严格依赖、逐级验收的开发阶段。它整合：

- 论文依据与公式审计；
- 当前 LBM 的 FullF/HOME、streaming、collision 和 boundary 架构；
- VOF 状态、时序与控制流设计；
- 小网格单元测试、数值基准和运行时不变量。

路线图的核心原则是：

> 每个阶段只引入一种新的错误来源，并在进入下一阶段前独立证明它正确。

本文是开发顺序和验收门禁的权威入口。公式细节仍以
[01-paper-audit.md](01-paper-audit.md) 和
[02-framework-and-formulas.md](02-framework-and-formulas.md) 为准；数据与时序语义以
[03-data-model-and-uml.md](03-data-model-and-uml.md) 和
[04-sequence-diagrams.md](04-sequence-diagrams.md) 为准。

## 2. 范围与“完整”定义

### 2.1 本路线图包含

完成本文全部阶段后，系统应支持：

- `LIQUID / INTERFACE / GAS` 三类占据状态；
- 守恒的自由表面质量输运；
- 开放大气压力下的 gas-to-interface population reconstruction；
- 界面跨网格移动时的类型转换与守恒质量重分配；
- 新界面格点合法的 kinetic state 初始化；
- 法向、PLIC、曲率和表面张力；
- FullF 正式路径，以及经过差分验收后的 HOME 路径；
- 静态墙面、周期边界和项目已有的普通 domain boundary；
- CPU/CUDA 正确性检查、数值基准与长期守恒监控。

完成这些能力后，可以称为：

> 完整的开放大气自由表面 VOF。

### 2.2 明确不包含

本文不开发：

- 封闭气泡识别、bubble ID、独立气泡压力；
- 移动固体、cut-cell 双向耦合、fresh/dead；
- 泡沫、溶解气体、D3Q7、disjoining pressure；
- 将 D3Q19 实现宣传为论文 D3Q27 高阶 HOME-FREE 的精确复现。

已有静态固壁和普通 domain boundary 可以继续使用，但不扩大为 VOF 固体耦合。

### 2.3 第一条兼容性边界

正式 VOF 第一版只打开：

```text
D3Q19
+ FullF
+ SRT 或 TRT
+ gravity/user force
+ fixed atmospheric pressure
+ static/periodic domain boundary
```

第一版必须关闭：

```text
Shan-Chen interaction
surface tension
HOME authoritative path
moving/cut-cell solid
bubble/foam
```

这些限制用于隔离错误，不表示最终架构只能支持上述组合。

## 3. 六个子系统与实施阶段

六个核心子系统为：

```text
状态与配置
  ↓
质量平流
  ↓
自由表面 LBM 边界
  ↓
类型转换与质量重分配
  ↓
新界面 kinetic 初始化
  ↓
法向、PLIC、曲率与表面张力
```

开发阶段在六个子系统之前增加基线冻结，在之后增加综合验收：

| 阶段 | 名称 | 核心问题 | 依赖 |
|---|---|---|---|
| P0 | 基线与证据冻结 | 当前系统究竟已经证明了什么 | 无 |
| P1 | 正式状态、配置与初始化 | 系统记住什么、字段属于哪个时间层 | P0 |
| P2 | 固定拓扑质量平流 | 液体质量去了哪里 | P1 |
| P3 | 零表面张力自由面边界 | 气体侧缺失 population 如何补全 | P2 |
| P4 | 类型转换与守恒重分配 | 界面如何跨过网格 | P3 |
| P5 | 新界面 kinetic 初始化 | 新流体格点怎样获得合法状态 | P4 |
| P6 | PLIC、曲率与表面张力 | 界面形状及毛细压力是什么 | P5 |
| P7 | 综合验收、HOME 与设备扩展 | 各模块组合后是否仍然正确 | P6 |

任何阶段只有在其退出门禁全部满足后，才能进入下一阶段的正式实现。

## 4. 全阶段统一规则

### 4.1 单一权威来源

- `state.debug_mock_sc_to_vof.phi/cell_type/normal` 只允许由 SC density 派生，
  只用于调试观察。
- authoritative VOF 的 `state.vof.phi/mass/cell_type` 只能由 VOF stepper 更新。
- `mass` 与 `phi` 不允许由两个 kernel 独立演化。
- 调试 normal 只有在 `normal_valid_epoch == epoch` 时可读；正式几何使用独立
  authoritative epoch。
- gas 单元中的 kinetic state 在正式 VOF 模式下必须明确视为无效。

### 4.2 时间语义

每个时间步使用：

```text
state_in  = n 时刻 post-collision kinetic + VOF 状态
scratch   = streamed populations、临时质量和转换请求
state_out = n+1 时刻最终 kinetic + VOF 状态
```

必须保持：

```text
kinetic fields 与 cell_type 属于同一时间层
phi 与 mass 属于同一时间层
有效 geometry 与 phi/cell_type 属于同一 epoch
所有 INTERFACE 都有合法 kinetic state
```

### 4.3 证据等级

每项能力必须同时具有四层证据：

1. **公式证据**：`PAPER / DERIVED / REFERRED / UNSPECIFIED` 标记明确。
2. **手算证据**：单格或小网格输入有人工可核对的期望值。
3. **独立 oracle**：NumPy/CPU 参考实现不调用生产 kernel。
4. **集成证据**：长期守恒、网格收敛和 CPU/CUDA 对照。

仅有截图、动画或“没有崩溃”不构成验收证据。

### 4.4 每阶段提交规则

一个提交只完成一种能力，例如：

```text
加入 mass 状态，但不加入平流
实现固定类型质量交换，但不重标记
实现 gamma=0 的自由面补全，但不加入曲率
产生转换候选，但不做质量重分配
```

每个提交说明必须回答：

1. 新增了什么物理或数据能力；
2. 读写了哪些时间层；
3. 可能破坏什么不变量；
4. 哪个测试会在实现错误时失败。

## 5. P0：基线与证据冻结

状态：`CPU_ACCEPTED`。冻结环境、SHA、命令、结果与限制见
[07-p0-baseline.md](07-p0-baseline.md)；CUDA 尚未验收。

### 5.1 阶段概要

固定当前 SC-to-VOF debug observation 和原 LBM 的行为，避免正式 VOF 接入后无法
判断回归来自新物理还是旧路径变化。

当前代码已经具有：

- 与 `state.vof` 隔离、由 Shan-Chen density 派生的调试 `phi/cell_type`；
- Parker–Youngs normal 的 observation 实现；
- 界面点和法向可视化；
- FullF/HOME 基础 LBM 路径。

这些能力不是正式守恒 VOF。

### 5.2 开发内容

- 保存 `test_lbm_debug_mock_sc_to_vof` 作为调试观察回归基线。
- 保存 FullF/HOME、collision、streaming、boundary 核心回归命令。
- 建立 VOF capability/status 表。
- 记录 CPU 和 CUDA 环境、精度与已知限制。
- 以 `interface_model` 作为界面物理唯一配置权威，gravity 保持正交。
- 给 `interface_model="vof"` 加入 fail-fast，防止未完成能力被误启用。

### 5.3 验收目标

- observation 开关不改变原 LBM 的 populations 和宏观量。
- 当前 FullF/HOME 回归测试通过。
- 当前 CPU/CUDA 能力明确记录；没有 CUDA 时不得声称 CUDA 验收完成。
- 文档明确标注“已有法向”只是 observation 能力。
- 未完成的 authoritative VOF 配置无法静默运行。

### 5.4 退出门禁

```text
[x] observation-only 与 authoritative 的命名和行为边界明确
[x] 旧 LBM 回归命令固定
[x] VOF 状态表建立
[x] 未实现正式模式 fail-fast
```

## 6. P1：正式状态、配置与初始化

### 6.1 阶段概要

建立后续所有 kernel 共同遵守的数据契约。该阶段不改变流动物理。

### 6.2 开发内容

P1 正式持久状态严格只有：

```text
mass
phi
cell_type
```

`normal/PLIC/curvature/epoch` 在 P6 首次消费时加入；`atmosphere_pressure` 在 P3、
`epsilon_phi` 在 P4、`surface_tension` 在 P6 首次消费时加入。P1 不保存无消费者的
配置或缓存。

还需要：

- 明确 `mass` 和 `phi` 的权威关系及时间语义；
- 明确 gas kinetic 无效、interface/liquid kinetic 有效；
- 使用现有 `state_in/state_out` 完成 VOF 双缓冲；
- 提供从预计算 `phi0` 初始化的正式 API；
- 先初始化 LBM kinetic/density，再初始化 `mass/phi/type`；
- 使用 candidate state，成功后才提交两个 domain buffer；
- authoritative `state.vof` 的分配不依赖 `debug_vof_observation`；
- clone、clear、copy 覆盖三个正式 VOF 数组；
- P1 的 `step()` 在任何状态写入前 fail-fast。

### 6.3 必须先做出的设计决策

P1 已冻结：

```text
谁是守恒权威：mass
初始化 density：LBM 实际写出的统一 state.density
初始化关系：mass0 = density0 * phi_input
phi0：立即由 mass0 / density0 反算
cell_type：float32 phi 的精确 0/1 分类
gas kinetic：无效，正式 VOF 物理路径禁止读取
```

在 P2 编码前仍必须冻结：

```text
mass scheme 选择严格 paper Eq.9-10 还是经典 FSLBM/Lehmann 变体
interface-gas 格链上的 logical population/flux 从哪里取得
质量子步使用的 rho 时间层
```

该 P2 决策必须证明与论文 Eq. (9) 等价或明确标为项目适配。无论采用哪一种，都禁止
从 gas 单元的持久 kinetic storage 读取“碰巧存在”的值。

### 6.4 验收目标

- 两个 LBM state buffer 中均存在独立 VOF 状态。
- clone/copy 后逐字段相等，且不存在共享可变数组。
- clear 后三个正式数组回到全 GAS 零质量状态。
- 首次或重复初始化失败时不提交半初始化 candidate。
- 初始化后：
  - 所有 density 有限且大于零；
  - `phi` 有限；
  - `0 <= phi <= 1`；
  - `mass` 与选定关系一致；
  - `LIQUID/INTERFACE/GAS` 分类一致；
  - 所有 active cell kinetic 合法。
- observation 和 authoritative 不可能同时写同一份 `phi/type`。

### 6.5 退出门禁

```text
[x] mass/phi 初始化权威与时间语义已冻结
[x] 正式 VOF 初始化可重复且事务式提交
[x] 所有生命周期测试通过
[x] VOF/SC 非法组合被拒绝
[x] 尚未启用质量平流和自由面补全
```

## 7. P2：固定拓扑的守恒质量平流

### 7.1 阶段概要

实现论文 Eq. (9)-(10) 的格链质量交换，但固定 `cell_type`，不执行 clamp、类型转换
或质量重分配。该阶段只回答“质量去了哪里”。

### 7.2 开发内容

- 为 FullF 和 HOME 定义统一的 logical population provider 契约。
- 第一实现只启用 FullF provider。
- 按 P1 已冻结的 mass scheme 处理 interface-gas 格链。
- 若该 scheme 需要重构 gas-link population，先实现无副作用的单格链纯 helper；
  P3 再把同一 helper 接入完整 streaming，而不是复制第二套公式。
- 每条格链从 `state_in(n)` 读取：
  - `rho/phi/cell_type`；
  - `f_i` 和对向 `f_bar`。
- 计算格链质量通量与 Eq. (10) 权重。
- 写入独立的 `mass_tmp/phi_tmp` 或 `state_out` 临时结果。
- 固定类型期间允许临时 `phi` 越过 `[0,1]`，但不得静默 clamp。
- 支持周期边界；普通开口边界质量通量必须单独记账。
- 增加每步：
  - 总质量；
  - domain boundary 净质量通量；
  - `phi` 最小/最大值；
  - 非有限值计数。

### 7.3 验收目标

手算测试：

- 静止均匀状态的所有格链净交换为零。
- 单一 population 扰动的增减方向和大小正确。
- `i ↔ opposite[i]` 的方向关系正确。
- LIQUID、INTERFACE、GAS 的 Eq. (10) 分支逐一命中。

守恒测试：

```text
M(n+1) - M(n) = domain_boundary_mass_flux
```

- 封闭/周期域右侧为零。
- 误差满足预先冻结的 float32/float64 容差。
- 局部格链交换在发送端和接收端符号相反。

oracle 测试：

- `3^3` 或 `5^3` 网格逐格匹配独立 NumPy 实现。
- 生产 kernel 与 oracle 不共享生产 helper。
- 把 gas kinetic storage 填入不同垃圾值，质量结果保持不变。

### 7.4 禁止提前加入

- 类型转换；
- clamp；
- 质量重分配；
- 曲率；
- 表面张力；
- normal 定向重分配。

### 7.5 退出门禁

```text
[ ] 固定类型单步手算全部通过
[ ] 周期/封闭域总质量守恒
[ ] boundary flux 记账闭合
[ ] NumPy oracle 逐格匹配
[ ] mass scheme 来源和 gas-link 语义已被测试锁定
[ ] 没有读取 gas kinetic storage
[ ] 任意 phi 越界会被观测而非静默修复
```

## 8. P3：零表面张力自由面 LBM 边界

### 8.1 阶段概要

实现 gas-to-interface 的 Eq. (11) reconstruction，使 gas 不参与 LBM 演化时，
interface 仍能获得完整 populations。第一版固定大气压力并令
`surface_tension = 0`、`curvature = 0`。

### 8.2 开发内容

推荐第一版采用低侵入接入：

```text
普通 streaming
→ 根据 state_in.cell_type 找 gas-to-interface 链
→ 覆盖对应缺失 population
→ 普通 open/domain boundary
→ collector
```

实现必须：

- 使用 `state_in(n)` 的 type 和 kinetic 状态；
- FullF 从持久 populations 读取对向 population；
- HOME 后续通过 Eq. (16)-(17) provider 重构；
- 使用固定 `p_atmos` 得到 `rho_g`；
- 检查 `rho_g > 0`；
- 确保每个 `_f_star[i,x]` 只有一个最终权威写入者；
- 区分 gas surface、solid wall 和 domain boundary；
- 禁止从 gas kinetic state 读取有效物理量。

### 8.3 验收目标

公式测试：

- 单个 interface cell、单个 gas link 的 Eq. (11) 手算一致。
- 19 个方向及其 opposite 映射全部覆盖。
- equilibrium 静态极限下 reconstruction 保持压力边界。
- 改写 gas kinetic 垃圾值不会改变输出。

数值测试：

- `gamma=0` 的平面自由面长期保持静止。
- 无持续密度漂移、质量流失或法向速度增长。
- 平移/镜像测试只产生对应的平移/镜像结果。
- 不重复施加半力速度修正。

### 8.4 禁止提前加入

- 曲率压力；
- 表面张力；
- bubble pressure；
- type transition；
- fresh/dead 初始化。

### 8.5 退出门禁

```text
[ ] 所有 gas-to-interface 链补全且方向正确
[ ] gas kinetic 垃圾值不影响结果
[ ] gamma=0 平面自由面稳定
[ ] rho_g 非法值有明确失败策略
[ ] 质量平流守恒没有被 surface completion 破坏
```

## 9. P4：类型转换与守恒质量重分配

### 9.1 阶段概要

让界面集合跨网格移动。该阶段是核心 VOF 中最容易产生长期漏液和 GPU 并发错误的
部分，必须拆成候选、拓扑、重分配三个子阶段。

### 9.2 P4A：转换候选

实现：

```text
INTERFACE with phi >= 1 + epsilon → TO_LIQUID candidate
INTERFACE with phi <= 0 - epsilon → TO_GAS candidate
```

验收目标：

- 阈值上下边界行为明确；
- 只产生请求，不立即原地修改 type；
- 输入 type/phi 只读，候选输出独立；
- 并行运行不依赖线程执行顺序。

### 9.3 P4B：拓扑修复与冲突解决

实现：

- 防止最终 LIQUID 与 GAS 直接邻接；
- 必要时把邻格提升为 INTERFACE；
- 定义同时 TO_LIQUID/TO_GAS 请求的优先级；
- 使用多 pass 或等价的确定性方案；
- 记录该方案是 `REFERRED`、`DERIVED` 还是项目决策。

验收目标：

- 最终非法 L-G 邻接计数为零；
- 旋转、镜像、线程调度不改变离散结果；
- 多个相邻候选同时转换时结果确定；
- pass 数达到上限时 fail-fast，而非提交半完成拓扑。

### 9.4 P4C：clamp 与质量重分配

实现：

- 最终 LIQUID `phi=1`，最终 GAS `phi=0`；
- 被 clamp 的正/负 excess 都守恒转移；
- 接收者集合只包含合法 INTERFACE；
- 没有接收者时使用显式失败或已审计 fallback；
- 不原地 scatter 到邻居造成数据竞争；
- 建议先计算 redistribution delta，再统一 gather/apply。

验收目标：

```text
mass_before_transition
= mass_after_transition_and_redistribution
+ explicit_boundary_flux
```

- 单个过满/欠满格点可手算。
- 多发送者、多接收者测试守恒。
- 接收者为零的退化情况有确定行为。
- CPU 重复运行结果一致。
- CUDA 无数据竞争，且总质量与 CPU 在容差内一致。

### 9.5 退出门禁

```text
[ ] 候选生成只读旧状态
[ ] 最终 L-G 非法邻接为零
[ ] clamp 不丢失 excess mass
[ ] 重分配无原地并发写冲突
[ ] 类型转换前后质量闭合
[ ] 新 interface 已被准确列出供 P5 初始化
```

## 10. P5：新界面 kinetic 初始化

### 10.1 阶段概要

为 `GAS → INTERFACE` 的格点构造下一步可用的合法 kinetic state。P4 只确定身份，
P5 才使新界面成为真正可演化的流体格点。

### 10.2 开发内容

第一版策略应简单且可审计：

```text
读取 state_out 中有效 LIQUID/INTERFACE 邻居
→ 加权得到 rho、u
→ FullF 使用 equilibrium populations 初始化
→ 写入新 interface kinetic
```

同时：

- `INTERFACE → GAS` 显式使 kinetic 无效；
- `INTERFACE → LIQUID` 保留或校正 collision 输出；
- unchanged active cells 直接保留 collision 输出；
- 没有有效邻居时 fail-fast 或使用已审计 fallback；
- 记录初始化对质量、动量的影响；
- 后续 HOME 路径单独定义 `rho/u/S` 初始化。

### 10.3 验收目标

- 单个 GAS→INTERFACE 后 density 有限且为正。
- population 非负性/可容许性符合项目策略。
- 初始化速度不产生孤立尖峰。
- 下一步 streaming/collision 不出现 NaN/Inf。
- 对称邻域得到对称初始化结果。
- 初始化前后质量守恒；若动量不严格守恒，误差被记录并受验收阈值约束。
- 连续多次创建/删除界面格点不积累异常 kinetic 值。

### 10.4 退出门禁

```text
[ ] 每个新 interface 都有合法 kinetic state
[ ] gas kinetic 不会被后续路径误读
[ ] 初始化无 NaN、负密度和不可接受速度尖峰
[ ] 初始化质量影响闭合
[ ] 固定 gamma=0 的移动界面可长期运行
```

## 11. P6：法向、PLIC、曲率与表面张力

### 11.1 阶段概要

把 observation normal 升级为 authoritative geometry，并完成
`phi → normal → PLIC → curvature → surface pressure`。在此之前，正式 VOF
始终使用 `gamma=0`。

### 11.2 P6A：法向

- 从最终 `phi/type(n+1)` 计算 normal。
- 固定 normal 的液体到气体方向约定。
- 处理 periodic/domain 边缘 stencil。
- 只有 interface normal 有效。

验收：

- 轴对齐平面方向正确；
- 倾斜平面方向误差随分辨率降低；
- 均匀场 normal 为零；
- 平移、镜像和旋转结果一致；
- `geometry_epoch == vof_epoch`。

### 11.3 P6B：PLIC

- 根据 normal 与 phi 求 interface plane offset。
- 保证重构平面切割出的液体体积与 phi 一致。
- 定义退化 normal、接近 0/1 的 phi 和数值容差策略。

验收：

- 轴对齐平面有解析结果；
- 随机 normal/phi 的重构体积误差满足冻结阈值；
- normal 翻转时 offset 与液体侧约定一致；
- PLIC 不读取过期 geometry。

### 11.4 P6C：曲率

- 使用已选择并记录来源的 PLIC/height-function/其他方法计算曲率。
- 明确 `kappa` 符号与 Eq. (12) 的关系。
- 对 stencil 不完整或几何退化定义显式行为。

验收：

- 平面曲率趋近零；
- 球面曲率符号正确；
- 球面/圆柱的曲率误差随网格细化下降；
- 平移和旋转不会改变收敛结论；
- 退化格点不会产生 NaN/Inf。

### 11.5 P6D：表面张力

- 在 Eq. (12) 中启用 `rho_g=(p_atmos-2 gamma kappa)/cs^2`。
- 检查 `rho_g > 0` 并实施已冻结的 limiter/fail-fast 策略。
- 不把 surface tension 作为第二套体力重复施加。

验收：

- `gamma=0` 精确退化回 P5 基线。
- Laplace pressure 的符号正确。
- 多分辨率液滴压差向解析关系收敛。
- 静态液滴伪流被量化且随分辨率/算法改进呈合理趋势。
- 表面张力开启后总质量仍满足守恒门禁。

### 11.6 退出门禁

```text
[ ] authoritative normal/PLIC/curvature epoch 一致
[ ] 平面和球面几何测试通过
[ ] 曲率符号与 Eq.12 锁定
[ ] Laplace pressure 通过网格收敛验收
[ ] gamma=0 与旧基线一致
```

## 12. P7：综合验收、HOME 与设备扩展

### 12.1 阶段概要

P0-P6 证明各子系统独立成立。P7 验证组合运行、编码扩展、设备一致性和长期稳定性。

### 12.2 综合数值场景

按顺序执行，不允许只运行 dam-break：

1. 静止平面自由面；
2. 周期域匀速平移界面；
3. 重力下静水自由面；
4. 球形液滴与 Laplace pressure；
5. 小振幅液面振荡；
6. dam-break；
7. 长时间质量守恒运行。

每个场景至少记录：

```text
initial_mass
current_mass
boundary_mass_flux
relative_mass_error
phi_min / phi_max
invalid_LG_adjacency_count
new_interface_count
non_finite_count
negative_density/population_count
max_velocity
interface_cell_count
```

### 12.3 HOME 扩展

只有 FullF 核心验收通过后才能启用 HOME：

- P2 logical population provider 加入 Eq. (16)-(17)；
- P3 对向 population 使用 HOME provider；
- P5 明确初始化 `rho/u/S`；
- 在低马赫、小网格、短时间范围与 FullF 做逐步差分；
- 分别验证 HOME 自身守恒和长期稳定性。

通过这些测试只能声明：

> 项目 D3Q19 HOME 编码与开放大气 VOF 兼容。

不能声明精确复现论文 D3Q27 高阶碰撞。

### 12.4 CPU/CUDA 验收

- 所有手算和 oracle 测试必须先在 CPU 通过。
- CUDA 必须运行相同输入并比较关键数组和不变量。
- 原子操作或 reduction 引起的浮点顺序差异使用预先冻结的容差。
- 类型与拓扑结果应保持离散一致；若不一致，不能仅用浮点容差解释。
- 重复运行检查非确定性和数据竞争。

### 12.5 性能与内存

性能优化必须在正确性冻结之后：

- 首先保留易审计的 multi-pass 和 scratch；
- profiling 后才决定 kernel fusion；
- fusion 前后逐数组差分；
- 不为了减少内存让 persistent/scratch 或 n/n+1 语义混淆；
- 所有性能变化必须继续通过质量、拓扑和几何门禁。

### 12.6 最终验收目标

```text
[ ] P0-P6 的全部回归持续通过
[ ] 综合场景不存在 NaN/Inf 和非法 L-G 邻接
[ ] 质量误差满足冻结阈值并与 boundary flux 闭合
[ ] 平移/镜像/旋转对称性通过
[ ] 几何和 Laplace pressure 展示网格收敛
[ ] dam-break 使用定量曲线而非仅截图验收
[ ] FullF CPU/CUDA 通过
[ ] 若宣称 HOME 支持，则 HOME 独立通过全部适用门禁
[ ] 不包含 bubble、moving solid 或 foam 隐式依赖
```

## 13. 建议代码边界

建议保持 `LbmSolver` 只负责顶层调度，VOF 物理由独立模块拥有：

```text
lbm/vof/
  contracts.py          模式、枚举和公共契约
  config.py             VofConfig 与能力校验
  state.py              持久 VOF/geometry 状态
  initialization.py     phi/type/mass/kinetic 初始化
  population.py         FullF/HOME logical population provider
  advection.py          Eq.9-10 质量平流调度
  surface.py            Eq.11-12 自由面补全
  transition.py         候选、拓扑和最终类型
  redistribution.py     守恒 excess mass 分配
  kinetic_init.py       新 interface kinetic 初始化
  geometry.py           normal/PLIC/curvature
  diagnostics.py        不变量、计数器和质量账本
  debug.py              observation-only 密度标签
  visualization.py      非权威显示
```

生产 kernel 可以按模块拆分，也可以在正确性冻结后融合。Python 调度层必须保留清晰
阶段边界和可单独调用的测试入口。

## 14. Solver 最终调度契约

完成 P6 后，单步建议固定为：

```text
1. copy static/domain boundary fields
2. advect mass from state_in into VOF scratch
3. stream logical populations from state_in
4. overwrite gas-to-interface missing links
5. complete ordinary domain boundaries
6. collect moments and compose force
7. collide into provisional state_out kinetic
8. generate and resolve type transitions
9. redistribute excess/deficit mass
10. initialize/invalidate kinetic state from final transitions
11. finalize state_out mass/phi/type
12. compute state_out normal/PLIC/curvature
13. write observables and diagnostics
14. validate epochs and enabled runtime invariants
15. swap state buffers
```

其中：

- 步骤 2 只读取 `state_in` 的 `rho/phi/type/logical f`；
- 步骤 4 只使用 `state_in` 的自由表面和几何；
- 步骤 8-12 构造完全一致的 `state_out(n+1)`；
- 下一步 Eq. (12) 读取刚完成的同 epoch curvature；
- 普通 domain boundary 与 gas surface boundary 必须有明确优先级。

## 15. 验收阈值管理

本文定义必须测什么，不凭空固定所有数值阈值。每个 benchmark 开始编码前，应在独立
acceptance 文件冻结：

- 网格分辨率；
- 时间步数；
- precision 和 device；
- mass absolute/relative tolerance；
- 允许的最大伪速度；
- 几何误差和期望收敛趋势；
- CPU/CUDA 差分容差；
- 失败时保存的最小诊断字段。

阈值不能在看到失败结果后临时放宽。若确需调整，必须记录原因、旧值、新值和证据。

## 16. 阶段状态表

建议使用以下状态：

```text
NOT_STARTED
IN_PROGRESS
CODE_COMPLETE
CPU_ACCEPTED
CUDA_ACCEPTED
BLOCKED
```

当前基线可登记为：

| 阶段 | 当前状态 | 说明 |
|---|---|---|
| P0 | CPU_ACCEPTED | observation/debug 边界和旧 LBM CPU 基线已冻结 |
| P1 | CPU_ACCEPTED | authoritative 状态、初始化、双缓冲和 fail-fast 已冻结 |
| P2 | NOT_STARTED | 无 authoritative mass advection |
| P3 | NOT_STARTED | `surface_completion` 仍为 fail-fast 占位 |
| P4 | NOT_STARTED | 无 transition/topology/redistribution |
| P5 | NOT_STARTED | 无通用 GAS→INTERFACE kinetic 初始化 |
| P6 | NOT_STARTED | observation normal 已有，但正式 PLIC/curvature 未完成 |
| P7 | NOT_STARTED | 尚无完整自由表面集成验收 |

只有状态达到 `CPU_ACCEPTED` 才能进入下一物理阶段；合并到声称支持 CUDA 的主路径前，
还必须达到 `CUDA_ACCEPTED`。

## 17. 下一阶段入口

P2 开始前先提交质量交换 ADR，然后实现固定类型 Eq. (9)-(10)：

```text
冻结 mass scheme、rho 时间层和 gas-link logical population 语义
只实现 FullF 质量平流
不转换、不 clamp、不重分配
加入单格链手算和 NumPy oracle
加入周期域总质量守恒测试
```

P1 已经使项目从 observation-only 进入可安全初始化的 authoritative VOF；只有完成
P2-P6 后才能声明自由表面能够正确推进。
