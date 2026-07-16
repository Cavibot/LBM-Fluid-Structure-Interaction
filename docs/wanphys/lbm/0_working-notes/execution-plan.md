# LBM 编码与碰撞架构重整执行计划

> 状态：已实施（2026-07-16）  
> 范围：D3Q19 LBM 核心；FullF/HOME 状态编码；SRT/TRT EPC；HOME-NOCM-MRT EMC  
> 暂不纳入：Raw MRT、FullF NOCM-MRT、HOME+Shan-Chen、HOME 刚体耦合、现有 momentum-exchange 重构

## 1. 目标

在保持 WanPhys/Newton 现有 `Domain / Model / State / Solver` 生命周期的前提下，将当前固定 FullF、融合 collide-stream 的 LBM 求解器重整为一条逻辑统一、可组合的流水线：

```text
EncodedState^n (post-collision)
  -> streaming
  -> macros / force / hydro
  -> collision
  -> encode
  -> EncodedState^(n+1) (post-collision)
```

该流水线支持两个正交选择：

1. 持久状态编码：`FullF | HOME`；
2. 碰撞执行形态：
   - EPC（Explicit Population Collision）：SRT、TRT；
   - EMC（Encoded Moment Collision）：HOME-NOCM-MRT。

最终形成以下 2x2 组合：

| 持久编码 | EPC：SRT/TRT | EMC：HOME-NOCM-MRT |
|---|---|---|
| FullF | 正式主路径 | 实验路径：闭式矩碰撞后重构 FullF |
| HOME | 实验路径：显式碰撞后投影 HOME | 正式主路径 |

本轮不追求所有组合具有相同数值地位；目标是先建立清晰、可验证的组合框架。

## 2. 已确定的设计决策

### 2.1 保留一个 Domain、一个 Model、一个 Solver

公开生命周期保持：

```text
LbmDomain
  |- LbmModel
  |- LbmSolver
  |- state_in
  `- state_out
```

- `LbmDomain` 继续负责状态创建、一步推进和双缓冲交换。
- `LbmModel` 保存公共静态参数与编码/碰撞选择。
- `LbmSolver` 负责组织一步的 streaming、物理量准备、collision 和 encode。
- 不创建两套 `FullFLbmDomain` / `HomeLbmDomain` 实现。
- 如未来需要，可提供薄工厂，但不得复制 Domain、Model 或 Solver 逻辑。

### 2.2 State 使用两个具体类型

State 的 GPU 内存布局确实不同，因此拆为：

```text
LbmStateBase
  |- FullFLbmState
  `- HomeLbmState
```

Model 暂不拆成两个具体类型，因为网格、边界、设备、力模型和大部分参数完全共享。

### 2.3 不设置 output_encoding

一次模拟中持久编码保持不变：

```text
FullF^n -> FullF^(n+1)
HOME^n  -> HOME^(n+1)
```

输出编码始终由 `model.encoding` 决定，不允许单独配置 `output_encoding`。

### 2.4 不创建独立 ExecutionPlan 类

本文中的 execution plan 是执行安排，不是本轮必须实现的 Python 类。

`LbmSolver.__init__()` 根据 Model 选择并缓存具体 streaming、collision 和 encode 调用；`step()` 直接编排四个阶段。只有在后续 GPU 融合方案显著增多时，才考虑抽取独立 Plan 类型。

### 2.5 coupling 暂不重构

- 当前轮次不设计 `LinkImpulse`。
- 不改造 momentum-exchange 算法。
- HOME 与现有 rigid coupling 组合时明确报不支持。
- FullF 继续保留 `.f`、`solid_phi`、`solid_body_id`、`vel_solid_*` 等兼容入口。
- coupling 的时间层和动量交换适配在核心流水线稳定后另立任务。

## 3. 持久状态时间语义

### 3.1 两种时间位置

一次 LBM 演化可写为：

```text
pre-collision f*
  -> collision
  -> post-collision fPost
  -> streaming
  -> next pre-collision f*
```

当前实现的活动 `state.f` 更接近 post-stream/pre-collision 状态，因为现有 kernel 执行 `collision + streaming` 后直接交换缓冲。

目标架构统一持久化 post-collision 状态：

```text
state_in (post-collision)
  -> streaming
  -> pre-collision temporary
  -> collision
  -> state_out (post-collision)
```

这是一次明确的内部时间语义调整，不是字段重命名。

### 3.2 FullF 持久状态

```text
FullFLbmState.f_post[Q * N]
```

其中 `f_post` 是 collision 后、streaming 前的完整 population。

为了保持现有 API：

```python
FullFLbmState.f  # 兼容别名，返回 f_post
LbmState = FullFLbmState
```

### 3.3 HOME 持久状态

HOME 保存 10 个三维矩分量：

```text
rho
rho*u_x, rho*u_y, rho*u_z
rho*S_xx, rho*S_yy, rho*S_zz
rho*S_xy, rho*S_xz, rho*S_yz
```

即：

\[
(\rho,\rho\mathbf u,\rho\mathbf S)^{post}
\]

`S` 使用论文定义的二阶 Hermite/速度矩：

\[
\rho S_{\alpha\beta}
=
\sum_i
(c_{i\alpha}c_{i\beta}-c_s^2\delta_{\alpha\beta})f_i.
\]

## 4. Model 最小配置

本轮只在现有 `LbmModel` 增加：

```python
encoding: str = "fullf"
collision: str | None = None
```

支持的编码标识：

```text
fullf
home
```

支持或预留的碰撞标识：

```text
srt          # 本轮实现，EPC
trt          # 本轮实现，EPC
home_nocm    # 本轮实现，EMC
raw_mrt      # 仅预留，选择时抛 NotImplementedError
nocm_mrt     # 仅预留，表示 FullF 显式 NOCM；选择时抛 NotImplementedError
```

兼容现有调用：

```text
collision is None and lambda_trt == 0 -> srt
collision is None and lambda_trt > 0  -> trt
```

本轮不拆分 `GridConfig`、`ForceConfig`、`BoundaryConfig` 等嵌套配置。

### 4.1 配置校验

构造 Model 时进行 fail-fast 校验：

- `encoding` 必须为 `fullf | home`；
- `collision` 必须属于已声明标识；
- `raw_mrt`、`nocm_mrt` 立即抛 `NotImplementedError`，不得静默回退；
- `home + G != 0` 暂时抛 `NotImplementedError`；
- `use_regularization` 仅允许 TRT EPC；
- HOME 与当前 rigid coupling 的拒绝放在 coupling 构造入口，而不是核心 Model 中。

## 5. State 设计

### 5.1 `LbmStateBase`

第一轮为了兼容现有代码，公共字段继续留在 State：

```text
density
velocity_x / velocity_y / velocity_z
vel_u / vel_v / vel_w
vel_solid_u / vel_solid_v / vel_solid_w
solid_phi
solid_body_id
force_x / force_y / force_z
```

`clear()`、公共数组 clone helper 和公共初始化放在基类。

`BoundaryFields` 本轮不移出 State。未来可能移到 Domain 所有，是为了避免非 kinetic 边界字段随双缓冲复制，并非这些字段不再需要。

### 5.2 `FullFLbmState`

新增：

```text
f_post: Q * N
```

提供 `.f` 兼容属性。

### 5.3 `HomeLbmState`

新增 10 个持久矩数组。HOME 不提供伪造的 `.f` 全域数组；需要 population 时必须通过 Hermite reconstruction 获取。

### 5.4 State 创建

`LbmDomain.create_state()` 改为调用：

```python
self._solver.create_state()
```

Solver 根据 `model.encoding` 创建相同类型的 `state_in` 和 `state_out`。

## 6. Solver 一步的最小结构

`LbmSolver.step()` 固定为四段：

```python
def step(state_in, state_out, dt, ...):
    pre_collision = self._stream(state_in)
    macros, force, hydro = self._prepare_physics(pre_collision, state_in)
    collision_output = self._collide(pre_collision, macros, force, hydro)
    self._encode(collision_output, state_out)
    self._write_observables(macros, force, hydro, state_out)
```

这里的返回值可以是内部 scratch 数组集合；本轮不要求创建 `CollisionResult` 数据类。

### 6.1 streaming

streaming 包括：

- 上游邻居寻址；
- 周期回绕；
- population 获取；
- static halfway bounce-back；
- 将 incoming population 交给相应输出形式。

streaming 不负责：

- collision；
- Shan-Chen 计算；
- equilibrium；
- 状态编码。

### 6.2 physics

`_prepare_physics()` 负责：

1. 获得 raw moments；
2. 计算物理力；
3. 计算 hydrodynamic velocity。

基本关系：

\[
\rho^*=\sum_i f_i^*,
\qquad
\mathbf j^*=\sum_i\mathbf c_i f_i^*,
\]

\[
\mathbf u=
\frac{\mathbf j^*+\mathbf F/2}{\rho^*}.
\]

固体字段由 coupling/调用方在 `solver.step()` 前准备；physics 阶段不负责栅格化或刚体速度写入。

### 6.3 collision

- EPC 接收完整 `f_star[Q]`，输出完整 `f_post[Q]`；
- EMC 接收 `rho_star / j_star / second_moment_star`，输出 post-collision HOME moments。

### 6.4 encode

encode 根据持久编码提交 collision 输出：

| Collision 输出 | `encoding=fullf` | `encoding=home` |
|---|---|---|
| populations | 直接保存 `f_post` | 投影为 HOME 10 个矩 |
| HOME moments | Hermite 重构 `f_post` | 直接保存 HOME moments |

## 7. 为什么 collision 决定 streaming 输出

输入编码只决定每个 incoming population 如何得到：

```text
FullF -> 直接读取 f_post_i
HOME  -> 由 rho、rho*u、rho*S 重构 f_post_i
```

collision 决定 streaming 后需要保留什么：

```text
EPC -> 需要完整 f_star[Q]
EMC -> 只需要 rho_star、j_star、second_moment_star
```

抽象关系为：

```text
for each direction i:
    f_i = provider.read(source, i)  # encoding 决定
    collector.consume(i, f_i)       # collision 决定
```

因此形成四种 streaming 特化：

```text
stream_fullf_to_populations
stream_home_to_populations
stream_fullf_to_moments
stream_home_to_moments
```

逻辑规则统一，但 Warp kernel 可以按 2x2 特化，避免万能 kernel 中的大量运行时分支。

## 8. Population 获取与 HOME 重构

### 8.1 FullF Provider

直接读取上游格点的 `f_post_i`。

### 8.2 HOME Provider

使用论文给出的 D3Q19 第三阶 Hermite reconstruction，由：

```text
rho
rho*u
rho*S
direction i
```

计算单个 `f_post_i`。

HOME Provider 必须是按方向计算的 `wp.func`，不得先构造全域 Q 分布再 streaming。

### 8.3 EMC streaming

EMC 路径边读取 population 边累计：

\[
\rho^* \mathrel{+}= f_i^*,
\]

\[
\mathbf j^* \mathrel{+}= \mathbf c_i f_i^*,
\]

\[
\rho^*\mathbf S^*
\mathrel{+}=
(\mathbf c_i\mathbf c_i-c_s^2\mathbf I)f_i^*.
\]

EMC 路径不得分配全域 `f_star[Q*N]`。

## 9. Collision 安排

### 9.1 本轮实际实现

```text
SrtCollision          # EPC
TrtCollision          # EPC
HomeNocmMrtCollision  # EMC
```

### 9.2 EPC 调用约定

SRT/TRT 具有相同输入输出形态：

```text
输入：f_star[Q]、rho、u、F
输出：f_post[Q]
```

第一轮不强制引入正式 Python `Protocol`；两个实现只需提供统一的 `input_kind="population"` 和 `collide()` 调用约定。

### 9.3 TRT regularization

现有 regularization 仅属于 TRT EPC：

```text
stream -> f_star -> regularization -> TRT collision
```

- SRT 不使用 TRT regularization；
- HOME-NOCM 不使用现有 TRT regularization；
- HOME reconstruction 自身承担 Hermite filtering，不重复套用该 kernel。

### 9.4 HOME-NOCM-MRT EMC

输入：

```text
rho_star
j_star
second_moment_star
physical force（第一版仅 Null，随后可加入论文局部力）
```

输出：

```text
rho_post
rho*u_post
rho*S_post
```

实现采用论文三维闭式更新，不显式执行完整 `M(u)`、全 Q 中心矩松弛和 `M^-1(u)`。

### 9.5 暂不实现的碰撞

```text
RawMrtCollision
FullNocmMrtCollision
```

仅预留 collision 标识和 EPC 扩展位置。不得添加无公式的占位 kernel；选择这些标识时必须明确报错。

## 10. Force 范围与顺序

### 10.1 顺序

force 位于 streaming 和 collision 之间，因为 Shan-Chen 依赖全域 post-stream 密度：

```text
stream
  -> rho_star / j_star
  -> force
  -> hydro velocity
  -> collision
```

### 10.2 第一版支持矩阵

| 组合 | Null | gravity/局部力 | Shan-Chen |
|---|---:|---:|---:|
| FullF + SRT/TRT | 支持 | 迁移现有行为 | 迁移现有行为 |
| HOME + SRT/TRT | 支持 | 后续补齐 | 不支持 |
| FullF + HOME-NOCM | 支持 | 后续补齐 | 不支持 |
| HOME + HOME-NOCM | 支持 | 按论文补齐 | 不支持 |

第一优先级是无力条件下的 collision/encoding 对齐。不得为赶进度把现有 SC velocity-shift 直接当成 HOME-NOCM 的论文 force closure。

### 10.3 GPU 融合

局部力可能允许：

```text
stream + moments + force + collision + encode
```

Shan-Chen 需要读取邻格 `rho_star`，通常要求全局 kernel 边界：

```text
kernel 1: streaming / rho_star
kernel 2: Shan-Chen force
kernel 3: collision / encode
```

这只影响 kernel 调度，不改变逻辑流水线。

## 11. Boundary 范围

新核心流水线首先正式支持：

```text
periodic
static halfway bounce-back
```

暂缓迁移：

```text
Zou-He inlet
convective outlet
moving wall
动态刚体 cut-link
```

原因是当前 bounce-back 与 collision-stream 融合，拆分后必须重新固定边界 population 的时间层。旧融合 kernel 在新 FullF 核心回归完成前保留作为数值对照，不在验证前删除。

## 12. 初始化与可观测量

### 12.1 FullF 初始化

`initialize_equilibrium()` 直接生成 post-collision equilibrium populations：

\[
f_{post,i}=f_i^{eq}(\rho_0,\mathbf u_0).
\]

### 12.2 HOME 初始化

初始化：

```text
rho = rho0
rho*u = rho0*u0
rho*S = equilibrium second moment
```

### 12.3 Observables

`density`、`velocity_*`、MAC velocity 和 force 字段继续作为公共可观测量。它们必须对应刚完成步骤的物理状态，不得继续保存“上一步计算出的宏观量但与当前 kinetic buffer 不同时间层”的隐含状态。

## 13. 文件修改范围

控制新增文件数量：

```text
wanphys/_src/fluid/fluid_grid/lbm/
  model.py       # encoding/collision 配置与校验
  state.py       # Base、FullF、HOME State
  domain.py      # 通过 solver.create_state 创建双缓冲
  solver.py      # 四阶段编排与函数选择
  constants.py   # 继续保存 D3Q19 常量
  kernels.py     # 旧 kernel 暂留，迁移后缩减

  streaming.py   # 四种 streaming 特化与公共 wp.func
  collisions.py  # SRT、TRT、HOME-NOCM
  encoding.py    # FullF/HOME commit、投影与重构
```

本轮不创建多层 `state/codec/transport/collision/execution/` 包结构。

## 14. 实施步骤

### 步骤一：最小骨架与 FullF 迁移

1. 修复当前 SC 测试与 kernel 签名、`psi_type=2` 行为不一致的基线问题。
2. 给 Model 增加 `encoding`、`collision` 和兼容解析。
3. 将 State 拆为 Base/FullF，保留 `LbmState` 与 `.f` 兼容别名。
4. Domain 改用 `solver.create_state()`。
5. 拆出 FullF post-collision streaming、SRT、TRT 和 FullF commit。
6. 将 Solver 固定为 stream -> physics -> collide -> encode。
7. 在 Null force、周期边界、静态 bounce-back 下与旧 kernel 对照。
8. 迁移 FullF gravity/Shan-Chen，不改变既有参数含义。

步骤一结束条件：FullF SRT/TRT 新流水线可独立运行；核心非 coupling 回归通过；旧融合 kernel 仍保留作对照。

### 步骤二：HOME 与 2x2

1. 实现 `HomeLbmState`。
2. 实现 D3Q19 HOME 第三阶 Hermite population reconstruction。
3. 实现四种 streaming 特化。
4. 实现 population -> HOME 投影。
5. 实现 HOME moments -> FullF reconstruction。
6. 实现 `HomeNocmMrtCollision` 闭式 EMC。
7. 点亮四种编码/碰撞组合。
8. 对交叉组合标注 experimental，并限制到已验证边界/力范围。

步骤二结束条件：四种组合在支持范围内可构造、step、保持有限；HOME 自然路径不物化全域 `f_star`。

### 步骤三：验证、兼容与清理

1. 完成状态工厂、streaming 矩阵、collision、encoding、HOME-NOCM 测试。
2. 明确所有非法/未实现组合的错误信息。
3. 对比 FullF/HOME 的质量、动量、二阶矩和均匀平衡保持。
4. 检查 HOME persistent storage 为每格点 10 个量。
5. 评估 FullF 拆分 kernel 的 scratch 内存和性能损失。
6. 在新旧 FullF 数值对照完成后，删除不再需要的旧核心分支；coupling 依赖的兼容入口继续保留。
7. 更新公开导出、README 和测试指南。

## 15. 测试计划

建议新增：

```text
newton/tests/test_lbm_state_encoding.py
newton/tests/test_lbm_streaming_matrix.py
newton/tests/test_lbm_collision_backends.py
newton/tests/test_lbm_home_nocm.py
```

### 15.1 State/Model

- 默认 Model 创建 FullF State；
- `encoding=home` 创建 Home State；
- state_in/state_out 类型一致；
- FullF `.f` 别名可用；
- HOME 不分配 Q population 持久数组；
- reserved MRT/NOCM 配置 fail-fast。

### 15.2 Streaming

- 四种 streaming 选择正确；
- 均匀场在周期边界下保持不变；
- FullF Provider 与等价 HOME Provider 的 incoming population 在容差内一致；
- static bounce-back 质量守恒；
- EMC 路径不物化全域 `f_star`。

### 15.3 Collision

- SRT equilibrium 固定点；
- TRT 在 `omega_plus == omega_minus` 时退化为 SRT；
- TRT regularization 仅在 TRT 路径执行；
- HOME-NOCM 与 CPU/NumPy 小规模参考公式一致；
- collision 后无 NaN/Inf。

### 15.4 Encoding

- population -> HOME 保持 `rho / rho*u / rho*S`；
- HOME -> population -> HOME 在保留矩上闭合；
- FullF+HOME-NOCM 与 HOME+HOME-NOCM 的宏观量在规定场景下对齐；
- HOME+SRT/TRT 的投影结果质量守恒。

### 15.5 回归

- 现有非 coupling LBM 测试；
- Shan-Chen wall force 和有限性测试；
- 代表性 D3Q19 SRT/TRT 示例 smoke test；
- coupling 套件仅用于确认 FullF 兼容入口未被意外删除，不作为 HOME 验收。

## 16. 验收标准

本轮完成需同时满足：

1. 公开仍只有一个 `LbmDomain`、`LbmModel`、`LbmSolver`。
2. FullF/HOME 使用不同具体 State，Domain 自动创建正确双缓冲。
3. 持久 kinetic state 统一为 post-collision 时间层。
4. Solver 代码可清楚读出 `stream -> physics -> collide -> encode`。
5. SRT/TRT 为 EPC，HOME-NOCM 为唯一 EMC。
6. 四种编码/碰撞组合均有明确执行路径或明确的支持限制。
7. HOME-NOCM 自然路径不分配全域 `f_star[Q*N]`。
8. Raw MRT、FullF NOCM-MRT 不存在伪实现，选择时 fail-fast。
9. HOME+Shan-Chen、HOME rigid coupling 不被错误宣称支持。
10. 核心测试通过，现有基线漂移已修复或有明确记录。

## 17. 明确非目标

本轮不做：

- D3Q27；
- 任意 D3QN；
- Raw MRT 数学实现；
- FullF NOCM-MRT 数学实现；
- HOME-Cumulant；
- HOME-SRT/TRT 闭式 collision；
- HOME+Shan-Chen 物理验证；
- free-surface population completion；
- coupling LinkImpulse/MEM 重构；
- HOME 量化；
- 为所有逻辑阶段强制拆成独立 GPU kernel；
- 大规模目录插件化。

## 18. 主要风险与控制

| 风险 | 控制措施 |
|---|---|
| 当前与目标持久时间层不同 | 保留旧融合 kernel 做逐场景对照；先验证周期/静态壁面 |
| FullF 拆分后显存和性能回退 | 正确性优先；记录 scratch 成本；后续允许重新融合 backend |
| HOME D3Q19 reconstruction 实现错误 | 对照论文公式；做 population -> moments 闭合测试 |
| HOME-NOCM force 语义混乱 | 第一版先 Null force；局部力单独验证；禁止直接复用 SC velocity shift |
| 交叉组合被误认为正式物理方法 | API/docstring 标记 experimental；测试只保证定义明确和数值有限 |
| coupling 被意外破坏 | 保留 FullF 兼容字段；HOME coupling fail-fast；不在本轮重构 MEM |
| 未实现 MRT 被静默降级 | reserved 标识统一抛 `NotImplementedError` |

## 19. 开始实施前检查清单

- [x] 确认 post-collision 为唯一持久 kinetic 时间层。
- [x] 确认本轮 collision 只有 SRT、TRT、HOME-NOCM。
- [x] 确认 Raw MRT、FullF NOCM 只预留标识并 fail-fast。
- [x] 确认 `output_encoding` 不存在。
- [x] 确认 coupling 不进入本轮 HOME 支持范围。
- [x] 确认新核心首批边界为 periodic/static bounce-back。
- [x] 确认 HOME+Shan-Chen 暂不支持。
- [x] 修复并记录当前 SC 测试基线漂移。

## 20. 实施结果

本计划已按上述范围落地：

- `FullFLbmState` 与 `HomeLbmState` 使用不同的持久 GPU 内存布局；
- `LbmSolver` 的新核心路径明确执行 `stream -> physics -> collide -> encode`；
- 四种 streaming 特化和 SRT/TRT/HOME-NOCM 三种 backend 已实现；
- HOME reconstruction 和 HOME-NOCM 闭式更新具有独立 NumPy 参考测试；
- Raw MRT、FullF NOCM、HOME+Shan-Chen、HOME rigid coupling 均 fail-fast；
- moving-wall/MEM 暂时继续使用旧融合 FullF 兼容路径，等待后续单独迁移。

当前 scratch 成本：FullF+EPC 为 `19N` 的 `f_star`；HOME+EPC 为
`19N f_star + 19N f_post`；HOME+HOME-NOCM 为 `10N` 的 pre-collision
moments；FullF+HOME-NOCM 为 `10N` pre + `10N` post moments。当前环境没有
CUDA 设备，因此本轮只完成内存结构评估，GPU 吞吐基准需在 CUDA 环境运行。

验证结果：架构、公式、SC 与刚体兼容共 34 项测试通过。另有 14 项既有
可视化测试引用仓库中已经不存在的示例模块，在导入阶段以
`ModuleNotFoundError` 失败；这些测试未进入本次 LBM 实现，不属于本轮回归。
