# LBM / VOF 重构实施计划（草案）

> 状态：Draft
>
> 架构依据：`docs/wanphys/lbm/lbm-vof-refactor-final-architecture.md`
>
> 阶段数：3
> 总原则：先建立可验证的等价结构，再切换生命周期与所有权，最后删除旧实现并收紧 API；任何一步都必须保持仓库可测试、可回退。

## 0. 全局约束

### 0.1 不变量

- 不引入 lattice descriptor、公开 Coupler 或 `VofStepContext`。
- `LbmDomain.initialize_vof()` 本轮保留，只做适配新 Solver 生命周期所需的最小修改。
- 不保留旧内部模块导入路径；但旧文件只在所有调用点迁移完成的步骤中删除。
- 高层模块不包含 `@wp.kernel`；Kernel 按物理阶段进入对应 `kernels/`。
- 结构移动和数值算法修改不得放在同一个提交中。
- Domain 不暴露私有双缓冲供测试调用；阶段测试直接构造 Solver 和 State。
- diagnostics 对 authoritative state 只读。
- LBM 本轮不接管 VOF `cell_type`、topology 或 kinetic 初始化。
- 正常时间步只执行一次最终 MAC `vel_u/v/w` 写回。

### 0.2 每步完成定义

每个步骤必须同时满足：

1. 目标文件落位且无双份 authoritative 实现。
2. 所列定向测试通过。
3. `python -m compileall` 或等效导入检查通过。
4. `rg` 不存在本步骤禁止保留的旧导入或旧符号调用。
5. Warp CPU 路径可编译；涉及 Kernel 移动时至少完成一次可用设备 smoke test。
6. 不修改与本步骤无关的数值公式、阈值、默认参数或时间步顺序。

---

# Phase 1：建立目标目录和依赖边界，保持数值行为等价

## Step 1.1：冻结基线与建立迁移清单

### 目标

在移动源码前记录当前可运行行为、实际调用点和测试覆盖，区分生产代码、测试基元、死代码候选，防止把未使用实现机械迁入新目录。

### 相关文件

- 当前源码：`wanphys/_src/fluid/fluid_grid/lbm/**/*.py`
- 核心测试：
  - `newton/tests/test_lbm_vof_p1.py` 至 `test_lbm_vof_p8.py`
  - `newton/tests/test_lbm_vof_fix1.py`
  - `newton/tests/test_lbm_vof_fix2.py`
  - `newton/tests/test_lbm_state_encoding.py`
  - `newton/tests/test_lbm_collision_backends.py`
  - `newton/tests/test_lbm_streaming_matrix.py`
  - `newton/tests/lbm_part123/`
  - `newton/tests/test_lbm_dambreak_examples.py`
- 示例：`wanphys/examples/lbm/fluid_grid_lbm_vof_dambreak.py`

### 实现边界

- 只增加测试、清单或基线记录，不移动生产文件。
- 不以“当前测试未覆盖”为理由修改算法。
- 对以下对象只标记状态，不在本步骤删除：
  - `epc_collision_kernel`
  - `compute_moments_kernel`
  - `collide_stream_bounceback_kernel`
  - `apply_guo_force_kernel`
  - `restore_physical_velocity_kernel`
  - `moment_velocity_kernel`
  - `VofMassTransport.finalize_fixed_topology`
  - `finalize_fixed_topology_vof_kernel`
  - `SurfaceCompletion` / `UnavailableSurfaceCompletion`
- host/device PLIC 重复实现视为有意的执行端差异，不列为普通重复代码。

### 验收规格

- 形成“符号 → 生产调用点 → 测试调用点 → 迁移动作”的清单。
- 记录至少一个小型 FullF VOF 和一个 HOME VOF 单步基线：mass、cell type 计数、rho/u、epoch、diagnostics。
- 现有定向测试在任何源码移动前通过；已有失败必须单独记录，不得混入重构修复。
- 明确上述死代码候选是“删除、测试专用或生产保留”中的一种。

## Step 1.2：建立 LBM `solver/` 与阶段 Kernel 目录

### 目标

把当前扁平 LBM 实现移动到 `solver/` 和 `solver/kernels/`，按物理阶段分组，同时保持 `LbmSolver.step()` 的调用顺序和计算结果不变。

### 相关文件

当前来源：

- `lbm/solver.py`
- `lbm/kernels.py`
- `lbm/streaming.py`
- `lbm/encoding.py`
- `lbm/collisions.py`
- `lbm/forcing.py`
- `lbm/moments.py`

目标文件：

- `lbm/solver/__init__.py`
- `lbm/solver/solver.py`
- `lbm/solver/collisions.py`
- `lbm/solver/forcing.py`
- `lbm/solver/moments.py`
- `lbm/solver/kernels/common.py`
- `lbm/solver/kernels/initialization.py`
- `lbm/solver/kernels/streaming.py`
- `lbm/solver/kernels/encoding.py`
- `lbm/solver/kernels/collision.py`
- `lbm/solver/kernels/moments.py`
- `lbm/solver/kernels/forcing.py`
- `lbm/solver/kernels/boundary.py`
- `lbm/solver/kernels/moving_wall.py`
- `lbm/solver/kernels/shan_chen.py`
- `lbm/solver/kernels/macroscopic.py`
- `lbm/solver/kernels/regularization.py`
- `lbm/solver/kernels/admissibility.py`

### 实现边界

- 本步骤只做等价移动、导入修正和必要的循环依赖拆除。
- `@wp.kernel` 全部迁入阶段 Kernel 文件。
- 单阶段 `wp.func` 与该阶段同文件；跨阶段 `wp.func` 进入唯一的 `kernels/common.py`。
- NumPy 矩阵、relaxation policy 和纯 Python 策略保留在 Solver 高层文件。
- 不改变 `_copy_boundary_fields()`、streaming、boundary、moments、force、collision、observable 的相对顺序。
- 不在本步骤删除第一次 MAC 写回，也不改变 VOF topology 或 kinetic 职责。
- 不创建每 Kernel 一个文件，不建立仅转发一次的包装类。

### 验收规格

- `rg '@wp.kernel' wanphys/_src/fluid/fluid_grid/lbm` 的结果只出现在 `solver/kernels/` 或 `vof/**/kernels/`。
- `lbm/solver/__init__.py` 只主动导出 `LbmSolver`。
- Part 1–3、collision、encoding、streaming、rigid coupling 定向测试通过。
- FullF/HOME 单步结果与 Step 1.1 基线在既定容差内一致。
- Warp Kernel 可在 CPU 路径重新编译；有 CUDA 时完成至少一组 FullF 和 HOME smoke test。

## Step 1.3：建立 VOF `model/state/solver/diagnostics` 结构和最小 API

### 目标

将 VOF 物理阶段、状态、模型和 diagnostics 分离，消除根 `vof/__init__.py` 对内部阶段对象的 eager export，同时不改变现有 VOF 执行顺序。

### 相关文件

当前来源：

- `lbm/vof/__init__.py`
- `lbm/vof/advection.py`、`advection_kernels.py`
- `lbm/vof/surface.py`、`surface_kernels.py`
- `lbm/vof/transition.py`、`transition_kernels.py`
- `lbm/vof/kinetic_init.py`、`kinetic_init_kernels.py`
- `lbm/vof/geometry.py`、`geometry_kernels.py`
- `lbm/vof/initialization.py`
- `lbm/vof/validation.py`
- `lbm/vof/diagnostics.py`
- `lbm/vof/runtime.py`
- `lbm/vof/debug.py`
- `lbm/vof/visualization.py`

目标范围：

- `lbm/vof/model.py`
- `lbm/vof/state.py`
- `lbm/vof/contracts.py`
- `lbm/vof/initial_conditions.py`
- `lbm/vof/validation.py`
- `lbm/vof/solver/` 与 `lbm/vof/solver/kernels/`
- `lbm/vof/diagnostics/` 与 `lbm/vof/diagnostics/kernels/`

### 实现边界

- 本步骤先做结构等价迁移；`VofSolver` 可先薄封装当前阶段对象，但不得增加第四个公开业务方法。
- `VofModel` 只接收 authoritative VOF 必需参数；gravity、速度上限、population positivity 等仍由 LBM 显式提供。
- `VofDiagnosticsConfig` 只描述 authoritative validation policy；Shan-Chen debug 参数本轮留在 `LbmModel`。
- `vof/__init__.py` 只导出 `VofModel`、`VofGridState`、`VofCellType`、`VofMassScheme`。
- `VofSolver` 只从 `vof/solver/__init__.py` 导出；diagnostics 对象只从 `vof/diagnostics/__init__.py` 导出。
- VOF 模块对 LBM state 类型只使用 `TYPE_CHECKING`、Protocol 或局部导入，禁止根包循环导入。
- `validation.py` 不反向依赖 `initial_conditions.py`。
- 保留 VOF 原有 kinetic 初始化逻辑及其所有权，本轮不迁入 LBM。

### 验收规格

- 五个入口可分别独立导入：`lbm`、`lbm.solver`、`lbm.vof`、`lbm.vof.solver`、`lbm.vof.diagnostics`。
- `vof/__init__.py` 不再导出 transport、surface、transition、geometry、kinetic initializer 和阶段 Result。
- 高层 `geometry.py`、`visualization.py`、`initialization.py` 中无 `@wp.kernel`。
- VOF P1–P8、fix1/fix2 和 diagnostics 相关测试通过。
- Step 1.1 的 VOF 单步基线保持一致。

### Phase 1 出口门

- 目标目录已建立，所有生产导入使用新路径。
- 数值时序和结果未发生有意变化。
- 不允许带着双份新旧实现进入 Phase 2。

---

# Phase 2：冻结 VOF 生命周期、事务、诊断和宏观发布顺序

## Step 2.1：实现 `VofSolver` 三方法生命周期与状态对事务

### 目标

让 `LbmSolver.step()` 只在两个固定时间点调用 VOF，并由 `VofSolver` 保护 `source/target/epoch` 事务。

### 相关文件

- `lbm/solver/solver.py`
- `lbm/vof/solver/solver.py`
- `lbm/vof/solver/advection.py`
- `lbm/vof/solver/surface.py`
- `lbm/vof/solver/transition.py`
- `lbm/vof/validation.py`
- `lbm/state.py`
- `newton/tests/test_lbm_vof_p4.py`
- `newton/tests/test_lbm_vof_p7.py`
- 新增或扩展 VOF transaction tests

### 实现边界

- 公开业务接口固定为：
  - `initialize(state, phi0)`
  - `complete_streaming(state_in, state_out, logical_f_post, streamed_populations)`
  - `finish_step(state_in, state_out)`
- `complete_streaming()` 接收 `state_out` 只为绑定事务，不得写入它。
- `complete_streaming()` 记录 source identity、target identity、source epoch 和本事务 validation cadence。
- 正常每步只检查 `state_in is not state_out`、prepared 标记、identity 和 epoch。
- type、shape、device、storage、数组别名和初始一致性移到双缓冲创建/初始化路径，只完整检查一次。
- 重复 prepare、无 prepare finish、错 source/target/epoch 均最简 `raise RuntimeError`。
- finish 失败不清除 prepared 标记；成功 `initialize()` 是明确恢复起点。
- 不引入公开 Context、回滚协议或额外 Coupler。

### 验收规格

- transaction tests 覆盖：同一 state、错 target、错 epoch、重复 prepare、跳过 prepare、finish 失败后重入。
- 测试证明 `complete_streaming()` 前后 `state_out` authoritative arrays 不变。
- instrumentation 证明每步不再扫描全部 VOF storage alias。
- mass exchange 仍只读旧时间层；将其放到普通 pull streaming 后不改变 Step 1.1 基线。
- `state_out.density` 产生前不执行最终 topology proposal/commit。

## Step 2.2：消除重复 MAC 计算

### 目标

把 cell-centered observable 与 MAC 发布拆开，使 VOF 和非 VOF 路径的最终 MAC 每步都只计算一次；不改变 topology、`cell_type` 或 kinetic 初始化职责。

### 相关文件

- `lbm/solver/solver.py`
- `lbm/solver/kernels/macroscopic.py`
- `lbm/state.py`
- `newton/tests/test_lbm_vof_p7.py`
- `newton/tests/test_lbm_state_encoding.py`

### 实现边界

- `_write_observables()` 拆为 cell-centered 写回和最终 MAC 写回；删除 VOF 修正前的 MAC 计算。
- VOF 路径在 `finish_step()` 完成现有处理后调用一次 `_write_mac_velocities(state_out)`。
- 非 VOF 路径在 cell-centered observable 写回后调用一次 `_write_mac_velocities(state_out)`。
- 不增加第二次全网格 cell-centered `rho/u` 重算。
- 不移动、不改写 VOF kinetic 初始化，不让 LBM 读取或管理 VOF `cell_type`。
- 不新增 topology 变化量、LBM 管理类、回调或耦合接口。

### 验收规格

- instrumentation 证明 `moments_to_mac_u/v/w` 每个分量每步各启动一次。
- 最终 MAC 数值由本步最终 `state_out.velocity_x/y/z` 生成。
- FullF/HOME、VOF/非 VOF、positivity on/off 的定向测试通过。
- VOF topology、kinetic 初始化结果及调用顺序与 Phase 1 基线一致。

## Step 2.3：将 diagnostics 内聚到 `finish_step()`

### 目标

让 diagnostics 在 VOF 提交完成后、transition scratch 清理前读取最终状态和 `proposed_type`，避免 LBM 暴露或重复计算 VOF scratch。

### 相关文件

- `lbm/vof/solver/solver.py`
- `lbm/vof/diagnostics/host.py`
- `lbm/vof/diagnostics/device.py`
- `lbm/vof/diagnostics/config.py`
- `lbm/vof/diagnostics/state.py`
- `lbm/vof/diagnostics/kernels/`
- `lbm/solver/solver.py`
- diagnostics、runtime profile 和 debug observer 相关测试

### 实现边界

- 调用链固定为 `Domain.step → LbmSolver.step → VofSolver.finish_step → diagnostics`。
- diagnostics 只读 authoritative LBM/VOF arrays；只允许写自己的 scratch、归约和结果缓冲。
- `proposed_type` 只读且仍属于 transition scratch，不进入 `state_out`。
- diagnostics 在 commit、geometry 和 target validation 之后执行，在 scratch cleanup 之前执行。
- validation cadence 在 `complete_streaming()` 时确定并随 prepared transaction 保存。
- `VofSolver.last_diagnostics` 是唯一缓存；`LbmSolver.last_vof_diagnostics` 只做 property 转发。
- diagnostics 显式持有 `VofModel`/配置，不通过 `state.model.vof_*` 取配置。

### 验收规格

- diagnostics 写保护测试证明所有 authoritative arrays 的 pointer 和内容在调用前后不变。
- device diagnostics 仍能读取本步 `previous_type/proposed_type/final_type`。
- STRICT、SAMPLED、DEVICE 三种 profile 的 cadence 与 Phase 1 基线一致。
- `LbmSolver` 不直接导入 diagnostics collect/validate 具体实现，也不持有第二份结果缓存。
- diagnostics 抛错时 Domain 不交换双缓冲，VOF prepared 标记保持 fail-stop。

### Phase 2 出口门

- 三方法生命周期成为唯一 VOF 步进路径。
- 双缓冲、topology、diagnostics、MAC 的时间顺序有直接测试保护。
- P1–P8、fix1/fix2、dambreak 单步和短程回归全部通过。

---

# Phase 3：边界语义收口、删除冗余并完成全仓迁移

## Step 3.1：拆分 boundary 配置、解析和 Kernel 执行

### 目标

清理当前 `boundaries.py` 的混合职责，移除自由表面占位抽象，并使六面边界、moving wall 和 cut link 的语义一致。

### 相关文件

- `lbm/model.py`
- `lbm/contracts.py`
- `lbm/boundaries.py`
- `lbm/solver/solver.py`
- `lbm/solver/kernels/boundary.py`
- `lbm/solver/kernels/moving_wall.py`
- `lbm/solver/kernels/streaming.py`
- `newton/tests/lbm_part123/test_part3_boundary_pipeline.py`
- `newton/tests/test_lbm_directional_streaming.py`
- `newton/tests/test_lbm_rigid_coupling.py`

### 实现边界

- 名称规范化和静态配置校验进入 `model.py`。
- 依赖 shape 的六面展开、索引解析和冲突检查保留在根 `boundaries.py`。
- population 执行只存在于 Solver Kernel 文件。
- 删除 `SurfaceCompletion`、`UnavailableSurfaceCompletion` 和 `model.surface_completion`。
- 从普通六面边界类型排除 `SURFACE`。
- `MOVING_WALL`、`CUT_LINK` 作为几何/链路能力处理，不再假装普通 per-face completion。
- `_copy_boundary_fields()` 仍在 streaming 前执行，且早于 force/collision 对 `state_out` 的读取。

### 验收规格

- `rg 'SurfaceCompletion|UnavailableSurfaceCompletion|surface_completion'` 在生产源码中无结果。
- 六面边界配置不能接受 `SURFACE`。
- moving wall、cut link、open boundary 和 periodic 组合测试通过。
- 边界重构前后的 streamed populations、rho/u 和力在既定容差内一致。
- `lbm/boundaries.py` 中不存在 `@wp.kernel`，Kernel 文件中不存在配置名称规范化逻辑。

## Step 3.2：删除死代码、收窄 API 并迁移所有调用方

### 目标

删除已确认无生产价值的旧实现，更新源码、测试、示例、文档和字符串 patch 路径，使新目录成为唯一事实来源。

### 相关文件

- `wanphys/_src/fluid/fluid_grid/lbm/`
- `newton/tests/test_lbm*.py`
- `newton/tests/lbm_part123/`
- `wanphys/examples/lbm/`
- `docs/wanphys/lbm/`
- `asv/benchmarks/` 中所有 LBM 引用

### 实现边界

- 只删除 Step 1.1 已确认无生产/测试调用的符号。
- 测试专用 Kernel 若保留，必须进入明确的测试 helper，不从生产包入口导出。
- 不建立旧模块 shim、alias module 或 deprecation 转发。
- `D3Q19_MOVING_DIRECTIONS` 等 host 常量只来自 `lbm/constants.py`；示例不复制方向表。
- device 方向/权重/opposite helper 只来自 LBM `kernels/common.py`。
- host/device PLIC 保持双实现并增加 parity test，不以去重名义跨执行端调用。

### 验收规格

- 对旧模块路径和旧导出符号执行全仓 `rg`，除迁移说明文档外无结果。
- `lbm/__init__.py`、`lbm/solver/__init__.py`、`lbm/vof/__init__.py`、`lbm/vof/solver/__init__.py`、`lbm/vof/diagnostics/__init__.py` 的导出与架构文档完全一致。
- dambreak 示例不再定义本地 D3Q19 方向常量，且能够启动和推进。
- 所有 mock/patch 字符串路径已更新。
- 删除后没有未引用的新包装层或空模块。

## Step 3.3：全矩阵验收、性能核对与文档冻结

### 目标

证明重构后的目录、时间步语义、数值结果、性能和公开 API 同时满足冻结架构，完成交付收口。

### 相关文件

- 全部 LBM/VOF 源码和测试
- `wanphys/examples/lbm/fluid_grid_lbm_vof_dambreak.py`
- `docs/wanphys/lbm/lbm-vof-refactor-final-architecture.md`
- 本实施计划
- 相关 benchmark 配置

### 实现边界

- 本步骤只修复验收暴露的重构缺陷，不引入新的物理模型或性能优化算法。
- 性能下降若来自额外检查，应优先确认是否违反“一次性结构验证、每步最小事务门禁”。
- 不用放宽容差、禁用 diagnostics 或跳过 HOME/FullF 路径来换取通过。
- CUDA 不可用时明确记录环境限制，但 CPU/Warp 编译和数值测试仍必须完成。

### 验收规格

- 测试矩阵至少覆盖：
  - FullF / HOME
  - SRT/TRT/NOCM 等当前受支持碰撞路径
  - VOF STRICT/SAMPLED/DEVICE diagnostics
  - periodic/open/moving-wall/cut-link 相关边界组合
  - positivity on/off
  - gravity、surface tension 和 geometry 更新
- dam-break 至少完成短程稳定性回归；mass、非法 topology、NaN/Inf、速度上限无退化。
- 与 Phase 1 基线比较：无未解释的数值偏差；允许偏差必须定位到已批准的 Phase 2 行为修正。
- 每步 MAC Kernel 启动次数从两组降为一组，正常每步无全 storage alias 扫描。
- 导入 smoke、静态 `rg` 规则、Markdown 链接和 Mermaid 文稿检查通过。
- 最终架构文档移除“草案/待定”表述；本计划记录每一步的实际结果和偏差。

### Phase 3 出口门

- 旧目录和旧内部 API 已完全消失。
- 所有受支持路径通过验收矩阵。
- 架构、代码、测试、示例和文档描述一致，可开始后续初始化流程的独立评估。

---

## 建议提交切分

每个 Step 至少独立一个提交；Step 内若同时包含结构移动和行为调整，继续拆成：

1. 等价移动与导入更新。
2. 生命周期或所有权调整。
3. 冗余删除与测试收口。

禁止跨 Phase 合并提交，以确保任何数值回归都能定位到单一职责变化。
