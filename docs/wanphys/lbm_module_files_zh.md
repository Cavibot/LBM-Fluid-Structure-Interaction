# LBM 模块文件说明

> 文档版本：2026-07-10  
> 路径根目录：`wanphys/_src/fluid/fluid_grid/lbm/`  
> 关联文档：[LBM HOME/HOME-FREE 扩展路线图](lbm_home_roadmap_zh.md)

本文档说明 LBM 子模块的**目录结构、各文件职责、依赖关系与调用入口**，便于分工开发与后续 G2–G4 扩展。

---

## 1. 目录总览

```text
wanphys/_src/fluid/fluid_grid/lbm/
├── __init__.py              # 对外公开 API
├── model.py                 # 静态配置 LbmModel
├── state.py                 # GPU 状态 LbmState
├── solver.py                # 时间步进 LbmSolver
├── domain.py                # 域封装 LbmDomain（双缓冲 + step）
├── constants.py             # 边界/SC 常量；格子几何再导出
├── kernels.py               # Warp GPU 核（D3Q19 分布型）
│
├── core/                    # [G1] 共享原语层
│   ├── __init__.py
│   ├── lattice.py           # LatticeSpec、D3Q19、NUM_DIRS
│   └── pipeline.py          # StepStats、LbmStepControl
│
├── phases/                  # [G1] 相界面插件层
│   ├── __init__.py
│   └── shan_chen.py         # ShanChenPhase（扩散界面 SC）
│
└── benchmark/               # [G1] 基准与指标层
    ├── __init__.py
    ├── metrics.py           # PerfMetrics、ValidationMetrics
    └── registry.py          # 变体注册 V0–V3

wanphys/_src/fluid/fluid_grid/coupling/
└── grid_lbm_rigid_coupling.py   # LBM ↔ 刚体耦合（未改动）

newton/tests/
└── test_lbm_g1_infrastructure.py  # G1 基础设施单测
```

**分层对应**（见路线图 §4）：

| 层 | 目录/文件 |
|----|-----------|
| L0 格子原语 | `core/lattice.py` |
| L1 状态表示 | `state.py` |
| L2 碰撞后端 | `solver.py` + `kernels.py` |
| L3 相界面 | `phases/` |
| L4 边界/FSI | `kernels.py`（BC/BB）+ `coupling/` |
| L5 基准 | `benchmark/` |

---

## 2. 顶层文件

### 2.1 `__init__.py`

**职责**：包对外入口，导出稳定 API。

**公开符号**：

| 符号 | 来源 | 说明 |
|------|------|------|
| `LbmModel` | `model.py` | 静态配置 |
| `LbmState` | `state.py` | 仿真状态 |
| `LbmSolver` | `solver.py` | 单步求解器 |
| `LbmDomain` | `domain.py` | 推荐使用的域对象 |
| `LatticeSpec`, `D3Q19`, `get_lattice_spec` | `core/lattice.py` | 格子规格 |
| `StepStats`, `LbmStepControl` | `core/pipeline.py` | 步进统计与控制 |

**使用方**：examples、`GridLbmRigidCoupling`、测试、未来 benchmark runner。

---

### 2.2 `model.py` — `LbmModel`

**职责**：不可变（dataclass）静态配置，继承 `FluidGridModelBase` 的网格属性。

**关键字段**：

| 字段 | 默认 | 说明 |
|------|------|------|
| `lattice` | `"D3Q19"` | 速度离散名称，经 `get_lattice_spec()` 解析 |
| `tau` | `0.55` | BGK 松弛时间 |
| `lambda_trt` | `0.0` | TRT 魔数，0 表示纯 BGK |
| `G` | `0.0` | Shan-Chen 强度，0 为单相 |
| `gravity_*` | `0.0` | 体 force（格子单位） |
| `bc_types` / `bc_velocity` / `bc_periodic` | 见源码 | 六面边界 |

**新增属性**（G1）：

- `lattice_spec` → `LatticeSpec`
- `num_dirs` → 分布函数个数（D3Q19 为 19）

**约束**：`__post_init__` 校验 τ、TRT、SC、周期 BC 一致性。

---

### 2.3 `state.py` — `LbmState`

**职责**：GPU 驻留、双缓冲之一的完整流体状态。

**主要数组**：

| 成员 | 形状 | 说明 |
|------|------|------|
| `f` | `(num_dirs × N,)` 扁平 | 分布函数 |
| `density`, `velocity_*`, `pressure` | `(nx,ny,nz)` | 宏观场 |
| `vel_u/v/w` | MAC 交错 | 可视化 / 耦合 |
| `vel_solid_u/v/w` | MAC 交错 | 移动壁面速度 |
| `solid_phi`, `solid_body_id` | `(nx,ny,nz)` | 刚体 SDF / 体 id |
| `force_x/y/z` | `(nx,ny,nz)` | SC 力（调试/可视化） |

**G1 变更**：`f` 长度由 `model.num_dirs * stride` 决定，不再硬编码 19。

---

### 2.4 `solver.py` — `LbmSolver`

**职责**：单 LBM 步的 host 编排；**不**直接实现 SC 细节。

**步进顺序**：

```text
复制 solid 场
→ compute_moments
→ ShanChenPhase.pre_collision   (若 G≠0)
→ reg_trt                       (可选)
→ collide_stream_bounceback
→ apply_boundary_conditions
→ apply_guo_force               (G=0 且有重力)
→ ShanChenPhase.post_collision  (若 G≠0)
→ 写 density/velocity/pressure/MAC
```

**G1 变更**：

- 持有 `ShanChenPhase` 实例
- `step(..., control=None) -> StepStats | None`
- 各阶段写入 `StepStats` 毫秒计时

**临时缓冲**（solver 私有，非 state）：`_rho`, `_ux/_uy/_uz`, `_fx/_fy/_fz`。

---

### 2.5 `domain.py` — `LbmDomain`

**职责**：符合 `Domain` 协议的高层入口；管理双缓冲 swap。

| 成员 | 说明 |
|------|------|
| `model`, `solver`, `state` | 标准三元组 |
| `step(dt)` | 调用 solver 并交换 `_state_in/_state_out` |
| `collect_step_stats` | 构造参数，默认 `False` |
| `last_step_stats` | 最近一步的 `StepStats`（需开启统计） |

**兼容性**：`step(dt)` 签名未变；现有 examples 无需修改即可运行。

---

### 2.6 `constants.py`

**职责**：

1. **再导出** `core/lattice.py` 中的格子几何（`NUM_DIRS`, `CX`, `W`, `CS2` 等），保持旧代码 `from .constants import NUM_DIRS` 可用。
2. 定义与格子无关的常量：`BC_*`、`PSI_*`、`FACE_*`。

**注意**：格子几何的**唯一数据源**是 `core/lattice.py`；勿在 `constants.py` 重复定义 CX/W。

---

### 2.7 `kernels.py`

**职责**：全部 Warp `@wp.kernel` / `@wp.func`；当前仅 **D3Q19 分布型** 实现。

**主要 kernel**：

| Kernel | 阶段 |
|--------|------|
| `compute_moments_kernel` | 矩提取 |
| `compute_shan_chen_force_kernel` | SC 力（由 phase 调用） |
| `apply_velocity_shift_kernel` / `restore_physical_velocity_kernel` | SC 速度修正 |
| `reg_trt_kernel` | TRT 正则化 |
| `collide_stream_bounceback_kernel` | 碰撞+迁移+半程 BB |
| `apply_boundary_conditions_kernel` | Zou-He / 出口等 |
| `apply_guo_force_kernel` | 重力 Guo 力 |
| `moments_to_mac_*` / `compute_pressure_kernel` | 后处理 |

**G1 变更**：BC 出口拷贝循环使用 `range(NUM_DIRS)`（来自 `core.lattice`）。碰撞核内部仍为 D3Q19 硬编码展开（G2 D3Q27 将新增或参数化 backend）。

**体量**：约 2300+ 行；G2 计划迁入 `backends/dist/d3q19/` 或并行新增 `d3q27/`。

---

## 3. `core/` — 共享原语（G1 新增）

### 3.1 `core/lattice.py`

**职责**：速度离散的几何与权重**单一数据源**。

**核心类型**：

```python
@dataclass(frozen=True)
class LatticeSpec:
    name: str
    num_dirs: int
    cx, cy, cz: tuple[int, ...]
    weights: tuple[float, ...]
    opposite: tuple[int, ...]
    cs2: float
```

**已注册格子**：

| 名称 | 常量 | num_dirs |
|------|------|----------|
| D3Q19 | `D3Q19` | 19 |

**API**：

- `get_lattice_spec("D3Q19")` — 按名查找，未知名抛 `ValueError`
- 模块级 `NUM_DIRS`, `CX`, `CY`, `CZ`, `W`, `OPPOSITE`, `CS2`

**扩展方式**（G2）：在 `_LATTICE_REGISTRY` 注册 `D3Q27`，并实现对应 kernel backend。

---

### 3.2 `core/pipeline.py`

**职责**：步进流水线跨 backend 共享的类型。

**`StepStats`** — 单步计时（毫秒）：

| 字段 | 含义 |
|------|------|
| `ms_total` | 整步 |
| `ms_moments` | 矩提取 |
| `ms_phase` | 相界面（SC pre+post） |
| `ms_regularization` | TRT 正则 |
| `ms_collision` | 碰撞+迁移 |
| `ms_bc` | 域边界 |
| `ms_body_force` | Guo 重力 |
| `ms_export` | 写回宏观/MAC |

**方法**：`to_dict()`、`with_num_cells(n)`、`mlups` 属性。

**`LbmStepControl`** — 传入 `solver.step(..., control=)`：

| 字段 | 默认 | 说明 |
|------|------|------|
| `collect_stats` | `True` | 是否返回 StepStats |
| `stats_out` | `None` | 可选：写入已有 StepStats 对象 |

---

## 4. `phases/` — 相界面插件（G1 新增）

### 4.1 `phases/shan_chen.py`

**职责**：Shan-Chen **扩散界面**多相模型的 pre/post collision 钩子。

**类型**：

| 类型 | 说明 |
|------|------|
| `MacroscopicBuffers` | 包装 solver 临时 `_rho/_ux/_fx` 等 |
| `ShanChenPhase` | 相界面逻辑类 |

**`ShanChenPhase` 方法**：

| 方法 | 调用时机 | 行为 |
|------|----------|------|
| `enabled` | 属性 | `model.G != 0` |
| `pre_collision` | 碰撞前 | 按 stride 算 SC 力；velocity shift |
| `post_collision` | 碰撞后 | restore 物理速度；拷贝 force 到 state |
| `reset` | 初始化后 | 重置 stride 计数 |

**未来**（G4）：新增 `phases/vof_sharp.py`，solver 通过同一 hook 接口切换，不修改主碰撞链。

---

## 5. `benchmark/` — 基准基础设施（G1 新增）

### 5.1 `benchmark/metrics.py`

**职责**：性能与物理指标的 dataclass 与采集函数。

| 类型/函数 | 用途 |
|-----------|------|
| `PerfMetrics` | ms/step、MLUPS、bytes/cell、kernel 分解 |
| `ValidationMetrics` | max_velocity、water_mass、front_position_x 等 |
| `bytes_per_cell(state)` | 估算每格点 GPU 常驻字节 |
| `perf_metrics_from_step_stats(...)` | StepStats → PerfMetrics |
| `collect_validation_metrics(state, ...)` | 从 density/velocity 采样 |

**输出**：`write_json(path)` 写入 benchmark 报告（G1 第 5 条 runner 将调用）。

---

### 5.2 `benchmark/registry.py`

**职责**：可对比变体的**注册表**（配置 metadata，非运行时 factory）。

**内置变体**：

| ID | name | lattice | backend | phase |
|----|------|---------|---------|-------|
| V0 | dist_d3q19_sc | D3Q19 | dist | shan_chen |
| V1 | dist_d3q27_sc | D3Q27 | dist | shan_chen |
| V2 | home_quant_d3q27 | D3Q27 | home_quant | none |
| V3 | home_free_quant_d3q27 | D3Q27 | home_quant | vof_sharp |

**API**：`register_variant`、`get_variant("V0")`、`list_variants()`。

**说明**：V1–V3 已注册供路线图使用；**当前仅 V0 有可运行实现**。

---

## 6. 依赖关系图

```text
                    examples / coupling
                           │
                           ▼
                      LbmDomain
                     /    │    \
                    /     │     \
              LbmModel  LbmSolver  LbmState
                  │         │         │
            core/lattice    │    num_dirs ← lattice
                  │         │
                  │    phases/shan_chen ──► kernels (SC)
                  │         │
                  └─────────┴────────► kernels (collide/BC/...)
                                    │
                              core/lattice.NUM_DIRS

benchmark/metrics ──► StepStats (from solver)
benchmark/registry ──► VariantSpec (metadata only)
```

---

## 7. 与流固耦合的接口

**文件**：`wanphys/_src/fluid/fluid_grid/coupling/grid_lbm_rigid_coupling.py`（G1 未改）

**数据流**：

1. Coupling 每步栅格化刚体 → `state.solid_phi`、`vel_solid_*`
2. `LbmSolver.step` 复制 solid 场到 `state_out`
3. `collide_stream_bounceback_kernel` 读取 `solid_phi` 做移动壁 BB
4. SC 力计算同样读取 `solid_phi`

量化 HOME（G3）将改为解耦固体校正核，coupling 对外 API 保持不变。

---

## 8. 测试

| 文件 | 覆盖 |
|------|------|
| `newton/tests/test_lbm_g1_infrastructure.py` | lattice、model、SC phase、StepStats、registry |
| `newton/tests/test_lbm_dambreak_examples.py` | 既有 dam-break 冒烟（依赖 examples 包） |

**运行**：

```bash
cd LBM-Fluid-Structure-Interaction
uv run python -m unittest newton.tests.test_lbm_g1_infrastructure -v
```

---

## 9. 启用步进统计（示例）

```python
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys._src.fluid.fluid_grid.lbm.benchmark.metrics import (
    bytes_per_cell,
    collect_validation_metrics,
    perf_metrics_from_step_stats,
)

model = LbmModel(fluid_grid_res=(128, 128, 128), fluid_grid_cell_size=0.02, G=-5.0)
domain = LbmDomain(model, collect_step_stats=True)
domain.create_state()

for step in range(100):
    domain.step(dt=1.0)

stats = domain.last_step_stats
perf = perf_metrics_from_step_stats(
    stats,
    variant="V0",
    lattice=model.lattice,
    num_cells=model.nx * model.ny * model.nz,
    bytes_per_cell_value=bytes_per_cell(domain.state),
)
val = collect_validation_metrics(domain.state, step_index=100)
```

---

## 10. 后续扩展落点（G2–G4）

| 目标 | 建议新增路径 | 改动现有文件 |
|------|--------------|--------------|
| G2 D3Q27 | `core/lattice.py` 注册 D3Q27；`backends/dist/d3q27/kernels.py` | `solver` 按 lattice 选 backend |
| G3 量化 HOME | `backends/moment/home_quant/` | `state` 增加矩缓冲；`solver` 委托 backend |
| G4 VOF | `phases/vof_sharp.py` | `solver` 注入 phase；去掉 SC 路径 |
| G1-⑤ 基准 runner | `benchmark/dambreak_two_spheres.py` | 使用 `registry` + `metrics` |

---

## 11. 相关文档

| 文档 | 内容 |
|------|------|
| [lbm_home_roadmap_zh.md](lbm_home_roadmap_zh.md) | 总体路线图、分工、变体矩阵 |
| [lbm_core_audit_zh.md](lbm_core_audit_zh.md) | 既有实现审计（含已知 Guo/SC 问题） |
| [lbm_rigid_coupling_guide_zh.md](lbm_rigid_coupling_guide_zh.md) | 刚体耦合使用 |
| [../wanphys/_src/fluid/fluid_grid/ARCHITECTURE.md](../../wanphys/_src/fluid/fluid_grid/ARCHITECTURE.md) | fluid_grid 总框架 |

---

*维护：新增/移动 LBM 文件时请同步更新本文档对应章节。*
