# LBM 扩展路线图：解耦架构与分工方案（含量化 HOME-LBM）

> 文档版本：2026-07-10  
> 适用范围：`wanphys/_src/fluid/fluid_grid/lbm/` 及流固耦合、基准算例  
> 参考论文：  
> - [高阶矩编码湍流动能模拟](../../公式修订版/高阶矩编码湍流动能模拟/高阶矩编码湍流动能模拟_中文.md)（HOME-LBM 算法基础）  
> - [稳定性引导量化的高性能矩编码LBM](../../公式修订版/稳定性引导量化的高性能矩编码LBM/稳定性引导量化的高性能矩编码LBM_中文.md)（**G3 主交付：量化 HOME-LBM**）  
> - [锐界面动力学自由表面流与泡沫](../../公式修订版/锐界面动力学自由表面流与泡沫/锐界面动力学自由表面流与泡沫_中文.md)（HOME-FREE LBM，G4）

---

## 1. 文档目的

本文档为 LBM 模块后续扩展提供**可分工、低耦合**的工程方案，回答：

- 如何在现有 D3Q19 分布型 LBM 上，逐步接入 D3Q27、**稳定性引导量化 HOME-LBM**、锐界面自由表面；
- 各子模块的**职责边界**与**接口契约**是什么；
- 统一基准算例 **dam-break + 双刚体小球 FSI** 下，如何对比效果与性能；
- 建议的**人员分工**、**里程碑**与**风险决策点**。

本文档面向算法实现者与集成负责人，不重复 LBM 教科书推导；公式细节以三篇论文中文修订版为准。

---

## 2. 项目目标与论文映射

### 2.1 四个开发目标

| ID | 目标 | 说明 |
|----|------|------|
| G1 | 精修普通 LBM + 性能评估 | 在现有 D3Q19 上重构分层、接入统一 benchmark 与指标 |
| G2 | D3Q27 分布型 LBM | 27 方向离散，与 D3Q19 对比精度与性能 |
| G3 | **量化 HOME-LBM** | 矩编码 + 三阶 Hermite + NOCM-MRT 碰撞 + **VN 稳定性引导 fp16 量化** + 解耦双核 FSI |
| G4 | 锐界面自由表面 | HOME-FREE：在量化 HOME 上叠 VOF 锐界面，不模拟气相动力学 |

### 2.2 统一基准算例

**Dam-break + 双刚体小球流固耦合**

- 参考实现：`wanphys/examples/lbm/fluid_grid_lbm_dambreak_two_spheres.py`
- 场景：封闭域、左侧水体、右侧空气；干燥地面上方放置一重球、一轻球；Invisible 刚性墙约束域边界
- 耦合：`GridLbmRigidCoupling`（SDF 栅格化 + 移动壁面 bounce-back + 可选动量交换反馈）

所有变体（V0–V3）共用**相同几何、网格分辨率、时间步与刚体参数**，仅替换流体后端与相界面模型。

### 2.3 论文贡献 → 代码模块

| 论文 | 核心贡献 | 对应代码层 | 目标 |
|------|----------|------------|------|
| HOME-LBM（基础） | 10 矩/节点；三阶 Hermite 重建；NOCM-MRT 矩碰撞 | `core/hermite.py`、`core/moments.py` | G3 子模块 |
| **稳定性引导量化** | VN 谱分析 → 矩范围/位宽；fp16 定点量化；解耦双核；混合精度 AtomicAdd；8³ tile SoA | `backends/moment/home_quant/`、`boundary/solid_correction.py` | **G3 主交付** |
| HOME-FREE | VOF 锐界面；L/G/I 单元；量化 HOME 流体推进 | `phases/vof_sharp.py` | G4 |

> **术语约定**：下文 **「HOME-LBM」若无特别说明，均指量化论文中的完整实现**（量化存储 + GPU 双核 + 解耦 FSI），而非 Li et al. 2023 原始 fp32 参考实现。开发阶段可保留 fp32 参考路径 `home_fp32_ref` 用于数值对照，但不作为 G3 验收变体。

---

## 3. 现状摘要

当前 `lbm/` 模块（2026-07）主要特征：

| 组件 | 现状 |
|------|------|
| 格子 | D3Q19（`constants.py`，kernel 内硬编码 19 路） |
| 状态 | 分布函数 `f`（19 × N 扁平数组）+ 宏观场 + MAC 面速度 |
| 碰撞 | BGK / TRT + 可选正则化（`reg_trt_kernel`） |
| 多相 | Shan-Chen 伪势（**扩散界面**） |
| 边界 | 半程 bounce-back、Zou-He、对流出口、周期 |
| FSI | `solid_phi` / `vel_solid_*` + `GridLbmRigidCoupling` |
| 入口 | `LbmModel` / `LbmSolver` / `LbmState` / `LbmDomain` |

**关键约束**：对外 API（`LbmDomain.step`、耦合接口）在重构期间应保持兼容；新能力通过配置项与 backend 切换接入。

---

## 4. 分层架构

### 4.1 设计原则

1. **表示与算法分离**：分布型（`f_i`）与矩型（`ρ, ρu, ρS`）是两种 L1 状态，共享 L0 格子原语。
2. **相界面可插拔**：Shan-Chen（过渡/对比）与 VOF 锐界面（最终目标）互不侵入碰撞核。
3. **FSI 策略可替换**：分布型用现有 bounce-back；量化 HOME 用**流体核 + 固体校正核**双 launch（面并行，混合精度 AtomicAdd）。
4. **一步流水线统一**：所有 backend 走相同 hook 顺序，便于 profiling 与 A/B 对比。

### 4.2 层次图

```text
L5  benchmark/          基准 harness、指标、变体注册、对比报告
L4  boundary/           域边界 BC、固体耦合策略
L3  phases/             单相 / Shan-Chen / VOF 锐界面
L2  backends/           碰撞+迁移（dist 或 moment）
L1  状态表示             DistributionState | QuantizedMomentState
L0  core/               LatticeSpec、Hermite、矩运算、StepContext
```

### 4.3 统一步进顺序

```text
① rasterize_solid          # 刚体 → solid_phi, vel_solid_*
② compute_moments          # f→(ρ,u) 或加载矩
③ phase.on_pre_collision   # SC 力 / VOF 对流准备
④ collision                # BGK/TRT 或 HOME 矩碰撞
⑤ stream                   # 迁移（分布重建+pull 或 f  pull）
⑥ domain_bc                # 非 BB 面边界
⑦ phase.on_post_collision  # VOF φ 更新、界面单元处理
⑧ solid.apply              # BB 或量化 HOME 固体校正核（独立 launch）
⑨ export_mac + metrics     # MAC 面速度、StepStats
```

---

## 5. 建议目录结构

在 `wanphys/_src/fluid/fluid_grid/lbm/` 下扩展（现有文件逐步迁入 `backends/dist/d3q19/`）：

```text
lbm/
├── core/
│   ├── lattice.py           # LatticeSpec: D3Q19 / D3Q27
│   ├── hermite.py           # H^[2], H^[3], 三阶重建 h(ρ,u,S)
│   ├── moments.py           # f↔矩、矩空间碰撞公式
│   └── pipeline.py          # StepContext, StepHooks, StepStats
│
├── backends/
│   ├── dist/
│   │   ├── d3q19/             # 现有 kernels / solver（迁入）
│   │   └── d3q27/             # G2
│   └── moment/
│       ├── home_quant/        # G3 主路径：量化 HOME-LBM
│       │   ├── solver.py      # 流体更新核（碰撞→重建→迁移）
│       │   ├── quant.py       # VN 边界、归一化、fp16 pack/unpack、抖动
│       │   ├── state.py       # 双缓冲：quant 矩 + fp32 迁移矩
│       │   └── stability.py   # VN 放大矩阵 / 矩范围导出（离线或启动时）
│       └── home_fp32_ref/     # 开发对照：全精度 HOME（非 G3 验收）
│
├── phases/
│   ├── none.py                # 单相
│   ├── shan_chen.py           # 现有 SC 逻辑抽取
│   └── vof_sharp.py           # G4 HOME-FREE
│
├── boundary/
│   ├── domain_bc.py
│   ├── bounceback_dist.py
│   └── solid_correction.py    # HOME 解耦固体核
│
├── benchmark/
│   ├── metrics.py
│   ├── dambreak_two_spheres.py
│   └── registry.py
│
├── model.py                   # 组合 backend + phase + lattice
├── solver.py                  # 委托 backend
├── state.py
├── domain.py
└── __init__.py
```

---

## 6. 接口契约

### 6.1 LatticeSpec（L0）

```python
@dataclass(frozen=True)
class LatticeSpec:
    name: str              # "D3Q19" | "D3Q27"
    num_dirs: int
    cx, cy, cz: tuple
    weights: tuple
    opposite: tuple
    cs2: float
```

**约定**：D3Q19 / D3Q27 仅在此层与对应 kernel 模板参数中分化；solver 主流程代码复用。

### 6.2 FluidBackend（L2）

```python
class FluidBackend(Protocol):
    def allocate_state(self, model: LbmModel) -> LbmStateBase: ...
    def step(
        self,
        state_in: LbmStateBase,
        state_out: LbmStateBase,
        ctx: StepContext,
    ) -> StepStats: ...
```

| 实现 | 说明 |
|------|------|
| `DistBackend` | 现有 `collide_stream_bounceback_kernel` 路径 |
| `HomeQuantBackend` | **碰撞→重建→迁移**（非经典迁移→碰撞）；fp16 矩读写；独立固体校正核 |
| `HomeFp32RefBackend` | 全精度 HOME，仅用于量化误差/稳定性对照 |

### 6.3 PhaseModel（L3）

```python
class PhaseModel(Protocol):
    def on_pre_collision(self, tmp, state, model) -> None: ...
    def on_post_collision(self, state_out, model) -> None: ...
```

| 变体阶段 | PhaseModel |
|----------|------------|
| V0, V1（SC dam-break） | `ShanChenPhase` |
| V3（锐界面 dam-break） | `VofSharpPhase` |

### 6.4 SolidCouplingStrategy（L4）

```python
class SolidCouplingStrategy(Protocol):
    def rasterize(self, rigid_state, fluid_state, model) -> None: ...
    def apply(self, state_in, state_out, model) -> ForceWrench | None: ...
```

| 后端 | 策略 |
|------|------|
| 分布型 | 现有 `GridLbmRigidCoupling` + halfway bounce-back |
| 量化 HOME | **核 1** 流体推进（无 solid 分支）+ **核 2** 面并行固体校正（fp32 AtomicAdd Δ矩，再 re-quantize） |

对外仍暴露 `GridLbmRigidCoupling`；内部按 `backend.kind` 分发。

### 6.5 StepContext

```python
@dataclass
class StepContext:
    lattice: LatticeSpec
    model: LbmModel
    phase: PhaseModel | None
    solid: SolidCouplingStrategy | None
    gravity: tuple[float, float, float]
    profiler: Profiler | None = None
```

### 6.6 StepStats（G1 交付）

```python
@dataclass
class StepStats:
    ms_total: float
    ms_moments: float = 0.0
    ms_collision: float = 0.0
    ms_stream: float = 0.0
    ms_phase: float = 0.0
    ms_solid: float = 0.0
    ms_bc: float = 0.0
```

---

## 7. 变体矩阵（对比实验配置）

统一分辨率建议 **128³**（`N=128`, `DH=0.02`），与现有算例一致。

| ID | 名称 | Lattice | 表示 | 碰撞 | 界面 | FSI | 目标 |
|----|------|---------|------|------|------|-----|------|
| **V0** | `dist_d3q19_sc` | D3Q19 | `f_i` | BGK/TRT | Shan-Chen | BB + 动量交换 | G1 基线 |
| **V1** | `dist_d3q27_sc` | D3Q27 | `f_i` | BGK/TRT | Shan-Chen | 同 V0 | G2 |
| **V2** | `home_quant_d3q27` | D3Q27 | **矩 fp16** (5×uint32) | HOME + NOCM-MRT | 单相 | 解耦双核 + 混合精度 AtomicAdd | **G3** |
| **V3** | `home_free_quant_d3q27` | D3Q27 | **矩 fp16** | HOME | **VOF 锐界面** | 同 V2 | G4 |
| V2-ref | `home_fp32_ref_d3q27` | D3Q27 | 矩 fp32 (10 float) | HOME | 单相 | 解耦双核 | 开发对照，非主对比 |

**G3 与 G4 的关系**：G4 在 G3 的 `HomeQuantBackend` 上仅替换 `PhaseModel` 为 VOF；量化路径、双核 FSI、矩 pack 格式保持不变。

### 7.1 配置 YAML 示例

```yaml
# benchmarks/dambreak_two_spheres/V1.yaml
variant: V1
lattice: D3Q27
backend: dist
phase: shan_chen

grid:
  res: [128, 128, 128]
  cell_size: 0.02

collision:
  tau: 0.51
  lambda_trt: 0.001
  use_regularization: true
  omega_reg: 0.01

phase:
  G: -5.0
  psi_type: 1
  psi_ref: 1.0
  sc_boundary_psi: -1.0

gravity_lbm: [0.0, 0.0, -0.005]

dambreak:
  dam_x_frac: 0.25
  rho_water: 1.8
  rho_air: 0.1

spheres:
  radius: 0.24
  heavy_density: 0.3
  light_density: 0.1

fsi:
  feedback_mode: momentum_exchange
  feedback_force_scale: 4.0
  buoyancy_force_scale: 0.13   # 可选经验辅助，非核心对比指标
```

```yaml
# benchmarks/dambreak_two_spheres/V2.yaml  — 量化 HOME-LBM（G3）
variant: V2
lattice: D3Q27
backend: home_quant
phase: none                    # 单相；G4 阶段换 vof_sharp

quant:
  bits: 16
  rho_range: [0.8, 1.5]
  u_range: [-0.4, 0.4]
  rho_s_neq_range: [-0.1, 0.1]
  dither: true
  mixed_precision_fsi: true    # 固体校正 fp32 AtomicAdd

collision:
  model: nocm_mrt              # HOME 底层碰撞
  tau: 0.51

fsi:
  mode: home_solid_correction  # 解耦双核，非格点内 BB
  surface_voxel_mask: true
```

---

## 8. 评测指标

### 8.1 性能指标（PerfMetrics）

| 指标 | 定义 |
|------|------|
| `ms_per_lbm_step` | 单次 `solver.step` wall time |
| `ms_rasterize` / `ms_solid` | FSI 子阶段耗时 |
| `mlups` | `N³ / (ms_per_step / 1000)` |
| `bytes_per_cell` | 稳态 GPU 内存 / `N³` |
| `peak_gpu_mem_mb` | 设备峰值内存（可选） |
| `kernel_breakdown` | 各 kernel 占比 |

### 8.2 物理与稳定性指标（ValidationMetrics）

| 指标 | 用途 |
|------|------|
| `max_velocity`, `max_density` | 发散检测 |
| `water_mass` / `phi_mass` | 质量守恒（VOF） |
| `sphere_pos(t)`, `sphere_vel(t)` | 与 V0 参考轨迹对比 |
| `front_position_x(t)` | dam-break 前沿位置 |
| `submerged_fraction` | 浸没比例一致性 |
| `energy_drift` | 长时间能量漂移 |

### 8.3 输出约定

```text
benchmarks/dambreak_two_spheres/
├── V0_dist_d3q19_sc.json
├── V1_dist_d3q27_sc.json
├── V2_home_quant_d3q27.json
├── V3_home_free_quant_d3q27.json
├── V2_ref_home_fp32_d3q27.json   # 可选：量化误差对照
├── comparison.md              # 汇总表格
└── plots/                     # 轨迹、前沿、性能柱状图
```

---

## 9. 各目标实现要点

### 9.1 G1：精修 D3Q19 + 指标

**任务**

- [x] 引入 `core/lattice.py`，将 kernel 中硬编码 `19` 参数化（先 D3Q19 路径验证）
- [x] `solver.step` 返回 `StepStats`（`domain.step` 签名保持不变，stats 可选写入 ctx）
- [x] 抽取 `phases/shan_chen.py`，solver 只负责碰撞-迁移主链
- [x] 实现 `benchmark/metrics.py` 与 `benchmark/registry.py`
- [ ] 跑通 V0，建立数值参考（球轨迹、前沿、性能基线）

**验收**：V0 行为与当前 `fluid_grid_lbm_dambreak_two_spheres.py` 一致（允许浮点误差）；产出首份 JSON 基准报告。

### 9.2 G2：D3Q27 分布型

**任务**

- [ ] `backends/dist/d3q27/constants.py` + 27 路 collide-stream kernel
- [ ] `LbmModel.lattice: Literal["D3Q19", "D3Q27"] = "D3Q19"`
- [ ] D3Q27 下 TRT `lambda_trt` 重新标定（参考 HOME-LBM 论文推荐值）
- [ ] 注册 V1，与 V0 同 SC 参数对比

**验收**：V1 无 NaN/发散；完成 V0 vs V1 性能与物理对比报告。

### 9.3 G3：稳定性引导量化 HOME-LBM（主交付）

G3 实现量化论文的**完整流水线**，不是单独的「fp32 HOME + 后加量化补丁」。

#### 9.3.1 算法组件

| 组件 | 论文依据 | 实现位置 |
|------|----------|----------|
| 三阶 Hermite 分布重建 | HOME-LBM Eq. 6–7 | `core/hermite.py` |
| NOCM-MRT 矩空间碰撞 | 量化论文附录 C.3 / A | `core/moments.py` |
| **碰撞→迁移** 顺序 | 量化论文 §4.2 Algorithm 2 | `home_quant/solver.py` |
| VN 稳定性 → 矩范围 | §5.1–5.4 | `home_quant/stability.py` |
| ρS 仅存 neq 分量 | Eq. 31–32 | `home_quant/quant.py` |
| fp16 定点量化 + 空间抖动 | §5.4 | `home_quant/quant.py` |
| 双缓冲：quant 矩 + fp32 迁移矩 | §5.4 混合精度 | `home_quant/state.py` |
| 解耦双核 FSI | §4.2 Algorithm 1→2 | `boundary/solid_correction.py` |
| 固体表面体素化预标记 | §4.2 | `boundary/solid_voxel_mask.py` |
| 8×8×8 tile SoA 布局 | §4.2 | `home_quant/state.py` |

#### 9.3.2 矩存储格式（量化 HOME）

每格点 **5 个 uint32**（10 个 fp16 矩分量打包），对比 fp32 的 10 float：

| 分量 | 存储 | 量化位宽 | 范围（论文） |
|------|------|----------|--------------|
| ρ | 直接 | 16-bit | [0.8, 1.5] |
| ρu_x, ρu_y, ρu_z | 直接 | 16-bit | u∈[-0.4,0.4] → ρu∈[-0.6,0.6] |
| ρS_αα^neq, ρS_αβ^neq | **仅 neq** | 16-bit | [-0.1, 0.1]；平衡部分在线重建 |

**内存目标**（相对 fp32 HOME）：矩缓冲区约 **50%**；含 FSI 混合精度缓冲后总体约 **25%** 节省（论文 §5.4）。

#### 9.3.3 任务清单

- [ ] `core/hermite.py` + `core/moments.py`（HOME 基础，fp32 路径先通）
- [ ] `home_fp32_ref/`：全精度双核 HOME，作为 V2-ref 对照
- [ ] `home_quant/stability.py`：离线/启动时导出各矩 `[m_min, m_max]`（可先用论文保守默认值，再补 VN 自动化）
- [ ] `home_quant/quant.py`：normalize → dither → pack uint32；unpack → denormalize
- [ ] `home_quant/state.py`：quant 双缓冲 + fp32 迁移临时缓冲（固体校正用）
- [ ] `home_quant/solver.py`：流体更新核（无 solid 分支，warp 无发散）
- [ ] `boundary/solid_correction.py`：面并行 + fp32 AtomicAdd + re-quantize
- [ ] 注册 **V2** `home_quant_d3q27`

#### 9.3.4 验收

- 2D Taylor 涡 / 3D 绕球：V2 与 V2-ref 视觉与谱误差可接受（参考论文 Fig. 10）
- `bytes_per_cell` 达到 fp32 HOME 的 ~50%（矩区）或 ~75%（含 FSI 混合精度）
- dam-break + 双球 FSI：≥5000 步无 NaN；球轨迹与 V2-ref 偏差在约定阈值内
- 性能：相对 V2-ref 有 measurable 加速（论文报告量化+双核合计约 4×，其中双核 ~2.8×、量化余量）

### 9.4 G4：锐界面 HOME-FREE（基于量化 HOME）

**任务**

- [ ] `phases/vof_sharp.py`：`phi` 场、L/G/I 单元分类
- [ ] 液体区走 `HomeQuantBackend`；气体区跳过；界面区质量交换（HOME-FREE Eq. 9）
- [ ] 去掉气相 Shan-Chen；VOF 与 quant 矩场并行存储
- [ ] FSI：在量化 HOME 双核上适配 HOME-FREE 的双侧 bounce + 新生单元处理（§4.3）
- [ ] 注册 **V3** `home_free_quant_d3q27`

**验收**：VOF 质量守恒；界面比 V0/V1 更锐；完整 dam-break + 双球；内存/性能指标相对 V2 增量可控。

**已知方法极限**：近静置时贴壁 **O(1) 格** 高度残余属单相 FSLBM 固有离散/模型边界（无气相、无润湿时尤甚），不应靠删界面启发式强行抹平。详见 [lbm_home_fslbm_one_cell_limit_zh.md](lbm_home_fslbm_one_cell_limit_zh.md)。

---

## 10. 人员分工（5 人示例）

| 角色 | 负责 | 交付物 | 前置依赖 |
|------|------|--------|----------|
| **A – 架构/基线** | `core/`、`benchmark/`、V0 重构 | 接口冻结 + V0 基准报告 | 无 |
| **B – D3Q27** | `backends/dist/d3q27/` | V1 | A 的 `LatticeSpec` |
| **C – 量化 HOME** | `backends/moment/home_quant/`、`core/hermite`、`quant/`、`solid_correction` | **V2** + V2-ref | A |
| **D – 锐界面** | `phases/vof_sharp.py`（叠在 V2 上） | **V3** | C 的 V2 稳定 |
| **E – 集成/FSI** | benchmark harness、coupling 适配 | 统一 CLI + 四变体对比 | A；与各 track 联调 |

### 10.1 并行规则

1. **Week 1**：A 冻结 `LatticeSpec`、`FluidBackend`、`StepStats` 接口。
2. **Week 2 起**：B、C 并行，互不改对方目录。
3. **D** 在 C 的 **V2 量化路径** dam-break 稳定后启动（非 fp32 ref）。
4. **E** 先用 V0 搭 benchmark；每完成 Vi 即挂入 `registry.py`。

### 10.2 三人压缩版

| 人员 | 职责 |
|------|------|
| P1 | A + E（core、benchmark、集成） |
| P2 | B + 部分 G1（D3Q27 + D3Q19 参数化） |
| P3 | C + D（**量化 HOME** + VOF） |

---

## 11. 里程碑时间表（建议 8–10 周）

| 周次 | 里程碑 |
|------|--------|
| 1–2 | core 接口 + metrics + **V0 基线报告** |
| 2–4 | **V1** (D3Q27+SC) ∥ **V2-ref** (fp32 HOME) → **V2** (量化 HOME) |
| 4–6 | V2 dam-break + 双球 FSI 稳定；量化 vs fp32 误差报告 |
| 5–7 | **V3** (VOF on 量化 HOME) |
| 7–8 | V0–V3 四变体对比报告 + 文档更新 |

---

## 12. 统一 CLI（规划）

```bash
# 单变体
uv run python -m wanphys.benchmark.dambreak_two_spheres \
    --variant V1 --steps 5000 --profile

# 批量对比
uv run python -m wanphys.benchmark.dambreak_two_spheres \
    --compare V0,V1,V2,V3 --out benchmarks/dambreak_two_spheres/
```

现有可视化入口保持不变：

```bash
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_two_spheres --viewer gl
```

---

## 13. 风险与决策点

| 风险 | 缓解措施 |
|------|----------|
| HOME+SC 与 HOME-FREE 两套界面并存 | **最终 dam-break 以 VOF (V3) 为准**；SC 仅作 V0/V1 过渡对比 |
| D3Q27 内存约为 D3Q19 的 1.42× | 以 **V2 量化**（5×uint32/格）作内存对照，强制报告 `bytes_per_cell` |
| 量化误差导致湍流发散 | V2-ref fp32 对照；VN 边界保守；固体校正 fp32 AtomicAdd |
| `kernels.py` 2300+ 行难拆 | B 负责复制+参数化；A 只做最小 extract |
| 双球「经验浮力」干扰对比 | benchmark 层标记为 optional helper；核心对比用动量交换 |
| HOME 固体校正 vs 现有 `solid_phi` | E 实现 `SolidCouplingAdapter`；量化路径用 `solid_phi` 做体素预标记 |

### 13.1 Week-1 需冻结的决策

1. 锐界面最终方案：**VOF**，不使用 Shan-Chen 作为生产路径。
2. G3 默认交付：**D3Q27 + 三阶 Hermite + fp16 量化矩 + 解耦双核 FSI**（非 fp32 HOME）。
3. 性能对比默认分辨率：**128³**（CI 可另跑 64³ 快速冒烟）。

---

## 14. 与现有文档的关系

| 文档 | 关系 |
|------|------|
| [fluid_grid/ARCHITECTURE.md](../../wanphys/_src/fluid/fluid_grid/ARCHITECTURE.md) | 网格流体总框架；LBM 为其中一支 |
| [lbm_core_audit_zh.md](lbm_core_audit_zh.md) | 当前实现审计；G1 重构时对照修复项 |
| [lbm_rigid_coupling_guide_zh.md](lbm_rigid_coupling_guide_zh.md) | FSI 使用说明；E 负责保持兼容 |
| [domain_glossary_zh.md](domain_glossary_zh.md) | 术语表 |

---

## 15. 检查清单（Definition of Done）

### 项目级

- [ ] V0–V3 均可通过 `benchmark/registry.py` 一键运行
- [ ] 每变体产出 JSON metrics + 对比 Markdown
- [ ] `LbmDomain` / `GridLbmRigidCoupling` 对外 API 无 breaking change
- [ ] 128³ dam-break + 双球完整模拟无 NaN（≥ 5000 LBM 步）

### 各变体

| 变体 | DoD |
|------|-----|
| V0 | 指标基线 + 与现有 example 行为一致 |
| V1 | D3Q27 kernel 正确；V0 vs V1 报告 |
| V2 | **fp16 量化矩** + 解耦双核 FSI；vs V2-ref 误差可接受；dam-break+双球稳定 |
| V3 | VOF on V2；质量守恒；完整 FSI |

---

## 16. 附录：量化 HOME-LBM 双核算法（参考）

### 16.1 流体更新核（Algorithm 2，每格点一线程）

```text
1. Unpack fp16 moments → fp32 (ρ, ρu, ρS^neq); reconstruct ρS from u
2. Moment-space NOCM-MRT collision → (ρ*, (ρu)*, (ρS)*)
3. Reconstruct f*_i = h_i(ρ*, u*, S*)          # 3rd-order Hermite, register/local
4. Stream: f_i(x) ← f*_i(x - c_i)
5. Recover moments from f → fp32 migrate buffer
6. Re-quantize → write fp16 packed moments to global memory
```

### 16.2 固体校正核（独立 launch，每三角面一线程）

```text
For each triangle T:
  For lattice cells x in bbox(T) ∩ surface_voxel_mask:
    For each link direction i:
      If link x→x+c_i intersects T at p:
        u^p ← solid velocity at p; S^p ← Eq. 8
        f^p_i ← h_i(ρ_x, u^p, S^p)
        Δf_i ← f^p_i - f_i(x - c_i)
        Δ(ρ, ρu, ρS) ← project Δf
        atomicAdd to fp32 migrate buffer at x    # 混合精度，避免 quant 原子舍入
After all faces:
  Re-quantize fp32 buffer → fp16 global moments
```

### 16.3 与分布型 LBM 的关键差异

| 项 | 分布型 (V0/V1) | 量化 HOME (V2/V3) |
|----|----------------|-------------------|
| 持久状态 | f_i × 2 缓冲 | fp16 矩 × 2 + fp32 迁移缓冲 |
| 步进顺序 | 矩→碰撞→迁移 | **碰撞→重建→迁移** |
| FSI | 格点 kernel 内 BB | **独立固体校正核** |
| 固体附近 warp | 随 solid_phi 发散 | 流体核无分支；发散仅在校正核 |
| 内存/cell | 19×4×2 ≈ 152 B (f) | 5×4×2 ≈ 40 B (quant) + 迁移临时 |

---

*维护者：在接口或里程碑变更时更新本文档，并在 PR 描述中链接对应章节。*
