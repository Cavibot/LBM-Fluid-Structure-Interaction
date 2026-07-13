# HOME-FSLBM DamBreak 样例 — 阶段性实现方案 v2.1

> **状态**: 待审核 | **日期**: 2026-07-13
> **输出**: 本文档经审核通过后作为唯一执行依据，后续任何修改须先更新本文档。

---

## 目录

1. [总体架构](#1-总体架构)
2. [数据模型](#2-数据模型)
3. [核心算法](#3-核心算法)
4. [阶段 1：基础设施与分布重建](#4-阶段-1基础设施与分布重建)
5. [阶段 2：中心矩 MRT 碰撞与单核融合](#5-阶段-2中心矩-mrt-碰撞与单核融合)
6. [阶段 3：自由表面三步](#6-阶段-3自由表面三步)
7. [阶段 4：DamBreak 样例](#7-阶段-4dambreak-样例)
8. [阶段 5：可视化与回归验证](#8-阶段-5可视化与回归验证)
9. [阶段 6（后续扩展）：气泡 / 表面张力 / 涡粘性 / 溶解气体](#9-阶段-6后续扩展气泡--表面张力--涡粘性--溶解气体)
10. [风险与约束](#10-风险与约束)

---

## 1. 总体架构

### 1.1 模块位置（零侵入）

```
wanphys/_src/fluid/fluid_grid/home_fslbm/   ← 新建，不修改 lbm/ 下任何文件
├── __init__.py           # 公开导出
├── constants.py          # D3Q27 格子常量 + Hermite 系数预计算
├── model.py              # HomeFSLbmModel（静态配置 dataclass）
├── state.py              # HomeFSLbmState（GPU 时变状态 SoA 布局）
├── solver.py             # HomeFSLbmSolver（step 流水线 + MomSwap）
├── domain.py             # HomeFSLbmDomain（双缓冲 + Domain ABC）
├── kernels.py            # Warp GPU kernels（stream_collide + surface_1/2/3 等）
├── boundary.py           # 反弹边界、速度限制、计算辅助函数
└── utils.py              # VTK 输出、定量诊断

wanphys/examples/
└── home_fslbm_dambreak.py  ← DamBreak 样例入口
```

### 1.2 继承关系

```
FluidGridModelBase (@dataclass)
  └── HomeFSLbmModel

DomainState (Protocol)
  └── HomeFSLbmState

FluidGridSolverBase
  └── HomeFSLbmSolver

Domain (ABC)
  └── HomeFSLbmDomain
```

### 1.3 论文与参考代码依据

| 标识 | 来源 | 用途 |
|------|------|------|
| [P4] | Wang et al. 2025 — *Kinetic Free-Surface Flows and Foams with Sharp Interfaces* | 主算法框架 |
| [P1] | Li et al. 2023 — *HOME-LBM* | 矩编码基础、Taylor-Green 验证 |
| [P3] | Li & Desbrun 2023 — *Fluid-Solid Coupling* | 固体边界处理 |
| [REF] | `docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/` | C++/CUDA 参考实现 |

---

## 2. 数据模型

### 2.1 矩存储布局（对标 [REF] `mrFlow3D::fMom`）

所有矩以 SoA 形式存储在 1D `wp.array` 中。`total_num = nx * ny * nz`。

| 偏移 (乘以 total_num) | 符号 | 含义 | 论文参照 |
|---|---|---|---|
| 0 | `rho` | 密度 ρ | [P4] Eq.(18) |
| 1 | `u_x` | x 方向速度 | [P4] Eq.(19) |
| 2 | `u_y` | y 方向速度 | |
| 3 | `u_z` | z 方向速度 | |
| 4 | `S_xx` | xx 应力分量（已减 cs²） | [P4] Eq.(20-21) |
| 5 | `S_xy` | xy 应力分量 | |
| 6 | `S_xz` | xz 应力分量 | |
| 7 | `S_yy` | yy 应力分量（已减 cs²） | |
| 8 | `S_yz` | yz 应力分量 | |
| 9 | `S_zz` | zz 应力分量（已减 cs²） | |

> **注意**: `S_αα` 的存储值 = (Σ c_α² f_i) / ρ − cs²，即已减去格子声速平方。
> 分布重建时需先还原为真二阶矩 `pixx = rho * (S_xx + cs2)`。

**双缓冲**: 维护两个 1D `wp.array`（`fMom` / `fMomPost`），通过指针交换（对标 [REF] `MomSwap`）实现零拷贝翻转。

### 2.2 附加状态字段（对标 [REF] `mrFlow3D`）

| 字段 | Warp 类型 | 用途 |
|------|----------|------|
| `mass` | `wp.array3d(dtype=float)` | 每单元实际流体质量 |
| `massex` | `wp.array3d(dtype=float)` | 待分配给邻居的过剩质量 |
| `phi` | `wp.array3d(dtype=float)` | VOF 体积分数 ∈ [0, 1] |
| `flag` | `wp.array3d(dtype=wp.uint8)` | 单元类型位掩码标志（见 2.3） |
| `force_x/y/z` | `wp.array3d(dtype=float)` | 外力分量（= `rho * g`） |
| `disjoin_force` | `wp.array3d(dtype=float)` | 分离力（阶段 6 启用，当前初始化为 0） |

### 2.3 单元类型位掩码（对标 [REF] `mlFluid.h`）

```python
TYPE_F  = 0b00000001   # Fluid cell (phi == 1)
TYPE_I  = 0b00000010   # Interface cell (0 < phi < 1)
TYPE_G  = 0b00000100   # Gas cell (phi == 0)
TYPE_S  = 0b00001000   # Solid cell (domain boundary or obstacle)
TYPE_IF = 0b00010000   # Transition flag: Interface -> Fluid
TYPE_IG = 0b00100000   # Transition flag: Interface -> Gas
TYPE_GI = 0b01000000   # Transition flag: Gas -> Interface

TYPE_SU = TYPE_F | TYPE_I | TYPE_G   # Surface type mask
TYPE_BO = TYPE_S                      # Boundary type mask
```

### 2.4 D3Q27 格子常量（对标 [REF] `mrConstantParamsGpu3D.h`）

```python
# Lattice weights
W0 = 8.0 / 27.0       # Rest particle (0,0,0)
W1 = 2.0 / 27.0       # Straight: (±1,0,0) etc., 6 directions
W2 = 1.0 / 54.0       # Edge: (±1,±1,0) etc., 12 directions
W3 = 1.0 / 216.0      # Corner: (±1,±1,±1), 8 directions

CS2 = 1.0 / 3.0       # Speed of sound squared
INV_CS2 = 3.0
INV_CS4 = 9.0
INV_CS6 = 27.0
INV_2CS4 = 4.5
INV_2CS6 = 13.5

# Opposite direction index map (27 directions)
OPPOSITE = [0,  2, 1,  4, 3,  6, 5,
            8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
            20, 19, 22, 21, 24, 23, 26, 25]
```

### 2.5 `HomeFSLbmModel` — 静态配置

```python
@dataclass
class HomeFSLbmModel(FluidGridModelBase):
    """Static configuration for HOME-FSLBM free-surface fluid simulation."""

    # Relaxation / viscosity
    tau: float = 0.55
    """BGK relaxation time.  nu = cs2 * (tau - 0.5) with cs2 = 1/3."""

    # Body force (lattice units)
    gravity_x: float = 0.0
    gravity_y: float = 0.0
    gravity_z: float = 0.0

    # Domain boundary conditions (per face: -x,+x,-y,+y,-z,+z)
    bc_types: tuple = ("bounce_back",) * 6
    """Boundary condition type per face."""

    # Maximum velocity clamp
    max_velocity: float = 0.4
    """Velocity magnitude clamp per [REF] normalizing_clamp."""
```

---

## 3. 核心算法

### 3.1 分布重建（对标 [REF] `mlCalDistributionFourthOrderD3Q27AtIndex`）

**论文参照**: [P4] Eq.(16-17) — 三阶 Hermite 展开。

```python
@wp.func
def reconstruct_fi_at_index(
    rho: float,
    ux: float, uy: float, uz: float,
    pixx: float, pixy: float, pixz: float,
    piyy: float, piyz: float, pizz: float,
    i: int,
) -> float:
    """Reconstruct f_i from 10 moments using 3rd-order Hermite expansion.

    Args:
        rho: Density.
        ux, uy, uz: Velocity components.
        pixx ... pizz: TRUE second-order moments
            (i.e. pixx = rho * (S_xx + CS2), NOT the stored stress S_xx).
        i: Lattice direction index (0..26).

    Returns:
        Reconstructed distribution value f_i.

    Reference:
        [REF] mrUtilFuncGpu3D.h:153-200
        [P4] Eq.(16): f_i = rho * w_i * [1 + (c_i·u)/cs2 + H^{[2]}(c_i):S/(2*cs4)
             + sum_{abg} H^{[3]}_{abg}(c_i) * T_{abg} / (2*cs6)]
        [P4] Eq.(17): T_{abg} = S_{ab}*u_g + S_{ag}*u_b + S_{bg}*u_a - 2*u_a*u_b*u_g

    Note:
        Uses the "bias subtraction" trick from [REF]:
        f_i = reconstruction - w_i, then f_i + w_i is used in moment summation.
        This improves floating-point precision for small non-equilibrium parts.
    """
    # Zero-th moment
    A0 = rho

    # First-order moments (momentum)
    Ax = ux * A0
    Ay = uy * A0
    Az = uz * A0

    # Second-order moments (TRUE, not stress)
    Axx = rho * pixx
    Ayy = rho * piyy
    Azz = rho * pizz
    Axy = rho * pixy
    Axz = rho * pixz
    Ayz = rho * piyz

    # Third-order moments via [P4] Eq.(17)
    Axxy = -2.0 * rho * uy * ux * ux + 2.0 * Axy * ux + Axx * uy
    Axyy = -2.0 * rho * ux * uy * uy + 2.0 * Axy * uy + Ayy * ux
    Axxz = -2.0 * rho * uz * ux * ux + 2.0 * Axz * ux + Axx * uz
    Axzz = -2.0 * rho * ux * uz * uz + 2.0 * Axz * uz + Azz * ux
    Ayyz = -2.0 * rho * uz * uy * uy + 2.0 * Ayz * uy + Ayy * uz
    Ayzz = -2.0 * rho * uy * uz * uz + 2.0 * Ayz * uz + Azz * uy
    Axyz = Axz * uy + Ayz * ux + Axy * uz - 2.0 * rho * ux * uy * uz

    # Pre-scaled quantities
    Ax3 = Ax * 3.0
    Ay3 = Ay * 3.0
    Az3 = Az * 3.0
    Axx3 = 3.0 * Axx
    Ayy3 = 3.0 * Ayy
    Azz3 = 3.0 * Azz
    Axy9 = 9.0 * Axy
    Axz9 = 9.0 * Axz
    Ayz9 = 9.0 * Ayz
    Axxy9 = 9.0 * Axxy
    Axyy9 = 9.0 * Axyy
    Axxz9 = 9.0 * Axxz
    Axzz9 = 9.0 * Axzz
    Ayyz9 = 9.0 * Ayyz
    Ayzz9 = 9.0 * Ayzz
    Axyz9 = 9.0 * Axyz

    # Hermite evaluation per direction (27 explicit cases)
    # The full 27-case switch is pre-generated.
    # Below is the general form; implementation uses explicit case-per-direction
    # to match [REF] exactly and enable compiler optimization.
    cx, cy, cz = C[i]
    cx2, cy2, cz2 = cx * cx, cy * cy, cz * cz

    # H^{[2]}_{ab} = c_a * c_b - CS2 * delta_{ab}
    Hxx = cx2 - CS2
    Hyy = cy2 - CS2
    Hzz = cz2 - CS2
    Hxy = cx * cy
    Hxz = cx * cz
    Hyz = cy * cz

    # Contribution weights from [REF]
    # f_i = w_i * [A0 + Ax3*cx + Ay3*cy + Az3*cz
    #      + (9/2)*Axx*Hxx + 9*Axy*Hxy + ... (6 terms for 2nd order)
    #      + (9/2)*Axxy*Hxx*cy + ... (7 terms for 3rd order)]
    result = A0
    result += Ax3 * cx + Ay3 * cy + Az3 * cz
    # 2nd-order terms (factor 9/2 for diagonal, 9 for off-diagonal)
    result += 4.5 * (Axx3 * Hxx / 3.0)  # simplified: Axx3 * (9/6) * Hxx
    # ... (full 27-case precomputed in constants.py)

    # Bias subtraction: f_i -= w_i (for precision, added back before use)
    result -= LATTICE_W[i]
    return result
```

### 3.2 中心矩 MRT 碰撞（对标 [REF] `mlGetPIAfterCollision`）

**论文参照**: [P4] Eq.(18-21) 的推广形式（中心矩而非简化 BGK），[P1] 中心矩 MRT。

```python
@wp.func
def mrt_collide(
    rho_star: float,
    ux_star: float, uy_star: float, uz_star: float,
    pixx_star: float, pixy_star: float, pixz_star: float,
    piyy_star: float, piyz_star: float, pizz_star: float,
    Fx: float, Fy: float, Fz: float,
    omega: float,
):
    """Central-moment MRT collision in moment space.

    Args:
        rho_star, u*_star, pixx_star...: Post-stream macroscopic moments.
        Fx, Fy, Fz: Force components (already multiplied by rho).
        omega: Relaxation frequency (1/tau).

    Returns:
        Tuple of 6 post-collision TRUE second-order moments
        (pixx, pixy, pixz, piyy, piyz, pizz).

    Reference:
        [REF] mrUtilFuncGpu3D.h (mlGetPIAfterCollision)
        [P4] Eq.(18-21): Moment-space collision with force terms.
    """
    inv_rho = 1.0 / rho_star

    # Convert TRUE moments to central moments
    # Central moment = Σ (c_a - u_a)(c_b - u_b) f_i
    pixx_cm = pixx_star * inv_rho - ux_star * ux_star
    pixy_cm = pixy_star * inv_rho - ux_star * uy_star
    pixz_cm = pixz_star * inv_rho - ux_star * uz_star
    piyy_cm = piyy_star * inv_rho - uy_star * uy_star
    piyz_cm = piyz_star * inv_rho - uy_star * uz_star
    pizz_cm = pizz_star * inv_rho - uz_star * uz_star

    # Equilibrium central moments: cs2 on diagonal, 0 off-diagonal
    pixx_eq = CS2
    piyy_eq = CS2
    pizz_eq = CS2
    pixy_eq = 0.0
    pixz_eq = 0.0
    piyz_eq = 0.0

    # MRT relaxation: omega-weighted blend toward equilibrium
    om1 = 1.0 - omega
    pixx_cm_post = om1 * pixx_cm + omega * pixx_eq
    pixy_cm_post = om1 * pixy_cm + omega * pixy_eq
    pixz_cm_post = om1 * pixz_cm + omega * pixz_eq
    piyy_cm_post = om1 * piyy_cm + omega * piyy_eq
    piyz_cm_post = om1 * piyz_cm + omega * piyz_eq
    pizz_cm_post = om1 * pizz_cm + omega * pizz_eq

    # Force contribution in central-moment space
    # (2*tau-1)/(2*tau) * (F_a * u_b + F_b * u_a) / rho
    coeff_f = (2.0 * tau - 1.0) / (2.0 * tau) * inv_rho

    # Convert back to TRUE moments + add force terms
    pixx_post = rho_star * (pixx_cm_post + ux_star * ux_star) \
                + coeff_f * 2.0 * Fx * ux_star
    pixy_post = rho_star * (pixy_cm_post + ux_star * uy_star) \
                + coeff_f * (Fx * uy_star + Fy * ux_star)
    pixz_post = rho_star * (pixz_cm_post + ux_star * uz_star) \
                + coeff_f * (Fx * uz_star + Fz * ux_star)
    piyy_post = rho_star * (piyy_cm_post + uy_star * uy_star) \
                + coeff_f * 2.0 * Fy * uy_star
    piyz_post = rho_star * (piyz_cm_post + uy_star * uz_star) \
                + coeff_f * (Fy * uz_star + Fz * uy_star)
    pizz_post = rho_star * (pizz_cm_post + uz_star * uz_star) \
                + coeff_f * 2.0 * Fz * uz_star

    return pixx_post, pixy_post, pixz_post, piyy_post, piyz_post, pizz_post
```

### 3.3 单步求解流水线（对标 [REF] `mrSolver3DGpu`）

```
HomeFSLbmSolver.step(state_in, state_out, dt):

  # ---- Step 1: Stream + Collide (fused kernel, [REF] stream_collide_bvh) ----
  stream_collide_kernel(
      fMom, fMomPost, mass, massex, phi, flag,
      force_x, force_y, force_z, tau, ...)

  # ---- Step 2: Free-surface step 1 — mark transitions ([REF] surface_1) ----
  surface_1_kernel(flag, ...)
      # G -> GI, prevent I -> G degradation

  # ---- Step 3: Free-surface step 2 — init new interface cells ([REF] surface_2) ----
  surface_2_kernel(fMomPost, flag, ...)
      # Average neighbor moments, write to GI cells

  # ---- Step 4: Free-surface step 3 — mass exchange + reflag ([REF] surface_3) ----
  surface_3_kernel(mass, massex, phi, flag, fMomPost, ...)
      # Mass redistribution, phi update, IF/IG/GI -> F/I/G conversion

  # ---- Step 5: Buffer swap ([REF] MomSwap) ----
  MomSwap(fMom, fMomPost)
```

### 3.4 自由表面三步详解

**surface_1** ([REF] 444–477):
- 遍历所有 `TYPE_I` 单元
- 邻居为 `TYPE_G` → 标记 `TYPE_GI`（gas → interface）
- 邻居即将变为 Fluid（`TYPE_IF`）→ 该邻居保持为 Interface 防止流体-气体直接相邻

**surface_2** ([REF] 479–601):
- 遍历所有 `TYPE_GI` 单元
- 从所有 Fluid / Interface 邻居平均 `ρ`, `u`
- 从平均 `(ρ, u)` 计算 27 个平衡态分布 `feq`
- 从 `feq` 计算二阶矩：`pixx = Σ c_x² * feq / ρ`
- 写回 `fMomPost`（新激活单元获得合理的初始矩）

**surface_3** ([REF] 604–701):
- 遍历所有非固体单元，按类型处理质量交换：
  - `TYPE_IF`: flag → `TYPE_F`, `mass = ρ`, `massex = mass_old − ρ`, `ϕ = 1`
  - `TYPE_IG`: flag → `TYPE_G`, `mass = 0`, `massex = mass_old`, `ϕ = 0`
  - `TYPE_GI`: flag → `TYPE_I`, `mass = clamp(mass_old, 0, ρ)`, `ϕ = calculate_phi(ρ, mass, TYPE_I)`
  - `TYPE_F`: `mass = ρ`, `massex = mass_old − ρ`, `ϕ = 1`
  - `TYPE_I`: `mass = clamp(mass_old, 0, ρ)`, `massex = 溢出量`, `ϕ = calculate_phi(ρ, mass, TYPE_I)`
  - `TYPE_G`: `mass = 0`, `massex = mass_old`, `ϕ = 0`
- 将 `massex` 均分到所有 Fluid/Interface 邻居
- 检测转换条件：
  - `mass > ρ` 且无 GAS 邻居 → 标记 `TYPE_IF`
  - `mass < 0` 且无 Fluid 邻居 → 标记 `TYPE_IG`

---

## 4. 阶段 1：基础设施与分布重建

**目标**: 建立模块骨架，实现 D3Q27 格子常量和三阶 Hermite 分布重建，通过单元测试精确验证。

**工期**: 3–4 天

### 4.1 产出文件

| 文件 | 内容 |
|------|------|
| `constants.py` | D3Q27 格子常量 + 27 个方向的 Hermite 系数预计算表 |
| `model.py` | `HomeFSLbmModel` dataclass |
| `state.py` | `HomeFSLbmState`（10 moments SoA + mass/massex/phi/disjoin_force/flag） |
| `domain.py` | `HomeFSLbmDomain`（model + solver + 双缓冲 state） |
| `solver.py` | `HomeFSLbmSolver` 骨架（`step` 流水线仅含 `MomSwap`、不含碰撞内核） |
| `kernels.py` | `reconstruct_fi_at_index`（Warp func）+ `compute_equilibrium_kernel` |
| `boundary.py` | `bounce_back_refill_kernel`（单方向反弹） |
| `__init__.py` | 模块公开导出 |
| `tests/test_home_fslbm_reconstruct.py` | 分布重建精度 + 反弹边界单元级测试 |
| `tests/test_home_fslbm_e2e_periodic.py` | 端到端周期边界小网格可运行回归测试 |

### 4.2 严格验收标准

#### A. 数学正确性（原标准，不动）

1. **D3Q27 权重归一化**: `sum(W) = 1.0`，误差 < `1e-15`
2. **反向索引**: 对任意方向 `i`，`C[i] + C[OPPOSITE[i]] = (0,0,0)`
3. **静止平衡态**: `ρ = 1, u = (0,0,0), S = 0` → `f_i = w_i`，最大误差 < `1e-12`
4. **匀速流**: `ρ = 1, u = (0.1, 0, 0)`，矩守恒重建后 `|Σ c_i*f_i − u| < 1e-10`
5. **三阶 Hermite 验核**: 对已知解析流场，从全分布函数计算的三阶矩与重建的三阶矩 `L2` 误差 < `1e-6`
6. **Model 参数校验**: `device` 解析正确、`tau > 0.5`、`bc_types` 长度为 6
7. **State 内存**: `12 × nx × ny × nz × 4` bytes（不含 Warp 开销），10 个矩 + 5 个附加字段全部归零

#### B. 端到端可运行（新增）

8. **最小周期网格运行**: `(8, 8, 8)` 网格、`ρ = 1, u = 0` 初始化、周期边界，运行 100 步无崩溃、无 NaN、`max|u| < 1e-10`
9. **Domain 双缓冲一致**: 连续调用 `domain.step()` 后 `state_in` 和 `state_out` 正确翻转（奇偶步验证）
10. **create_state 可重复调用**: 两次调用 `create_state()` 返回两个独立 State，互不影响
11. **Solver 骨架正确接入**: `solver.step()` 被 `domain.step()` 调用且不抛出异常

#### C. 边界处理正确性（新增）

12. **反弹边界单元级验证 — 静止壁面**: 在固体邻居方向 `d`，`f_hn[d] = f_eq(rho_cur, 0, 0, 0, d)`，误差 < `1e-12`
13. **反弹边界单元级验证 — 反向一致性**: `f_hn[d]`（来自固体）= `f_on[OPPOSITE[d]]`（流出到固体），误差 < `1e-12`（无穿透验证）
14. **域边界 TYPE_S 标记正确**: `flag` 数组在域边界 6 个面上均为 `TYPE_S`，内部均不为 `TYPE_S`
15. **无中文注释**: 所有代码注释和 docstring 为英文

---

## 5. 阶段 2：中心矩 MRT 碰撞与单核融合

**目标**: 实现 `stream_collide_kernel`（单核融合迁移+碰撞），通过 Taylor-Green 涡旋验证精度。

**工期**: 4–5 天

### 5.1 产出文件

| 文件 | 内容 |
|------|------|
| `kernels.py` | `stream_collide_kernel`（重建→迁移→矩计算→MRT碰撞→写回，单核） |
| `solver.py` | `HomeFSLbmSolver`（完整 step 流水线 + `MomSwap` + 边界条件路由） |
| `boundary.py` | `bounce_back` 处理、`normalizing_clamp` 速度限制、周期边界 wrap helper |
| `tests/test_home_fslbm_collision.py` | 碰撞守恒 + 静止稳定性 + 力场终端速度验证 |
| `tests/test_home_fslbm_periodic_tgv.py` | 周期边界 Taylor-Green 涡旋精度验证 |
| `tests/test_home_fslbm_poiseuille.py` | 外力驱动 Poiseuille 流（周期+反弹混合边界） |
| `tests/test_home_fslbm_cavity_decay.py` | 全反弹封闭腔体衰减流（边界行为正确性） |

### 5.2 `stream_collide_kernel` 行内流程

```
For each cell (i,j,k):
  1. Early exit if TYPE_S or TYPE_G
  2. For each direction d = 0..26:
     a. Read neighbor cell index
     b. If neighbor is TYPE_S: f_hn[d] = feq(rho_cur, 0,0,0, d)  (bounce-back)
        Else: f_hn[d] = reconstruct_fi_at_index(neighbor_moments, d) - W[d]
     c. f_on[d] = reconstruct_fi_at_index(current_moments, d) - W[d]
  3. If TYPE_I: handle gas-side closure, mass exchange via neighbor phi weights
  4. Compute post-stream moments rho*, u*, pixx*...piyz* from f_hn + W
  5. Velocity clamp: if |u*| > 0.4: u* *= 0.4/|u*|
  6. mrt_collide(...) -> post-collision TRUE moments
  7. Write back to fMomPost (10 moments, S_aa -= CS2)
```

### 5.3 严格验收标准

#### A. 守恒与稳定性（原标准，不动）

1. **周期边界质量守恒**: `Σρ` 每步变化 < `1e-14`
2. **无外力动量守恒**: `Σρu` 每步变化 < `1e-14`
3. **静止稳定性**: `ρ = 1, u = 0`，全反弹边界，运行 1000 步后 `max|u| < 1e-10`
4. **Taylor-Green 涡旋**: `Re = 2000, 128³`，周期边界，动能衰减曲线与 [P1] Fig.7 参考解的最大偏差 < 3%
5. **球体绕流**: `Re = 1000`，阻力系数 `Cd` 与经验公式偏差 < 5%
6. **重力自由落体**: `g = (0, -1e-4, 0), τ = 0.55`，周期边界，`t` 步后 `|u_y| ≈ g * τ * t`，误差 < 0.5%
7. **速度限制**: 任意单步中 `max|u| ≤ 0.4` 始终成立

#### B. 边界行为正确性（新增）

8. **反弹壁面静止流体 — 无穿透**: 全反弹封闭腔体 `(32, 32, 32)`，`ρ = 1, u = 0` 初始化，运行 500 步，壁面相邻单元速度始终为零（`|u_wall| < 1e-10`），边界单元 ρ 与内部一致（无边界密度漂移，偏差 < 1e-6）
9. **Poiseuille 流 — 周期+反弹混合边界**: 外力 `g_x = 1e-5` 驱动，y/z 方向周期边界，x 方向两端反弹壁面，`(64, 32, 32)` 网格。稳态后：
   - yz 平面速度剖面为抛物型 `u_x(y,z) = U_max * (1 - (2y/L_y)²) * (1 - (2z/L_z)²)`
   - 与解析解逐点 L2 误差 < 2%
   - 中心线速度 `U_max` 与理论值 `g_x * L_y² / (8 * ν)` 偏差 < 3%
   - 壁面处无滑移：`|u_wall| < 1e-8`
10. **Courant 条件**: Poiseuille 流稳态下 `max|u| * dt / dx < 0.4` 始终成立
11. **全反弹腔体衰减流 — 长时间闭域行为**: `(32, 32, 32)` 全反弹边界、初始涡量场（双剪切层），运行 2000 步：
    - 动能单调递减（无虚假增长）
    - 最终 `max|u| → 0`（< 1e-6，2000 步内达到）
    - 全程无 NaN/Inf
12. **反弹无穿透 — 逐方向验证**: 对每个与固体相邻的方向 `d`，`f_hn[d]`（来自固体）= `f_on[OPPOSITE[d]]`（流出到固体），误差 < `1e-10`

#### C. 端到端回归（新增）

13. **阶段 1 回归**: 所有阶段 1 验收标准在阶段 2 完成后仍然全部通过
14. **周期边界 TGV 确定性**: 同一参数两次运行 `T_MAX` 步后 `max|Δu| < 1e-10`
15. **无中文注释**: 所有代码注释和 docstring 为英文

---

## 6. 阶段 3：自由表面三步

**目标**: 实现 `surface_1/2/3` 三步，支持 VOF 追踪和自由表面边界，通过静止水柱和质量守恒验证。

**工期**: 3–4 天

### 6.1 产出文件

| 文件 | 内容 |
|------|------|
| `kernels.py` | `surface_1_kernel`, `surface_2_kernel`, `surface_3_kernel`, `calculate_disjoint_kernel` |
| `boundary.py` | `calculate_phi`, `calculate_curvature` (skeleton, σ=0 for now) |
| `tests/test_home_fslbm_free_surface.py` | 自由表面单元测试 |

### 6.2 严格验收标准

1. **VOF 有界性**: `0 ≤ ϕ ≤ 1` 全程不违反
2. **单元类型拓扑**: `TYPE_F` 和 `TYPE_G` 在任何时刻不直接相邻
3. **静止水柱**: 矩形水柱在无重力下 `ϕ` 不变 ≥ 500 步
4. **质量守恒**: 封闭域内 `Σmass` 每步变化 < `1e-6`%
5. **排水测试**: 底部开口容器，排水速率与 Torricelli 定律 `v = sqrt(2gh)` 偏差 < 10%
6. **新生单元初始化**: `TYPE_GI` 单元初始化后的 `ρ` 与邻居平均值偏差 < 1%
7. **界面过渡平稳**: 每步 `IF/IG/GI` 转换数 < 总单元数的 5%
8. **无中文注释**

---

## 7. 阶段 4：DamBreak 样例

**目标**: 完整可运行的 dam break 样例脚本，正确初始化、运行、输出结果。

**工期**: 2–3 天

### 7.1 产出文件

| 文件 | 内容 |
|------|------|
| `wanphys/examples/home_fslbm_dambreak.py` | DamBreak 样例入口 |

### 7.2 样例配置

```python
# ============================================================
# DamBreak configuration — [P4] Fig.8, [REF] testMrLBM2D.cpp
# ============================================================
GRID_RES = (200, 100, 50)      # nx, ny, nz
TAU = 0.55                      # Relaxation time
GRAVITY = (0.0, -5e-5, 0.0)    # Lattice-unit gravity

DAM_X = 50                      # Dam wall x-position
WATER_W = 50                    # Initial water column width
WATER_H = 80                    # Initial water column height
WATER_D = 50                    # Initial water column depth

T_MAX = 8000                    # Total simulation steps
OUTPUT_INTERVAL = 50            # Steps between VTK dumps
```

### 7.3 初始化逻辑

```
1. All cells: flag = TYPE_G, phi = 0, mass = 0, rho = 1.0, u = 0, S = 0
2. Water column region (x < DAM_X, y < WATER_H): flag = TYPE_F, phi = 1, mass = rho
3. 1-cell border around water column: flag = TYPE_I, phi = 0.5, mass = 0.5 * rho
4. Domain faces: flag |= TYPE_S (based on bc_types)
5. All fMom entries: rho = 1.0, u = 0, S_aa = 0
```

### 7.4 严格验收标准

1. **脚本正常运行**: 无异常退出，无段错误
2. **初始化验证**: `Σϕ` 初始值 = `WATER_W × WATER_H × WATER_D`，误差 < `1e-6`
3. **溃坝前缘速度**: 前缘传播速度 ≈ `sqrt(g × WATER_H)`（浅水波理论），误差 < 15%
4. **涌浪自相似**: `x_front(t) ∝ sqrt(t)`（log-log 斜率 ≈ 0.5 ± 0.1）
5. **质量守恒**: 全程 `Σmass` 变化率 < 0.01%
6. **非负密度**: `ρ_min ≥ 0` 始终成立
7. **速度有界**: `max|u| ≤ 0.4`（clamp 保护）
8. **无 NaN/Inf**: 8000 步内数值完全稳定
9. **最终稳态**: 左右水面等高，误差 < 2% 域高度
10. **VTK 可读**: ParaView 可直接打开 `.vti` 文件
11. **无中文注释**

---

## 8. 阶段 5：可视化与回归验证

**目标**: VTK/NPY 输出 + 定量诊断工具 + 回归测试。

**工期**: 1–2 天

### 8.1 产出文件

| 文件 | 内容 |
|------|------|
| `utils.py` | `dump_vtk`, `compute_front_position`, `compute_total_mass`, `compute_total_energy` |
| `tests/test_home_fslbm_dambreak_regression.py` | 回归测试（确定性验证） |

### 8.2 严格验收标准

1. **VTK 输出可读**: ParaView 默认打开，所有字段（rho, u, S_xx, phi, flag）可视化
2. **前缘检测**: `compute_front_position()` 返回 ϕ=0.5 等值面的最大 x 坐标
3. **能量诊断**: `compute_total_energy()` 输出 `E_k + E_p`，单调递减（粘性耗散）
4. **确定性回归**: 相同参数两次运行，ϕ 场 `max|Δϕ| < 1e-10`
5. **无中文注释**

---

## 9. 阶段 6（后续扩展）：气泡 / 表面张力 / 涡粘性 / 溶解气体

> **触发条件**: 阶段 4 DamBreak 样例通过全部验收标准后方可启动。
> **工期**: 每子项 2–4 天，可独立并行。

### 9.1 子阶段 6a：表面张力

- **论文参照**: [P4] Sec 4.4, Eq.(22-25); [REF] `calculate_curvature` + `stream_collide_bvh` 中 `rho_laplace` 项
- **产出**: `boundary.py` 中 `calculate_curvature` 完整实现（曲率 → Laplace 压力）
- **验收**: 静态液滴 Laplace 压力 `Δp = σκ`，直径与 `Δp` 关系偏差 < 5%

### 9.2 子阶段 6b：涡粘性湍流模型

- **论文参照**: [P4] Sec 4.2; [REF] `stream_collide_bvh:1021-1023` — `ν_e = 4 × ||S||_F`
- **产出**: 在 `stream_collide_kernel` 中按局部 Frobenius 范数动态调整 `omega`
- **验收**: 高 Re 溃坝（`τ = 0.5005`）下无振铃伪影，速度场光滑

### 9.3 子阶段 6c：气泡追踪与压力模型

- **论文参照**: [P4] Sec 4.2, Eq.(22-25); [REF] CCL + `tag_matrix` + `bubble.rho/volume/init_volume`
- **产出**: CCL 连通分量标记、气泡体积追踪、理想气体压力更新
- **验收**: 单气泡上升体积守恒（变化 < 0.1%）；两气泡合并前后总体积一致

### 9.4 子阶段 6d：溶解气体输运 (D3Q7 gMom)

- **论文参照**: [P4] Sec 4.4; [REF] `g_reconstruction` + `g_stream_collide` + `gMom[7]` D3Q7
- **产出**: `g_stream_collide_kernel` + `bubble_volume_g_update_kernel`
- **验收**: 恒定浓度场不衰减；气泡在过饱和液体中体积线性增长

> **说明**: 子阶段 6a–6d 互不依赖，可任意顺序实施。每项均须通过对应验收标准后方可合入。

---

## 10. 风险与约束

### 10.1 严格执行规则

1. **零侵入**: 禁止修改 `wanphys/_src/fluid/fluid_grid/lbm/` 下任何文件
2. **禁止降级**: 必须使用 D3Q27 + 三阶 Hermite 展开 + 中心矩 MRT 碰撞 + 单核融合 + 完整自由表面三步
3. **英文注释**: 所有代码注释和 docstring 为英文
4. **公式可追溯**: 所有数据模型和伪代码标注论文公式编号或参考代码行号
5. **禁止冒烟通过**: 每个阶段必须通过**全部**列出的严格验收标准

### 10.2 技术风险

| 风险 | 缓解 |
|------|------|
| Warp 不支持 `wp.uint8` 位掩码 | 降级为 `wp.int32`，使用整数比较替代位运算 |
| 单核融合寄存器压力过大 | 拆分为 `reconstruct_stream_kernel` + `collide_kernel`（2 核，非降级） |
| D3Q27 27 路分支 warp divergence | 按方向类型分 4 组处理（rest/straight/edge/corner），减少分支 |
| Warp kernel 不支持递归 `wp.func` 循环 | 将 27 方向展开为显式赋值（对标 [REF] 预计算法） |

---

> **审批记录**
>
> | 日期 | 审批人 | 意见 |
> |------|--------|------|
> | 2026-07-13 | （待审核） | — |
