# HOME-FSLBM 阶段 2 自由表面测试补全计划

> **日期**: 2026-07-21
> **范围**: 自由表面子系统（surface_1/2/3 kernel + Phase C/D 曲率/质量交换）的单元测试、金标准回归测试、集成测试
> **前置**: [阶段 1 状态报告](phase1_status_zh.md)（17/17 通过）、[阶段 2 设计方案](phase2_design_zh.md)

---

## 目录

1. [当前测试覆盖现状](#1-当前测试覆盖现状)
2. [金标准数据格式规范](#2-金标准数据格式规范)
3. [单元测试补全](#3-单元测试补全)
4. [金标准回归测试](#4-金标准回归测试)
5. [集成测试](#5-集成测试)
6. [金标准生成器扩展](#6-金标准生成器扩展)
7. [文件清单与执行顺序](#7-文件清单与执行顺序)
8. [风险与注意事项](#8-风险与注意事项)
9. [成功门禁](#9-成功门禁)

---

## 1. 当前测试覆盖现状

### 1.1 已实现代码

| 模块 | 文件 | 行数 | 覆盖内容 |
|------|------|------|---------|
| 表面辅助函数 | `kernels_fluid.py` | ~300 行 | `calculate_phi`, `calculate_normal`(内联), `calculate_curvature_from_grid`, `_lu_solve_5x5_scalar`, `plic_cube`/`plic_cube_reduced` |
| surface_1 | `kernels_surface.py` | ~55 行 | 标记传播（IF→阻止 IG、IF→GI 候选） |
| surface_2 | `kernels_surface.py` | ~170 行 | GI 初始化（ρ/u 平均→HOME 矩→D3Q7 平衡态）+ IG 邻居转换 |
| surface_3 | `kernels_surface.py` | ~160 行 | VOF 质量重分配、标记过渡（IF→F/IG→G/GI→I）、余量分发、delta_phi/delta_g 累积 |
| Phase C/D | `kernels_fluid.py` | ~250 行 | PLIC 曲率、气泡气体压力、表面张力调制、Guo 力修正、气侧边界填充、质量交换累积 |
| Solver 集成 | `solver.py` | ~30 行新增 | stream_collide_bvh → surface_1 → surface_2 → surface_3 → swap_moments |

### 1.2 已实现测试（`test_kernels_surface.py`，9 个方法）

| 测试类 | 方法数 | 覆盖内容 | 状态 |
|--------|--------|---------|------|
| `TestCalculatePhi` | 3 | TYPE_F→1.0, TYPE_I→mass/rho, TYPE_G→0.0 | ✅ 通过 |
| `TestPlicCube` | 3 | V=0.5/n=(1,0,0), V=0.1/n=(1,0,0), V=0.5/n=(1,1,1) | ✅ 通过 |
| `TestSurface1` | 1 | TYPE_IF 邻居 TYPE_G → TYPE_GI（仅单向） | ✅ 通过 |
| `TestSurface3` | 2 | TYPE_F mass→rho+massex 均分, TYPE_I mass 钳制 | ✅ 通过 |

### 1.3 覆盖率缺口

| 模块 | 代码行 | 测试方法 | 缺口 |
|------|--------|---------|------|
| surface_1 IG→I 传播 | 6 行 | 0 | 完全缺失 |
| surface_2 GI 初始化 + IG 分支 | 170 行 | 0 | 完全缺失 |
| surface_3 边界情况 | 160 行 | 2 | 缺少 4 个关键边界测试 |
| `calculate_normal` | ~30 行 | 0 | 完全缺失（内联在 `calculate_curvature_from_grid` 中） |
| `calculate_curvature` | ~180 行 | 0 | 完全缺失 |
| `_lu_solve_5x5_scalar` | ~100 行 | 0 | 完全缺失 |
| Phase C/D 集成 | 250 行 | 0 | 完全缺失 |
| Solver 调用顺序 | 30 行 | 0 | 完全缺失 |
| 端到端金标准回归 | — | 0 | 完全缺失 |

---

## 2. 金标准数据格式规范

### 2.1 格式选择

与阶段 1 保持一致，使用 **文本格式（`.txt`）、每行一个值**：

```
格式: 每行一个数值（float32 / int32 / uint8），无 header，无注释
加载: np.loadtxt(filename, dtype=dtype)
编码: UTF-8, LF 换行
```

**阶段 1 证明此格式可行**：14 个文件（`collision_force.txt` 等），`conftest.py` 中 `load_golden(name)` 直接 `np.loadtxt` 加载，零问题。

### 2.2 目录结构

每个金标准场景一个子目录，按场量分文件：

```
wanphys/_src/fluid/fluid_grid/home_fslbm/tests/golden_data/
├── collision_force.txt                    # 阶段 1 已有
├── collision_omega0.5.txt                 # 阶段 1 已有
├── ...                                    # 阶段 1 已有（共 14 个 .txt）
│
├── droplet_r4/                            # ★ 阶段 2 新增
│   ├── f_mom_post.txt                     # 10*N 个 float32，按 [mom0..mom9] 交错
│   ├── flag.txt                           # N 个 int32（存储 uint8 flag 值）
│   ├── mass.txt                           # N 个 float32
│   ├── phi.txt                            # N 个 float32
│   └── tag_matrix.txt                     # N 个 int32
│
├── droplet_r8/                            # 同上
├── droplet_r12/                           # 同上
├── flat_interface/                        # 同上
├── near_droplets/                         # 同上
├── ellipsoid/                             # 同上
├── falling_droplet/                       # 同上
├── droplet_wall/                          # 同上
├── droplet_tau055/                        # 同上
└── droplet_tau2/                          # 同上
```

### 2.3 场量存储约定

| 文件名 | 元素类型 | 元素数量 | 布局说明 |
|--------|---------|---------|---------|
| `f_mom_post.txt` | float32 | 10 × N | 第 i 个单元的第 m 个矩在偏移 `m * N + i` |
| `flag.txt` | int32 | N | 存储为 int32（C 风格 `unsigned char` 的零扩展） |
| `mass.txt` | float32 | N | |
| `phi.txt` | float32 | N | |
| `tag_matrix.txt` | int32 | N | |

其中 `N = nx × ny × nz`，单元索引 `idx = iz * ny * nx + iy * nx + ix`（C 风格 row-major，与参考代码一致）。

> **注意**: 对于重力下落、椭球振荡等动态场景，φ 场在仿真过程中会演化。金标准验证以**末步状态**为准（按 [§4](#4-金标准回归测试) 各场景指定的步数）。

### 2.4 加载函数扩展

在 `conftest.py` 中新增 `load_surface_golden(scene_name)`：

```python
def load_surface_golden(scene_name: str) -> dict[str, np.ndarray]:
    """加载阶段 2 表面金标准数据（多文件分目录 .txt 格式）。

    Returns
    -------
    dict
        'f_mom_post'  : shape (10*N,)  float32
        'flag'        : shape (N,)      int32
        'mass'        : shape (N,)      float32
        'phi'         : shape (N,)      float32
        'tag_matrix'  : shape (N,)      int32
    """
    scene_dir = GOLDEN_DIR / scene_name
    return {
        "f_mom_post": np.loadtxt(scene_dir / "f_mom_post.txt", dtype=np.float32),
        "flag":       np.loadtxt(scene_dir / "flag.txt", dtype=np.int32),
        "mass":       np.loadtxt(scene_dir / "mass.txt", dtype=np.float32),
        "phi":        np.loadtxt(scene_dir / "phi.txt", dtype=np.float32),
        "tag_matrix": np.loadtxt(scene_dir / "tag_matrix.txt", dtype=np.int32),
    }
```

---

## 3. 单元测试补全

### 3.1 优先级定义

| 优先级 | 含义 | 阻塞条件 |
|--------|------|---------|
| **P0** | 阻塞性 — 无此测试无法确认核心算法正确性 | 阻塞 Phase C/D 集成和金标准回归 |
| **P1** | 重要 — 覆盖边界情况和数值稳定性 | 阻塞阶段 2 验收 |
| **P2** | 辅助 — 增加对集成层的信心 | 不阻塞验收，可后续补 |

---

### 3.2 P0：法向计算 — `TestCalculateNormal`（新增测试类）

**参考**: `mrUtilFuncGpu3D.h:357-369`，`phase2_design_zh.md` §2.1.2

**实现前置条件**: `calculate_normal` 目前内联在 `calculate_curvature_from_grid` 中。需先提取为独立 `@wp.func`，并创建测试 wrapper kernel `_kernel_calculate_normal`，输入 27 个 φ 值（按邻居方向排列），返回法向。

| # | 测试方法 | 网格 | 输入 φ 场 | 预期 | 容差 |
|---|---------|------|----------|------|------|
| 1 | `test_planar_z` | 8³ | z<4 φ=1.0, z≥4 φ=0.0 | n ≈ (0, 0, 1) | 方向夹角 < 1° |
| 2 | `test_planar_x` | 8³ | x<4 φ=1.0, x≥4 φ=0.0 | n ≈ (1, 0, 0) | 方向夹角 < 1° |
| 3 | `test_planar_diagonal` | 8³ | φ 沿 (1,1,0) 方向线性变化 | n ≈ (1/√2, 1/√2, 0) | 方向夹角 < 3° |
| 4 | `test_spherical_r8` | 16³ | R=8 球体液滴中心 (8,8,8) | n 径向向外，|n|≈1.0 | 径向投影 > 0.95 |
| 5 | `test_uniform_no_nan` | 8³ | 全 φ=1.0 或全 φ=0.0 | 无 NaN/Inf | 精确 |

---

### 3.3 P0：曲率计算 — `TestCalculateCurvature`

**参考**: `mrUtilFuncGpu3D.h:371-420`，`phase2_design_zh.md` §2.1.6

PLIC-Monge 曲率公式是微分几何平均曲率 H = (κ₁+κ₂)/2。但由于 VOF 场的平滑宽度 δ 和 PLIC 偏移的数学性质（详见 §3.3.1），Monge 拟合的输入 `z` 坐标会产生 `−1/δ` 的系统偏置。测试不直接断言 Warp 输出的原始曲率值，而是用 **z 重建法** 验证管线其余部分正确：

1. 对轴对齐的 φ=0.5 细胞，从 Warp 获取法向和 PLIC 偏移；
2. 计算 `z_corrected = dot(ei,bz) − δ × offset`；
3. 用修正后的 z 做 Monge 拟合，断言曲率匹配几何期望（误差 < ±5%）。

#### 3.3.1 PLIC z 偏置的来源

对轴对齐法向 `n`，`plic_cube(φ, n) = φ − 0.5`（PLIC 数学恒等式）。VOF 场在接口附近线性过渡：`φ ≈ 0.5 − Δr/δ`，其中 Δr 是邻居到球心的几何径向偏差。PLIC 偏移 `d0 = φ−0.5 = −Δr/δ`。但 Monge 重建期望 `z = Δr`。因此 Warp 给 Monge 的输入是真实值的 `−1/δ` 倍且符号相反。`z_corrected = −δ × z_plic` 恢复正确的几何 z。

#### 3.3.2 测试方法

| # | 测试 | 网格 | 验证方式 |
|---|------|------|---------|
| 1 | `test_sphere_r4` | 16³, R=4 | 选最佳 φ=0.5 细胞，z 重建后 Monge 拟合 → `|K| ≈ 1/R` |
| 2 | `test_sphere_r8` | 32³, R=8 | 同上 |
| 3 | `test_sphere_r12` | 32³, R=12 | 同上 |
| 4 | `test_plane_zero` | 16³, z=8 | 原始输出：`median|K| < 0.05`（平面 z 偏置为零，无需重建） |
| 5 | `test_cylinder_r6` | 32³, R=6 | 硬编码轴对齐细胞 `(22,16,16)`，z 重建后 → `|K| ≈ 1/(2R)` |
| 6 | `test_isolated_interface_zero` | 8³ | 邻居不足 5 → fallback `K=0.0` |

#### 3.3.3 金标准对比

Warp 曲率的原始值（含 PLIC 偏置）应完全匹配参考 C++ 代码在相同输入上的输出。在参考代码可运行后，用金标准数据对比验证 Warp 曲线的逐细胞一致性——如果偏置，两者同偏，说明移植正确。

---

### 3.4 P0：surface_1 补全 — IG→I 传播

**当前**: `TestSurface1` 只覆盖了 G→GI 方向。  
**补全**: 在现有 `TestSurface1` 类中新增 3 个方法。

| # | 测试方法 | 输入 | 预期 |
|---|---------|------|------|
| 2 | `test_if_to_ig_converts_to_i` | 中心 TYPE_IF，指定邻居 TYPE_IG | 邻居 flag → TYPE_I（TYPE_SU 子域从 IG 变为 I） |
| 3 | `test_if_does_not_affect_solid` | 中心 TYPE_IF，邻居 TYPE_S | 邻居 flag 保持不变 |
| 4 | `test_if_does_not_affect_fluid` | 中心 TYPE_IF，邻居 TYPE_F | 邻居 flag 保持 TYPE_F |

**实现要点**:
- 使用 8³ 网格，在角落单元精确设置 flag。
- 验证 TYPE_IG 被正确转为 TYPE_I 时，flag 的其他 bit（如 TYPE_BO）不被影响。
- 多次独立运行验证确定性——surface_1 写入邻居 flag 存在并行竞态，5 次运行结果必须完全一致。

---

### 3.5 P0：surface_2 — `TestSurface2Initialization`（新增测试类）

**参考**: `mrLbmSolverGpu3D.cu:479-602`，`phase2_design_zh.md` §2.2.2

**实现**: 在 8³ 网格上精确控制单 TYPE_GI 单元及其 26 个邻居的 flag/f_mom_post/c_value，验证 GI 初始化分支和 IG 分支。

**GI 分支**（参考 `mrLbmSolverGpu3D.cu:494-567`）:

| # | 测试方法 | 输入 | 预期 |
|---|---------|------|------|
| 1 | `test_gi_averages_fluid_neighbors` | 6 个面邻居为已知 (ρ,u) 的 TYPE_F | `f_mom_post[0]=avg(ρ)`, `f_mom_post[1-3]=avg(u)` |
| 2 | `test_gi_stress_from_equilibrium` | 同上，u=(0.1, 0, 0) | `f_mom_post[4..9]` 通过 feq·cx_i·cx_j 求和后除以 ρ 再减 cs² 得到 |
| 3 | `test_gi_c_value_from_face_neighbors` | DI<7 的面邻居 c_value 已知 | `c_value = avg(邻居 c_value)` |
| 4 | `test_gi_g_mom_d3q7_equilibrium` | 同上 | `g_mom` 为 D3Q7 平衡态（ρ_g = avg(c_value)，u = avg(u)） |
| 5 | `test_gi_no_fluid_neighbors_default` | 无任何流体/接口邻居 | `rhon=1.0`, `u=(0,0,0)`, `c_value=0.0` |

**IG 分支**（参考 `mrLbmSolverGpu3D.cu:569-600`）:

| # | 测试方法 | 输入 | 预期 |
|---|---------|------|------|
| 6 | `test_ig_converts_f_neighbor_to_i` | 中心 TYPE_IG，邻居 TYPE_F（islet=0） | 邻居 flag → TYPE_I，`merge_detector=1` |
| 7 | `test_ig_respects_islet` | 中心 TYPE_IG，邻居 TYPE_IF（islet=1） | 中心 flag → TYPE_I（不是邻居），邻居 flag 不变 |

**关键验证点**: `f_mom_post[4] = pixx/rhon - cs²`（非 `pixx - rhon*cs²`），与参考代码 HOME 矩存储约定一致。

---

### 3.6 P1：PLIC 已知案例补全 — `TestPlicCube`（+3 个）

在现有 `TestPlicCube` 类中新增：

| # | 测试方法 | 输入 | 预期 | 覆盖分支 |
|---|---------|------|------|---------|
| 4 | `test_axis_aligned_large_volume` | V=0.9, n=(1,0,0) | d=+0.4 | 对称于 V=0.1 案例 |
| 5 | `test_diagonal_small_volume` | V=0.1, n=(1/√3,1/√3,1/√3) | — | `plic_cube_reduced` case(1) cbrt 分支 |
| 6 | `test_diagonal_large_volume` | V=0.9, n=(1/√3,1/√3,1/√3) | d(V=0.9) ≈ -d(V=0.1) | 对称性验证 |

---

### 3.7 P1：surface_3 边界情况补全（+4 个）

在现有 `TestSurface3` 类中新增：

| # | 测试方法 | 输入 | 预期 |
|---|---------|------|------|
| 3 | `test_mass_conservation_global` | 混合 TYPE_F/I/G 网格，已知初始 Σmass | 执行后 `Σ(mass + massex)` 不变（容差 1e-10） |
| 4 | `test_isolated_i_self_absorb` | 孤立 TYPE_I（无任何流体/接口邻居） | `counter=0`, `massn += massexn`, `massexn=0` |
| 5 | `test_flag_transitions_full` | IF/IG/GI 单元各至少一个 | IF→F、IG→G、GI→I 逐一验证 |
| 6 | `test_delta_phi_delta_g` | TYPE_I φ 0.5→0.8（mass 增加） | `delta_phi = +0.3`, `delta_g` 扣减 `rhon_g * delta_phi` |

---

### 3.8 P1：LU 分解 — `TestLuSolve5x5`（新增测试类）

**参考**: `mrUtilFuncGpu3D.h:124-140`，`phase2_design_zh.md` §2.1.5

**实现**: 创建测试 wrapper kernel，传入 5×5 对称矩阵（作为 15 个标量）和 5 元素 RHS，返回解向量。

| # | 测试方法 | 输入 | 预期 |
|---|---------|------|------|
| 1 | `test_identity` | M=I_5, b=(1,2,3,4,5) | x = b（容差 1e-10） |
| 2 | `test_known_2x2` | M=[[4,1],[1,3]], b=(1,2) (Nsol=2) | x ≈ (0.0909, 0.6364) |
| 3 | `test_known_3x3` | M=[[4,1,0],[1,3,1],[0,1,2]], b=(1,2,3) (Nsol=3) | 对照 numpy 验证 |
| 4 | `test_roundtrip` | 随机对称正定 5×5 M，随机 b | M@x ≈ b（容差 1e-8） |
| 5 | `test_nsol_lt_5` | Nsol=3（曲率拟合点数 3-4 的 fallback 路径） | 仅使用前 3×3 子矩阵，正确求解 |

---

### 3.9 P2：流体内核曲率与质量守恒集成 — `TestStreamCollideBvhSurface`（新增测试类）

| # | 测试方法 | 输入 | 预期 |
|---|---------|------|------|
| 1 | `test_sphere_curvature_via_kernel` | 球面 φ 场 + `stream_collide_bvh_kernel` 1 步 | TYPE_I 单元 κ 中位数 ≈ 2/R |
| 2 | `test_plane_curvature_via_kernel` | 平面 φ 场 + `stream_collide_bvh_kernel` 1 步 | 所有 TYPE_I 单元 κ ≈ 0 |
| 3 | `test_mass_conservation_one_step` | 随机 φ 场 + `stream_collide_bvh_kernel` + `surface_3_kernel` 1 步 | Σmass 漂移 ≤ 1e-10 |

---

### 3.10 回归：阶段 1 测试重新验证

阶段 1 的全部 17 个测试（`test_kernels_fluid.py`）在阶段 2 补全后**必须重新执行并通过**。特别关注 `TestStreamCollideBvh` 和 `TestStreamCollideBvhShearDecay`——Phase C/D 代码在 TYPE_I 路径中有大量新增逻辑，可能影响纯流体（TYPE_F）场景的数值路径。

```powershell
uv run pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_kernels_fluid.py -v
```

---

## 4. 金标准回归测试

### 4.1 测试文件

新建 `wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_regression_surface.py`（~400 行）。

### 4.2 对比规则

| 场量类型 | 对比方法 | 容差 |
|---------|---------|------|
| 浮点场量（f_mom_post, mass, phi） | 相对误差 `|a-b|/max(|a|,|b|,1e-10)` | ≤ 1e-4 |
| 近零值（参考值 < 1e-8） | 绝对误差 `|a-b|` | ≤ 1e-10 |
| flag | 逐位精确匹配 | 不允许任何差异 |
| tag_matrix | 标签重编号置换后匹配 | 提取唯一标签集，双射映射，最小不一致率 = 0 |

### 4.3 测试场景（8 个金标准场景）

| # | 测试方法 | 网格 | 初始条件 | 步数 | 核心验证点 |
|---|---------|------|---------|------|-----------|
| 1 | `test_droplet_r4` | 16³ | R=4 球面 φ 场 | 5 | flag/mass/phi 逐单元匹配（曲率含 PLIC 偏置，见 §3.3.1） |
| 2 | `test_droplet_r8` | 32³ | R=8 球面 φ 场 | 10 | TYPE_I 环带宽度 2-3 单元；质量守恒 |
| 3 | `test_droplet_r12` | 32³ | R=12 球面 φ 场 | 10 | 低曲率处界面平滑；质量守恒 |
| 4 | `test_flat_interface` | 32³ | z<16=TYPE_F, z≥16=TYPE_G | 10 | κ≈0，z=16 处 TYPE_I 环带不漂移 |
| 5 | `test_near_droplets` | 32³ | 两 R=6 液滴中心距 14 | 20 | surface_1 G→GI 传播正确，两液滴 tag 不同（不误合并） |
| 6 | `test_ellipsoid` | 32³ | rx=10, ry=6, rz=6 椭球 | 50 | 曲率驱动向球形松弛；质量守恒；重心不漂移 |
| 7 | `test_falling_droplet` | 32³ | R=6 液滴, gz=-0.001 | 30 | 重心 z 坐标减小（向下移动）；质量守恒 |
| 8 | `test_droplet_wall` | 32³ | R=6 液滴邻接 z=0 壁面 | 10 | 壁面反弹正确；surface_1 在固体边界行为一致 |

### 4.4 参数变化场景（+2 个）

| # | 测试方法 | 网格 | R | ω | 步数 | 验证点 |
|---|---------|------|---|-----|------|--------|
| 9 | `test_droplet_tau055` | 32³ | 8 | 1/0.55 ≈ 1.818 | 20 | 低粘性，球面振荡幅度 > 高粘性情况 |
| 10 | `test_droplet_tau2` | 32³ | 8 | 0.5 | 20 | 高粘性，快速阻尼至球形 |

---

## 5. 集成测试

### 5.1 测试文件

新建 `wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_solver_surface.py`（~150 行）。

### 5.2 测试方法

| # | 测试方法 | 验证内容 |
|---|---------|---------|
| 1 | `test_surface_kernel_launch_order` | 单步执行，验证 kernel 启动顺序：`stream_collide_bvh` → `surface_1` → `surface_2` → `surface_3` → `swap_moments` |
| 2 | `test_split_flag_accumulation` | IF→F 过渡且 `tag > 0` → `split_flag` 原子递增 |
| 3 | `test_surface_state_consistency` | 单步后 flag 不含 TYPE_GI（全部已转为 TYPE_I） |
| 4 | `test_phase1_regression` | 阶段 1 全部 17 个测试在 surface kernel 集成后仍全部通过 |

---

## 6. 金标准生成器扩展

### 6.1 修改文件

`docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/export_solver_golden.cpp`

### 6.2 新增生成函数

```cpp
// ---- 球体液滴 ----
void gen_droplet(int N, double R, int steps, const char* out_dir);

// ---- 平面界面 ----
void gen_flat_interface(int N, int steps, const char* out_dir);

// ---- 近接双液滴 ----
void gen_near_droplets(int N, double R, double center_dist, int steps, const char* out_dir);

// ---- 椭球体液滴 ----
void gen_ellipsoid(int N, double rx, double ry, double rz, int steps, const char* out_dir);

// ---- 重力下落液滴 ----
void gen_falling_droplet(int N, double R, double gz, int steps, const char* out_dir);

// ---- 壁面接触液滴 ----
void gen_droplet_wall(int N, double R, double z_wall, int steps, const char* out_dir);

// ---- 参数变化 ----
void gen_droplet_omega(int N, double R, double omega, int steps, const char* out_dir);
```

### 6.3 生成函数内部流程

每个生成函数执行以下步骤：

1. **分配**: `mrFlow3D` + `mrLbmSolver3D`，网格尺寸 N³
2. **初始化**: 设置 φ 场（球面用 `φ = clamp((R - dist)/δ + 0.5, 0, 1)`），设定 TYPE_F/I/G flag，初始 fMom 为 feq(ρ=1, u=0)，tag_matrix = -1
3. **仿真循环**: 调用 `mrSolver3DGpu()` 运行 `steps` 步（完整管线，含 surface_1/2/3）
4. **输出**: 在指定步数后，逐场量写出 `.txt` 文件到 `out_dir/`

> 对于 `gen_droplet_omega`：构造 model 时覆盖默认 ω 参数。

### 6.4 输出约定

生成器逐场量写出 `.txt` 文件到对应子目录。示例伪代码（`gen_droplet`）：

```cpp
void write_float_field(const char* path, float* data, int N) {
    FILE* f = fopen(path, "w");
    for (int i = 0; i < N; ++i) fprintf(f, "%.9g\n", data[i]);
    fclose(f);
}

void gen_droplet(int N, double R, int steps, const char* out_dir) {
    // ... 初始化 + 仿真 steps 步 ...
    int total = N * N * N;
    write_float_field("droplet_rN/f_mom_post.txt", flow->fMomPost, 10 * total);
    write_int_field(  "droplet_rN/flag.txt",        (int*)flow->flag, total);
    write_float_field("droplet_rN/mass.txt",         flow->mass, total);
    write_float_field("droplet_rN/phi.txt",          flow->phi, total);
    write_int_field(  "droplet_rN/tag_matrix.txt",   flow->tag_matrix, total);
}
```

---

## 7. 文件清单与执行顺序

### 7.1 新增/修改文件

```
wanphys/_src/fluid/fluid_grid/home_fslbm/
├── kernels_fluid.py                     # 可能微调：提取 calculate_normal 为独立 @wp.func
├── tests/
│   ├── conftest.py                      # 修改：新增 load_surface_golden()
│   ├── test_kernels_surface.py          # 修改：+4 个测试类（含补全），+20 个测试方法
│   ├── test_regression_surface.py       # ★ 新增：~400 行，10 个金标准回归测试
│   ├── test_solver_surface.py           # ★ 新增：~150 行，4 个集成测试
│   └── golden_data/                     # ★ 新增子目录
│       ├── droplet_r4/                  #   (5 个 .txt 文件)
│       ├── droplet_r8/
│       ├── droplet_r12/
│       ├── flat_interface/
│       ├── near_droplets/
│       ├── ellipsoid/
│       ├── falling_droplet/
│       ├── droplet_wall/
│       ├── droplet_tau055/
│       └── droplet_tau2/
│
docs/wanphys/home_fslbm/
└── phase2_test_plan_zh.md               # ★ 本文档

docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/
└── export_solver_golden.cpp             # 修改：新增 7 个生成函数
```

### 7.2 执行顺序与依赖

```
阶段 A — 单元测试（P0，阻塞性）
  A1. 提取 calculate_normal 为独立 @wp.func（若需要）
  A2. TestCalculateNormal（5 个测试）
  A3. TestCalculateCurvature（6 个测试）
  A4. TestSurface1 补全（+3 个测试）
  A5. TestSurface2Initialization（7 个测试）

阶段 B — 单元测试（P1）
  B1. TestPlicCube 补全（+3 个测试）
  B2. TestSurface3 边界情况补全（+4 个测试）
  B3. TestLuSolve5x5（5 个测试）
  B4. 阶段 1 回归重新验证

阶段 C — 金标准回归（依赖 A + 生成器扩展完成）
  C1. 扩展 export_solver_golden.cpp
  C2. 运行参考代码生成金标准 .txt 文件
  C3. 实现 conftest.py load_surface_golden()
  C4. 实现 test_regression_surface.py（10 个测试）

阶段 D — 集成验证（P2）
  D1. test_solver_surface.py（4 个测试）
  D2. TestStreamCollideBvhSurface（3 个测试）
  D3. 阶段 1 最终回归验证
```

---

## 8. 风险与注意事项

| # | 风险 | 缓解措施 |
|---|------|---------|
| 1 | `calculate_normal` 未独立暴露 | 若提取成本高，可暂不提取——在曲率测试中隐式验证；但需在计划中标记为技术债务 |
| 2 | surface_1/surface_2 并行写竞态 | 每个相关测试独立运行 5 次，要求结果完全一致 |
| 3 | PLIC 曲率在粗网格上离散误差大 | 容差按网格分辨率分档（R=4 容差 30%, R=8 容差 15%, R=12 容差 10%） |
| 4 | 阶段 1 测试在 Phase C/D 补全后回归失败 | 阶段 B4 提前回归验证；若失败则优先排查 TYPE_I 路径对 TYPE_F 路径的副作用 |
| 5 | 球面 φ 场在离散网格上 TYPE_I 环带宽度不足（< 5 接口邻居） | 使用 `φ = clamp((R - dist) / δ + 0.5, 0, 1)`，δ = 1.5·Δx，确保 2-3 单元宽的过渡带 |
| 6 | 金标准数据量过大（32³ × 10 场景 × 5 场量 ≈ 1.6M 个 float） | `.txt` 文本格式每个 float 约 12 字节，总计约 20 MB——可接受。若超过 100 MB 再考虑压缩 |

---

## 9. 成功门禁

阶段 2 测试补全视为完成的标准：

1. **P0 测试全部通过**：TestCalculateNormal（5）、TestCalculateCurvature（6）、surface_1 补全（3）、TestSurface2Initialization（7）——合计 21 个
2. **P1 测试全部通过**：TestPlicCube 补全（3）、surface_3 补全（4）、TestLuSolve5x5（5）——合计 12 个
3. **金标准回归全部通过**：10 个场景，浮点相对误差 ≤ 1e-4，flag 逐位精确，标签拓扑一致
4. **阶段 1 回归通过**：17/17 测试仍然通过
5. **P2 测试通过**：TestStreamCollideBvhSurface（3）、test_solver_surface（4）
6. **代码覆盖率**: `kernels_surface.py` 分支覆盖率 ≥ 90%，`kernels_fluid.py` Phase C/D 分支覆盖率 ≥ 85%

---

## 参考

| 资源 | 路径 |
|------|------|
| 阶段 2 设计方案 | [phase2_design_zh.md](phase2_design_zh.md) |
| 阶段 1 状态报告 | [phase1_status_zh.md](phase1_status_zh.md) |
| 迁移架构计划 | [migration_plan_zh.md](migration_plan_zh.md) |
| 迁移执行手册 | [migration_execution_zh.md](migration_execution_zh.md) |
| 二次审核意见书 | [migration_reaudit_zh.md](migration_reaudit_zh.md) |
| 参考 CUDA 实现 | `docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/` |
| PLIC 曲率参考 | `mrUtilFuncGpu3D.h:320-420` |
| 表面 kernel 参考 | `mrLbmSolverGpu3D.cu:444-937` |
