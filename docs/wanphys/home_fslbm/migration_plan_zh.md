# Home-FSLBM 迁移至 WanPhys 框架计划（修订版）

> **目标目录**: `wanphys/_src/fluid/fluid_grid/home_fslbm/`
> **参考代码**: `docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/`
> **参考文献**: 见下方 [论文编号对照表](#论文编号对照表)
> **引擎**: Newton（通过 WanPhys 核心层）
> **GPU 后端**: NVIDIA Warp（替代 CUDA）
> **日期**: 2026-07-15
> **状态**: 已按审核意见修订（审核引用: `home_fslbm_migration_audit_zh.md`）

---

## 目录

1. [概述](#1-概述)
2. [参考系统分析](#2-参考系统分析)
3. [目标架构](#3-目标架构)
4. [逐模块迁移方案](#4-逐模块迁移方案)
5. [算法详述与模块标注](#5-算法详述与模块标注)
6. [单元测试方案](#6-单元测试方案)
7. [文件清单](#7-文件清单)
8. [迁移路线图](#8-迁移路线图)
9. [风险登记](#9-风险登记)
10. [金标准数据生成流程](#10-金标准数据生成流程)

---

## 论文编号对照表

本文中使用的"论文一"至"论文五"简称对应的完整信息如下：

| 编号 | 目录名 | 中文全称 | 英文简称 | 对本迁移的核心贡献 |
|------|--------|---------|---------|-------------------|
| **论文一** | `流体-固体耦合动力学两相流模拟` | 流体-固体耦合动力学两相流模拟 | Fluid-Solid Coupling Two-Phase Flow (Li & Desbrun) | 双分辨率网格；相场 LBM；HMBB 流固耦合 |
| **论文二** | `湍流多流体动力学模拟` | 湍流多流体动力学模拟 | Turbulent Multi-Fluid Dynamics | N 相可混溶/不可混溶 LBM；HOME-LBM 多流体适配 |
| **论文三** | `稳定性引导量化的高性能矩编码LBM` | 稳定性引导量化的高性能矩编码LBM | Stability-Guided Quantized HOME-LBM | 分裂 kernel GPU 设计；16 位量化；von Neumann 稳定性分析 |
| **论文四** | `锐界面动力学自由表面流与泡沫` | 锐界面动力学自由表面流与泡沫（HOME-FREE LBM） | Kinetic Free-Surface Flows and Foams with Sharp Interfaces (Wang et al., ACM TOG 2025) | **本迁移主要参考文献**：D3Q27 HOME-LBM 自由表面求解器；VOF 锐界面；YACCLAB CCL 气泡追踪；CMR-MRT 气体模型；分离压力泡沫模型；涡粘性湍流。参考代码：`Home-FSLBM/` |
| **论文五** | `高阶矩编码湍流动能模拟` | 高阶矩编码湍流动能模拟（HOME-LBM） | High-Order Moment-Coded Turbulent Kinetic Energy (HOME-LBM) | HOME-LBM 基础理论：矩编码（10 矩/节点）；三阶 Hermite 重建；NOCM-MRT 闭式碰撞；5-10 倍加速证明 |

---

## 1. 概述

### 1.1 目标

将 **HOME-FSLBM** 参考实现从原始 CUDA C++ 迁移至基于 Newton 引擎与 NVIDIA Warp GPU 后端的 WanPhys 框架。

### 1.2 范围

本次迁移覆盖**完整的三维 D3Q27 自由表面 LBM 求解器**，包括：

- HOME-LBM 矩编码流体动力学（每格点存储 10 个矩）
- NOCM-MRT 碰撞算子
- VOF 锐界面自由表面追踪
- PLIC 曲率估计
- D3Q7 **CMR-MRT** 溶解气体对流扩散（非 BGK——按审核项 B3 更正）
- 并行 **YACCLAB** CCL 气泡追踪与气体定律（非简化并查集——按审核项 B1 更正）
- **双精度**气泡体积/密度数组（非 float——按审核项 B2 更正）
- 涡粘性湍流模型（内联于 stream_collide_bvh）
- 分离压力泡沫模型
- 流固双向反弹耦合
- 确定性标签冲突解决的气泡合并/分裂检测

### 1.3 非范围（延期项）

- 二维变体
- 16 位量化优化（论文三）
- 双分辨率网格（论文一）
- N 相可混溶/不可混溶泛化（论文二）
- 润湿/接触角

---

## 2. 参考系统分析

### 2.1 HOME-FSLBM 算法管线

参考求解器使用**两阶段调用结构**（非单一管线）：

```
// ---- 阶段一: coupling() — 气泡 + 气体子系统 ----
coupling(lbm_dev_gpu, param, N, l0p, roup, labma, u0p, time_step):
  ├── get_tag_kernel               // 邻居标签传播
  ├── assign_tag_kernel            // 通过 previous_merge_tag 解决并行标签冲突
  ├── recheck_merge_kernel         // 区分真合并与气泡移动导致的标签变化
  ├── update_bubble(...):
  │     ├── bubble_volume_update   // 按气泡累加 (1-phi)
  │     ├── bubble_rho_update_kernel // 气体定律: rho_new = rho_old * V_init / V
  │     ├── MergeSplitDetectorKernel  // 全局合并/分裂标志检查
  │     └── [条件执行] handle_merge_spilt → ClearDectector
  └── g_handle(...):
        ├── g_reconstruction       // 亨利定律边界交换
        ├── g_stream_collide       // D3Q7 CMR-MRT（非 BGK！）
        ├── bubble_volume_g_update_kernel  // 气体通量 → 气泡体积
        ├── mrSolver3D_g_step2Kernel       // gMom 交换
        └── bubble_rho_update_kernel       // 气体交换后更新密度

// ---- 阶段二: mrSolver3DGpu() — 流体 + 自由表面子系统 ----
mrSolver3DGpu(lbm_dev_gpu, param, N, l0p, roup, labma, u0p, time_step):
  ├── calculate_disjoint           // 射线投射分离压力
  ├── [条件执行] clear_inlet       // 移除入口单元（time_step == 57595）
  ├── atmosphere_rho_update_kernel // 开敞式水箱大气压力
  ├── atmosphere_volme_update_kernel  // 大气体积修正
  ├── stream_collide_bvh ★         // 唯一流体 kernel（内联湍流 + 自由表面）
  ├── ResetDisjoinForce            // 清零分离力
  ├── surface_1                    // 标记传播
  ├── surface_2                    // GI 初始化
  ├── surface_3                    // 质量重分配 + 标记转换
  └── mrSolver3D_step2Kernel       // fMom 交换
```

**源文件**: `mrSolver3D.h:121-125`（coupling 调用），`mrLbmSolverGpu3D.cu:1973-2075`（mrSolver3DGpu 函数体）

### 2.2 关键数据结构

#### mrFlow3D（逐节点状态，源自 `mrFlow3D.h:31-91`）

| 字段 | 类型 | 大小 | 用途 |
|------|------|------|------|
| fMom | REAL* | 10 × N | HOME 矩 |
| fMomPost | REAL* | 10 × N | 碰撞后矩 |
| flag | MLLATTICENODE_SURFACE_FLAG* | N | 单元类型位域 |
| mass | REAL* | N | VOF 质量 |
| massex | REAL* | N | 余量质量 |
| phi | REAL* | N | 体积分数 [0,1] |
| forcex/y/z | REAL* | N | 体积力分量 |
| gMom | float* | 7 × N | D3Q7 气体分布函数 |
| gMomPost | float* | 7 × N | 碰撞后气体分布函数 |
| delta_g | float* | N | 气体浓度变化 |
| c_value | float* | N | 溶解气体浓度 |
| **src** | float* | N | 气体源项（用于 CMR-MRT 碰撞力矩空间外力项） |
| delta_phi | float* | N | 体积分数变化 |
| tag_matrix | int* | N | 气泡 ID |
| previous_tag | int* | N | 上一步气泡 ID |
| **previous_merge_tag** | int* | N | 合并前气泡 ID（并行标签冲突解决） |
| input_matrix | unsigned char* | N | CCL 二值输入 |
| label_matrix | int* | N | CCL 输出标签 |
| **merge_detector** | bool* | N | 逐单元合并候选标记 |
| disjoin_force | float* | N | 分离压力 |
| **islet** | int* | N | 孤立气泡标记（入口移除用） |
| bubble | mlBubble3D | 1 | **全部双精度：** volume[]、init_volume[]、rho[]、label_* |

#### D3Q27 格子常数

| 常数 | 值 | 描述 |
|------|-----|------|
| c_s | 1/√3 ≈ 0.57735 | 声速 |
| w0 | 8/27 | 静止粒子权重 |
| w1-6 | 2/27 | 面邻居权重（×6） |
| w7-18 | 1/54 | 边邻居权重（×12） |
| w19-26 | 1/216 | 角邻居权重（×8） |
| def_6_sigma | 6×4e-3 | 默认表面张力 ×6 |
| K_h | 1e-3 | 亨利常数 |

#### 标记位域

```
TYPE_S=0x01, TYPE_F=0x08, TYPE_I=0x10, TYPE_G=0x20
TYPE_IF=0x18 (I|F), TYPE_IG=0x30, TYPE_GI=0x38
TYPE_SU=0x38 (G|I|F 掩码), TYPE_BO=0x03 (S|E 掩码)
```

### 2.3 HOME-LBM 碰撞模型

[ALGO: NOCM-MRT 碰撞，论文五 Eq.(21-23)，论文四 Eq.(18-21)]
[SRC: `mlGetPIAfterCollision`, `mrUtilFuncGpu3D.h:424-471`]

**核心原理**（论文五）：仅存储前 3 个速度矩（共 10 个标量）和完整的 D3Q27 分布通过三阶 Hermite 展开按需重建。

碰撞在应力张量上直接以闭式执行——这是 HOME-LBM 的核心效率所在。

**碰撞更新公式**（NOCM-MRT）：

```
输入: ρ, u, F, ω, Π^(45)_xx, Π^(45)_yy, Π^(45)_zz, Π^(90)_xy, Π^(90)_xz, Π^(90)_yz

// 1. 将对角应力分解为无迹部分
Π_xx_part = (2Π_xx - Π_yy - Π_zz) / 3
Π_yy_part = (2Π_yy - Π_xx - Π_zz) / 3
Π_zz_part = (2Π_zz - Π_xx - Π_yy) / 3

// 2. 计算平衡态应力（由速度导出）
RU2 = ρ u_x², RV2 = ρ u_y², RW2 = ρ u_z²
RUVW2 = (RU2 + RV2 + RW2) / 3

// 3. 碰撞更新（对角分量）
Π_xx_new = ρ/3 + Π_xx_part·(1-ω) + RUVW2 + ω·(2RU2-RV2-RW2)/3 + F_x·u_x
Π_yy_new = ρ/3 + Π_yy_part·(1-ω) + RUVW2 + ω·(2RV2-RU2-RW2)/3 + F_y·u_y
Π_zz_new = ρ/3 + Π_zz_part·(1-ω) + RUVW2 + ω·(2RW2-RU2-RV2)/3 + F_z·u_z

// 4. 碰撞更新（非对角分量）
Π_xy_new = Π_xy·(1-ω) + ω·ρ·u_x·u_y + (F_y·u_x + F_x·u_y)/2
Π_xz_new = Π_xz·(1-ω) + ω·ρ·u_x·u_z + (F_z·u_x + F_x·u_z)/2
Π_yz_new = Π_yz·(1-ω) + ω·ρ·u_y·u_z + (F_z·u_y + F_y·u_z)/2

// 5. 存储碰撞后矩（速度分量含 1/2 力修正）
fMomPost[0] = ρ
fMomPost[1] = u_x + F_x/(2ρ)
fMomPost[2] = u_y + F_y/(2ρ)
fMomPost[3] = u_z + F_z/(2ρ)
fMomPost[4] = Π_xx_new/ρ - c_s²
fMomPost[5] = Π_xy_new/ρ
fMomPost[6] = Π_xz_new/ρ
fMomPost[7] = Π_yy_new/ρ - c_s²
fMomPost[8] = Π_yz_new/ρ
fMomPost[9] = Π_zz_new/ρ - c_s²
```

完整伪代码见附录 B。

### 2.4 D3Q7 气体模型 — CMR-MRT（已更正）

[ALGO: 论文四 Eq.(33-38)]
[SRC: `g_stream_collide`, `mrLbmSolverGpu3D.cu:1750-1848`]

参考代码使用 **CMR-MRT（中心矩松弛——多松弛时间）**，非 BGK。关键代码：

```cpp
// mrLbmSolverGpu3D.cu:1822-1848 — g_stream_collide 碰撞部分
mrutilfunc.mlConvertCmrMoment_d3q7(uxn, uyn, uzn, pop_g);   // 分布 → 中心矩
mrutilfunc.mlConvertCmrMoment_d3q7(uxn, uyn, uzn, g_eq);

float s[7];  // 7 个独立松弛率
s[0] = 1.0f;                          // 密度矩: 瞬时平衡
s[1] = s[2] = s[3] = 1.0f / 0.9f;   // 动量: omega ≈ 1.111
s[4] = 1.5f;                          // 正应力差
s[5] = s[6] = 1.5f;                   // 二阶矩

for (int i = 0; i < 7; i++) {
    src = src_Q[i] * (1 - s[i] / 2);
    pop_out[i] = fma(1.0f - s[i], pop_g[i], fma(s[i], g_eq[i], src));
}
mrutilfunc.mlConvertCmrF_d3q7(uxn, uyn, uzn, pop_out);  // 中心矩 → 分布
```

两个关键变换函数：

1. **`mlConvertCmrMoment_d3q7`**（`mrUtilFuncGpu3D.h:474-496`）：7 个分布 → 7 个中心矩。将格子分布变换到以局部流速 u 为中心的速度空间，计算各阶中心矩（密度、动量、正应力差、二阶矩模）。变换中每个格点速度减去 u 后计算伪距多项式。

```
算法 (mlConvertCmrMoment_d3q7):
  Input:  ux, uy, uz (局部流速), node_in_out[7] (D3Q7 分布)
  Output: node_in_out[7] (7 个中心矩: k0=密度, k1=动量_x, k2=动量_y, k3=动量_z,
                           k4=正应力差(xx-yy), k5=正应力差(xx-zz), k6=二阶矩模)
  
  保存输入副本: node[0..6] = node_in_out[0..6]
  清零输出:    node_in_out[0..6] = 0
  
  For k = 0..6:
    CX = ex3d_gpu[k] - ux    // 以 u 为中心的格子速度
    CY = ey3d_gpu[k] - uy
    CZ = ez3d_gpu[k] - uz
    ftemp = node[k]
    node_in_out[0] += ftemp                     // k0: 密度 = Σ f_k
    node_in_out[1] += ftemp * CX                // k1: x-动量 = Σ f_k · (c_kx - ux)
    node_in_out[2] += ftemp * CY                // k2: y-动量
    node_in_out[3] += ftemp * CZ                // k3: z-动量
    node_in_out[4] += ftemp * (CX² - CY²)       // k4: 正应力差 (xx-yy)
    node_in_out[5] += ftemp * (CX² - CZ²)       // k5: 正应力差 (xx-zz)
    node_in_out[6] += ftemp * (CX²+ CY²+ CZ²)   // k6: 二阶矩模 (迹)
```

2. **`mlConvertCmrF_d3q7`**（`mrUtilFuncGpu3D.h:499-518`）：7 个中心矩 → 7 个分布。反变换，将碰撞后的中心矩重建为格子分布。包含二次速度展开（U²/2 项）和高阶耦合项。

```
算法 (mlConvertCmrF_d3q7):
  Input:  U, V, W (局部流速), node_in_out[7] (7 个中心矩: k0..k6)
  Output: node_in_out[7] (重建的 7 个 D3Q7 格子分布)
  
  加载矩: k0..k6 = node_in_out[0..6]
  
  // 重建 7 个方向上的分布（带速度展开）
  node_in_out[0] = -k0*U² - 2*k1*U - k0*V² - 2*k2*V - k0*W² - 2*k3*W + k0 - k6
  //方向0: 静止 (0,0,0) — 含全速度二次项和二阶矩迹
  
  node_in_out[1] = k1/2 + k4/6 + k5/6 + k6/6 + (U*k0)/2 + U*k1 + (U²*k0)/2
  //方向1: (+1,0,0) — 含 x-动量线性项、正应力差耦合、速度二次展开
  
  node_in_out[2] = k4/6 - k1/2 + k5/6 + k6/6 - (U*k0)/2 + U*k1 + (U²*k0)/2
  //方向2: (-1,0,0) — 含 x-反方向线性项
  
  node_in_out[3] = k2/2 - k4/3 + k5/6 + k6/6 + (V*k0)/2 + V*k2 + (V²*k0)/2
  //方向3: (0,+1,0)
  
  node_in_out[4] = k5/6 - k4/3 - k2/2 + k6/6 - (V*k0)/2 + V*k2 + (V²*k0)/2
  //方向4: (0,-1,0)
  
  node_in_out[5] = k3/2 + k4/6 - k5/3 + k6/6 + (W*k0)/2 + W*k3 + (W²*k0)/2
  //方向5: (0,0,+1)
  
  node_in_out[6] = k4/6 - k3/2 - k5/3 + k6/6 - (W*k0)/2 + W*k3 + (W²*k0)/2
  //方向6: (0,0,-1)
```

3. **松弛率物理含义**（`g_stream_collide` 第 1822-1824 行硬编码）：
   - `s[0] = 1.0`：密度矩瞬间松弛至平衡态（质量守恒严格满足）
   - `s[1-3] = 1/0.9 ≈ 1.111`：动量矩以 `τ_m = (1/s - 0.5) = 0.4` 的松弛时间汇聚，对应气体动量扩散系数 `ν_g = c_s²·(τ_m - 0.5) = (1/3)·0.4 ≈ 0.133`
   - `s[4] = 1.5` 和 `s[5-6] = 1.5`：高阶矩快速松弛（`τ = 0.833`），耗散数值伪振荡但保留足够的高阶精度

### 2.5 气泡模型 — YACCLAB CCL

[ALGO: 论文四 Sec.5.2]
[SRC: `tDCCL.cu`（578 行），`mrLbmSolverGpu3D.cu:1089-1631`]

参考代码使用 YACCLAB GPU CCL：InitLabeling（2×2×2 block 初始化，8 邻域连接检查）→ MergeMasks（block 级 atomicMin 合并）→ LabelAnalysis（合并链消解为密集标签 0..N-1）→ LabelReduction（全局传播）→ FinalLabeling。密集标签编号对 `tag_matrix` 作为数组索引至关重要（`bubble.rho[tag_matrix[curind] - 1]`）。

**所有气泡属性使用 `double` 精度**（`mrFlow3D.h:17-27`）：
- `bubble.volume`、`bubble.init_volume`、`bubble.rho`：`double*`
- 体积累加：`atomicAdd(&bubble.init_volume[tag-1], (double)(1.f - phi[curind]));`（第 1083 行）
- 气体更新：`atomicAdd(&bubble.init_volume[tag-1], (double)1.f/4.f * factor * delta_g * phi);`（第 1873 行）

**float vs double 定量误差分析**：
- 网格可达 600×300×300 = 5.4×10⁷ 格点
- 5×10⁷ 次 float 累加舍入误差：O(√N)·ε ≈ 7×10³·1.2×10⁻⁷ ≈ 8×10⁻⁴（相对误差）
- 10⁵ 时间步累积可漂移 > 1%
- Double 提供约 15 位有效数字（vs float 的 7 位）
- **结论**：气泡体积/密度数组必须使用 `wp.float64`

---

## 3. 目标架构

### 3.1 WanPhys 编码范式

WanPhys 中每个物理域遵循严格的**四类分解**：

```
Model  （dataclass，继承 FluidGridModelBase → DomainModel）
   └── 静态配置（网格尺寸、物理参数、格子常数）
State  （class，继承 DomainState）
   └── GPU 数组（warp 数组）、clear()、clone()
Solver （class，继承 FluidGridSolverBase → DomainSolver）
   └── Warp kernel 启动、step(state_in, state_out, dt)
Domain （class，继承 Domain）
   └── 双缓冲状态管理、step()、create_state()
```

**关键更正（审核项 B9）**: `HomeFslbmState` 继承自 **`DomainState`**（非 `FluidGridStateBase`）。遵循现有 `LbmState`（`lbm/state.py`）的模式。`FluidGridStateBase` 会自动分配 MAC staggered 网格数组（`vel_u/v/w`、`pressure` 等），HOME-FSLBM 使用自己的场量体系（`flag`、`mass`、`phi`、`f_mom` 等），与 MAC staggered grid 完全不兼容。MAC staggered 速度场如有可视化需要，可通过 `post_step` 钩子在 domain 层单独计算。

**编码规范**（源自现有 `wanphys/_src/fluid/fluid_grid/lbm/`）：
- SPDX 头部声明：`# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers` / `# SPDX-License-Identifier: Apache-2.0`
- 每个文件包含 `from __future__ import annotations`
- 扁平分布数组使用 `wp.array(dtype=float)` 配合步长索引
- 三维场使用 `wp.array3d(dtype=float)`
- Kernel 启动方式：`wp.launch(kernel, dim=(nx, ny, nz), inputs=[...])`
- 热路径中不引入 NumPy 依赖
- 类型提示使用 `float` 而非 `wp.float32`
- 文档字符串采用 NumPy/Sphinx 风格
- 常数模块独立管理格子参数
- 所有 `@wp.kernel` 和 `@wp.func` 定义放在独立的 `kernels*.py` 文件中
- 无中文注释；标识符中不使用特殊字符

### 3.2 文件结构

```
wanphys/_src/fluid/fluid_grid/home_fslbm/
├── __init__.py
├── constants.py             # D3Q27/D3Q7 格子常数、CMR-MRT 松弛率向量
├── model.py                 # HomeFslbmModel（turbulence_radius=3）
├── state.py                 # HomeFslbmState（继承 DomainState，+4 字段，wp.float64）
├── solver.py                # HomeFslbmSolver（两阶段管线结构）
├── domain.py                # HomeFslbmDomain
├── kernels_fluid.py         # stream_collide_bvh（单一 kernel，内联湍流）
├── kernels_surface.py       # VOF surface_1/2/3、PLIC
├── kernels_bubble.py        # YACCLAB CCL、气泡更新、合并/分裂（15 kernel）
├── kernels_gas.py           # D3Q7 CMR-MRT、气体体积更新
└── kernels_foam.py          # 分离压力、大气、入口
```

### 3.3 与参考代码的关键差异

| 方面 | 参考代码（CUDA） | 目标（Warp） |
|------|-----------------|-------------|
| GPU API | CUDA kernel, cudaMalloc | Warp kernel, wp.zeros |
| 双精度 | `double`（气泡体积） | `wp.float64`（气泡数组） |
| CCL 后端 | YACCLAB via OpenCV GpuMat3 | Warp 原生 YACCLAB 重实现 |
| 气体碰撞 | CMR-MRT（7 松弛率） | CMR-MRT（忠实复现） |

---

## 4. 逐模块迁移方案

### 4.1 `constants.py` [P0]

**用途**：D3Q27/D3Q7 格子常数、标记枚举、物理常数。

**内容清单**（全部源自 `mrConstantParamsGpu3D.h` 和 `mlLbmCommon.h`）：

- `NUM_DIRS = 27`：D3Q27 离散速度数量
- `CX`, `CY`, `CZ`：27 方向的速度分量列表
- `W0 = 8/27`：静止粒子权重
- `WS = 2/27`：面邻居权重（6 个方向：±x, ±y, ±z）
- `WE = 1/54`：边邻居权重（12 个方向：±x±y, ±x±z, ±y±z）
- `WC = 1/216`：角邻居权重（8 个方向：±x±y±z）
- `OPPOSITE`：反向方向索引（27 个整数）
- `CS2 = 1/3`：声速平方
- `CS = 1/√3 ≈ 0.57735`：声速
- `INV_CS2 = 3.0`：声速平方的倒数
- `NUM_MOMENTS = 10`：HOME-LBM 存储的矩数量
- `CellFlag` 类：`TYPE_S=0x01`, `TYPE_F=0x08`, `TYPE_I=0x10`, `TYPE_G=0x20`，静态方法 `TYPE_IF()`/`TYPE_IG()`/`TYPE_GI()` 返回组合值
- `TYPE_SU_MASK = 0x38`, `TYPE_BO_MASK = 0x03`：位掩码
- `NUM_DIRS_GAS = 7`：D3Q7 离散速度数量
- `W_GAS = [1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8]`：D3Q7 权重
- `SURFACE_TENSION = 6.0 * 4e-3`：默认表面张力（`def_6_sigma`）
- `HENRY_CONSTANT = 1e-3`：亨利常数（`K_h`）
- `DISJOINT_FACTOR = 0.032`：分离压力因子
- `GAS_CMR_S = [1.0, 1.0/0.9, 1.0/0.9, 1.0/0.9, 1.5, 1.5, 1.5]`：D3Q7 CMR-MRT 7 个松弛率（审核项 B3，源自 `g_stream_collide` 第 1822-1824 行的硬编码 `s[0..6]`）

### 4.2 `model.py` [P0]

**用途**：静态仿真配置 dataclass。

**类**：`HomeFslbmModel(FluidGridModelBase)`，继承自 `base.py` 的网格属性（`fluid_grid_res`、`fluid_grid_cell_size`、`device`）。

**完整字段表**（源自 `mrFlow3D.h:78-85` 的 `Create()` 参数以及 `MLFluidParam3D` 和 `MLMappingParam`）：

| 字段 | 类型 | 默认值 | 描述 | 参考 |
|------|------|--------|------|------|
| omega | float | 1.0 | NOCM-MRT 松弛频率（ω = 1/τ，τ > 0.5 为正粘性） | 论文五 Eq.(21) |
| gravity_x/y/z | float | 0.0 | 格子单位体积力（对应 `param->gx/gy/gz`） | `MLFluidParam3D` |
| surface_tension | float | 6·4e-3 | 表面张力系数（`def_6_sigma`，D3Q27 需 ×6） | 论文四 Eq.(12) |
| gas_omega | float | 1.0 | D3Q7 CMR-MRT 基础松弛频率（实际使用 7 个独立松弛率） | 论文四 Eq.(34) |
| henry_constant | float | 1e-3 | 亨利定律常数 `K_h` | 论文四 Eq.(38) |
| disjoin_factor | float | 0.032 | 分离压力强度乘数 | 论文四 Eq.(40) |
| turbulence_factor | float | 4.0 | 涡粘性乘数（ν_e = factor · ‖S‖_F） | 论文四 Sec.5.1 |
| turbulence_radius | int | **3** | 气泡邻近搜索半径（±3，对应 6×6×6=216 邻居，审核项 S6 更正） | `stream_collide_bvh:1001` |
| max_bubbles | int | 65536 | 最大追踪气泡数（`mlBubble3D::max_bubble_count`） | `mrFlow3D.h:25` |
| atmosphere_open | bool | False | 顶部边界是否向大气开放（影响 `atmosphere_rho_update_kernel` 条件） | 第 1274 行 |
| bc_types | tuple[int,...] | (0,)*6 | 各面边界条件类型（0=反弹，1=Zou-He，2=对流出口，3=周期性） | `LbmModel` 模式 |
| bc_periodic | tuple[bool,bool,bool] | (False,)*3 | 逐轴周期标记 | `LbmModel` 模式 |
| initial_density | float | 1.0 | 均匀初始密度 ρ₀，用于平衡态初始化 | `LbmModel` 模式 |

**计算属性**：`tau`（1/omega）、`kinematic_viscosity`（(τ-0.5)/3）、`_periodic_ints`（供 kernel 传参）。

**`__post_init__` 校验**：ω > 0（确保 τ > 0.5），网格分辨率 > 0，周期性标记与边界类型一致性检查（遵循现有 `LbmModel` 模式）。

### 4.3 `state.py` [P0]

**用途**：GPU 端仿真状态（`mrFlow3D.h:31-91` 的 Warp 等价实现）。

**类**：`HomeFslbmState(DomainState)`——从 `FluidGridStateBase` 更正为 `DomainState`（审核项 B9）。`__init__` 自行分配所有 warp 数组，`clear()` 归零全部数组，`clone()` 通过 `wp.copy` 深拷贝。

**完整 Warp 数组**（**粗体**为审核项 B6 新增字段）：

| 数组名 | 形状 | Dtype | 参考源 | 用途 |
|--------|------|-------|--------|------|
| `f_mom` | (10×N,) flat | float | `mrFlow3D.h:38` | HOME 矩 [ρ, ρux, ρuy, ρuz, ρSxx, ρSxy, ρSxz, ρSyy, ρSyz, ρSzz]（当前帧） |
| `f_mom_post` | (10×N,) flat | float | `mrFlow3D.h:39` | HOME 矩（下一帧/后缓冲区） |
| `flag` | (nx,ny,nz) | wp.uint8 | `mrFlow3D.h:36` | 逐单元类型位域（TYPE_F/TYPE_I/TYPE_G/TYPE_S + 转换标记） |
| `mass` | (nx,ny,nz) | float | `mrFlow3D.h:47` | VOF 质量 |
| `massex` | (nx,ny,nz) | float | `mrFlow3D.h:48` | 余量质量（surface_3 中用于邻居间重分配） |
| `phi` | (nx,ny,nz) | float | `mrFlow3D.h:49` | 体积分数 [0,1]（= mass/ρ 钳制到 [0,1]） |
| `force_x/y/z` | (nx,ny,nz) | float | `mrFlow3D.h:42-44` | 体积力分量（含重力和分离压力贡献） |
| `g_mom` | (7×N,) flat | float | `mrFlow3D.h:71` | D3Q7 溶解气体分布函数（当前帧） |
| `g_mom_post` | (7×N,) flat | float | `mrFlow3D.h:72` | D3Q7 溶解气体分布函数（下一帧） |
| `delta_g` | (nx,ny,nz) | float | `mrFlow3D.h:73` | 气体浓度变化累加器（g_reconstruction 中从流体邻居收集） |
| `c_value` | (nx,ny,nz) | float | `mrFlow3D.h:75` | 溶解气体浓度 |
| **`src`** | (nx,ny,nz) | float | `mrFlow3D.h:76` | 气体外部源项（g_stream_collide CMR-MRT 碰撞中作为力矩空间外力项，`src[curind] * d3q7_w[i]` 经 CMR 变换后按 `src_Q[i] * (1 - s[i]/2)` 加入） |
| `delta_phi` | (nx,ny,nz) | float | `mrFlow3D.h:66` | 体积分数变化（气泡体积追踪用） |
| `tag_matrix` | (nx,ny,nz) | wp.int32 | `mrFlow3D.h:55` | 气泡 ID（从 1 开始，0=无，-1=边界/流体） |
| `previous_tag` | (nx,ny,nz) | wp.int32 | `mrFlow3D.h:56` | 上一时间步气泡 ID（合并/分裂检测用） |
| **`previous_merge_tag`** | (nx,ny,nz) | wp.int32 | `mrFlow3D.h:57` | 合并前气泡 ID（assign_tag_kernel 中用于恢复因并行竞争丢失的旧标签，第 1415/1438-1446 行） |
| `input_matrix` | (nx,ny,nz) | wp.uint8 | `mrFlow3D.h:58` | CCL 二值输入（TYPE_I/TYPE_G → 255，否则 → 0） |
| `label_matrix` | (nx,ny,nz) | wp.int32 | `mrFlow3D.h:59` | CCL 输出标签（密集编号 0..N-1） |
| **`merge_detector`** | (nx,ny,nz) | wp.bool | `mrFlow3D.h:61` | 逐单元合并候选标记（surface_1 第 587 行标记 IF 邻居；assign_tag_kernel 第 1438 行检查决定是否恢复旧标签；clear_detector 第 76 行清零） |
| `disjoin_force` | (nx,ny,nz) | float | `mrFlow3D.h:69` | 泡沫分离压力（calculate_disjoint 中通过 atomicAdd 累加） |
| **`islet`** | (nx,ny,nz) | wp.int32 | `mrFlow3D.h:70` | 孤立气泡标记（clear_inlet 第 123-125 行中转 TYPE_G + 清零矩；stream_collide_bvh 第 620/723 行中跳过或转回 TYPE_F；g_stream_collide 第 1767 行中跳过） |
| `bubble_volume` | (max_bubbles,) | **wp.float64** | `mrFlow3D.h:19` | 各气泡当前体积（双精度 atomicAdd 累加，审核项 B2） |
| `bubble_init_volume` | (max_bubbles,) | **wp.float64** | `mrFlow3D.h:20` | 各气泡初始体积（气体定律 PV=const 的参考值） |
| `bubble_rho` | (max_bubbles,) | **wp.float64** | `mrFlow3D.h:21` | 各气泡气体密度（ρ_new = ρ_old · V_init / V） |
| `bubble_count` | 标量 | wp.int32 | `mrFlow3D.h:26` | 活跃气泡计数 |
| `merge_flag` | 标量 | wp.int32 | `mrFlow3D.h:62` | 全局合并事件标记（MergeSplitDetectorKernel 设置） |
| `split_flag` | 标量 | wp.int32 | `mrFlow3D.h:63` | 全局分裂事件标记（report_split 通过 atomicExch 设置） |

**方法**：
- `clear()`：归零所有数组。`flag` 全部设为 `TYPE_G`（0x20，与参考代码 `Create()` 中初始化一致，见 `mrFlow3D.h:201`），`tag_matrix`/`previous_tag`/`previous_merge_tag` 全部设为 -1，`merge_detector` 全部设为 `False`，`islet` 全部设为 0。注意：HOME-FSLBM 不使用 `solid_phi`/`solid_body_id`（与现有 `LbmState` 不同）——固体边界通过 `flag` 字段的 `TYPE_S` 位表示
- `clone()`：通过 `wp.copy` 深拷贝所有 warp 数组
- `clear_forces()`：DomainState 协议要求的方法。HOME-FSLBM 的力场在每个时间步由 kernel 直接写入，不积累，故此方法为空操作

### 4.4 `kernels_fluid.py` [P0]

**文件用途**：HOME-LBM 流体求解器的核心 kernel 文件。实现从 HOME 矩（10 个标量/格点）出发的完整 D3Q27 碰撞-流传输管线。

**依赖**：`constants.py`（格子常数、权重）、`model.py`（松弛参数）、`state.py`（flag/mass/phi 场量访问）

**Kernel 清单**：

| 名称 | 类型 | 参考行号 | 功能 |
|------|------|---------|------|
| `f_eq_d3q27` | @wp.func | `mrUtilFuncGpu3D.h:292-320` | D3Q27 Maxwell-Boltzmann 平衡态。输入 (ρ, ux, uy, uz)，通过预计算的 `u0=ux+uy` 等组合项和 fma 指令优化，返回 27 个方向的 feq[i]。权重使用 W0/W1/W2/W3 常量 |
| `reconstruct_distribution` | @wp.func | `mrUtilFuncGpu3D.h:153-266` | 四阶 Hermite 展开：从 HOME 矩 (ρ, ρu, ρS) 重建单个方向 i 的分布 f_i。计算 A0/Ax/Ay/Az/Axx/Axy/Axz/Ayy/Ayz/Azz/Axxy/... 等展开系数，对 27 个方向用 switch-case 展开 |
| `compute_pi_from_moments` | @wp.func | `mrUtilFuncGpu3D.h:424-471` | NOCM-MRT 闭式碰撞更新。输入 (ρ, u, F, ω, Π_αβ)，输出碰撞后 6 个应力分量。公式见附录 B |
| **`stream_collide_bvh_kernel`** | @wp.kernel | `mrLbmSolverGpu3D.cu:703-1057` | **唯一流体 kernel**——融合拉式流传输、Hermite重建、自由表面质量交换、涡粘性湍流、NOCM-MRT碰撞 |

**`stream_collide_bvh_kernel` 完整实现计划**：

```
@wp.kernel
def stream_collide_bvh_kernel(
    f_mom: wp.array(dtype=float),           # [in]  HOME moments (10×N flat)
    f_mom_post: wp.array(dtype=float),      # [out] post-collision moments
    flag: wp.array3d(dtype=wp.uint8),       # cell type bitfield
    mass: wp.array3d(dtype=float),          # [in/out] VOF mass
    phi: wp.array3d(dtype=float),           # [out] VOF fill level
    tag_matrix: wp.array3d(dtype=wp.int32), # bubble ID
    previous_tag: wp.array3d(dtype=wp.int32),
    bubble_volume: wp.array(dtype=wp.float64),
    bubble_init_volume: wp.array(dtype=wp.float64),
    bubble_rho: wp.array(dtype=wp.float64),
    force_x: wp.array3d(dtype=float),
    force_y: wp.array3d(dtype=float),
    force_z: wp.array3d(dtype=float),
    disjoin_force: wp.array3d(dtype=float),
    vel_solid_u: wp.array3d(dtype=float),   # moving-wall solid velocity
    vel_solid_v: wp.array3d(dtype=float),
    vel_solid_w: wp.array3d(dtype=float),
    solid_phi: wp.array3d(dtype=float),     # SDF for rigid-body coupling
    omega_plus: float, omega_minus: float,  # TRT collision frequencies
    has_moving_walls: int,                   # bool flag
    px: int, py: int, pz: int,              # periodic flags
    nx: int, ny: int, nz: int,
    stride: int,                             # = nx*ny*nz
):
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    
    # ---- STEP 0: early-out for solids and pure gas (lines 703-728) ----
    flagsn = flag[i, j, k]
    flagsn_bo = flagsn & TYPE_BO_MASK
    flagsn_su = flagsn & TYPE_SU_MASK
    
    if flagsn_bo == TYPE_S:
        # solid: copy rho, zero velocity/stress, return
        f_mom_post[idx] = f_mom[idx]  # keep density
        for d in range(1, 10):
            f_mom_post[d * stride + idx] = 0.0
        return
    
    if flagsn_su == TYPE_G:
        return  # pure gas: no evolution
    
    # ---- STEP 1: Extract HOME moments (lines 729-740) ----
    rho = f_mom[0 * stride + idx]
    ux  = f_mom[1 * stride + idx]
    uy  = f_mom[2 * stride + idx]
    uz  = f_mom[3 * stride + idx]
    Sxx = f_mom[4 * stride + idx] + CS2
    Sxy = f_mom[5 * stride + idx]
    Sxz = f_mom[6 * stride + idx]
    Syy = f_mom[7 * stride + idx] + CS2
    Syz = f_mom[8 * stride + idx]
    Szz = f_mom[9 * stride + idx] + CS2
    
    # ---- STEP 2: Pull-streaming — gather from 27 neighbors (lines 741-777) ----
    f_streamed = array_27()
    for di in range(27):
        ni = (i - CX[di])  # pull from neighbor
        nj = (j - CY[di])
        nk = (k - CZ[di])
        # periodic wrapping
        if px: ni = (ni + nx) % nx
        if py: nj = (nj + ny) % ny
        if pz: nk = (nk + nz) % nz
        # bounds check
        if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
            f_streamed[di] = reconstruct_distribution(rho, ux, uy, uz,
                          Sxx, Sxy, Sxz, Syy, Syz, Szz, OPPOSITE[di])
        else:
            nflag = flag[ni, nj, nk]
            nflag_bo = nflag & TYPE_BO_MASK
            if nflag_bo == TYPE_S:
                # bounce-back with moving-wall correction
                vw_dot = solid_wall_velocity_dot(
                    vel_solid_u, vel_solid_v, vel_solid_w, ni, nj, nk, di)
                # reconstruct from neighbor moments and bounce-back
                f_streamed[di] = bounce_back_reconstruct(
                    f_mom, ni, nj, nk, OPPOSITE[di], vw_dot, stride)
            else:
                # normal stream: reconstruct from neighbor moments
                nidx = ni * ny * nz + nj * nz + nk
                nr = f_mom[0*stride+idx]; nu = f_mom[1*stride+idx]
                nv = f_mom[2*stride+idx]; nw = f_mom[3*stride+idx]
                nSxx = f_mom[4*stride+idx]+CS2
                nSxy = f_mom[5*stride+idx]
                nSxz = f_mom[6*stride+idx]
                nSyy = f_mom[7*stride+idx]+CS2
                nSyz = f_mom[8*stride+idx]
                nSzz = f_mom[9*stride+idx]+CS2
                f_streamed[di] = reconstruct_distribution(
                    nr, nu, nv, nw, nSxx, nSxy, nSxz, nSyy, nSyz, nSzz, di)
    
    # ---- Also reconstruct outgoing distributions from current cell (fon, lines 790-803) ----
    fon = array_27()
    for di in range(27):
        cr = f_mom[0 * stride + cur_idx]
        cu = f_mom[1 * stride + cur_idx]
        cv = f_mom[2 * stride + cur_idx]
        cw = f_mom[3 * stride + cur_idx]
        cSxx = f_mom[4 * stride + cur_idx] + CS2
        cSxy = f_mom[5 * stride + cur_idx]
        cSxz = f_mom[6 * stride + cur_idx]
        cSyy = f_mom[7 * stride + cur_idx] + CS2
        cSyz = f_mom[8 * stride + cur_idx]
        cSzz = f_mom[9 * stride + cur_idx] + CS2
        fon[di] = reconstruct_distribution(
            cr, cu, cv, cw, cSxx, cSxy, cSxz, cSyy, cSyz, cSzz, di)
    
    # ---- STEP 3: Compute macroscopic moments from streamed distributions ----
    rho_new, ux_new, uy_new, uz_new = compute_rho_u(f_streamed)
    
    # ---- STEP 4: Free-surface mass exchange (lines 780-999) ----
    if flagsn_su == TYPE_I:
        # Collect neighbor phi values for curvature calculation
        phij = array_27()
        for di in range(27):
            ni = clamp_coord(i - CX[di], nx, px)
            nj = clamp_coord(j - CY[di], ny, py)
            nk = clamp_coord(k - CZ[di], nz, pz)
            phij[di] = phi[ni, nj, nk]
        
        curv = calculate_curvature(phij)  # PLIC Monge patch fit
        normal = calculate_normal(phij)   # 27-pt stencil
        
        # Gas pressure from bubble + Laplace + disjoining
        tag = tag_matrix[i, j, k]
        rho_k = 1.0
        sigma_k = SURFACE_TENSION
        if tag > 0:
            rho_k = bubble_rho[tag - 1]
            if bubble_init_volume[tag - 1] > 5000000.0:
                sigma_k = 1e-6
            if bubble_volume[tag - 1] < 64.0:
                sigma_k = 2e-4
        
        disjoint = disjoin_force[i, j, k]
        rho_laplace = sigma_k * curv
        gas_pressure = rho_k - rho_laplace - DISJOINT_FACTOR * disjoint
        
        # Velocity with Guo force correction for gas equilibrium
        fx = force_x[i, j, k]; fy = force_y[i,j,k]; fz = force_z[i,j,k]
        inv_2rho = 0.5 / rho_new
        u_eq_x = clamp(ux_new + fx * inv_2rho, -0.4, 0.4)
        u_eq_y = clamp(uy_new + fy * inv_2rho, -0.4, 0.4)
        u_eq_z = clamp(uz_new + fz * inv_2rho, -0.4, 0.4)
        
        feg = array_27()
        f_eq_d3q27(gas_pressure, u_eq_x, u_eq_y, u_eq_z, feg)
        
        # Mass exchange: accumulate mass from fluid/interface neighbors
        massn = mass[i, j, k]
        for di in range(1, 27):
            ni = clamp_coord(i - CX[di], nx, px)
            nj = clamp_coord(j - CY[di], ny, py)
            nk = clamp_coord(k - CZ[di], nz, pz)
            nflag_su = flag[ni, nj, nk] & TYPE_SU_MASK
            if nflag_su & (TYPE_F | TYPE_I):
                nphi = phi[ni, nj, nk]
                inv_di = OPPOSITE[di]
                dflux = f_streamed[di] - fon[inv_di]
                if nflag_su == TYPE_F:
                    massn += dflux
                else:  # TYPE_I
                    massn += 0.5 * (nphi + phij[0]) * dflux
            elif nflag_su == TYPE_G:
                # Fill gas-side incoming distribution
                f_streamed[di] = feg[OPPOSITE[di]] - fon[OPPOSITE[di]] + feg[di]
        
        mass[i, j, k] = massn
        phi[i, j, k] = calculate_phi(rho_new, massn, TYPE_I)
    
    # ---- STEP 5: Turbulence eddy viscosity (lines 1001-1028) ----
    Omega = omega_minus  # default
    for dij in range(-3, 3):
        for djk in range(-3, 3):
            for dkh in range(-3, 3):
                ni, nj, nk = i + djk, j + dij, k + dkh
                if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                    ntag = tag_matrix[ni, nj, nk]
                    if ntag > 0 and bubble_volume[ntag - 1] < 5000000.0:
                        # Frobenius norm of strain rate
                        invR = 1.0 / rho_new
                        xx = Sxx * invR - CS2
                        yy = Syy * invR - CS2
                        zz = Szz * invR - CS2
                        xy = Sxy * invR
                        xz = Sxz * invR
                        yz = Syz * invR
                        nu_e = 4.0 * sqrt(xx*xx + 2*xy*xy + 2*xz*xz + yy*yy + 2*yz*yz + zz*zz)
                        Omega = 1.0 / ((nu_e + 1e-4) * 3.0 + 0.5)
                        break  # first small bubble found
            if Omega != omega_minus: break
        if Omega != omega_minus: break
    
    # ---- STEP 6: Compute stress tensor from streamed distributions ----
    Pixx, Pixy, Pixz, Piyy, Piyz, Pizz = compute_stress(f_streamed)
    
    # ---- STEP 7: NOCM-MRT collision (lines 1030-1055) ----
    fx = force_x[i,j,k]; fy = force_y[i,j,k]; fz = force_z[i,j,k]
    Pixx, Pixy, Pixz, Piyy, Piyz, Pizz = compute_pi_from_moments(
        rho_new, ux_new, uy_new, uz_new, fx, fy, fz, Omega,
        Pixx, Pixy, Pixz, Piyy, Piyz, Pizz)
    
    # Store post-collision moments
    invR = 1.0 / rho_new
    f_mom_post[0*stride + idx] = rho_new
    f_mom_post[1*stride + idx] = ux_new + fx * invR * 0.5
    f_mom_post[2*stride + idx] = uy_new + fy * invR * 0.5
    f_mom_post[3*stride + idx] = uz_new + fz * invR * 0.5
    f_mom_post[4*stride + idx] = Pixx * invR - CS2
    f_mom_post[5*stride + idx] = Pixy * invR
    f_mom_post[6*stride + idx] = Pixz * invR
    f_mom_post[7*stride + idx] = Piyy * invR - CS2
    f_mom_post[8*stride + idx] = Piyz * invR
    f_mom_post[9*stride + idx] = Pizz * invR - CS2
```

**启动配置**：`dim=(nx, ny, nz)`，每个线程处理一个格点单元。

**Warp 特定注意事项**：
- `@wp.func` 辅助函数必须在 kernel 外部定义为独立函数，Warp 会自动内联
- D3Q27 的 27 个方向在 kernel 内部以 for 循环处理（Warp 的编译器会展开），不需要像参考代码的 switch-case 那样手动展开全部 27 个方向
- 原子操作使用 `wp.atomic_add`（对应 CUDA `atomicAdd`），`wp.atomic_min`（对应 `atomicMin`）
- 周期性包装使用 `%` 运算符（Warp 支持有符号整数取模）

### 4.5 `kernels_surface.py` [P0]

**文件用途**：VOF 自由表面追踪和 PLIC 界面几何计算。包含三阶段表面标记传播、VOF 质量重分配、Monge 曲面片曲率拟合、PLIC 单位立方体相交等全 GPU kernel。

**依赖**：`constants.py`（flag 枚举、CS2）、`state.py`（flag/mass/phi 场量）

**Kernel 清单**：

| 名称 | 类型 | 参考行号 | 功能 |
|------|------|---------|------|
| `calculate_phi` | @wp.func | `mrUtilFuncGpu3D.h` | 由 (ρ, mass, flag) 计算填充水平。TYPE_F → 1.0，TYPE_G → 0.0，TYPE_I → clamp(mass/ρ, 0, 1) |
| `calculate_normal` | @wp.func | `mrUtilFuncGpu3D.h:320-380` | 27 点模板有限差分计算 ∇φ。`bx = Σ w_i c_i φ_i` 等三方向加权求和，L2 归一化 |
| `calculate_curvature` | @wp.func | `mrUtilFuncGpu3D.h:320-420` | Monge 曲面片二次拟合求平均曲率。收集邻居接口点 → 旋转至局部坐标 → 5×5 LU 最小二乘 → κ 公式 → 钳制 [-1,1] |
| `plic_cube` | @wp.func | `mrUtilFuncGpu3D.h:104-149` | SZ & Kawano 2022 方法：对称化简 → 5 情形分支 → 解析 d。详见附录 C |
| `surface_1_kernel` | @wp.kernel | `mrLbmSolverGpu3D.cu:444-477` | 接口标记传播。TYPE_IF 单元将邻居 TYPE_G 设为 TYPE_GI，阻止 TYPE_IG 出现在接口附近 |
| `surface_2_kernel` | @wp.kernel | `479-603` | GI 初始化。TYPE_GI 单元从邻居 TYPE_F/TYPE_I 平均 (ρ, u)，设为平衡态分布 |
| `surface_3_kernel` | @wp.kernel | `604-937` | VOF 质量重分配核心。逐单元按 flag 类型处理 mass/phi 转换，执行邻居间质量交换（VOF 平流），对接口单元施加 Laplace+气泡压力边界 |

**`surface_3_kernel` 完整实现计划**（对应参考代码第 604-937 行）：

```
@wp.kernel
def surface_3_kernel(
    f_mom: wp.array(dtype=float),           # [in]  HOME moments pre-surface
    f_mom_post: wp.array(dtype=float),      # [out] post-surface moments
    flag: wp.array3d(dtype=wp.uint8),       # [in/out] cell type bitfield
    mass: wp.array3d(dtype=float),          # [in/out] VOF mass
    massex: wp.array3d(dtype=float),        # [out] excess mass for redistribution
    phi: wp.array3d(dtype=float),           # [out] VOF fill level
    tag_matrix: wp.array3d(dtype=wp.int32), # [in/out] bubble ID
    previous_tag: wp.array3d(dtype=wp.int32), # [out] previous bubble ID
    force_x: wp.array3d(dtype=float),
    force_y: wp.array3d(dtype=float),
    force_z: wp.array3d(dtype=float),
    disjoin_force: wp.array3d(dtype=float),
    bubble_volume: wp.array(dtype=wp.float64),
    bubble_init_volume: wp.array(dtype=wp.float64),
    bubble_rho: wp.array(dtype=wp.float64),
    stride: int, nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    flagsn = flag[i, j, k]
    flagsn_sus = flagsn & (TYPE_SU_MASK | TYPE_S)
    
    if flagsn_sus & TYPE_S: return  # skip solids
    
    rhon = f_mom_post[0 * stride + idx]  # density from post-collision moments
    massn = mass[i, j, k]
    massexn = 0.0
    phin = 0.0
    
    # === Phase A: mass redistribution by flag type (lines 618-670) ===
    if flagsn_sus == TYPE_F:
        massexn = massn - rhon
        massn = rhon
        phin = 1.0
        previous_tag[i, j, k] = tag_matrix[i, j, k]
        tag_matrix[i, j, k] = -1
    elif flagsn_sus == TYPE_I:
        massexn = massn - rhon if massn > rhon else (massn if massn < 0.0 else 0.0)
        massn = wp.clamp(massn, 0.0, rhon)
        phin = calculate_phi(rhon, massn, TYPE_I)
    elif flagsn_sus == TYPE_G:
        massexn = massn
        massn = 0.0
        phin = 0.0
    elif flagsn_sus == TYPE_IF:
        flag[i, j, k] = wp.uint8((flagsn & ~TYPE_SU_MASK) | TYPE_F)
        previous_tag[i, j, k] = tag_matrix[i, j, k]
        tag_matrix[i, j, k] = -1
        massexn = massn - rhon
        massn = rhon
        phin = 1.0
    elif flagsn_sus == TYPE_IG:
        flag[i, j, k] = wp.uint8((flagsn & ~TYPE_SU_MASK) | TYPE_G)
        massexn = massn
        massn = 0.0
        phin = 0.0
    elif flagsn_sus == TYPE_GI:
        flag[i, j, k] = wp.uint8((flagsn & ~TYPE_SU_MASK) | TYPE_I)
        massexn = massn - rhon if massn > rhon else (massn if massn < 0.0 else 0.0)
        massn = wp.clamp(massn, 0.0, rhon)
        phin = calculate_phi(rhon, massn, TYPE_I)
    
    # === Phase B1: pull-streaming from neighbor moments (lines 729-777) ===
    fhn = array_27()
    for di in range(27):
        ni = wp.clamp(i - CX[di], 0, nx-1)
        nj = wp.clamp(j - CY[di], 0, ny-1)
        nk = wp.clamp(k - CZ[di], 0, nz-1)
        n_idx = ni * ny * nz + nj * nz + nk
        nr = f_mom_post[0 * stride + n_idx]
        nu = f_mom_post[1 * stride + n_idx]
        nv = f_mom_post[2 * stride + n_idx]
        nw = f_mom_post[3 * stride + n_idx]
        nSxx = f_mom_post[4 * stride + n_idx] + CS2
        nSxy = f_mom_post[5 * stride + n_idx]
        nSxz = f_mom_post[6 * stride + n_idx]
        nSyy = f_mom_post[7 * stride + n_idx] + CS2
        nSyz = f_mom_post[8 * stride + n_idx]
        nSzz = f_mom_post[9 * stride + n_idx] + CS2
        fhn[di] = reconstruct_distribution(nr, nu, nv, nw,
                    nSxx, nSxy, nSxz, nSyy, nSyz, nSzz, di)
    
    # === Phase B2: reconstruct outgoing distributions from current cell (fon, lines 790-803) ===
    ux = f_mom_post[1 * stride + idx]
    uy = f_mom_post[2 * stride + idx]
    uz = f_mom_post[3 * stride + idx]
    Sxx = f_mom_post[4 * stride + idx] + CS2
    Sxy = f_mom_post[5 * stride + idx]
    Sxz = f_mom_post[6 * stride + idx]
    Syy = f_mom_post[7 * stride + idx] + CS2
    Syz = f_mom_post[8 * stride + idx]
    Szz = f_mom_post[9 * stride + idx] + CS2
    
    fon = array_27()
    for di in range(27):
        fon[di] = reconstruct_distribution(rhon, ux, uy, uz,
                    Sxx, Sxy, Sxz, Syy, Syz, Szz, di)
    
    # === Phase C: interface pressure boundary (lines 840-910) ===
    if flagsn_sus == TYPE_I:
        phij = array_27()
        for di in range(27):
            ni = wp.clamp(i - CX[di], 0, nx-1)
            nj = wp.clamp(j - CY[di], 0, ny-1)
            nk = wp.clamp(k - CZ[di], 0, nz-1)
            phij[di] = phi[ni, nj, nk]
        
        curv = calculate_curvature(phij)
        tag = tag_matrix[i, j, k]
        rho_k = 1.0
        sigma_k = SURFACE_TENSION
        if tag > 0:
            rho_k = bubble_rho[tag - 1]
            if bubble_init_volume[tag - 1] > 5000000.0:
                sigma_k = 1e-6
            if bubble_volume[tag - 1] < 64.0:
                sigma_k = 2e-4
        
        disjoint = disjoin_force[i, j, k]
        rho_laplace = sigma_k * curv
        gas_pressure = rho_k - rho_laplace - DISJOINT_FACTOR * disjoint
        
        fx = force_x[i,j,k]; fy = force_y[i,j,k]; fz = force_z[i,j,k]
        inv_2rho = 0.5 / rhon
        u_eq = normalize_clamp(
            wp.vec3(ux + fx * inv_2rho, uy + fy * inv_2rho, uz + fz * inv_2rho), 0.4)
        
        feg = array_27()
        f_eq_d3q27(gas_pressure, u_eq.x, u_eq.y, u_eq.z, feg)
    
    # === Phase D: mass exchange with 27 neighbors (lines 926-935) ===
    for di in range(1, 27):
        ni = wp.clamp(i - CX[di], 0, nx-1)
        nj = wp.clamp(j - CY[di], 0, ny-1)
        nk = wp.clamp(k - CZ[di], 0, nz-1)
        nflags_su = flag[ni, nj, nk] & TYPE_SU_MASK
        if nflags_su & (TYPE_F | TYPE_I):
            nphi = phi[ni, nj, nk]
            opp_di = OPPOSITE[di]
            dflux = fhn[di] - fon[opp_di]
            if nflags_su == TYPE_F:
                massn += dflux
            else:
                massn += 0.5 * (nphi + phin) * dflux
    
    # === Phase E: gas-side boundary fill for interface cells ===
    if flagsn_sus == TYPE_I:
        for di in range(1, 27):
            ni = wp.clamp(i - CX[di], 0, nx-1)
            nj = wp.clamp(j - CY[di], 0, ny-1)
            nk = wp.clamp(k - CZ[di], 0, nz-1)
            nflags_su = flag[ni, nj, nk] & TYPE_SU_MASK
            if nflags_su == TYPE_G:
                opp_di = OPPOSITE[di]
                fhn[di] = feg[opp_di] - fon[opp_di] + feg[di]
    
    # === Phase F: write-back (lines 937-1055 continued in stream_collide_bvh) ===
    mass[i, j, k] = massn
    massex[i, j, k] = massexn
    phi[i, j, k] = phin
    # Recompute moments from updated fhn and store in f_mom_post
    rho_new, ux_new, uy_new, uz_new = compute_rho_u(fhn)
    Pixx, Pixy, Pixz, Piyy, Piyz, Pizz = compute_stress(fhn)
    invR = 1.0 / rho_new
    f_mom_post[0*stride + idx] = rho_new
    f_mom_post[1*stride + idx] = ux_new
    f_mom_post[2*stride + idx] = uy_new
    f_mom_post[3*stride + idx] = uz_new
    f_mom_post[4*stride + idx] = Pixx * invR - CS2
    f_mom_post[5*stride + idx] = Pixy * invR
    f_mom_post[6*stride + idx] = Pixz * invR
    f_mom_post[7*stride + idx] = Piyy * invR - CS2
    f_mom_post[8*stride + idx] = Piyz * invR
    f_mom_post[9*stride + idx] = Pizz * invR - CS2
```

**`calculate_curvature` 完整实现计划**（对应参考代码 `mrUtilFuncGpu3D.h:320-420`）：

```
@wp.func
def calculate_curvature(phij: array_27) -> float:
    # Step 1: compute interface normal n = -grad(phi) via 27-pt stencil
    # Using D3Q27 weights: w_rest=0, w_face=4, w_edge=2, w_corner=1
    bx = wp.vec3(0.0, 0.0, 0.0)
    for di in range(1, 27):
        weight = 4.0 if di <= 6 else (2.0 if di <= 18 else 1.0)
        bx.x += weight * float(CX[di]) * phij[di]
        bx.y += weight * float(CY[di]) * phij[di]
        bx.z += weight * float(CZ[di]) * phij[di]
    
    if wp.length(bx) < 1e-10:
        return 0.0  # degenerate normal
    bz = wp.normalize(bx)  # align primary axis with normal
    
    # Step 2: build orthonormal local frame (bx, by, bz)
    rn = wp.vec3(0.56270900, 0.32704452, 0.75921047)
    by = wp.normalize(wp.cross(bz, rn))
    bx_local = wp.cross(by, bz)
    
    # Step 3: collect neighbor interface points in local frame
    center_offset = plic_cube(phij[0], bz)
    p = array_24()  # max 24 interface neighbors (27 - 1 gas - 1 fluid - 1 self)
    number = 0
    for di in range(1, 27):
        if 0.0 < phij[di] < 1.0:  # interface neighbor
            ei = wp.vec3(float(CX[di]), float(CY[di]), float(CZ[di]))
            offset = plic_cube(phij[di], bz) - center_offset
            p_i = wp.vec3(wp.dot(ei, bx_local), wp.dot(ei, by), wp.dot(ei, bz) + offset)
            p[number] = p_i
            number += 1
    
    if number < 5:
        return 0.0  # fallback: too few points for quadratic fit
    
    # Step 4: least-squares Monge patch fit: z = A*x^2 + B*y^2 + C*x*y + H*x + I*y
    M = zeros_5x5()
    b_vec = wp.vec5(0.0, 0.0, 0.0, 0.0, 0.0)
    for idx in range(number):
        x, y, z = p[idx].x, p[idx].y, p[idx].z
        x2, y2 = x*x, y*y
        x3, y3 = x2*x, y2*y
        # Accumulate upper triangular (M is symmetric)
        M[0][0] += x2*x2; M[0][1] += x2*y2; M[0][2] += x3*y; M[0][3] += x3; M[0][4] += x2*y
        M[1][1] += y2*y2; M[1][2] += x*y3; M[1][3] += x*y2; M[1][4] += y3
        M[2][2] += x2*y2; M[2][3] += x2*y; M[2][4] += x*y2
        M[3][3] += x2; M[3][4] += x*y
        M[4][4] += y2
        b_vec[0] += x2*z; b_vec[1] += y2*z; b_vec[2] += x*y*z
        b_vec[3] += x*z; b_vec[4] += y*z
    
    # Fill lower triangular by symmetry (50% arithmetic savings per reference comment)
    for r in range(1, 5):
        for c in range(r):
            M[r][c] = M[c][r]
    
    # Solve M*x = b via LU decomposition (5x5, in-place)
    x_sol = lu_solve_5x5(M, b_vec, number if number < 5 else 5)
    A, B, C, H, I = x_sol[0], x_sol[1], x_sol[2], x_sol[3], x_sol[4]
    
    # Step 5: mean curvature of Monge patch
    denom = pow(H*H + I*I + 1.0, 1.5)
    K = (A*(I*I + 1.0) + B*(H*H + 1.0) - C*H*I) / max(denom, 1e-10)
    
    return wp.clamp(K, -1.0, 1.0)
```

### 4.6 `kernels_bubble.py` [P1]

**文件用途**：气泡子系统的全部 GPU kernel。包含 YACCLAB 三维连通分量标记（CCL）的完整 Warp 重实现、气泡体积/密度归约与更新、合并/分裂检测与标签冲突解决。

**依赖**：`constants.py`（flag 枚举、CCL 阈值）、`state.py`（tag_matrix/input_matrix/label_matrix 等气泡场量）

**完整 YACCLAB CCL 复现**（审核项 B1）。共 15 个 kernel：

| # | 名称 | 类型 | SRC 行号 | 备注 |
|---|------|------|---------|------|
| 1 | **`report_split`** | @wp.func | 53-61 | ← 新增 |
| 2 | **`clear_detector`** | @wp.kernel | 64-107 | ← 新增 |
| 3 | **`clear_inlet`** | @wp.kernel | 110-142 | ← 新增 |
| 4 | `init_tag_kernel` | @wp.kernel | 1089-1112 | |
| 5 | `convert_flag_to_input_kernel` | @wp.kernel | 1115-1135 | |
| 6 | `parse_label_kernel` | @wp.kernel | 1066-1088 | |
| 7 | `create_bubble_label_kernel` | @wp.kernel | 1139-1148 | |
| 8 | `update_init_tag_kernel` | @wp.kernel | 1150-1169 | |
| 9 | **`ResetLabelVolume`** | @wp.kernel | 1185-1193 | ← 新增 |
| 10 | `bubble_volume_update_kernel` | @wp.kernel | 1301-1355 | |
| 11 | `bubble_rho_update_kernel` | @wp.kernel | 1356-1365 | |
| 12 | **`assign_tag_kernel`** | @wp.kernel | 1424-1450 | ← 新增 |
| 13 | **`recheck_merge_kernel`** | @wp.kernel | 1452-1475 | ← 新增 |
| 14 | **`MergeSplitDetectorKernel`** | @wp.kernel | 1366-1374 | ← 新增 |
| 15 | `handle_merge_spilt` | host+kernel | 1576-1613 | |

**YACCLAB CCL Warp 重实现策略**：
- 使用 `wp.array3d` 替代 OpenCV `GpuMat3`
- `wp.atomic_min` 替代 CUDA `atomicMin`
- CUDA 使用 `__syncthreads()` 处拆分为独立 kernel 启动
- **原型验证子阶段**：正式实现前，在 16³ 网格上用 3 个分离球体验证 CCL 输出与参考代码一致

**YACCLAB CCL 五阶段实现计划**（逐阶段对应 `tDCCL.cu`）：

```
阶段 1: InitLabeling (2×2×2 block)
  - 每个 block 处理 8 个体素（2×2×2），检查 8 邻域连通性
  - 为每个体素分配初始标签（基于其 2×2×2 内的最小邻接标签）
  - 使用原子操作写入全局 label 数组（对应 CUDA atomicMin）

阶段 2: MergeMasks (block-level merge)
  - 构建 block 间的等价标签表：对每个 block 边界上的体素对，
    若两个相邻 block 中的体素均被标记且标签不同，
    通过 atomicMin 将较大标签归并到较小标签
  - 输出的等价表是部分排序的——标签 i 的"父标签"≤i

阶段 3: LabelAnalysis (merge chain resolution)
  - 在等价表上运行传递闭包：对每个标签 i，
    若 parent[i] != i，则 parent[i] = parent[parent[i]]
  - 第二轮扫描将标签重新映射为密集的 0..N-1 编号

阶段 4: LabelReduction (global propagation)
  - 对全部体素索引启动 kernel，将 label 替换为 parent[label]
  - 输出全局一致的密集标签

阶段 5: FinalLabeling
  - 将最终标签写入 label_matrix
  - 确保标签从 1 开始（0 保留给无标签体素）
```

**关键 bubble kernel 详细实现计划**：

`bubble_volume_update_kernel` 完整伪代码（对应参考代码第 1301-1355 行）：
```
@wp.kernel
def bubble_volume_update_kernel(
    phi: wp.array3d(dtype=float),
    tag_matrix: wp.array3d(dtype=wp.int32),
    previous_tag: wp.array3d(dtype=wp.int32),
    bubble_volume: wp.array(dtype=wp.float64),
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    tag = tag_matrix[i, j, k]
    if tag > 0:
        wp.atomic_add(bubble_volume, tag - 1,
            wp.float64(1.0 - float(phi[i, j, k])))
    previous_tag[i, j, k] = tag_matrix[i, j, k]
    tag_matrix[i, j, k] = -1
```

`bubble_rho_update_kernel` 完整伪代码（对应参考代码第 1356-1365 行）：
```
@wp.kernel
def bubble_rho_update_kernel(
    bubble_volume: wp.array(dtype=wp.float64),
    bubble_init_volume: wp.array(dtype=wp.float64),
    bubble_rho: wp.array(dtype=wp.float64),
    bubble_count: int,
):
    b = wp.tid()
    if b >= bubble_count: return
    if bubble_volume[b] > 0.0:
        bubble_rho[b] = bubble_rho[b] * bubble_init_volume[b] / bubble_volume[b]
    if bubble_init_volume[b] > 5000000.0 and bubble_rho[b] == 1.0:
        bubble_init_volume[b] = bubble_rho[b] * bubble_volume[b]  # atmosphere
```

`assign_tag_kernel` + `recheck_merge_kernel` 并行标签冲突解决（对应参考代码第 1424-1475 行）：
```
@wp.kernel
def assign_tag_kernel(
    tag_matrix: wp.array3d(dtype=wp.int32),
    previous_merge_tag: wp.array3d(dtype=wp.int32),
    merge_detector: wp.array3d(dtype=wp.bool),
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    if merge_detector[i, j, k] and tag_matrix[i, j, k] <= 0:
        tag_matrix[i, j, k] = previous_merge_tag[i, j, k]  # restore lost tag

@wp.kernel
def recheck_merge_kernel(
    tag_matrix: wp.array3d(dtype=wp.int32),
    previous_tag: wp.array3d(dtype=wp.int32),
    merge_flag: wp.array(dtype=wp.int32),
    nx: int, ny: int, nz: int,
):
    tag = wp.tid() + 1  # tags are 1-based
    old_tags_found = 0
    first_old = -1
    for (i, j, k) in grid_range:  # scan all cells with this tag
        if tag_matrix[i, j, k] == tag:
            pt = previous_tag[i, j, k]
            if pt > 0 and pt != first_old:
                if first_old == -1:
                    first_old = pt
                else:
                    old_tags_found += 1
    if old_tags_found > 0:
        wp.atomic_add(merge_flag, 0, 1)  # true merge detected
```

### 4.7 `kernels_gas.py` [P1]

**CMR-MRT 碰撞**（审核项 B3）：

| 名称 | 类型 | SRC 行号 |
|------|------|---------|
| `g_reconstruction_kernel` | @wp.kernel | 1644-1748 |
| `g_stream_collide_kernel` | @wp.kernel | 1750-1848 |
| `convert_to_central_moment_d3q7` | @wp.func | `mrUtilFuncGpu3D.h:474-496` |
| `convert_from_central_moment_d3q7` | @wp.func | `mrUtilFuncGpu3D.h:499-518` |
| **`bubble_volume_g_update_kernel`** | @wp.kernel | **1853-1879** ← 新增 |

**`g_stream_collide_kernel` 完整伪代码**（对应参考代码第 1750-1848 行）：

```
@wp.kernel
def g_stream_collide_kernel(
    g_mom: wp.array(dtype=float),          # [in]  D3Q7 distributions (7*N flat)
    g_mom_post: wp.array(dtype=float),     # [out] post-collision gas
    flag: wp.array3d(dtype=wp.uint8),
    src: wp.array3d(dtype=float),          # gas source term
    c_value: wp.array3d(dtype=float),      # [out] dissolved gas concentration
    delta_g: wp.array3d(dtype=float),      # [in/out] gas flux accumulator
    stride: int, nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    flagsn = flag[i, j, k]
    flagsn_bo = flagsn & TYPE_BO_MASK
    flagsn_su = flagsn & TYPE_SU_MASK
    
    if flagsn_bo == TYPE_S or flagsn_su == TYPE_G:
        return  # skip solids and gas cells
    
    # === Pull-streaming from 7 neighbors ===
    pop_g = array_7()
    for di in range(7):
        ni = max(0, min(nx-1, i - CX[di]))
        nj = max(0, min(ny-1, j - CY[di]))
        nk = max(0, min(nz-1, k - CZ[di]))
        nidx = ni * ny * nz + nj * nz + nk
        nflag_bo = flag[ni, nj, nk] & TYPE_BO_MASK
        if nflag_bo == TYPE_S:
            pop_g[di] = g_eq_d3q7(c_value[i,j,k], 0.0, 0.0, 0.0, di)
        else:
            pop_g[di] = g_mom[nidx + di * stride]
    
    # === CMR-MRT collision in moment space ===
    ux = f_mom_post[1*stride + idx]  # use fluid velocity for CMR shift
    uy = f_mom_post[2*stride + idx]
    uz = f_mom_post[3*stride + idx]
    g_eq_k = array_7()
    rhon_g = pop_g[0]+pop_g[1]+pop_g[2]+pop_g[3]+pop_g[4]+pop_g[5]+pop_g[6]
    g_eq_d3q7_all(rhon_g, ux, uy, uz, g_eq_k)
    
    # Transform to central moments
    convert_to_central_moment_d3q7(ux, uy, uz, pop_g)
    convert_to_central_moment_d3q7(ux, uy, uz, g_eq_k)
    
    # Source term in moment space
    src_Q = array_7()
    for di in range(7):
        src_Q[di] = src[i, j, k] * W_GAS[di]
    convert_to_central_moment_d3q7(ux, uy, uz, src_Q)
    
    # Apply 7 independent relaxation rates
    s = [1.0, 1.0/0.9, 1.0/0.9, 1.0/0.9, 1.5, 1.5, 1.5]
    pop_out = array_7()
    for mi in range(7):
        src_m = src_Q[mi] * (1.0 - s[mi] * 0.5)
        pop_out[mi] = (1.0 - s[mi]) * pop_g[mi] + s[mi] * g_eq_k[mi] + src_m
    
    # Transform back to distributions and store
    convert_from_central_moment_d3q7(ux, uy, uz, pop_out)
    for di in range(7):
        g_mom_post[idx + di * stride] = pop_out[di]
    c_value[i, j, k] = rhon_g
```

**`bubble_volume_g_update_kernel` 完整伪代码**（对应参考代码第 1853-1879 行）：

```
@wp.kernel
def bubble_volume_g_update_kernel(
    delta_g: wp.array3d(dtype=float),
    phi: wp.array3d(dtype=float),
    tag_matrix: wp.array3d(dtype=wp.int32),
    bubble_init_volume: wp.array(dtype=wp.float64),
    factor: float,
    nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    tag = tag_matrix[i, j, k]
    if tag > 0:
        contrib = wp.float64(0.25 * factor * float(delta_g[i,j,k]) * float(phi[i,j,k]))
        wp.atomic_add(bubble_init_volume, tag - 1, contrib)
```

### 4.8 `kernels_foam.py` [P2]

**用途**：分离压力、大气压力、入口清理。

**Kernel 清单**（全部源自 `mrLbmSolverGpu3D.cu`）：

| 名称 | 类型 | SRC 行号 | 用途 |
|------|------|---------|------|
| `calculate_disjoint_kernel` | @wp.kernel | 162-272 | 对每个 TYPE_I 单元沿法向反方向进行射线投射（步长 0.2，最多 20 步=4.0 单元距离），命中不同气泡标签的接口时计算分离压力 P_disjoin = max(0, 1-d/4)，通过 atomicAdd 累加到 `disjoin_force` |
| `reset_disjoin_force_kernel` | @wp.kernel | 144-160 | 清零 `disjoin_force` 和 `massex` 累加器，为下一时间步准备 |
| `atmosphere_rho_update_kernel` | @wp.kernel | 1259-1283 | 开敞式水箱大气压力更新：对顶部边界附近的大气泡（体积 > 1,000,000），通过 `atomicExch(double)` 将气泡密度重置为 1.0，涉及 `N`、`l0p`、`roup`、`labma`、`u0p` 物理映射参数 |
| `atmosphere_volme_update_kernel` | @wp.kernel | 1286-1298 | 单线程 kernel，在大气密度更新后修正大气初始体积 `init_volume = rho * volume`（审核项 B10 新增） |

**`calculate_disjoint_kernel` 完整伪代码**（对应参考代码第 162-272 行）：

```
@wp.kernel
def calculate_disjoint_kernel(
    flag: wp.array3d(dtype=wp.uint8),
    phi: wp.array3d(dtype=float),
    mass: wp.array3d(dtype=float),
    f_mom: wp.array(dtype=float),
    tag_matrix: wp.array3d(dtype=wp.int32),
    disjoin_force: wp.array3d(dtype=float),
    stride: int, nx: int, ny: int, nz: int,
):
    i, j, k = wp.tid()
    idx = i * ny * nz + j * nz + k
    flagsn = flag[i, j, k]
    if (flagsn & TYPE_SU_MASK) != TYPE_I:
        return  # only interface cells
    
    # Collect 27 phi values from neighbors (with solid-neighbor fallback)
    phij = array_27()
    massn = mass[i, j, k]
    for di in range(1, 27):
        ni = max(0, min(nx-1, i - CX[di]))
        nj = max(0, min(ny-1, j - CY[di]))
        nk = max(0, min(nz-1, k - CZ[di]))
        nflag = flag[ni, nj, nk]
        if (nflag & TYPE_BO_MASK) == TYPE_S:
            # Solid neighbor: search 6 face neighbors for non-solid phi
            found = False
            for fj in range(1, 7):
                si = ni - CX[fj]; sj = nj - CY[fj]; sk = nk - CZ[fj]
                if 0 <= si < nx and 0 <= sj < ny and 0 <= sk < nz:
                    if (flag[si, sj, sk] & TYPE_BO_MASK) != TYPE_S:
                        phij[di] = phi[si, sj, sk]
                        found = True
                        break
            if not found:
                phij[di] = phi[ni, nj, nk]
        else:
            phij[di] = phi[ni, nj, nk]
        massn += massex[ni, nj, nk]  # accumulate excess mass for fill level
    phij[0] = calculate_phi(f_mom[0*stride+idx], massn, flagsn)
    
    # Compute normal and ray-cast
    normal = calculate_normal(phij)
    tag_cur = tag_matrix[i, j, k] - 1
    disjoint = 0.0
    
    for step in range(1, 20):
        rx = int(round(float(i) - float(step) * 0.2 * normal.x))
        ry = int(round(float(j) - float(step) * 0.2 * normal.y))
        rz = int(round(float(k) - float(step) * 0.2 * normal.z))
        if 0 <= rx < nx and 0 <= ry < ny and 0 <= rz < nz:
            rtag = tag_matrix[rx, ry, rz]
            if rtag > 0 and flag[rx, ry, rz] == TYPE_I:
                rtag_idx = rtag - 1
                if tag_cur != rtag_idx:
                    center_off = plic_cube(phij[0], normal)
                    alpha = phi[rx, ry, rz]
                    d = abs(float(step) * 0.2 * normal.x) - (1.0 - alpha)
                    dist = abs(d / max(abs(normal.x), 1e-8)) - center_off
                    dj = 1.0 - dist / 4.0
                    if dj > disjoint:
                        disjoint = dj
    
    if disjoint > 0.0:
        wp.atomic_add(disjoin_force, idx, disjoint)
```

### 4.9 `solver.py` [P0]

按审核项 B5 重写为两阶段结构：

```python
def step(self, state_in, state_out, dt, contacts=None, control=None):
    # === 阶段一: coupling() — 气泡 + 气体子系统 ===
    # 1a. 气泡标签传播与冲突解决
    wp.launch(kernels_bubble.get_tag_kernel, ...)
    wp.launch(kernels_bubble.assign_tag_kernel, ...)
    wp.launch(kernels_bubble.recheck_merge_kernel, ...)
    # 1b. 气泡体积/密度更新
    wp.launch(kernels_bubble.bubble_volume_update_kernel, ...)
    wp.launch(kernels_bubble.bubble_rho_update_kernel, ...)
    wp.launch(kernels_bubble.MergeSplitDetectorKernel, ...)
    if merge_flag or split_flag:
        self._handle_merge_split(state_out)  # CCL + 标签处理
    # 1c. 气体子系统
    wp.launch(kernels_gas.g_reconstruction_kernel, ...)
    wp.launch(kernels_gas.g_stream_collide_kernel, ...)  # CMR-MRT
    wp.launch(kernels_gas.bubble_volume_g_update_kernel, ...)
    wp.copy(state_out.g_mom, state_out.g_mom_post)
    wp.launch(kernels_bubble.bubble_rho_update_kernel, ...)

    # === 阶段二: mrSolver3DGpu() — 流体 + 自由表面子系统 ===
    wp.launch(kernels_foam.calculate_disjoint_kernel, ...)
    if self._step_count == 57595:  # 条件 clear_inlet
        wp.launch(kernels_bubble.clear_inlet, ...)
    wp.launch(kernels_foam.atmosphere_rho_update_kernel, ...)
    wp.launch(kernels_foam.atmosphere_volme_update_kernel, ...)
    # ★ 唯一流体 kernel（内联湍流 + 自由表面）
    wp.launch(kernels_fluid.stream_collide_bvh_kernel, ...)
    wp.launch(kernels_foam.reset_disjoin_force_kernel, ...)
    wp.launch(kernels_surface.surface_1_kernel, ...)
    wp.launch(kernels_surface.surface_2_kernel, ...)
    wp.launch(kernels_surface.surface_3_kernel, ...)
    wp.copy(state_out.f_mom, state_out.f_mom_post)
```

**辅助方法**：

- `_handle_merge_split(state)`：当 `merge_flag` 或 `split_flag` 置位时触发。内部执行完整 CCL 管线：`convertIntToUnsignedChar` → `connectedComponentLabeling` → `parse_label` → `ResetLabelVolume` → `reduce_label_rho` → `bubble_list_swap` → `num_rho_update_kernel` → `ClearDectector`（对应 `handle_merge_spilt` 第 1576-1631 行的完整流程）
- `_sync_bc_from_model()`：将 model 中的 BC 参数同步至 solver 端 device array
- `initialize_equilibrium(state, rho0=1.0, u0=(0,0,0))`：将 `f_mom` 初始化为指定 (ρ₀, u₀) 的平衡态

### 4.10 `domain.py` [P0]

**用途**：顶层 Domain 编排器，封装模型/求解器/双缓冲状态。

**类**：`HomeFslbmDomain(Domain)`

遵循现有 `LbmDomain`（`lbm/domain.py`）的精确模式：

```python
class HomeFslbmDomain(Domain):
    def __init__(self, model: HomeFslbmModel, solver: HomeFslbmSolver | None = None):
        self._model = model
        self._solver = solver or HomeFslbmSolver(model)
        self._state_in: HomeFslbmState | None = None
        self._state_out: HomeFslbmState | None = None

    @property
    def name(self) -> str:
        return "fluid_grid_home_fslbm"

    @property
    def model(self) -> HomeFslbmModel:
        return self._model

    @property
    def solver(self) -> HomeFslbmSolver:
        return self._solver

    @property
    def state(self) -> HomeFslbmState:
        if self._state_in is None:
            self.create_state()
        return self._state_in

    def create_state(self) -> HomeFslbmState:
        self._state_in = HomeFslbmState(self._model)
        self._state_out = HomeFslbmState(self._model)
        return self._state_in

    def step(self, dt: float, contacts=None) -> None:
        self._solver.step(self._state_in, self._state_out, dt)
        self._state_in, self._state_out = self._state_out, self._state_in

    def pre_step(self, dt: float) -> None:
        pass  # 预留钩子：外部力注入、控制更新等

    def post_step(self, dt: float) -> None:
        pass  # 预留钩子：MAC 速度场插值（如可视化需要）、导出量计算等
```

### 4.11 `__init__.py` [P0]

**用途**：公开 API 接口。

```python
from .domain import HomeFslbmDomain
from .model import HomeFslbmModel
from .solver import HomeFslbmSolver
from .state import HomeFslbmState

__all__ = [
    "HomeFslbmDomain",
    "HomeFslbmModel",
    "HomeFslbmSolver",
    "HomeFslbmState",
]
```

**注册更新**：
- `wanphys/_src/fluid/fluid_grid/__init__.py` —— 添加 `from .home_fslbm import HomeFslbmDomain, HomeFslbmModel, HomeFslbmSolver, HomeFslbmState`
- `wanphys/_src/fluid/__init__.py` —— 按需重导出

---

## 5. 算法详述与模块标注

### 5.1 HOME-LBM 矩编码 [kernels_fluid.py]

```
[ALGO: HOME-LBM 矩编码]
参考文献: 论文五 Eq.(1-17), 论文四 Eq.(16-17)
模块: kernels_fluid.py

HOME-LBM 的核心创新在于仅存储前三个速度矩（共 10 个标量：ρ, ρu_α, ρS_αβ），
而非全部 27 个分布函数。完整的 D3Q27 分布通过三阶 Hermite 展开按需重建。

矩定义:
  ρ  = Σ_i f_i                          （密度）
  ρu_α = Σ_i c_iα · f_i                 （动量，α∈{x,y,z}）
  ρS_αβ = Σ_i c_iα · c_iβ · f_i - ρ·c_s²·δ_αβ  （应力张量，6 个分量）

分布重建:
  f_i = w_i · [A₀ + 3·c_iα·A_α + (9/2)·c_iα·c_iβ·A_αβ + ...]

碰撞更新 (NOCM-MRT):
  Π_new = Π_eq + (1-ω)·(Π - Π_eq) + 力项

其中 Π = Σ_i c_iα·c_iβ·f_i 为二阶矩（能量张量）。
```

### 5.2 NOCM-MRT 碰撞 [kernels_fluid.py]

```
[ALGO: NOCM-MRT 碰撞算子]
参考文献: 论文五 Eq.(21-23), 论文四 Eq.(18-21)
模块: kernels_fluid.py (compute_pi_from_moments)
源文件: mrUtilFuncGpu3D.h:424-471

碰撞直接在应力张量上以闭式执行，而非对单个分布函数进行。
这是 HOME-LBM 的核心效率提升所在。

输入:  ρ, u_α, F_α, ω, Π_αβ（碰撞前应力）
输出: Π_αβ_new（碰撞后应力）

对角分量:
  Π_αα_new = ρ/3 + Π_αα^part·(1-ω) + (ρ/3)·u·u + ω·ρ·u_α²/3_修正 + F_α·u_α

非对角分量:
  Π_αβ_new = Π_αβ·(1-ω) + ω·ρ·u_α·u_β + (F_α·u_β + F_β·u_α)/2

其中 Π_αα^part 为对角应力的无迹分解。
完整伪代码见附录 B。
```

### 5.3 VOF 自由表面追踪 [kernels_surface.py]

```
[ALGO: VOF 锐界面自由表面追踪]
参考文献: 论文四 Eq.(9-11), Sec.4.1
模块: kernels_surface.py
源文件: mrLbmSolverGpu3D.cu:444-937

三相追踪，采用单元标记:
  TYPE_F（流体）:  φ = 1, mass = ρ
  TYPE_I（接口）:  0 < φ < 1, mass ∈ [0, ρ]
  TYPE_G（气体）:  φ = 0, mass = 0

三阶段表面更新:
  surface_1: 向外传播标记转换。对每个 TYPE_IF 单元，沿 27 个方向检查邻居
             ——若为 TYPE_IG 则保留为 TYPE_I（防止气体捕获），
             若为 TYPE_G 则设为 TYPE_GI（气体→接口转换）。
  surface_2: 初始化新创建的接口单元。对每个 TYPE_GI 单元，
             从其流体/接口邻居处平均 ρ、u，据此设置平衡态分布。
  surface_3: 质量重分配与标记状态转换。
             计算填充水平 φ = mass/ρ → 按标记类型转换状态 →
             执行邻居间质量交换（Δm 基于流入/流出分布差与填充水平加权）→
             对接口单元施加气体边界条件（Laplace 压力 + 分离压力）。

质量交换 (VOF 平流):
  Δm_i = φ_邻居为_F · (f_i^n - f_inv(i)^n)
       + φ_邻居为_I · 0.5(φ_j+φ_n)(f_i^n - f_inv(i)^n)

接口压力边界:
  P_gas = ρ_bubble - σ·κ - P_disjoin
```

### 5.4 PLIC 曲率 [kernels_surface.py]

```
[ALGO: PLIC（分段线性界面计算）]
参考文献: 论文四 Eq.(12)
模块: kernels_surface.py (calculate_curvature, calculate_normal, plic_cube)
源文件: mrUtilFuncGpu3D.h:104-149 (plic_cube), mrUtilFuncGpu3D.h:320-420 (curvature)

1. 法向估计: n = -∇φ / |∇φ|（27 点有限差分）
2. 局部坐标系旋转: 使 n 对齐 z 轴
3. PLIC 偏移: 各接口邻居偏移 d_i = plic_cube(φ_i, n)
4. Monge 曲面片拟合: z(x,y) = A·x² + B·y² + C·xy + H·x + I·y
   通过最小二乘法（5×5 LU 分解）对邻居接口点求解
5. 平均曲率:
   κ = [A(1+I²) + B(1+H²) - C·H·I] / (1+H²+I²)^(3/2)

plic_cube: 单位立方体-平面相交体积的解析解
（SZ & Kawano 2022 方法，采用简化对称情形）。
完整伪代码见附录 C。
```

### 5.5 气泡模型 — YACCLAB CCL [kernels_bubble.py]

```
[ALGO: YACCLAB 并行气泡追踪]
参考文献: 论文四 Sec.5, Eq.(22-25)
模块: kernels_bubble.py
源文件: tDCCL.cu（578 行），mrLbmSolverGpu3D.cu:1089-1631

管线:
  初始化: 在 TYPE_I/TYPE_G 单元上执行三维连通分量标记 (CCL)
          InitLabeling → MergeMasks → LabelAnalysis → LabelReduction → FinalLabeling
          → 为每个连通域分配唯一气泡 ID (tag_matrix, 密集编号 0..N-1)

  每步:
    1. 体积更新:   V_b = Σ_{i∈bubble} (1 - φ_i)  [通过 atomicAdd(double) 实现]
    2. 气体定律:   ρ_b_new = ρ_b_old · V_init / V_new  [PV = const]
    3. 表面张力调制:
       - 大气泡 (V > 5,000,000):  σ → 1e-6  （无表面张力）
       - 小气泡 (V < 64):        σ → 2e-4  （降低表面张力）
       - 分离状态（邻近其他气泡）: σ 保持不变

  合并检测:
    previous_tag[i] → tag_matrix[i]（对所有 TYPE_I 单元）
    若两个不同的 previous_tag 映射到同一当前 tag:
      守恒合并: V_new·ρ_new = V₁·ρ₁ + V₂·ρ₂
    assign_tag_kernel + recheck_merge_kernel 提供确定性标签冲突解决

  分裂检测:
    若一个 previous_tag 映射到多个不同的当前 tag:
      设置 split_flag = 1

  涡粘性（湍流模型）:
    ν_e = 4 · ‖S‖_F  （应用于气泡周围 3 个单元以内的所有单元）
    ‖S‖_F = sqrt(S_xx² + 2S_xy² + 2S_xz² + S_yy² + 2S_yz² + S_zz²)
    仅对小气泡启用（体积 < 5,000,000 格子单位³）
```

### 5.6 溶解气体 — CMR-MRT [kernels_gas.py]

```
[ALGO: D3Q7 CMR-MRT 溶解气体对流扩散]
参考文献: 论文四 Eq.(33-38)
模块: kernels_gas.py
源文件: mrLbmSolverGpu3D.cu:1750-1848（碰撞），mrUtilFuncGpu3D.h:474-518（变换函数）

D3Q7 CMR-MRT 碰撞:
  1. 正变换: 7 个分布 → 7 个中心矩
     mlConvertCmrMoment_d3q7(ux, uy, uz, distributions)
     每个格子速度减去局部流速 u 后计算伪距多项式
  2. 矩空间松弛: 7 个独立松弛率
     s[0]=1.0（密度矩瞬时平衡）
     s[1-3]=1/0.9≈1.111（动量矩高速松弛）
     s[4-6]=1.5（应力矩中速松弛）
  3. 反变换: 7 个中心矩 → 7 个分布
     mlConvertCmrF_d3q7(ux, uy, uz, moments)
     包含二次速度展开和高阶修正项

边界条件:
  - 固体: 零通量平衡态
  - 气体邻居: 含气体侧压力的平衡态
  - 接口: 亨利定律源项 c_sat = K_h · ρ_bubble / 4

气体通量→气泡体积更新:
  bubble_volume_g_update_kernel (mrLbmSolverGpu3D.cu:1853-1879)
  通过 atomicAdd(double) 将溶解气体通量累加到初始气泡体积
```

### 5.7 泡沫分离压力 [kernels_foam.py]

```
[ALGO: 泡沫分离压力]
参考文献: 论文四 Eq.(39-41)
模块: kernels_foam.py
源文件: mrLbmSolverGpu3D.cu:162-272

对每个 TYPE_I 单元:
  1. 计算接口法向 n
  2. 沿 -n 方向进行射线投射（步长 0.2，最多 20 步 = 4.0 单元距离）
  3. 当命中另一个具有不同气泡标签的接口时:
     d = 到另一接口的距离（考虑 PLIC 中心偏移修正）
     P_disjoin = max(0, 1 - d/4)
  4. 分离力抵消表面张力，阻止气泡聚并
```

### 5.8 湍流模型（内联）[kernels_fluid.py]

```
[ALGO: 涡粘性湍流模型]
参考文献: 论文四 Sec.5.1
模块: kernels_fluid.py（stream_collide_bvh_kernel 内部，第 1001-1028 行）

嵌入在 stream_collide_bvh_kernel 中，紧邻碰撞前执行:
  1. 搜索 ±3 邻域（6×6×6 = 216 个邻居单元）
  2. 寻找第一个包含小气泡（体积 < 5,000,000）的邻居
  3. 计算应变率张量 Frobenius 范数:
     ‖S‖_F = sqrt(S_xx² + 2S_xy² + 2S_xz² + S_yy² + 2S_yz² + S_zz²)
  4. 涡粘性系数 = 4 × ‖S‖_F
  5. 修改碰撞松弛率:
     Ω_new = 1 / ((4·‖S‖_F + 1e-4) · 3 + 0.5)
  6. 找到第一个小气泡后即停止搜索（break）

不可拆分为独立 kernel —— Ω 必须在修改后立即传入
mlGetPIAfterCollision 进行碰撞更新。
```

---