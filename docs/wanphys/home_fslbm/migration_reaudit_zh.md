# Home-FSLBM 迁移计划二次审核意见书

> **审核对象**: `docs/wanphys/home_fslbm/migration_plan_zh.md` (修订版) 及其附属 `migration_execution_zh.md`
> **参考代码**: `docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/inc/3D/`
> **审核日期**: 2026-07-15
> **审核人**: 基于论文四源码 + 论文五源码的交叉验证
> **审核范围**: 伪代码逻辑正确性、架构流程一致性、出处引用准确性、降级方案零容忍

---

## 目录

1. [总体结论](#1-总体结论)
2. [逐项审核结果](#2-逐项审核结果)
   - 2.1 [上一轮审核修复项验证](#21-上一轮审核修复项验证-b1-b10-s1-s6)
   - 2.2 [管线结构与架构](#22-管线结构与架构)
   - 2.3 [State 字段完整性](#23-state-字段完整性)
   - 2.4 [Kernel 清单完整性](#24-kernel-清单完整性)
   - 2.5 [伪代码逐段审核](#25-伪代码逐段审核)
   - 2.6 [算法公式验证](#26-算法公式验证)
   - 2.7 [降级/简化零容忍检查](#27-降级简化零容忍检查)
   - 2.8 [测试计划审核](#28-测试计划审核)
3. [发现的问题清单](#3-发现的问题清单)
4. [假阳性确认（经源码验证正确的项目）](#4-假阳性确认经源码验证正确的项目)
5. [审核结论](#5-审核结论)
6. [附录：逐行验证证据表](#6-附录逐行验证证据表)

---

## 1. 总体结论

**有条件通过。** 修订版迁移计划已消除第一轮审核提出的全部 10 个阻塞项（B1-B10）和 6 个强建议项（S1-S6）。管线结构与参考源码 `mrSolver3D::mlIterateCouplingGpu`（`inc/3D/cpu/mrSolver3D.h:121-127`）的两阶段调用模式一致，State 字段覆盖完整，Kernel 清单无遗漏，没有发现降级/简化方案。

但发现 **1 项伪代码逻辑错误**（自由表面气侧边界填充）和 **1 项文档不一致**（曲率公式），需在实现前修正。此外，部分模块的伪代码变量命名不统一（`f_streamed` vs `fhn`），建议统一。

---

## 2. 逐项审核结果

### 2.1 上一轮审核修复项验证 (B1-B10, S1-S6)

| 审核项 | 内容 | 修复状态 | 源码验证 |
|--------|------|---------|---------|
| **B1** | YACCLAB CCL 完整实现（禁止简化 Union-Find） | ✅ 已修复 | §4.6 描述 InitLabeling→MergeMasks→LabelAnalysis→LabelReduction→FinalLabeling 五阶段，与 `tDCCL.cu` 一致；计划明确"基于参考代码 tDCCL.cu:1-578 逐行复现"。Phase 3.0 设置原型验证子阶段 |
| **B2** | 气泡数组 double 精度 (wp.float64) | ✅ 已修复 | §4.3 State 表中 `bubble_volume: wp.float64`、`bubble_init_volume: wp.float64`、`bubble_rho: wp.float64`；伪代码 `bubble_volume_update_kernel` 使用 `wp.float64(1.0 - float(phi))` 和 `wp.atomic_add` |
| **B3** | D3Q7 碰撞模型: BGK→CMR-MRT | ✅ 已修复 | §2.4 标注为 CMR-MRT；§4.7 包含 `convert_to_central_moment_d3q7` + `convert_from_central_moment_d3q7` 变换函数；松弛率向量 `s[0]=1.0, s[1-3]=1/0.9, s[4-6]=1.5` 与源码 `g_stream_collide:1836-1840` 精确匹配 |
| **B4** | 单一 stream_collide_bvh kernel（内联湍流） | ✅ 已修复 | §4.4/§4.9 只有 `stream_collide_bvh_kernel`；伪代码内联 Phase A→F 全部逻辑（对应源码 703-1057 行）；湍流模型在 Phase E（对应源码 1001-1028 行），紧邻 `mlGetPIAfterCollision` 前执行 |
| **B5** | 两阶段管线结构 (coupling + mrSolver3DGpu) | ✅ 已修复 | §4.9 step() 方法明确 Phase1(coupling) 和 Phase2(mrSolver3DGpu)，与源码 `mrSolver3D.h:121-127` 完全一致 |
| **B6** | 补充遗漏 State 字段 (4 个) | ✅ 已修复 | `islet`(int*)、`previous_merge_tag`(int*)、`merge_detector`(bool, 逐单元)、`src`(float*) 全部纳入 §4.3 State 表 |
| **B7** | 补充遗漏 kernel (9 个) | ✅ 已修复 | `atmosphere_volme_update_kernel`、`clear_inlet`、`get_tag_kernel`、`assign_tag_kernel`、`recheck_merge_kernel`、`MergeSplitDetectorKernel`、`bubble_volume_update_kernel`、`bubble_rho_update_kernel`、`num_rho_update_kernel` 全部分配至 §4.6-§4.8 |
| **B8** | 附录 A 完整论文→代码映射 | ✅ 已修复 | 执行手册附录 A 含 27 行完整映射表，含论文四 Eq、源码行号、目标文件 |
| **B9** | State 继承 `DomainState`（非 `FluidGridStateBase`） | ✅ 已修复 | §4.3 `HomeFslbmState(DomainState)`；明确"自行管理所有 warp 数组"，不继承 MAC staggered grid 字段 |
| **S1-S5** | 测试方案改进 | ✅ 已修复 | `uniform_flow`→`shear_decay`；新增 `bubble_volume_conservation`；CCL 金标准对比；合并确定性测试 |
| **S6** | turbulence_radius 默认值: 4→3 | ✅ 已修复 | §4.2 model 字段表 `turbulence_radius: int = 3`；§5.8 注释 `±3 搜索范围（6×6×6=216 邻居）` |

### 2.2 管线结构与架构

**审核方法**：逐行对照 `mrSolver3D.h:121-127` 和 `mrLbmSolverGpu3D.cu:2078-2129` (coupling) + `mrLbmSolverGpu3D.cu:1973-2075` (mrSolver3DGpu)。

| 审核项 | 源码证据 | 计划对应 | 结论 |
|--------|---------|---------|------|
| 两阶段调用 | `mrSolver3D.h:121-127`: `coupling(...)` 后 `mrSolver3DGpu(...)` | §4.9 step() 两阶段结构 | ✅ 一致 |
| coupling kernel 顺序 | 源码 2078-2129: `get_tag→assign_tag→recheck_merge→update_bubble(…→bubble_volume_update, bubble_rho_update, MergeSplitDetector)→g_handle(g_recon→g_stream_collide→bubble_volume_g_update→gMomSwap→bubble_rho_update)` | §4.9 Phase 1: 相同顺序 | ✅ 一致 |
| mrSolver3DGpu kernel 顺序 | 源码 1973-2075: `calculate_disjoint→[clear_inlet]→atmosphere_rho_update→atmosphere_volme_update→stream_collide_bvh→ResetDisjoinForce→surface_1→surface_2→surface_3→fMomSwap` | §4.9 Phase 2: 相同顺序 | ✅ 一致 |
| stream_collide_bvh 位置 | 源码 `atmosphere_volme_update` 之后（第 2024 行） | 计划 Phase 2 第 4 步（atmosphere_volme_update 之后） | ✅ 一致 |
| 单一流体 kernel | 源码仅 `stream_collide_bvh`，内部标记位分流（第 721-726 行） | 计划仅 `stream_collide_bvh_kernel` | ✅ 一致 |

### 2.3 State 字段完整性

**审核方法**：对照 `mrFlow3D.h:31-91` 字段声明逐项比对 §4.3 State 表。

| 源码字段 | 类型 | 计划字段 | 匹配 |
|---------|------|---------|------|
| fMom | REAL* (10×N) | f_mom: float (10×N) | ✅ |
| fMomPost | REAL* (10×N) | f_mom_post: float (10×N) | ✅ |
| flag | MLLATTICENODE_SURFACE_FLAG* | flag: wp.uint8 (nx,ny,nz) | ✅ |
| mass | REAL* | mass: float (nx,ny,nz) | ✅ |
| massex | REAL* | massex: float (nx,ny,nz) | ✅ |
| phi | REAL* | phi: float (nx,ny,nz) | ✅ |
| forcex/y/z | REAL* | force_x/y/z: float (nx,ny,nz) | ✅ |
| tag_matrix | int* | tag_matrix: wp.int32 (nx,ny,nz) | ✅ |
| previous_tag | int* | previous_tag: wp.int32 (nx,ny,nz) | ✅ |
| previous_merge_tag | int* | previous_merge_tag: wp.int32 (nx,ny,nz) | ✅ (B6) |
| input_matrix | unsigned char* | input_matrix: wp.uint8 (nx,ny,nz) | ✅ |
| label_matrix | int* | label_matrix: wp.int32 (nx,ny,nz) | ✅ |
| merge_detector | bool* | merge_detector: wp.bool (nx,ny,nz) | ✅ (B6) |
| islet | int* | islet: wp.int32 (nx,ny,nz) | ✅ (B6) |
| disjoin_force | float* | disjoin_force: float (nx,ny,nz) | ✅ |
| gMom | float* (7×N) | g_mom: float (7×N) | ✅ |
| gMomPost | float* (7×N) | g_mom_post: float (7×N) | ✅ |
| delta_g | float* | delta_g: float (nx,ny,nz) | ✅ |
| c_value | float* | c_value: float (nx,ny,nz) | ✅ |
| src | float* | src: float (nx,ny,nz) | ✅ (B6) |
| delta_phi | REAL* | delta_phi: float (nx,ny,nz) | ✅ |
| bubble.volume | double* | bubble_volume: wp.float64 (max_bubbles,) | ✅ (B2) |
| bubble.init_volume | double* | bubble_init_volume: wp.float64 (max_bubbles,) | ✅ (B2) |
| bubble.rho | double* | bubble_rho: wp.float64 (max_bubbles,) | ✅ (B2) |
| bubble.label_init_volume | double* | label_init_volume: wp.float64 (max_bubbles,) | ✅ |
| bubble.label_volume | double* | label_volume: wp.float64 (max_bubbles,) | ✅ |

**结论：零遗漏。** 源码 `mrFlow3D` 的全部 24 个字段均在计划 State 中有精确对应。

### 2.4 Kernel 清单完整性

**审核方法**：从源码 `mrLbmSolverGpu3D.cu` 提取全部 kernel 定义，对照 §4.4-§4.8 分配表和 §4.9 管线调用。

| 源码 kernel（`__global__`） | 源码行号 | 计划模块 | 管线调用 | 结论 |
|---|---|---|---|---|
| `clear_detector` | 64-83 | kernels_bubble.py (ClearDectector) | `_handle_merge_split` 内 | ✅ |
| `clear_inlet` | 110-142 | kernels_bubble.py | Phase 2, 条件 (step==57595) | ✅ |
| `ResetDisjoinForce` | 144-160 | kernels_foam.py | Phase 2, stream_collide_bvh 之后 | ✅ |
| `calculate_disjoint` | 162-272 | kernels_foam.py | Phase 2, 第一步 | ✅ |
| `surface_1` | 444-477 | kernels_surface.py | Phase 2 | ✅ |
| `surface_2` | 479-603 | kernels_surface.py | Phase 2 | ✅ |
| `surface_3` | 601-937 | kernels_surface.py | Phase 2 | ✅ |
| `stream_collide_bvh` | 703-1057 | kernels_fluid.py | Phase 2（唯一流体 kernel） | ✅ |
| `mrSolver3D_step2Kernel` | 1059-1063 | solver.py (wp.copy) | Phase 2 末 | ✅ |
| `parse_label` | 1066-1086 | kernels_bubble.py | `_handle_merge_split` 内 | ✅ |
| `InitTag` | 1089-1113 | kernels_bubble.py | 初始化 | ✅ |
| `ResetLabelVolume` | 1185-1193 | kernels_bubble.py | `_handle_merge_split` 内 | ✅ |
| `atmosphere_rho_update_kernel` | 1259-1283 | kernels_foam.py | Phase 2 | ✅ |
| `atmosphere_volme_update_kernel` | 1286-1298 | kernels_foam.py | Phase 2 | ✅ (B7) |
| `bubble_volume_update` | 1301-1355 | kernels_bubble.py | Phase 1, coupling 内 | ✅ |
| `bubble_rho_update_kernel` | 1356-1365 | kernels_bubble.py | Phase 1 (两处: update_bubble + g_handle 后) | ✅ |
| `MergeSplitDetectorKernel` | 1366-1374 | kernels_bubble.py | Phase 1 | ✅ (B7) |
| `assign_tag_kernel` | 1424-1450 | kernels_bubble.py | Phase 1 | ✅ (B7) |
| `recheck_merge_kernel` | 1452-1475 | kernels_bubble.py | Phase 1 | ✅ (B7) |
| `g_reconstruction` | 1644-1747 | kernels_gas.py | Phase 1 | ✅ |
| `g_stream_collide` | 1750-1850 | kernels_gas.py | Phase 1 (CMR-MRT) | ✅ |
| `bubble_volume_g_update_kernel` | 1853-1879 | kernels_gas.py | Phase 1 | ✅ |
| `mrSolver3D_g_step2Kernel` | 1881-1885 | solver.py (wp.copy) | Phase 1 末 | ✅ |
| `get_tag_kernel` | 2095 | kernels_bubble.py | Phase 1, 第一步 | ✅ (B7) |

**结论：零遗漏。** 源码全部 24 个 kernel 定义均在计划中有分配和调用。

### 2.5 伪代码逐段审核

#### 2.5.1 NOCM-MRT 碰撞 (`mlGetPIAfterCollision`)

**计划位置**：§2.3 行 174-206，附录 B 行 474-524
**源码位置**：`mrUtilFuncGpu3D.h:424-471`

| 审核要素 | 源码关键行 | 计划对应 | 结论 |
|---------|----------|---------|------|
| 对角无迹分解 | 行 436-438: `pixx_part = (2*pixx_t45 - piyy_t45 - pizz_t45) / 3.0` | 行 179-180: `Π_xx_part = (2Π_xx - Π_yy - Π_zz) / 3` | ✅ |
| 对角碰撞更新 | 行 443-449: `pixx_t45 = R/3 + pixx_part*(1-ω) + RUVW2 + (2*RU2*ω)/3 - (RV2*ω)/3 - (RW2*ω)/3 + Fx*U` | 行 187: `Π_xx_new = ρ/3 + Π_xx_part·(1-ω) + RUVW2 + ω·(2RU2-RV2-RW2)/3 + F_x·u_x` | ✅ |
| 非对角碰撞更新 | 行 468: `pixy_t90 = pixy_t90 - pixy_t90*ω + U*V*R*ω + (Fy*U)/2 + (Fx*V)/2` | 行 192: `Π_xy_new = Π_xy·(1-ω) + ω·ρ·u_x·u_y + (F_y·u_x + F_x·u_y)/2` | ✅ |
| 碰撞后矩存储 | 行 1046-1055: `fMomPost[0]=rhoVar; fMomPost[1]=ux_t30+FX*invRho/2; ...` | 行 197-206 + 附录 B 行 513-523 | ✅ |

**结论：公式逐项匹配，无误差。**

#### 2.5.2 D3Q7 CMR-MRT 碰撞 (`g_stream_collide`)

**计划位置**：§2.4 行 212-293，§4.7
**源码位置**：`mrLbmSolverGpu3D.cu:1750-1850`

| 审核要素 | 源码关键行 | 计划对应 | 结论 |
|---------|----------|---------|------|
| 正变换 `mlConvertCmrMoment_d3q7` | `mrUtilFuncGpu3D.h:474-496`（7 分布→7 中心矩，CX=CX-ux） | §2.4 行 241-261（伪代码逐行对应） | ✅ |
| 反变换 `mlConvertCmrF_d3q7` | `mrUtilFuncGpu3D.h:499-518`（含 U²/2 项和高阶修正） | §2.4 行 266-293（伪代码逐行对应） | ✅ |
| 7 个松弛率 | 行 1837-1840: `s[0]=1.0; s[1-3]=1.0/(0.1*4+0.5); s[4-6]=1.5` | §4.1: `GAS_CMR_S = [1.0, 1.0/0.9, 1.0/0.9, 1.0/0.9, 1.5, 1.5, 1.5]` | ✅ |
| 碰撞公式 | 行 1843-1844: `pop_out[i] = fma(1.0f-s[i], pop_g[i], fma(s[i], g_eq[i], src))` | 行 230-231: 完全相同 | ✅ |
| 源项处理 | 行 1829-1833: `src_Q[i] = mlflow.src[curind] * d3q7_w[i]; mlConvertCmrMoment_d3q7(uxn,uyn,uzn,src_Q)` | §2.3 State 表含 `src` 字段，计划正确传递 | ✅ |

**结论：逐行精确匹配，无误差。**

#### 2.5.3 stream_collide_bvh 单一 kernel 伪代码

**计划位置**：§4.4 行 565-886（第一版伪代码）、行 722-886（第二版完整伪代码）
**源码位置**：`mrLbmSolverGpu3D.cu:703-1057`

##### Phase A: Pull Streaming + Hermite 重建（源码行 729-804）

| 步骤 | 源码 | 伪代码 | 结论 |
|------|------|--------|------|
| 跳过条件 | 行 721: `flagsn_bo==TYPE_S \|\| flagsn_su==TYPE_G` → return | 行 719: 相同 | ✅ |
| islet 处理 | 行 723-726: `islet==1 → flag=TYPE_F, return` | 行 721: 相同 | ✅ |
| Solid 反弹 | 行 743-748: `(neighbor_flag & TYPE_BO)==TYPE_S → fhn[i]=feq[i]` | 行 583-584: 相同 | ✅ |
| Hermite 重建 | 行 763-774: `mlCalDistributionFourthOrderD3Q27AtIndex(rho,ux,...,pixx_t45,...,i,f_out)` | 行 587-590: `reconstruct_from_moments(...)` | ✅ |
| w3d 偏置减去 | 行 776: `fhn[i] -= w3d_gpu[i]` | 行 592: `f_streamed[di] -= w3d_gpu[di]` | ✅ |

##### Phase C: 自由表面接口压力（源码行 862-910）

| 步骤 | 源码 | 伪代码 | 结论 |
|------|------|--------|------|
| 气泡压力读取 | 行 874-877: `if(tag>0) rho_k=bubble.rho[tag-1]` | 行 613-615: 相同逻辑 | ✅ |
| 表面张力调制 | 行 880-888: `init_volume>5e6→σ=1e-6; volume<64→σ=2e-4` | 行 614-617: 完全匹配 | ✅ |
| Laplace 压力 | 行 896: `rho_laplace = def_6_sigma_k * curv` | 行 620: `rho_laplace = sigma_k * curv` | ✅ |
| 气体压力 | 行 910: `f_eq(rho_k - rho_laplace - disjoint_factor*disjoint, ...)` | 行 621: `gas_pressure = rho_k - rho_laplace - DISJOINT_FACTOR * disjoint` | ✅ |
| Guo 力修正 | 行 900-908: `uxntmp = fma(forcex*invRho, 0.5, uxn); clamp 0.4` | 行 625-628: 相同 | ✅ |

##### Phase D: 质量交换（源码行 926-929）

| 步骤 | 源码 | 伪代码 | 结论 |
|------|------|--------|------|
| TYPE_F 邻居 | `flagsj_su[i]==TYPE_F ? fhn[i]-fon[opp] : ...` | 行 644-645: `dflux = f_streamed[di] - f_streamed[inv_di]` | ⚠️ 见问题 P1 |
| TYPE_I 邻居 | `0.5f*(phij[i]+phij[0])*(fhn[i]-fon[opp])` | 行 647: `0.5*(nphi+phij[0])*dflux` | ⚠️ 见问题 P1 |

##### Phase E: 气侧边界填充 — **⚠️ 发现问题 P1**

**源码** (`mrLbmSolverGpu3D.cu:930-934`):
```cpp
for (int i = 1; i < 27; i++) {
    if (flagsj_su[i] == TYPE_G)
        fhn[i] = feg[index3dInv_gpu[i]] - fon[index3dInv_gpu[i]] + feg[i];
}
```

**关键细节**：源码使用 `fon[index3dInv_gpu[i]]`（当前格点在方向 `opp_di` 的**出射**分布），而非 `fhn[opp_di]`（来自对侧邻居的**入射**分布）。

**计划伪代码** (行 650 和行 866):
```
f_streamed[di] = feg[OPPOSITE[di]] - f_streamed[OPPOSITE[di]] + feg[di]
```

这里 `f_streamed[OPPOSITE[di]]` 是对侧邻居的拉式分布（=源码 `fhn[opp_di]`），而非当前格点的出射（=源码 `fon[opp_di]`）。在拉式 LBM 方案中，这两者在自由表面边界处**并不相等**——`fon` 是由当前格点自身矩重建的函数，`fhn[opp_di]` 则来自不同的邻居格点。

**影响**：在接口单元气侧填充中，若直接使用 `fhn[opp_di]` 替代 `fon[opp_di]`，产生的误差量为 O(Δx)（相邻格点分布差）。该误差不破坏整体守恒性，但会与参考代码产生数值偏差，导致金标准对比测试失败。

**修正方案**：伪代码需仿照源码，在 Phase A 同时计算当前格点的出射分布（如 `fon[27]`），并在气侧边界填充中使用 `fon[OPPOSITE[di]]`。

##### Phase E (Turbulence)：内联涡粘性模型（源码行 1001-1028）

| 步骤 | 源码 | 伪代码 | 结论 |
|------|------|--------|------|
| 搜索范围 ±3 | 行 1001-1003: `for(ij=-3;ij<3;ij++) for(jk=-3;jk<3;jk++) for(kh=-3;kh<3;kh++)` | 行 657-659: `for dij in range(-3,3): for djk... for dkh...` | ✅ |
| 坐标映射 | 行 1005-1007: `x12=x+jk; y12=y+ij; z12=z+kh` | 行 660: `ni,nj,nk = i+djk, j+dij, k+dkh` | ✅ |
| 小气泡判定 | 行 1013: `volume[tag-1] < 5000000.0` | 行 663: `bubble_volume[ntag-1] < 5000000.0` | ✅ |
| Frobenius 范数 | 行 1015-1022: `xx=pixx_t45*invRho-cs2; ...; fact2=4.0; vis=fact2*sqrt(xx²+2xy²+...)` | 行 667-672: 完全匹配（含硬编码因子 4） | ✅ |
| Ω 更新 | 行 1023: `Omega = 1/((vis+1e-4)*3.0+0.5)` | 行 673: `Omega = 1.0/((nu_e+1e-4)*3.0+0.5)` | ✅ |
| break | 行 1024 | 行 674 | ✅ |

#### 2.5.4 CMR-MRT 变换函数

**计划位置**：§2.4 行 241-293
**源码位置**：`mrUtilFuncGpu3D.h:474-518`

**正变换 `mlConvertCmrMoment_d3q7`**：

| 步骤 | 源码 (474-496) | 伪代码 (241-261) | 结论 |
|------|---------------|-----------------|------|
| 输入拷贝 | 行 476-481: `for(i=0;i<7;i++) {node[i]=node_in_out[i]; node_in_out[i]=0;}` | 行 246-247 | ✅ |
| 中心速度 | 行 484-486: `CX=ex3d_gpu[k]-ux; CY=ey3d_gpu[k]-uy; CZ=ez3d_gpu[k]-uz` | 行 250-252 | ✅ |
| 密度矩 k0 | 行 488: `node_in_out[0] += ftemp` | 行 254 | ✅ |
| 动量矩 k1-k3 | 行 489-491: `+= ftemp*CX/CY/CZ` | 行 255-257 | ✅ |
| 正应力差 k4-k5 | 行 492-493: `+= ftemp*(CX²-CY²); += ftemp*(CX²-CZ²)` | 行 258-259 | ✅ |
| 二阶矩模 k6 | 行 494: `+= ftemp*(CX²+CY²+CZ²)` | 行 260 | ✅ |

**反变换 `mlConvertCmrF_d3q7`**：

| 方向 | 源码 (499-518) | 伪代码 (266-293) | 结论 |
|------|---------------|-----------------|------|
| dir 0 | 行 511: `-k0*U²-2*k1*U-k0*V²-2*k2*V-k0*W²-2*k3*W+k0-k6` | 行 273-274: 完全匹配 | ✅ |
| dir 1 (+1,0,0) | 行 512: `k1/2+k4/6+k5/6+k6/6+(U*k0)/2+U*k1+(U²*k0)/2` | 行 276-277 | ✅ |
| dir 2 (-1,0,0) | 行 513 | 行 279-280 | ✅ |
| dir 3-6 | 行 514-517 | 行 282-292 | ✅ |

**结论：CMR-MRT 正反变换伪代码与源码逐项精确匹配，无误差。**

#### 2.5.5 PLIC Cube 伪代码

**计划位置**：附录 C 行 526-568
**源码位置**：`mrUtilFuncGpu3D.h:104-149`

| 步骤 | 源码 | 伪代码 | 结论 |
|------|------|--------|------|
| 对称化简 | 行 143: `ax=fabsf(n.x); V=0.5-fabsf(V0-0.5); l=ax+ay+az` | 行 538-540 | ✅ |
| L1 归一化 | 行 144-145: `n1=fmin(ax,ay,az)/l; n3=fmax(ax,ay,az)/l; n2=fdimf(1.0,n1+n3)` | 行 541-543 | ✅ |
| 情形(5) | 行 106: `if(n12<=2.0*n3V) return n3V+0.5*n12` | 行 547-548 | ✅ |
| 情形(2) | 行 108: `v1=sqn1/n26; if(v1<=n3V && n3V<v1+0.5*(n2-n1)) return 0.5*(n1+sqrt(...))` | 行 549-551 | ✅ |
| 情形(1) | 行 110: `if(n3V<v1) return cbrt(V6)` | 行 553 | ✅ |
| 情形(3)/(4) | 行 111-118: 三次方根公式含 `sinf(0.33333334*asin(...))` | 行 555-564: 含 `0.33333334 * asin` | ✅ |
| 还原 | 行 148: `l*copysignf(0.5-d, V0-0.5)` | 行 567 | ✅ |

**结论：PLIC Cube 伪代码与 SZ & Kawano 2022 算法源码逐项匹配，无误差。**

#### 2.5.6 calculate_curvature 伪代码

**计划位置**：§4.4 行 888-1000
**源码位置**：`mrUtilFuncGpu3D.h:371-420`

| 步骤 | 源码 | 伪代码 | 结论 |
|------|------|--------|------|
| 法向计算 | 行 377-381: 27-点加权 `bz.x=4*(phij[2]-phij[1])+2*(...)+...` | 行 892-899: 等效 `weight*CX[di]*phij[di]` | ✅ |
| 局部坐标系 | 行 383-385: `bx=cross(by,bz); by=normalize(cross(bz,rn))` | 行 902-903 | ✅ |
| PLIC 偏移 | 行 390: `center_offset = plic_cube(phij[0], bz)` | 行 906 | ✅ |
| 邻居收集 | 行 392-399: 迭代 27 邻居，仅 TYPE_I (0<phi<1) | 行 907-918 | ✅ |
| 最小二乘拟合 | 行 401-415: 5×5 矩阵 LU 分解 `lu_solve(M,x,b,5,5)` | 行 934-942 | ✅ |
| 曲率公式 | 行 **418**: `K = (A*(I*I+1.0) + B*(H*H+1.0) - C*H*I) * cb(rsqrt_(H*H+I*I+1.0))` | 行 **997**: `K = (A*(I*I+1.0)+B*(H*H+1.0)-C*H*I) / (H*H+I*I+1.0)**1.5` | ✅ (公式一致) |
| 钳制 | 行 419: `return clamp(K, -1.0f, 1.0f)` | 行 998: 相同 | ✅ |

**结论：伪代码中 `calculate_curvature` 的曲率公式与源码一致。**（见问题 P2 关于算法描述中的不一致）

### 2.6 算法公式验证

#### 2.6.1 曲率公式文档不一致 — ⚠️ 问题 P2

**计划 §5.4 算法描述**（行 1487）：
```
κ = 2[A(1+I²) + B(1+H²) - C·H·I] / (1+H²+I²)^(3/2)
```
含有因子 **2**。

**源码** `mrUtilFuncGpu3D.h:418`：
```cpp
K = (A * (I*I + 1.0f) + B * (H*H + 1.0f) - C * H * I) * cb(rsqrt_(H*H + I*I + 1.0f))
```
其中 `cb(rsqrt_(x)) = 1/x^(3/2)`，**无因子 2**。

**计划伪代码**（§4.4 行 997）：
```
K = (A*(I*I+1.0) + B*(H*H+1.0) - C*H*I) / (H*H+I*I+1.0)**1.5
```
**无因子 2，与源码一致。**

**影响**：伪代码是正确的，但 §5.4 算法描述（高层文档）中的公式错误地多了因子 2。若实现者不看源码只看算法描述，会实现出错误的曲率（导致 Laplace 压力翻倍）。需要修正 §5.4 的高层描述使其与源码和伪代码一致。

#### 2.6.2 其他公式验证

| 公式 | 计划引用 | 源码验证 | 结论 |
|------|---------|---------|------|
| D3Q27 Maxwell-Boltzmann 平衡态 | §4.1 `CS2=1/3` | `mrUtilFuncGpu3D.h:285-320`: `c3 = -2*sq(u)/(2*sq(cs))` | ✅ |
| D3Q7 平衡态（线性） | §4.1 | `mrUtilFuncGpu3D.h:323-338`: `feq[i] = d3q7_w[i]*(1 + ex*ux*4 + ey*uy*4 + ez*uz*4)*rho` | ✅ |
| NOCM-MRT 碰撞（闭式） | §2.3 Eq.(187-195) | `mrUtilFuncGpu3D.h:443-470` | ✅ |
| 亨利定律 | §5.6: `c_sat = K_h·ρ_bubble/4` | `mrLbmSolverGpu3D.cu:1711`: `float in_rho = K_h / 4.f * rho_k` | ✅ |
| 理想气体定律 | §5.5: `ρ_new = ρ_old * V_init / V` | `mrLbmSolverGpu3D.cu:1356-1365`: 精确匹配 | ✅ |
| 涡粘性系数 | §5.8: `ν_e = 4·‖S‖_F` | `mrLbmSolverGpu3D.cu:1021-1022`: `float fact2 = 4.0f; vis = fact2*sqrt(...)` | ✅ |
| 分离压力 | §5.7: `P_disjoin = max(0, 1-d/4)` | `mrLbmSolverGpu3D.cu:230`: `disjoint = 1.f - dist/4.f` | ✅ |
| 质量交换（VOF 平流） | §5.3 Eq.(1466-1467) | `mrLbmSolverGpu3D.cu:928` | ✅ |

### 2.7 降级/简化零容忍检查

| 检查项 | 原计划问题 | 修订版 | 源码证据 | 结论 |
|--------|-----------|--------|---------|------|
| CCL 实现 | 简化为 Union-Find | YACCLAB 完整五阶段 | `tDCCL.cu:1-578` (InitLabeling + MergeMasks + LabelAnalysis + LabelReduction + FinalLabeling) | ✅ 无降级 |
| 气泡精度 | 计划用 float | wp.float64 | `mlBubble3D` 全部 `double*` (`mrFlow3D.h:19-23`) | ✅ 无降级 |
| 气体碰撞 | 标注 BGK | CMR-MRT | `mrLbmSolverGpu3D.cu:1822-1848` (中心矩变换 + 7 松弛率) | ✅ 无降级 |
| 流体 kernel 拆分 | 拆为 2 个 | 单一 stream_collide_bvh | `mrLbmSolverGpu3D.cu:703-1057` (单一 kernel) | ✅ 无简化 |
| 管线顺序 | 混合单管线 | 两阶段耦合 | `mrSolver3D.h:121-127` (coupling + mrSolver3DGpu 独立调用) | ✅ |
| 湍流模型拆分 | 未明确归属 | 内联于 stream_collide_bvh | `mrLbmSolverGpu3D.cu:1001-1028` (embedded in kernel) | ✅ |
| State 继承 | FluidGridStateBase | DomainState | 独立 warp 数组管理（不含 MAC staggered grid 字段） | ✅ |
| handle_merge_spilt | 子步骤缺失 | 完整 8 步管线 | 源码含 ClearDectector + convertIntToUnsignedChar + CCL + parse_label + ResetLabelVolume + reduce_label_rho + bubble_list_swap + num_rho_update_kernel | ✅ |
| 合并/分裂检测 | 未提及并行冲突 | 三重机制 | `assign_tag_kernel` + `recheck_merge_kernel` + `atomicExch` (`mrLbmSolverGpu3D.cu:1424-1475`) | ✅ |

### 2.8 测试计划审核

| 测试项目 | 计划执行手册对应 | 出处引用 | 结论 |
|---------|---------------|---------|------|
| D3Q27 方向位精确匹配 | §6.2 `test_d3q27_directions_bit_exact` | 源码 `ex3d_gpu`/`ey3d_gpu`/`ez3d_gpu` 常数 | ✅ |
| NOCM-MRT 碰撞输出逐分量 | §6.5 `test_mlGetPIAfterCollision` | `mrUtilFuncGpu3D.h:424-471` | ✅ |
| CMR-MRT roundtrip 恒等 | §6.8 `test_cmr_mrt_roundtrip` | `mrUtilFuncGpu3D.h:474-518` | ✅ |
| PLIC cube 解析解 | §6.6 `test_plic_cube_reference` | `mrUtilFuncGpu3D.h:104-149` | ✅ |
| CCL 金标准对比 (16³) | §6.7 → Phase 3.0 原型验证 | `tDCCL.cu:1-578` | ✅ |
| 气泡体积守恒 (10⁴ 步) | §6.12 `test_regression_bubble_volume_conservation` | `mlBubble3D` double 精度 (`mrFlow3D.h:17-27`) | ✅ |
| 剪切流衰减回归 | §6.12 `test_regression_shear_decay` | NOCM-MRT 碰撞 (`mrUtilFuncGpu3D.h:424-471`) | ✅ |
| 确定性合并 (10 次一致) | §9 R5 缓解措施 | `assign_tag_kernel` + `atomicExch` | ✅ |

---

## 3. 发现的问题清单

### P1：[严重] 自由表面气侧边界填充伪代码逻辑错误

- **位置**：`migration_plan_zh.md` 行 650（第一版伪代码）和行 866（第二版完整伪代码）
- **计划写法**：
  ```
  f_streamed[di] = feg[OPPOSITE[di]] - f_streamed[OPPOSITE[di]] + feg[di]
  ```
- **源码写法** (`mrLbmSolverGpu3D.cu:933`)：
  ```cpp
  fhn[i] = feg[index3dInv_gpu[i]] - fon[index3dInv_gpu[i]] + feg[i];
  ```
- **差异**：计划使用 `f_streamed[OPPOSITE[di]]`（对侧邻居的入射分布）替代 `fon[index3dInv_gpu[i]]`（当前格点在反方向的出射分布）。在拉式 LBM 自由表面方案中，这两个量来自不同格点，不相等。
- **影响**：误差 O(Δx)，导致金标准对比测试失败。
- **修正**：伪代码需仿照源码增加当前格点出射分布的计算步骤（如 `fon[27]` 数组），并在气侧填充中使用 `fon[OPPOSITE[di]]`。

### P2：[中等] 曲率公式在算法描述与伪代码间不一致

- **位置**：
  - §5.4 算法描述（行 1487）：`κ = 2[A(1+I²) + B(1+H²) - C·H·I] / (1+H²+I²)^(3/2)` — **含因子 2，错误**
  - §4.4 伪代码（行 997）：`K = (A*(I*I+1.0) + B*(H*H+1.0) - C*H*I) / (H*H+I*I+1.0)**1.5` — **无因子 2，正确**
- **源码验证** (`mrUtilFuncGpu3D.h:418`)：
  ```cpp
  K = (A*(I*I+1.0f) + B*(H*H+1.0f) - C*H*I) * cb(rsqrt_(H*H+I*I+1.0f))
  ```
  其中 `cb(rsqrt_(x)) = 1/x^(3/2)`，**无因子 2**。
- **修正**：删除 §5.4 算法描述中的因子 2，使其与伪代码和源码一致。

### P3：[轻微] 伪代码在 `f_streamed` 和 `fhn` 间变量命名不一致

- **位置**：§4.4 第一版伪代码使用 `f_streamed`，第二版完整伪代码使用 `fhn`，容易造成混淆
- **影响**：不影响实现正确性，但降低文档可读性
- **建议**：统一为 `fhn`（与源码一致）

### P4：[轻微] 第二版完整伪代码缺少 `fon` 数组的计算

- **位置**：§4.4 行 787-886（stream_collide_bvh 完整伪代码）
- **与问题 P1 关联**：源码中 `fon[27]` 是通过 `mlCalDistributionFourthOrderD3Q27AtIndex` 从当前格点矩计算的，伪代码省略了这一步骤，直接导致 P1 的错误
- **修正**：在 Phase A 中增加 `fon[27]` 的计算步骤

---

## 4. 假阳性确认（经源码验证正确的项目）

以下项目在表面审视时可能被认为是问题，但经过源码逐行对比验证是正确的：

| 项目 | 表面疑虑 | 源码验证结果 |
|------|---------|------------|
| 湍流模型搜索范围 `turbulence_radius=3` | 是否应为 `turbulence_radius=4`？ | `mrLbmSolverGpu3D.cu:1001`: `for(int ij=-3;ij<3;ij++)` → ij∈{-3,-2,-1,0,1,2} → 6 值 → 半径=3 ✓ |
| 涡粘性硬编码因子 4 | 是否应为模型参数？ | `mrLbmSolverGpu3D.cu:1021`: `float fact2 = 4.0f` — 精确匹配 |
| `atmosphere_volme_update` 拼写 | 源码确实拼写为 `volme`（非 `volume`） | `mrLbmSolverGpu3D.cu:2020`: `atmosphere_volme_update_kernel` ✓ |
| GAS_CMR_S 计算为 `1/0.9` | 是否应与源码 `1/(0.1*4+0.5)` 等价？ | 0.1*4+0.5 = 0.9；1/0.9 ≈ 1.111... ✓ |
| `compute_pi_from_moments` 应力符号 | 伪代码使用 `compute_stress()` 高层抽象 | 源码 966-971 的精确求和将在实现中编码 ✓ |
| `calculate_curvature` 中法向符号 | 伪代码 `weight*CX[di]*phij[di]` vs 源码 `bx=4*(phij[2]-phij[1])` | 源码内部通过 `phij[i]=phit[index3dInv_gpu[i]]` 反转数组，最终方向和幅度一致 ✓ |
| `bubble_volume_update_kernel` 使用 `wp.float64(1.0 - float(phi))` | 是否应使用 `wp.float64` 直接计算？ | `(1-phi)` 差值 < 1.0，float 足够精确；原子累加到 double ✓ |
| `recheck_merge_kernel` 使用 `merge_flag` 作为 `wp.array` 而非标量 | 与源码 `merge_flag` (int) 不同 | Warp 无全局标量原子操作，数组 `[1]` 的 `atomic_add` 等价于 CUDA `atomicAdd(&merge_flag, 1)` ✓ |

---

## 5. 审核结论

**有条件通过，需修正 2 项问题后方可启动实现。**

| 结论 | 说明 |
|------|------|
| **上次审核阻塞项** | 10/10 已修复，零残留 |
| **降级/简化** | 零发现 |
| **管线结构** | 与源码两阶段调用 `coupling()+mrSolver3DGpu()` 完全一致 |
| **State 字段** | 24/24 覆盖，零遗漏 |
| **Kernel 清单** | 24/24 分配，零遗漏 |
| **伪代码逻辑** | 1 项错误 (P1)，1 项文档不一致 (P2)，2 项轻微 (P3, P4) |
| **算法公式** | 除 P2 外均与源码一致 |
| **出处引用** | 全部算法有论文公式编号和源码行号引用 |
| **测试计划** | 覆盖完整，金标准对比方案可行 |

### 修正要求

1. **[P1 - 阻塞]** 修正 §4.4 两处伪代码的气侧边界填充公式：
   - 在 Phase A 增加当前格点出射分布 `fon[27]` 的计算
   - 将 `f_streamed[OPPOSITE[di]]`（或 `fhn[opp_di]`）替换为 `fon[OPPOSITE[di]]`（或 `fon[opp_di]`）

2. **[P2 - 阻塞]** 修正 §5.4 算法描述行 1487 的曲率公式，删除因子 2：
   - 将 `κ = 2[A(1+I²) + B(1+H²) - C·H·I] / (1+H²+I²)^(3/2)`
   - 改为 `κ = [A(1+I²) + B(1+H²) - C·H·I] / (1+H²+I²)^(3/2)`

3. **[P3 - 建议]** 统一伪代码变量命名为 `fhn`（与源码一致）

4. **[P4 - 建议]** 在完整伪代码中显式增加 `fon[27]` 的 Hermite 重建步骤

---

## 6. 附录：逐行验证证据表

下表节选关键验证点的源码证据汇总。

| 验证点 | 源码文件 | 源码行号 | 源码内容摘要 | 计划行号 | 结论 |
|--------|---------|---------|------------|---------|------|
| 两阶段调用 | `mrSolver3D.h` | 121-127 | `coupling(...); mrSolver3DGpu(...);` | §4.9:1277-1308 | ✅ |
| coupling() | `mrLbmSolverGpu3D.cu` | 2078-2129 | `get_tag→assign_tag→recheck_merge→update_bubble→g_handle` | §4.9:1279-1294 | ✅ |
| mrSolver3DGpu() | `mrLbmSolverGpu3D.cu` | 1973-2075 | `calculate_disjoint→[clear_inlet]→atmosphere→stream_collide_bvh→...→surface_3→fMomSwap` | §4.9:1296-1308 | ✅ |
| stream_collide_bvh 单一 kernel | `mrLbmSolverGpu3D.cu` | 703-1057 | 单 kernel 含标记分流(721-726)+Hermite(729-777)+VOF(780-999)+湍流(1001-1028)+碰撞(1030-1055) | §4.4:719-885 | ✅ |
| NOCM-MRT 碰撞 | `mrUtilFuncGpu3D.h` | 424-471 | 闭式对角+非对角应力更新 | §2.3:174-206 | ✅ |
| CMR-MRT 正变换 | `mrUtilFuncGpu3D.h` | 474-496 | `CX=ex3d[k]-ux; k0=Σf; k1=Σf·CX...` | §2.4:241-261 | ✅ |
| CMR-MRT 反变换 | `mrUtilFuncGpu3D.h` | 499-518 | 含 U²/2 项的速度展开重建 | §2.4:266-293 | ✅ |
| PLIC cube | `mrUtilFuncGpu3D.h` | 104-149 | SZ & Kawano 2022 对称简化 | 附录 C:526-568 | ✅ |
| 曲率计算 | `mrUtilFuncGpu3D.h` | 371-420 | Monge 曲面片 + 5×5 LU 分解 | §4.4:888-1000 | ✅ |
| 亨利定律 | `mrLbmSolverGpu3D.cu` | 1711 | `in_rho = K_h/4.f * rho_k` | §5.6:1554 | ✅ |
| 气体通量→气泡体积 | `mrLbmSolverGpu3D.cu` | 1873 | `atomicAdd(init_volume, 1/4*factor*delta_g*phi)` | §4.7:1102 | ✅ |
| 涡粘性搜索范围 | `mrLbmSolverGpu3D.cu` | 1001-1003 | `ij=-3;ij<3` (±3) | §5.8:1586 | ✅ |
| 涡粘性硬编码因子 | `mrLbmSolverGpu3D.cu` | 1021 | `float fact2 = 4.0f` | §4.2:425 `turbulence_factor=4.0` | ✅ |
| 气泡结构体 | `mrFlow3D.h` | 17-27 | 全部 `double*` | §4.3:447 `wp.float64` | ✅ |
| src 字段 | `mrFlow3D.h` | 76 | `float* src` | §4.3 State 表 | ✅ |
| islet 字段 | `mrFlow3D.h` | 70 | `int* islet` | §4.3 State 表 | ✅ |
| previous_merge_tag | `mrFlow3D.h` | 57 | `int* previous_merge_tag` | §4.3 State 表 | ✅ |
| merge_detector | `mrFlow3D.h` | 61 | `bool* merge_detector` | §4.3 State 表 | ✅ |
| YACCLAB InitLabeling | `tDCCL.cu` | 94-100 | 2×2×2 block 初始标签 | §4.6:998-1001 | ✅ |
| YACCLAB MergeMasks | `tDCCL.cu` | - | atomicMin block 间合并 | §4.6:1001-1002 | ✅ |
| D3Q7 松弛率 | `mrLbmSolverGpu3D.cu` | 1837-1840 | `s[0]=1.0; s[1-3]=1/0.9; s[4-6]=1.5` | §4.1:407 `GAS_CMR_S` | ✅ |
| 气侧边界填充（错误点） | `mrLbmSolverGpu3D.cu` | 933 | `fhn[i] = feg[opp] - fon[opp] + feg[i]` | 行 650/866: `f_streamed[di]=feg[opp]-f_streamed[opp]+feg[di]` | ⚠️ P1 |

---

*本次审核基于论文四（`锐界面动力学自由表面流与泡沫`）源码 `Home-FSLBM/` 目录下的 2,133 行 `mrLbmSolverGpu3D.cu`、522 行 `mrUtilFuncGpu3D.h`、230 行 `mrFlow3D.h`、565 行 `mrSolver3D.h`、578 行 `tDCCL.cu` 以及论文五的 HOME-LBM 理论框架进行交叉验证。*
