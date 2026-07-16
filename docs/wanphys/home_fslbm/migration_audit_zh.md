# Home-FSLBM 迁移计划审核意见书

> **审核对象**: [迁移计划](migration_plan_zh.md)
> **参考代码**: `docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/`
> **审核日期**: 2026-07-15
> **审核标准**: 禁止简化、禁止降级；所有模块必须精确复现参考代码行为
> **审核结论**: **驳回，需逐条修复后重新提交**

---

## 目录

1. [总体结论](#1-总体结论)
2. [降级/简化项（驳回）](#2-降级简化项驳回)
   - 2.1 CCL 实现降级
   - 2.2 双精度降级
   - 2.3 D3Q7 碰撞模型错误
3. [管线结构错误](#3-管线结构错误)
   - 3.1 虚构独立 stream_collide_kernel
   - 3.2 管线顺序与参考代码不一致
4. [State 字段遗漏](#4-state-字段遗漏)
5. [Kernel 遗漏](#5-kernel-遗漏)
6. [架构问题](#6-架构问题)
   - 6.1 State 继承链错误
   - 6.2 Turbulence 模型归属不明确
7. [管线步骤遗漏](#7-管线步骤遗漏)
8. [测试计划缺陷](#8-测试计划缺陷)
9. [修复清单](#9-修复清单)
10. [附录：参考代码完整 kernel 清单](#10-附录参考代码完整-kernel-清单)

---

## 1. 总体结论

**驳回。** 该迁移计划存在 **3 项必须拒绝的降级/简化、1 项管线结构错误（虚构了不存在的 kernel）、4 项 State 字段遗漏、9 项 kernel 遗漏、D3Q7 碰撞模型的类型错误（计划声称 BGK，实际为 CMR-MRT），以及 State 继承链架构性错误**。

迁移计划在算法公式标注和论文-代码映射表（附录 A）方面有显著优点，论文公式和行号索引总体准确。但在代码复现的精确性——管线结构还原、数据字段完整性、碰撞模型正确性、以及降级项的零容忍原则——方面存在根本性缺陷。在当前形态下不可执行。

---

## 2. 降级/简化项（驳回）

### 2.1 CCL 实现降级

> **计划 §4.6**: "CCL 策略：参考代码使用 YACCLAB（基于 OpenCV GpuMat3 的 GPU CCL）。对于 Warp，采用简化的两遍并查集（Union-Find）实现。"

**驳回理由**：

参考代码 `tDCCL.cu`（578 行）是完整的 YACCLAB 三维 GPU 连通分量标记实现，利用 OpenCV `GpuMat3` 的 pitched memory 分配、CUDA 原子操作（`atomicMin`、`atomicExch`）和 `__syncthreads()` 屏障同步。该实现经过论文验证（参考论文四 Sec.5.2），其标签一致性、并行正确性和跨 block 合并逻辑有严格数学保证。

"简化两遍 Union-Find" 不等价于 YACCLAB：

1. **标签编号规则不同**：Union-Find 产生任意顺序标签，YACCLAB 产生密集标签（0..N-1），参考代码因依赖 `tag_matrix` 从 1 开始的密集编号（如 `bubble.rho[tag_matrix[curind] - 1]`，`mrLbmSolverGpu3D.cu:1709`），标签编号方式直接影响气泡属性索引的正确性。
2. **合并顺序不同**：并行 Union-Find 的合并方向取决于线程调度，而 YACCLAB 通过确定性扫描确保标签一致性。
3. **跨 block 处理**：Union-Find 需要多轮迭代才能收敛跨 block 的标签，YACCLAB 在两轮 `labeling` + `analysis` 中完成。

**结论**：迁移必须完整复现 YACCLAB CCL 算法逻辑，不得简化。如果 Warp 无 `GpuMat3` 等价物，需基于 `wp.array3d` 重建等价的 pitched memory 分配方案。

**修复要求**：

- 基于参考代码 `tDCCL.cu:1-578` 逐行复现 CCL 逻辑
- 使用 Warp 的 `wp.atomic_min`、`wp.atomic_add` 替代 CUDA 原子操作
- 若 Warp 原子操作有差异，需在计划中逐项标识并给出具体替代方案
- 增加 Phase 3 的原型验证子阶段：在正式实现前用 16³ 网格验证标签输出与参考代码完全一致

---

### 2.2 双精度降级

> **计划 §9 风险登记**: "以 `float` 累加，对大量气泡采用 Kahan 求和；监控体积漂移。"

**驳回理由**：

参考代码 `mlBubble3D` 结构（`mrFlow3D.h:17-27`）中气泡属性 **全部使用 `double`**：

```cpp
// mrFlow3D.h:17-27
struct mlBubble3D {
    double* volume;
    double* init_volume;
    double* rho;
    double* label_init_volume;
    double* label_volume;
    int label_num;
    unsigned int max_bubble_count = 65536;
    int bubble_count = 0;
};
```

气泡体积累加使用 `double` 精度的原子操作：

```cpp
// mrLbmSolverGpu3D.cu:1873 — bubble_volume_g_update_kernel
atomicAdd(&mlflow[0].bubble.init_volume[tag - 1],
    (double)1.f / 4.f * factor * mlflow[0].delta_g[curind] * mlflow[0].phi[curind]);
```

以及 `parse_label` 中的体积归约（`mrLbmSolverGpu3D.cu:1066-1088`）：

```cpp
// parse_label 内
atomicAdd(&mlflow[0].bubble.init_volume[label_id - 1],
    (double)(1.f - mlflow[0].phi[curind]));
```

**float 精度的累积误差分析**：

- 网格尺寸可达 600×300×300 = 5.4×10⁷ 格点
- 每个气泡的 (1-φ) 贡献约 1/V_cell 量级（格子单位）
- 5×10⁷ 次 float 累加的舍入误差约为 O(√N)·ε ≈ 7×10³·1.2×10⁻⁷ ≈ 8×10⁻⁴（相对误差）
- 在 10⁵ 时间步仿真中，此误差累积可导致气泡体积漂移 > 1%
- Kahan 求和仅将误差从 O(N·ε) 降至 O(ε)，**不改变 float 相对 double 的基准精度差距**（float: ~7 位有效数字 vs double: ~15 位）

**结论**：使用 `float` 是一种不可接受的精度降级。气泡体积/密度数组必须维持 `double` 精度。

**修复要求**：

- 气泡体积/密度/初始体积数组使用 `wp.float64`
- 若 Warp 后端不支持 `wp.float64` 原子操作：
  - 在计划中 **明确说明替代策略**（如使用 `wp.atomic_add(wp.int64)` 配合定点化累加方案）
  - 提供 **定量误差分析**（蒙特卡洛模拟或最坏情况分析），证明替代方案的累积误差 ≤ double 的 2 倍
- 在 `test_regression.py` 中增加气泡体积守恒测试：N=10⁴ 步后 ΣV_bubble 相对变化 ≤ 1×10⁻⁶

---

### 2.3 D3Q7 碰撞模型错误：BGK → 实际为 CMR-MRT

> **计划 §5.6**: "D3Q7 BGK 碰撞: g_i_post = g_i - ω_g · (g_i - g_i^eq(c, u))"

**驳回理由**：

参考代码 `g_stream_collide`（`mrLbmSolverGpu3D.cu:1822-1848`）使用的是 **CMR-MRT（Central Moment Relaxation — Multiple Relaxation Time）** 碰撞模型，而非 BGK。完整代码如下：

```cpp
// mrLbmSolverGpu3D.cu:1822-1848 — 实际 g_stream_collide 碰撞部分
// cmr
float w = 1.0f / 0.53f;
float src = 0.f;
mrutilfunc.mlConvertCmrMoment_d3q7(uxn, uyn, uzn, pop_g);   // 分布 → 中心矩
mrutilfunc.mlConvertCmrMoment_d3q7(uxn, uyn, uzn, g_eq);

float src_Q[7];
for (int i = 0; i < 7; i++)
    src_Q[i] = mlflow[0].src[curind] * d3q7_w[i];
mrutilfunc.mlConvertCmrMoment_d3q7(uxn, uyn, uzn, src_Q);

float pop_out[7];
float s[7];
s[0] = 1.0f;                                        // 密度矩松弛率
s[1] = s[2] = s[3] = 1.0f / (0.1 * 4 + 0.5);      // 动量矩松弛率 ≈ 1.111
s[4] = 1.5f;                                         // 正应力差松弛率
s[5] = s[6] = 1.5f;                                  // 二阶矩松弛率

for (int i = 0; i < 7; i++) {
    src = src_Q[i] * (1 - s[i] / 2);
    pop_out[i] = fma(1.0f - s[i], pop_g[i], fma(s[i], g_eq[i], src));
}
mrutilfunc.mlConvertCmrF_d3q7(uxn, uyn, uzn, pop_out);  // 中心矩 → 分布
```

CMR-MRT 与 BGK 的关键差异：

| 属性 | BGK（计划声称） | CMR-MRT（实际参考） |
|------|---------------|-------------------|
| 松弛参数数 | 1 个（ω_g） | 7 个独立 s[0..6] |
| 碰撞空间 | 分布函数（速度空间） | 中心矩（以 u 为中心的矩空间） |
| 变换 | 无 | `mlConvertCmrMoment_d3q7`（正变换）+ `mlConvertCmrF_d3q7`（反变换） |
| 数值行为 | 所有矩等速松弛 | 密度矩瞬间平衡、动量矩高速松弛、应力矩中速松弛 |

**CMR-MRT 的两个关键变换函数在计划中完全未被提及**：

1. **`mlConvertCmrMoment_d3q7`**（`mrUtilFuncGpu3D.h:474-496`）：7 个分布 → 7 个中心矩，每个方向的格子速度减去局部流速 u 后计算伪距多项式。
2. **`mlConvertCmrF_d3q7`**（`mrUtilFuncGpu3D.h:499-518`）：7 个中心矩 → 7 个分布，包含二次速度展开和高阶修正项。

**结论**：计划中将 D3Q7 碰撞标注为 BGK 是一个 **事实错误**，会导致实现与参考代码在数值上完全不兼容。

**修复要求**：

- 重新标注 D3Q7 气体碰撞为 CMR-MRT
- 在 `kernels_gas.py` 中实现以下函数：
  - `convert_to_central_moment_d3q7(ux, uy, uz, distributions)` → 7 个中心矩
  - `convert_from_central_moment_d3q7(ux, uy, uz, moments)` → 7 个分布
- 将单一 `gas_omega` 参数替换为 7 元素松弛率向量或硬编码常量：`s = [1.0, ω_m, ω_m, ω_m, 1.5, 1.5, 1.5]`，其中 `ω_m = 1/(0.1*4+0.5) = 1/0.9 ≈ 1.111...`
- 在附录 A 中补充这两个函数的行号映射

---

## 3. 管线结构错误

### 3.1 虚构独立 `stream_collide_kernel`

> **计划 §4.4** 和 **§4.9** 将 `stream_collide_kernel` 和 `stream_collide_free_surface_kernel` 当作两个独立的 kernel，分别在管线步骤 3 和步骤 6 中调用。

**错误性质**：

参考代码 **并不存在独立的 `stream_collide` kernel**。`mrSolver3DGpu` 函数（`mrLbmSolverGpu3D.cu:1973-2075`）中只有一个流体碰撞 kernel——`stream_collide_bvh`（第 2024-2030 行）。该 kernel 内部通过标记位分流逻辑处理所有单元类型：

- `flagsn_bo == TYPE_S || flagsn_su == TYPE_G` → 跳过（固体/纯气体单元不需处理）
- `flagsn_su == TYPE_F` 或 `TYPE_I` → 执行完整的 Hermite 重建 + 拉式流传输 + NOCM-MRT 碰撞
- `flagsn_su == TYPE_I` → 额外执行质量交换 + 气体边界条件（自由表面逻辑）

**拆分造成的后果**：

1. **湍流涡粘性修正丢失**：参考代码的涡粘性模型（基于应变率 ‖S‖_F 修改 Omega）**嵌入在 `stream_collide_bvh` 内部**（第 1001-1028 行）。将其拆分为 `stream_collide_kernel`（纯流体）和 `stream_collide_free_surface_kernel`（接口），会导致纯流体单元的 Omega 不被湍流模型修改——这在参考代码中不存在此问题。
2. **多余的 kernel 启动开销**：增加一个不必要的 GPU kernel 启动。
3. **代码冗余**：两个 kernel 有大量重复的 Hermite 重建和 NOCM-MRT 碰撞逻辑。

**修复要求**：

- **删除** `stream_collide_kernel`
- 将 `stream_collide_free_surface_kernel` **重命名**为 `stream_collide_bvh_kernel`
- 该 kernel 内部**完整实现**参考代码第 703-1057 行全部逻辑：
  - Pull streaming + Hermite 重建（第 729-777 行）
  - 自由表面质量交换（第 780-999 行）
  - 涡粘性湍流模型（第 1001-1028 行）
  - NOCM-MRT 碰撞（第 1030-1055 行）
- 在附录 A 中补充具体行号分段

---

### 3.2 管线顺序与参考代码不一致

计划 §4.9 描述的 step() 管线顺序为：

```
g_reconstruction → get_tag → stream_collide → g_stream_collide
→ calculate_disjoint → stream_collide_bvh → ResetDisjoinForce
→ surface_1/2/3 → MomSwap → [周期性] bubble pipeline
```

但参考代码的实际调用结构（`mrSolver3D::mlIterateCouplingGpu`，见 `mrSolver3D.h:121-125`）是：

```
mlIterateCouplingGpu(timestep):
  ① coupling(lbm_dev_gpu, ...) ——— 气泡/气体子系统
       ├─ get_tag_kernel
       ├─ assign_tag_kernel
       ├─ recheck_merge_kernel
       ├─ update_bubble(...)
       │    ├─ bubble_volume_update
       │    ├─ bubble_rho_update_kernel
       │    ├─ MergeSplitDetectorKernel
       │    └─ [条件] handle_merge_spilt → ClearDectector
       └─ g_handle(...)
            ├─ g_reconstruction
            ├─ g_stream_collide (CMR-MRT)
            ├─ bubble_volume_g_update_kernel
            ├─ mrSolver3D_g_step2Kernel (gMom swap)
            └─ bubble_rho_update_kernel

  ② mrSolver3DGpu(lbm_dev_gpu, ...) ——— 流体/自由表面子系统
       ├─ calculate_disjoint
       ├─ [条件] clear_inlet
       ├─ atmosphere_rho_update_kernel
       ├─ atmosphere_volme_update_kernel
       ├─ stream_collide_bvh ★ (唯一流体 kernel)
       ├─ ResetDisjoinForce
       ├─ surface_1
       ├─ surface_2
       ├─ surface_3
       └─ mrSolver3D_step2Kernel (fMom swap)
```

**关键差异**：

1. `coupling()` 和 `mrSolver3DGpu()` 是**两个独立调用**，计划将其混合为单一管线，掩盖了函数边界的结构和数据依赖。
2. `stream_collide_bvh` 在 `calculate_disjoint` 和 `atmosphere_*` **之后**执行，而非之前。
3. 气泡管线（CCL + merge/split）在 `g_handle` 之前执行，而非之后。

**修复要求**：

- 按照参考代码的两阶段结构重写 solver.step() 管线
- 阶段一对应 `coupling()`：bubble tagging → CCL → merge/split → gas reconstruction + CMR-MRT
- 阶段二对应 `mrSolver3DGpu()`：disjoint pressure → atmosphere → stream_collide_bvh → surface tracking → buffer swap
- 附录 A 补充完整的调用关系图

---

## 4. State 字段遗漏

对照 `mrFlow3D.h:31-91` 和迁移计划 §4.3，以下字段在目标 State 中**缺失**，但在参考代码中被活跃使用：

### 4.1 `islet`（孤立气泡标记）

- **参考类型**: `int*`
- **目标类型**: `wp.array3d(dtype=wp.int32)`
- **用途**: 标记孤立气泡单元。当气泡因注入/分离过程变为孤立状态时设为 1。
- **使用位置**:
  - `clear_inlet`（第 123, 125 行）：islet 单元转为 TYPE_G，质量/矩清零
  - `stream_collide_bvh`（第 620, 723 行）：islet 单元被跳过或转回 TYPE_F
  - `surface_3`（第 585 行）：islet 邻居不被标记为 merge 候选
  - `g_stream_collide`（第 1767 行）：islet 单元被跳过

### 4.2 `previous_merge_tag`（合并前气泡ID）

- **参考类型**: `int*`
- **目标类型**: `wp.array3d(dtype=wp.int32)`
- **用途**: 存储合并事件中"被吸收"的气泡 ID。在 `assign_tag_kernel` 中用于恢复因并行竞争丢失的旧标签。
- **使用位置**:
  - `handle_merge_spilt`（第 1415 行）：写入 `thisCellID`
  - `assign_tag_kernel`（第 1438, 1441, 1442, 1446 行）：将旧标签写回 `tag_matrix`，解决并行标签冲突

### 4.3 `merge_detector`（逐单元合并事件标记）

- **参考类型**: `bool*`
- **目标类型**: `wp.array3d(dtype=wp.bool)`
- **用途**: 逐单元记录可能的合并事件。`surface_1` 标记候选单元，`clear_detector` 清零。
- **使用位置**:
  - `clear_detector`（第 76 行）：全部归零
  - `surface_1`（第 587 行）：标记 IF 邻居为 merge 候选
  - `assign_tag_kernel`（第 1438 行）：检查 merge_detector 决定是否恢复旧标签

### 4.4 `src`（溶解气体源项）

- **参考类型**: `float*`
- **目标类型**: `wp.array3d(dtype=float)`
- **用途**: 溶解气体的外部源项/汇项。在 `g_stream_collide` CMR-MRT 碰撞中作为外力项加入矩空间。
- **使用位置**:
  - `g_stream_collide`（第 1831 行）：`src[curind] * d3q7_w[i]`，经 CMR 变换后按 `src_Q[i] * (1 - s[i]/2)` 加入碰撞

**修复要求**：上述 4 个字段必须加入 `HomeFslbmState.__init__` 的 warp 数组分配中，并同步更新 `clear()` 和 `clone()` 方法。

---

## 5. Kernel 遗漏

以下参考代码中的 kernel/函数在迁移计划中 **未被明确分配模块**（搜索了全部 §4.4-4.8 和附录 A）：

### 5.1 气泡子系统遗漏（应归入 `kernels_bubble.py`）

| # | 遗漏的 kernel/函数 | 参考代码行 | 用途 |
|---|-------------------|-----------|------|
| 1 | **`clear_detector`** | 第 64-107 行 | 清零 `merge_detector`、`split_flag`、`merge_flag`；需要在 grid 级调用 + 单线程清零标志 |
| 2 | **`assign_tag_kernel`** | 第 1424-1450 行 | 将 `previous_merge_tag` 写回 `tag_matrix`，解决并行标签竞争（恢复被合并吸收的气泡旧标签） |
| 3 | **`recheck_merge_kernel`** | 第 1452-1475 行 | 二次验证合并事件——遍历同一标签所有单元，区分"真合并"与"气泡移动导致的标签变化" |
| 4 | **`reset_label_volume`** | 第 1598-1601 行 | 清零 `label_volume` 和 `label_init_volume`（临时归约缓冲），为新一轮 CCL 准备 |
| 5 | **`reduce_label_rho`** | 第 1601-1603 行 | 按新标签归约合并后的气泡体积和密度（`atomicAdd(double)`） |
| 6 | **`bubble_list_swap`** | 第 1604-1608 行 | 交换 `bubble.volume` ↔ `bubble.label_volume`、`bubble.init_volume` ↔ `bubble.label_init_volume` 指针 |
| 7 | **`num_rho_update_kernel`** | 第 1607-1611 行 | 更新活跃气泡计数 `bubble_count` |
| 8 | **`MergeSplitDetectorKernel`** | 第 1629-1631 行 | 全局合并/分裂检测——检查任一单元的 `merge_detector`，设置全局标志位 |

### 5.2 气体子系统遗漏（应归入 `kernels_gas.py`）

| # | 遗漏的 kernel/函数 | 参考代码行 | 用途 |
|---|-------------------|-----------|------|
| 9 | **`bubble_volume_g_update_kernel`** | 第 1853-1879 行 | 从 `delta_g`（溶解气体通量）更新初始气泡体积：`init_volume += (1/4)*factor*delta_g*phi`，使用 `atomicAdd(double)` |

### 5.3 参考代码 kernel 全覆盖清单

参考代码 `mrLbmSolverGpu3D.cu` 共定义了以下 GPU kernel（按出现顺序，排除辅助 swap 函数）：

| 顺序 | kernel 名称 | 行号 | 迁移计划中 | 归属模块 |
|------|-----------|------|----------|---------|
| 1 | `clear_detector` | 64 | ❌ 遗漏 | kernels_bubble.py |
| 2 | `clear_inlet` | 110 | ✓ 管线中提及但未分配 kernel | kernels_foam.py |
| 3 | `ResetDisjoinForce` | 144 | ✓ | kernels_foam.py |
| 4 | `calculate_disjoint` | 162 | ✓ | kernels_foam.py |
| 5 | `surface_1` | ~444 | ✓ | kernels_surface.py |
| 6 | `surface_2` | ~520 | ✓ | kernels_surface.py |
| 7 | `report_split` | 53 | ❌ 遗漏（设备函数） | kernels_bubble.py |
| 8 | `surface_3` | ~601 | ✓ | kernels_surface.py |
| 9 | `stream_collide_bvh` | 703 | ✓（作为 free_surface） | kernels_fluid.py |
| 10 | `parse_label` | 1066 | ✓ | kernels_bubble.py |
| 11 | `InitTag` | 1089 | ✓ | kernels_bubble.py |
| 12 | `convertIntToUnsignedChar` | 1115 | ✓ | kernels_bubble.py |
| 13 | `create_bubble_label` | 1139 | ✓ | kernels_bubble.py |
| 14 | `update_init_tag` | 1150 | ✓ | kernels_bubble.py |
| 15 | `ResetLabelVolume` | ~1195 | ❌ 遗漏 | kernels_bubble.py |
| 16 | `atmosphere_rho_update_kernel` | 1259 | ✓（但管线未调用） | kernels_foam.py |
| 17 | `atmosphere_volme_update_kernel` | 1286 | ❌ 遗漏 | kernels_foam.py |
| 18 | `bubble_volume_update_kernel` | 1301 | ✓ | kernels_bubble.py |
| 19 | `bubble_rho_update_kernel` | 1320 | ✓ | kernels_bubble.py |
| 20 | `handle_merge_spilt` 内部子 kernel | 1576-1613 | ❌ 遗漏 4 个子 kernel | kernels_bubble.py |
| 21 | `g_reconstruction` | 1644 | ✓ | kernels_gas.py |
| 22 | `g_stream_collide` | 1750 | ✓ | kernels_gas.py |
| 23 | `bubble_volume_g_update_kernel` | 1853 | ❌ 遗漏 | kernels_gas.py |
| 24 | `mrSolver3D_g_step2Kernel` | 1881 | ✓（solver 管线中作为 wp.copy） | solver.py |
| 25 | `assign_tag_kernel` | 1424 | ❌ 遗漏 | kernels_bubble.py |
| 26 | `recheck_merge_kernel` | 1452 | ❌ 遗漏 | kernels_bubble.py |
| 27 | `MergeSplitDetectorKernel` | 1629 | ❌ 遗漏 | kernels_bubble.py |

**修复要求**：

- 上述 9 个遗漏的 kernel/函数必须分配到对应模块文件
- 在附录 A 的论文公式→代码映射表中补充每个遗漏 kernel 的精确行号
- `atmosphere_volme_update_kernel` 的缺失尤为严重——kernel 定义和功能描述均不存在于计划中

---

## 6. 架构问题

### 6.1 State 继承链错误

> **计划 §4.3**: `HomeFslbmState(FluidGridStateBase)`

**错误**：`FluidGridStateBase.__init__`（`base.py:64-107`）会自动创建 MAC staggered 网格数组：

```python
self.vel_u = wp.zeros((nx+1, ny, nz), ...)
self.vel_v = wp.zeros((nx, ny+1, nz), ...)
self.vel_w = wp.zeros((nx, ny, nz+1), ...)
self.pressure = wp.zeros((nx, ny, nz), ...)
self.density = wp.zeros((nx, ny, nz), ...)
self.solid_phi = wp.zeros((nx, ny, nz), ...)
self.solid_body_id = wp.full((nx, ny, nz), -1, ...)
```

这些字段 HOME-FSLBM **全部不需要**。HOME-FSLBM 使用自己的场量体系（`flag`、`mass`、`phi`、`f_mom` 等），与 MAC staggered grid 完全不兼容。

**正确做法**：现有 `LbmState`（`lbm/state.py:18`）继承 `DomainState` 而非 `FluidGridStateBase`，自行管理所有 warp 数组。迁移计划必须效仿此模式。

**修复要求**：

- 将 `HomeFslbmState` 基类从 `FluidGridStateBase` 改为 `DomainState`
- 在 constructor 中自行调用 `super().__init__()`（DomainState 无参数 __init__）
- MAC staggered 速度场如有可视化需要，可通过 `post_step` 钩子在 domain 层单独计算

### 6.2 Turbulence 模型归属不明确

> **计划 §5.5** 描述了涡粘性湍流模型但未明确指定归属模块。

参考代码中该逻辑 **嵌入在 `stream_collide_bvh` 内部**（`mrLbmSolverGpu3D.cu:1001-1028`）：

```cpp
// 遍历 ±3 邻域（6×6×6 = 216 个邻居）
for (int ij = -3; ij < 3; ij++)
    for (int jk = -3; jk < 3; jk++)
        for (int kh = -3; kh < 3; kh++) {
            // ...
            if (mlflow[0].tag_matrix[ind_back] > 0) {
                if (mlflow[0].bubble.volume[tag - 1] < 5000000.0) {  // 小气泡
                    float xx = pixx_t45 * invRho - cs2;
                    // ... (计算 Frobenius 范数)
                    float vis = 4.0f * sqrtf(xx²+2xy²+2xz²+yy²+2yz²+zz²);
                    Omega = 1 / ((vis + 1e-4) * 3.0f + 0.5f);  // 修改松弛率
                    break;
                }
            }
        }
```

该逻辑：
1. 仅对小气泡（体积 < 5,000,000 格子单位³）启用
2. 计算 Frobenius 范数 ‖S‖_F（应变率张量）
3. 涡粘性系数 = 4 × ‖S‖_F（因子 4 来自计划中的 `turbulence_factor`）
4. 在邻域中找到第一个小气泡即停止搜索（`break`）

**修复要求**：

- 在 `kernels_fluid.py` 的 `stream_collide_bvh_kernel` 中**内联实现**
- 不可拆分为独立 kernel——涡粘性修正必须紧邻碰撞前执行（因为在计算好 Ω 后立即调用 `mlGetPIAfterCollision(Omega, ...)`）
- 修正计划中 `turbulence_radius=4` 默认值 → 应为 3（对应 ±3 搜索范围）

---

## 7. 管线步骤遗漏

参考代码 `mrSolver3DGpu()` 函数（`mrLbmSolverGpu3D.cu:1973-2075`）有以下步骤在计划管线中缺失：

### 7.1 `clear_inlet`（条件执行）

- **参考代码行**: 第 110-142 行（kernel 定义），第 1998-2008 行（调用）
- **触发条件**: `time_step == 180 * 320 - 5`（即第 57595 步时执行一次）
- **功能**: 将 `islet == 1` 的单元转为 TYPE_G（气体），清空质量和矩，恢复为平衡态。用于注入仿真中移除入口残留。
- **计划状态**: 在 §2.1 管线中标注了 `[条件执行] clear_inlet`，但在 §4.8-4.9 中既无 kernel 分配，也无管线调用

### 7.2 `atmosphere_rho_update`（kernel 已定义但管线遗漏）

- **参考代码行**: 第 1259-1283 行（kernel 定义），第 2010-2018 行（调用）
- **功能**: 开敞式水箱顶部边界的大气压力更新，涉及 `N`、`l0p`、`roup`、`labma`、`u0p` 等物理映射参数
- **计划状态**: §4.8 的 `kernels_foam.py` kernel 清单中列出了 `atmosphere_rho_update_kernel`，但 §4.9 solver.step() 管线中 **未调用**

### 7.3 `atmosphere_volme_update`（kernel 完全缺失）

- **参考代码行**: 第 1286-1290 行（kernel 定义），第 2020-2022 行（调用）
- **功能**: 大气体积修正——单线程 kernel，在大气压力更新后修正大气体积
- **计划状态**: 在 §4.8 kernel 清单中**完全未出现**，§4.9 管线中也无调用。kernel 名称在计划中**全文仅出现一次**（§2.1 管线描述 `atmosphere_volume_update`），但无定义、无算法、无源代码引用

**修复要求**：

- 为 `clear_inlet` 补充 kernel 定义（→ `kernels_bubble.py` 或新的 `kernels_inlet.py`）
- 在 solver.step() 中按参考代码顺序插入三者
- 补充 `atmosphere_volme_update_kernel` 的完整算法描述和源代码引用（`mrLbmSolverGpu3D.cu:1286-1290`）

---

## 8. 测试计划缺陷

### 8.1 CCL 测试无对照验证 [阻塞性]

`test_kernels_bubble.py` 的 10 个测试全部为自验证（如 "tag=-1"、"输入 → 255"），**没有任何对照参考代码 CCL 输出的金标准测试**。鉴于 CCL 是最高风险的核心模块（计划自身也将其列为头号风险），这是不可接受的。

**修复要求**: 增加至少 1 个金标准 CCL 标签对比测试——在已知的简单二值体素配置（如 16³ 含 3 个分离球体）上运行参考代码和 Warp 实现，逐单元比较 `label_matrix` 输出，允许标签重编号映射但禁止拓扑差异（连通域数量必须一致）。

### 8.2 泡沫/气体回归测试缺失 [阻塞性]

`test_regression.py` 的 5 个场景（`uniform_flow`、`droplet`、`dambreak`、`rising_bubble`、`two_bubbles_merge`）全部涉及流体力学核心，但**没有任何一个涉及溶解气体或分离压力**。这两个模块（P1-P2）完全没有端到端验证。

**修复要求**: 增加至少 1 个泡沫场景——例如"两个彼此接近的气泡因分离压力而停止聚并"，网格 64³，500 步，关键检查项：Σ disjoin_force > 0 且气泡间最小距离 > 0（即没有发生合并）。

### 8.3 `test_regression_uniform_flow` 区分度为零

32³ 网格上 100 步均匀流——对于任何正确实现的 LBM（无论碰撞模型是 BGK、MRT 还是 NOCM-MRT），均匀平衡态流场都是稳态解。该测试的区分度近乎为零，无法发现任何算法实现错误。

**修复要求**: 替换为**剪切流衰减测试**（如初始 u_x = sin(2πz/L_z)），在 32³ 网格上运行 100 步后比较速度剖面。剪切流的粘性衰减速率直接依赖碰撞算子，可区分 NOCM-MRT 与简化实现。

### 8.4 `atmosphere_volme_update_kernel` 无测试

此 kernel 在计划中完全缺失定义，自然也无可对应的测试。

**修复要求**: kernel 补全后，需增加测试验证大气体积修正的单步计算。

### 8.5 金标准数据生成方法缺失

`test_regression.py` 要求对照参考代码比较输出，但未说明：

- 如何编译/运行参考 CUDA C++ 代码
- 金标准数据将存储在何处（现有 `newton/tests/golden_data/` 目录适用于 Newton，WanPhys 测试目录未规划 golden_data）
- 自动化流程如何实现（手动运行参考代码还是 CI 集成）

**修复要求**: 在计划中增加"金标准数据生成流程"一节，明确：
1. 参考代码编译环境（CUDA 版本、编译器、依赖库）
2. 参考代码输出格式和保存位置
3. 测试的数据加载和比较逻辑

---

## 9. 修复清单

### 阻塞项（修复后重新提交审核）

- [ ] **B1** — 驳回 CCL 简化：改为完整复现 YACCLAB CCL 逻辑，增加原型验证子阶段
- [ ] **B2** — 驳回 double→float 降级：气泡体积/密度使用 `wp.float64`，补充定量误差分析
- [ ] **B3** — 修正 D3Q7 碰撞模型：BGK → CMR-MRT，实现 `convert_to_central_moment_d3q7` / `convert_from_central_moment_d3q7`
- [ ] **B4** — 修正管线结构：删除虚构的 `stream_collide_kernel`，统一为 `stream_collide_bvh_kernel`（内联涡粘性模型）
- [ ] **B5** — 按参考代码两阶段结构（coupling + mrSolver3DGpu）重写 solver.step()
- [ ] **B6** — 补充遗漏的 State 字段：`islet`、`previous_merge_tag`、`merge_detector`、`src`
- [ ] **B7** — 补充 9 个遗漏 kernel（见 §5 清单）到对应模块
- [ ] **B8** — 在管线中插入 `clear_inlet`、`atmosphere_rho_update`、`atmosphere_volme_update`
- [ ] **B9** — 修正 State 继承：`DomainState` 替代 `FluidGridStateBase`
- [ ] **B10** — 补充 `atmosphere_volme_update_kernel` 的完整定义、算法描述和源文件引用

### 强建议项（审核通过前应完成）

- [ ] **S1** — 增加 CCL 金标准标签对比测试（test_kernels_bubble.py）
- [ ] **S2** — 增加泡沫/分离压力端到端回归测试（test_regression.py）
- [ ] **S3** — 替换 `test_regression_uniform_flow` 为剪切流衰减测试
- [ ] **S4** — 在附录 A 中补充遗漏 kernel 的行号映射
- [ ] **S5** — 增加"金标准数据生成流程"说明节
- [ ] **S6** — 修正 `turbulence_radius` 默认值：4 → 3

---

## 10. 附录：参考代码完整 kernel 清单

以下为 `mrLbmSolverGpu3D.cu` 中所有 GPU kernel 的完整清单（按行号排序），含迁移计划覆盖状态：

| # | 行号 | kernel 名称 | 类型 | 计划状态 | 应归属 |
|---|------|-----------|------|---------|--------|
| 1 | 53 | `report_split` | `__device__` | ❌ 遗漏 | kernels_bubble.py |
| 2 | 64 | `clear_detector` | `__global__` | ❌ 遗漏 | kernels_bubble.py |
| 3 | 110 | `clear_inlet` | `__global__` | ⚠️ 管线提及，无 kernel | kernels_bubble.py |
| 4 | 144 | `ResetDisjoinForce` | `__global__` | ✓ | kernels_foam.py |
| 5 | 162 | `calculate_disjoint` | `__global__` | ✓ | kernels_foam.py |
| 6 | ~444 | `surface_1` | `__global__` | ✓ | kernels_surface.py |
| 7 | ~520 | `surface_2` | `__global__` | ✓ | kernels_surface.py |
| 8 | ~601 | `surface_3` | `__global__` | ✓ | kernels_surface.py |
| 9 | 703 | `stream_collide_bvh` | `__global__` | ✓（被拆分为两个 kernel） | kernels_fluid.py |
| 10 | 1066 | `parse_label` | `__global__` | ✓ | kernels_bubble.py |
| 11 | 1089 | `InitTag` | `__global__` | ✓ | kernels_bubble.py |
| 12 | 1115 | `convertIntToUnsignedChar` | `__global__` | ✓ | kernels_bubble.py |
| 13 | 1139 | `create_bubble_label` | `__global__` | ✓ | kernels_bubble.py |
| 14 | 1150 | `update_init_tag` | `__global__` | ✓ | kernels_bubble.py |
| 15 | ~1195 | `ResetLabelVolume` | `__global__` | ❌ 遗漏 | kernels_bubble.py |
| 16 | 1259 | `atmosphere_rho_update_kernel` | `__global__` | ⚠️ 已定义，管线遗漏 | kernels_foam.py |
| 17 | 1286 | `atmosphere_volme_update_kernel` | `__global__` | ❌ 遗漏 | kernels_foam.py |
| 18 | 1301 | `bubble_volume_update_kernel` | `__global__` | ✓ | kernels_bubble.py |
| 19 | 1320 | `bubble_rho_update_kernel` | `__global__` | ✓ | kernels_bubble.py |
| 20 | 1424 | `assign_tag_kernel` | `__global__` | ❌ 遗漏 | kernels_bubble.py |
| 21 | 1452 | `recheck_merge_kernel` | `__global__` | ❌ 遗漏 | kernels_bubble.py |
| 22 | 1576-1613 | `handle_merge_spilt` 内子 kernel | host + global | ⚠️ 部分覆盖 | kernels_bubble.py |
| 23 | 1629 | `MergeSplitDetectorKernel` | `__global__` | ❌ 遗漏 | kernels_bubble.py |
| 24 | 1644 | `g_reconstruction` | `__global__` | ✓ | kernels_gas.py |
| 25 | 1750 | `g_stream_collide` | `__global__` | ✓（但碰撞模型标注错误） | kernels_gas.py |
| 26 | 1853 | `bubble_volume_g_update_kernel` | `__global__` | ❌ 遗漏 | kernels_gas.py |
| 27 | 1881 | `mrSolver3D_g_step2Kernel` | `__global__` | ✓（solver 中作为 wp.copy） | solver.py |

**统计**：27 个 kernel 中，11 个完整覆盖，4 个存在严重问题（标注错误/结构错误/管线遗漏），**9 个完全遗漏**，3 个部分覆盖（子步骤不完整）。

---

*本审核意见基于参考代码 `Home-FSLBM/inc/3D/gpu/mrLbmSolverGpu3D.cu`（2133 行）、`mrUtilFuncGpu3D.h`（522 行）、`mrFlow3D.h`（230 行）、`tDCCL.cu`（578 行）等源文件的逐行审查。所有行号引用均经过交叉验证。*
