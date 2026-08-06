# HOME-FSLBM 阶段 3 设计与实现报告

> **日期**: 2026-08-06  
> **范围**: YACCLAB CCL 气泡追踪（`kernels_bubble.py`）及 solver Phase 1（`coupling`）接入  
> **前置**: 阶段 1（流体核心）+ 阶段 2（自由表面）  
> **测试结果**: **15 / 15** `test_kernels_bubble.py` 通过；阶段 1/2 回归 **104** 通过

---

## 1. 交付物

| 文件 | 内容 |
|------|------|
| [`kernels_bubble.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/kernels_bubble.py) | YACCLAB 风格 CCL + 气泡生命周期 / 合并分裂 kernel |
| [`solver.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/solver.py) | Phase 1 `coupling` 气泡路径；`init_bubbles`；`_handle_merge_split` |
| [`model.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/model.py) | `clear_inlet_enabled`（默认关闭） |
| [`tests/test_kernels_bubble.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_kernels_bubble.py) | CCL 金标准 / 确定性 / 气体定律 / 合并守恒 / solver 触发 |

**不在本阶段**：`g_handle` / CMR-MRT 气体、`calculate_disjoint`、大气更新（阶段 4）。

---

## 2. 与参考代码符号对照

### 2.1 YACCLAB CCL（`tDCCL.cu`）

| 参考 | Warp |
|------|------|
| `InitLabeling` | `ccl_init_labeling_kernel` |
| `Merge` | `ccl_merge_kernel`（显式 3D 邻域 Union，等价 26-连通跨 2×2×2 block） |
| `PathCompression` | `ccl_path_compression_kernel` |
| `FinalLabeling` | `ccl_final_labeling_kernel` |
| `thrust::sort` + `unique` + `renumber_*` | host `np.unique` + `ccl_apply_renumber_kernel` |
| `connectedComponentLabeling` | `connected_component_labeling(input, label) -> label_num` |
| OpenCV `GpuMat3` | flat `wp.array`，索引 `x + nx*(y + ny*z)` |
| `atomicMin` UF | `wp.atomic_min` |
| `__syncthreads__` | 独立 `wp.launch`（全局隐式同步） |

### 2.2 气泡业务（`mrLbmSolverGpu3D.cu`）

| 参考 | Warp |
|------|------|
| `clear_detector` | `clear_detector_kernel` |
| `clear_inlet` | `clear_inlet_kernel`（`model.clear_inlet_enabled`） |
| `InitTag` | `init_tag_kernel` |
| `convertIntToUnsignedChar` | `convert_flag_to_input_kernel` |
| `parse_label` | `parse_label_kernel` |
| `create_bubble_label` | `create_bubble_label_kernel` |
| `update_init_tag` | `update_init_tag_kernel` |
| `InitBubble` | `init_bubbles` / `HomeFslbmSolver.init_bubbles` |
| `get_tag` / `assign_tag` / `recheck_merge` | 同名 kernel |
| `bubble_volume_update` | `bubble_volume_update_kernel`（**按 `delta_phi`**，非裸 `1-φ`） |
| `bubble_rho_update` | `bubble_rho_update_kernel`（`ρ = V_init / V`） |
| `MergeSplitDetectorKernel` | `merge_split_detector_kernel` |
| `handle_merge_spilt` | `handle_merge_split`：`convert → CCL → reset_label_volume → reduce_label_rho → bubble_list_swap → num_rho_update` |
| `MomSwap` 气泡指针 | Python 端交换 `state.bubble_volume` ↔ `label_volume` 等 array 引用 |

---

## 3. Solver 时序

```
step():
  Phase 1 coupling:
    seed merge/split GPU flags from state_in
    copy bubble + gas + flag/phi/delta_phi → state_out
    get_tag → assign_tag → recheck_merge
    bubble_volume_update → bubble_rho_update
    MergeSplitDetector
    if merge|split: handle_merge_split → clear_detector
    (gas copy only — stage 4)

  Phase 2 mrSolver3DGpu:
    [optional] clear_inlet
    gravity → stream_collide_bvh
    surface_1 → surface_2 → surface_3 (writes split_flag)
    f_mom swap
```

跨步分裂信号：`surface_3` 写入的 `split_flag` 存入 `state_out`，下一拍 Phase 1 读入。

---

## 4. 关键决策记录

### R1 — CCL / pitched memory

未移植 `reinterpret_cast` 的 pitched ushort/ulong 加载；以 **2×2×2 block UF + 26-连通邻域 Union** 保持 YACCLAB 结构与确定性。稠密重编号在 host 完成。

门禁：16³ × 3 球与 `scipy.ndimage.label`（26-连通）**拓扑一致**；5 次运行位级相同。

### R2 — float64 atomic

探测通过：`wp.atomic_add(wp.float64)` 在 CUDA 后端可用。体积累加直接使用 double atomic，无需 int64 定点。

### R7 — 文档 vs 源码

以 `.cu` 为准：

- `bubble_volume_update` 使用 `-delta_phi`，不是计划伪代码中的 `1-φ`
- `handle_merge_spilt` 使用 `reduce_label_rho`，不是 `parse_label`
- `bubble_rho` 更新为 `init_volume / volume`

---

## 5. 门禁结果

| 门禁 | 测试 | 结果 |
|------|------|------|
| A — CCL 16³ 三球拓扑 | `test_ccl_golden_16cube_3_spheres` | 通过 |
| A — 确定性 | `test_ccl_deterministic` | 通过 |
| B — 合并后 V·ρ 守恒 | `test_merge_detection_two_old_one_new` | 通过（`Σ(1-φ)·ρ` → `init`；`ρ·V = init`） |
| Solver 触发 CCL | `test_solver_handle_merge_split_trigger` | 通过 |
| 阶段 1/2 回归 | fluid/surface/state/model | 104 passed |

---

## 6. 使用提示

```python
solver.init_bubbles(state)   # 仿真开始前：CCL + 创建气泡属性
domain.step(dt)              # 每步自动 coupling 气泡路径
```

CCL 前景 = `TYPE_G | TYPE_I`（非固体）；背景流体应为 `TYPE_F`，否则整域气体会连成单个气泡。

---

## 7. 金标准回归（测试 + 导出器已就绪，数据待用户生成/迁移）

- 用例：[`test_regression_bubble.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_regression_bubble.py)（B1–B13，无数据则 skip）
- 导出器：[`docs/Home-FSLBM/export_bubble_golden.cpp`](../../Home-FSLBM/export_bubble_golden.cpp)（CMake 目标 `export_bubble_golden`）
- 运行与拷贝说明：[phase3_bubble_golden_zh.md](phase3_bubble_golden_zh.md)

## 8. 后续（阶段 4）

- `kernels_gas.py`：CMR-MRT D3Q7 + `g_handle` 接入 Phase 1 末尾  
- `kernels_foam.py`：分离压力 + 大气  
- 长期体积守恒回归（10⁴ 步）归阶段 5
