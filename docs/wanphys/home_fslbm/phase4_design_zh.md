# HOME-FSLBM 阶段 4 设计与实现报告

> **日期**: 2026-08-10  
> **范围**: D3Q7 CMR-MRT 溶解气体（`kernels_gas.py`）+ 泡沫分离压力/大气（`kernels_foam.py`）及 solver 接入  
> **前置**: 阶段 1（流体）+ 阶段 2（自由表面）+ 阶段 3（气泡 CCL）  
> **真理源**: 参考 CUDA（风险 R7），非论文简化公式

---

## 1. 交付物

| 文件 | 内容 |
|------|------|
| [`kernels_gas.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/kernels_gas.py) | `g_eq`、CMR 正/反变换、`g_reconstruction`、`g_stream_collide`、`bubble_volume_g_update` |
| [`kernels_foam.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/kernels_foam.py) | `calculate_disjoint`、`reset_disjoin_force`、`atmosphere_rho/volme_update` |
| [`solver.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/solver.py) | Phase 1 末尾 `g_handle`；Phase 2 前 disjoint/大气，后 reset |
| 测试 / 示例 / 金标准说明 | 见 [测试计划](phase4_test_plan_zh.md)、[金标准导出](phase4_gas_foam_golden_zh.md) |

**不在本阶段**: 10^4 步气泡体积守恒长期回归（阶段 5）。

---

## 2. 与参考代码符号对照

### 2.1 气体（`mrUtilFuncGpu3D.h` / `mrLbmSolverGpu3D.cu`）

| 参考 | Warp |
|------|------|
| `calculate_g_eq` | `g_eq_d3q7` |
| `mlConvertCmrMoment_d3q7` | `convert_to_central_moment_d3q7` |
| `mlConvertCmrF_d3q7` | `convert_from_central_moment_d3q7` |
| `g_reconstruction` | `g_reconstruction_kernel` |
| `g_stream_collide` | `g_stream_collide_kernel` |
| `bubble_volume_g_update_kernel` | `bubble_volume_g_update_kernel` |
| `mrSolver3D_g_step2Kernel` | `wp.copy(g_mom, g_mom_post)` |
| `g_handle` | solver Phase 1 末尾顺序调用 |

松弛率（`.cu:1837-1840`）:

```
s[0] = 1.0
s[1]=s[2]=s[3] = 1.0 / (0.1*4 + 0.5)   # == 1/0.9
s[4] = 1.5
s[5]=s[6] = 1.5
```

对应 `constants.GAS_CMR_S`。

### 2.2 泡沫（`mrLbmSolverGpu3D.cu`）

| 参考 | Warp |
|------|------|
| `calculate_disjoint` | `calculate_disjoint_kernel` |
| `ResetDisjoinForce` | `reset_disjoin_force_kernel`（清零 `disjoin_force` **与** `massex`） |
| `atmosphere_rho_update_kernel` | `atmosphere_rho_update_kernel` |
| `atmosphere_volme_update_kernel` | `atmosphere_volme_update_kernel`（保留参考拼写 volme） |

---

## 3. Solver 时序

```
step():
  Phase 1 coupling:
    ... bubble get_tag / assign / recheck / volume / rho / merge-split ...
    g_reconstruction
    g_stream_collide          # writes g_mom_post, c_value
    bubble_volume_g_update    # init_volume += (1/4)*delta_g*phi; clear delta_g
    copy g_mom <- g_mom_post
    bubble_rho_update         # after gas volume change

  Phase 2 mrSolver3DGpu:
    calculate_disjoint        # before stream_collide (reads current f_mom)
    [optional] clear_inlet
    atmosphere_rho_update
    atmosphere_volme_update
    gravity -> stream_collide_bvh
    reset_disjoin_force       # after collide, before surface
    surface_1 -> surface_2 -> surface_3
    swap f_mom <- f_mom_post  # NOT g_mom (g already swapped in Phase 1)
```

---

## 4. 关键决策记录

### R3 — CMR 变换必须先过门禁

正/反变换逐行对照 `mrUtilFuncGpu3D.h:474-518`。单元测试 `test_cmr_mrt_*` 与金标准 G1 为阻塞门禁。

### R7 — 以 `.cu` 为准的细节

- `bubble_volume_g_update` 使用 **`flag == TYPE_I` 精确相等**（非 `TYPE_SU` 掩码）
- `calculate_disjoint` 距离公式使用 `normal.x` 分母与 PLIC 中心偏移，禁止“各向同性改进”
- 亨利饱和：`c_sat = K_h / 4 * rho_bubble`
- `g_eq`：速度先乘 4 再与权重组合（`calculate_g_eq`）

### gMom 交换

参考 `mrSolver3D_step2Kernel` **只交换 fMom**；gMom 仅在 `g_handle` 内交换。阶段 4 修正原先 `swap_moments_kernel` 同时覆盖 g 的行为，避免用陈旧 `g_mom_post` 覆盖 `surface_2` 写入的 `g_mom`。

### 索引

Warp 平坦索引与现有 fluid/surface 一致：`idx = i*ny*nz + j*nz + k`（`array3d[i,j,k]`）。金标准参考布局仍为 x-fastest，经 `reorder_ref_scalar_to_warp` 转换。

---

## 5. 门禁

| 门禁 | 测试 | 预期 |
|------|------|------|
| CMR 变换精确匹配 | `test_convert_*` / G1 | 容差 <= 1e-10 |
| 泡沫阻止聚并 | F2 / foam 可视化 | merge_flag=0，两泡间距保持 |
| 既有回归 | `pytest home_fslbm/tests` | 阶段 1–3 全绿 |

---

## 6. 后续（阶段 5）

- Domain 级完整管线顺序断言
- 泡沫无聚并长程金标准、10^4 步体积守恒
- 性能剖析 / graph capture（风险 R6）
