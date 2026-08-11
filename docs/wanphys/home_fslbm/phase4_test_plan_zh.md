# HOME-FSLBM 阶段 4 测试计划

> **日期**: 2026-08-10  
> **范围**: CMR-MRT 气体 + 泡沫分离压力/大气的单元、集成、金标准回归与可视化验收  
> **前置**: [阶段 4 设计](phase4_design_zh.md)

---

## 1. 单元测试

### 1.1 `test_kernels_gas.py`

| 测试 | 验收 |
|------|------|
| `test_g_eq_d3q7` | 对照解析/`calculate_g_eq`，容差 1e-12 |
| `test_convert_to_central_moment_d3q7` | 对照参考向量（门禁 R3） |
| `test_convert_from_central_moment_d3q7` | 对照参考向量 |
| `test_cmr_mrt_roundtrip` | 正→反，容差 1e-10 |
| `test_g_reconstruction_henry_law` | 接口 `c_sat = K_h/4 * rho_b`；`delta_g` 对 TYPE_F 邻居累加 |
| `test_g_stream_collide_relaxation` | 均匀流体非平衡 g 松弛；`c_value` 守恒量级合理 |
| `test_bubble_volume_g_update` | `init_volume` 增量正确且 `delta_g` 清零 |

### 1.2 `test_kernels_foam.py`

| 测试 | 验收 |
|------|------|
| `test_disjoint_raycast_hit` | 两 TYPE_I 近邻球 → `disjoin_force > 0` |
| `test_disjoint_no_neighbor` | 单球 → 全 0 |
| `test_disjoint_distance_scaling` | 已知 d → `max(0, 1-d/4)` |
| `test_reset_disjoin_force` | force 与 massex 归零 |
| `test_atmosphere_rho_update` | 边界大泡 `rho -> 1.0` |
| `test_atmosphere_volme_update` | `init_volume = rho * volume` |

### 1.3 `test_solver_gas_foam.py`

| 测试 | 验收 |
|------|------|
| `test_gas_handle_updates_g_mom` | 单步后 `g_mom`/`c_value` 有限且变化符合碰撞 |
| `test_disjoint_cleared_after_step` | 步末 `disjoin_force` 全 0（reset 已执行） |
| `test_f_mom_swap_not_g_post_overwrite` | surface 写入的 g 不被错误的末步 g 交换覆盖 |

---

## 2. 金标准回归（`test_regression_gas_foam.py`）

无数据则 **skip**。格式：每场景子目录、每场一个 `.txt`（与阶段 2/3 一致）。

| ID | 场景目录 | 检查 |
|----|----------|------|
| G1 | `gas_cmr_transform_vectors` | 正/反变换向量 |
| G2 | `gas_henry_interface_step1` | `delta_g`, `g_mom` |
| G3 | `gas_stream_collide_step10` | `g_mom`, `c_value` |
| G4 | `gas_volume_g_update` | `bubble_init_volume` |
| F1 | `foam_disjoint_two_spheres` | `disjoin_force` |
| F2 | `foam_no_coalescence` | 无合并 + 间距；门禁 S2 |
| F3 | `foam_atmosphere_open_tank` | `bubble_rho` / `init_volume` |

容差：浮点相对 1e-4（近零绝对 1e-10）；`flag` 精确；`tag` 允许置换。

---

## 3. 可视化验收

| 示例 | 人工检查 |
|------|----------|
| `fluid_grid_home_fslbm_foam_pair.py` | 数百步 `bubble_count==2`，质心距不塌缩 |
| `fluid_grid_home_fslbm_rising_bubble_gas.py` | 气泡上升；`c_value` 有限；无发散 |
| `fluid_grid_home_fslbm_dambreak.py` | 接入 gas/foam 后仍稳定 |

---

## 4. 成功门禁

1. CMR 正/反变换与参考匹配（<=1e-10）
2. F2 / foam_pair：分离压力阻止聚并
3. 全量 `pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests` 无回归
