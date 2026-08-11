# 阶段四气体/泡沫金标准导出说明

> **导出器**: [`docs/Home-FSLBM/export_gas_foam_golden.cpp`](../../Home-FSLBM/export_gas_foam_golden.cpp)  
> **CMake 目标**: `export_gas_foam_golden`（链接 CUDA，与 `export_bubble_golden` 同级）  
> **用户职责**: 编译运行后将 `gas_*` / `foam_*` 目录拷贝到 Warp 测试金标准路径。

---

## 1. 构建与运行

```powershell
cmake --build <your-build-dir> --target export_gas_foam_golden --config Release
<path-to>\export_gas_foam_golden.exe
# 或指定输出根目录:
# export_gas_foam_golden.exe C:\Users\...\docs\Home-FSLBM\golden_data
```

期望输出子目录（与 `droplet_*` / `bubble_*` 并列）：

```text
docs/Home-FSLBM/golden_data/
  gas_cmr_transform_vectors/     # G1
  gas_henry_interface_step1/     # G2
  gas_stream_collide_step10/     # G3
  gas_volume_g_update/           # G4
  foam_disjoint_two_spheres/     # F1
  foam_no_coalescence/           # F2
  foam_atmosphere_open_tank/     # F3
```

---

## 2. 数据迁移

```powershell
Copy-Item -Recurse docs\Home-FSLBM\golden_data\gas_* `
  wanphys\_src\fluid\fluid_grid\home_fslbm\tests\golden_data\
Copy-Item -Recurse docs\Home-FSLBM\golden_data\foam_* `
  wanphys\_src\fluid\fluid_grid\home_fslbm\tests\golden_data\
```

```powershell
uv run pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_regression_gas_foam.py -v
```

无数据时对应用例 **skip**；拷贝后自动启用。

---

## 3. 场景与字段

- 平坦索引（`mrFlow3D`）：`idx = x + nx*(y + ny*z)`（x 最快）
- 每场一个 `.txt`，一行一个数
- Warp 侧经 `reorder_ref_scalar_to_warp` 转为 `array3d[x,y,z]`
- `g_mom.txt` 为 `7 * N` 连续块（方向外层）

| ID | 场景 | 网格/步 | 参考路径 | 关键文件 |
|----|------|---------|----------|----------|
| G1 | `gas_cmr_transform_vectors` | N/A | host `mlConvertCmr*` | `pop_in`, `moment_out`, `moment_in`, `pop_out`, `uxuyuz` |
| G2 | `gas_henry_interface_step1` | 16³ / reconstr. 1 | `launch_g_reconstruction` | `nx/ny/nz`, `flag`, `g_mom`, `delta_g`, `c_value` |
| G3 | `gas_stream_collide_step10` | 16³ / 10 | `launch_g_stream_collide` + swap | `nx/ny/nz`, `flag`, `g_mom`, `c_value`, `steps` |
| G4 | `gas_volume_g_update` | 16³ / 1 | `launch_bubble_volume_g_update` | `bubble_init_volume`（后）、`delta_g`（IC）、`phi`, `tag_matrix`, `flag` |
| F1 | `foam_disjoint_two_spheres` | 32³ / 1 | `launch_calculate_disjoint`（**不**走 Reset） | `disjoin_force`, `phi`, `flag`, `tag_matrix`, `sum_disjoin` |
| F2 | `foam_no_coalescence` | 64³ / 250 | 全耦合 `mlIterateCouplingGpu` | `steps`, `bubble_count`, `merge_flag`, `com_distance` |
| F3 | `foam_atmosphere_open_tank` | 32³ / 1 | `launch_atmosphere_*` | `bubble_rho`, `bubble_volume`, `bubble_init_volume` |

### 场景初值要点

- **G2/G3/G4**：软球 `r=4` 于 16³ 中心；均匀溶解气 `c0=0.01`（G3 中心 `gMom[1]` ×1.3）
- **G4**：在 TYPE_I 且 `tag>0` 上种子 `delta_g=0.02`；`delta_g.txt` 为初值（kernel 会清零）
- **F1**：两球 `r=6`，球心 `(12,16,16)` / `(20,16,16)`（表面间隙 ~2）
- **F2**：两球 `r=6`，球心 `(24,32,32)` / `(39,32,32)`（间隙 ~3），`disjoin_factor=0.032`（参考硬编码）
- **F3**：泡近 `z>nz-10`；host 将 `volume[0]=2e6` 后跑大气 rho/volume 核

---

## 4. 实现备注

参考 CUDA 中 `g_handle` / `calculate_disjoint` 等为内部符号；为导出器在 [`mrLbmSolverGpu3D.h`](../../Home-FSLBM/inc/3D/gpu/mrLbmSolverGpu3D.h) 增加了 `launch_*` / `g_handle` 的 `extern "C"` 入口，避免全步 `mrSolver3DGpu` 在 dump 前把 `disjoin_force` / `delta_g` 清掉。
