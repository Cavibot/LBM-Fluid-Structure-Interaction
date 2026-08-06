# 阶段三气泡金标准导出说明

> **导出器已实现**：[`docs/Home-FSLBM/export_bubble_golden.cpp`](../../Home-FSLBM/export_bubble_golden.cpp)  
> **用户职责**：编译运行导出器，并将 `bubble_*` 目录拷贝到 Warp 测试金标准路径。

---

## 1. 构建与运行

在 `docs/Home-FSLBM` 工程中配置 CMake（与现有 `export_solver_golden` 相同依赖：CUDA / OpenCV）。

```powershell
# 配置并编译目标 export_bubble_golden
# （CMake 已将 Home-FSLBM-cuda 升到 C++17；改过 CMake 后请先 reconfigure）
cmake --build <your-build-dir> --target export_bubble_golden --config Release

# 默认输出与 export_solver_golden 相同：docs/Home-FSLBM/golden_data/
# （相对路径 ../../../golden_data；从 out/build/x64-Debug 等构建目录运行即可）
<path-to>\export_bubble_golden.exe
# 或显式指定：
# export_bubble_golden.exe C:\Users\...\docs\Home-FSLBM\golden_data
```

运行结束后应在 `docs/Home-FSLBM/golden_data/` 看到 13 个子目录（与 `droplet_*` 并列）：

```text
docs/Home-FSLBM/golden_data/
  bubble_ccl_3spheres_r3/
  bubble_ccl_touching_face/
  bubble_ccl_touching_edge/
  bubble_ccl_diagonal_gap/
  bubble_ccl_odd_dims/
  bubble_ccl_near_wall/
  bubble_ccl_many_small/
  bubble_init_two_bubbles/
  bubble_init_interface_shell/
  bubble_coupling_volume_delta_phi/
  bubble_two_bubbles_merge/
  bubble_split_via_surface/
  bubble_translate_no_merge/
```

---

## 2. 数据迁移（用户执行）

将上述 `bubble_*` 目录整体复制到：

```text
wanphys/_src/fluid/fluid_grid/home_fslbm/tests/golden_data/
```

例如：

```powershell
Copy-Item -Recurse docs\Home-FSLBM\golden_data\bubble_* `
  wanphys\_src\fluid\fluid_grid\home_fslbm\tests\golden_data\
```

然后运行：

```powershell
uv run pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_regression_bubble.py -v
```

无数据时对应用例会 **skip**；拷贝后自动启用。

---

## 3. 布局约定

- 平坦索引（与 `mrFlow3D` 一致）：`idx = x + nx*(y + ny*z)`（x 最快）
- 每场一个 `.txt`，一行一个数
- Warp 侧经 `reorder_ref_scalar_to_warp` 转为 `array3d[x,y,z]`

### 场景与文件

| 场景 | 关键文件 |
|------|----------|
| CCL（7 个） | `nx/ny/nz.txt`, `input_matrix.txt`, `label_matrix.txt` |
| Init（2 个） | `flag`, `phi`, `tag_matrix`, `bubble_count`, `bubble_volume`, `bubble_rho`, `bubble_init_volume`, … |
| Coupling（4 个） | 同上 + `steps.txt`；`bubble_coupling_volume_delta_phi` 含**初始** `delta_phi.txt`（非末态） |

初始条件与 Warp [`test_regression_bubble.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_regression_bubble.py) 中 builder **逐格一致**（硬球 `TYPE_I`/`TYPE_F` + 六面 `TYPE_S`）。

---

## 4. 导出器内部要点

| 步骤 | API |
|------|-----|
| CCL | 主机填 `input_matrix` → `mlTransData2Gpu` → `connectedComponentLabeling` |
| `bubble_ccl_odd_dims` | 逻辑 17³；参考 YACCLAB 奇数 pitch 会对齐失败，故在 **18³** 上跑 CCL 再裁回 17³ 写出 |
| Init | 填 `flag`/`phi` → upload → `InitBubble` |
| Coupling | `InitBubble` → `mlIterateCouplingGpu` × steps |
| 回传 | 自定义 `trans_bubble_to_host`（标准 `mlTransData2Host` **不含**气泡场） |

---

## 5. 优先验证顺序

1. `bubble_ccl_3spheres_r3`
2. `bubble_init_two_bubbles`
3. `bubble_two_bubbles_merge`
