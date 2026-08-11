# 阶段五稳定性金标准导出说明

> **用途**: 单泡体积时间序列、泡沫 500 步摘要（可选硬比）  
> **用户职责**: 有参考导出器输出后拷贝到 Warp `tests/golden_data/`

---

## 1. 期望目录

```text
docs/Home-FSLBM/golden_data/
  stability_volume_single_bubble/   # 可选
    nx.txt ny.txt nz.txt steps.txt
    volume_series.txt               # 每行一步的 bubble_volume[0]
    volume_rel_drift.txt            # 标量 |V_T-V_0|/V_0
  foam_no_coalescence/              # 可与阶段四共用
    bubble_count.txt merge_flag.txt steps.txt com_distance.txt
```

也可复用阶段四已导出的 `foam_no_coalescence/`。

---

## 2. 迁移

```powershell
Copy-Item -Recurse docs\Home-FSLBM\golden_data\stability_* `
  wanphys\_src\fluid\fluid_grid\home_fslbm\tests\golden_data\
```

```powershell
uv run pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_regression_stability.py -v
uv run pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_regression_stability.py -v -m slow
```

---

## 3. 备注

- 功能门禁 S2/S3 **不依赖**金标准即可跑（Warp 自洽）
- 金标准用于与参考 CUDA 逐点对照时启用
- `10⁴` 步标记 `slow`，CI 默认可跳过
