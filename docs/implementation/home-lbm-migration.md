# HOME-LBM 后端迁移与兼容

本文说明 M6 后的公共入口、默认后端、legacy 回退和旧 `lbm/` 场景边界。迁移目标是让新场景明确使用 HOME 与几何 VOF，同时保留旧实现供一个兼容发布周期复现，不用参数补偿伪造旧结果。

## 公共入口

生产代码从 `wanphys.fluid` 导入：

```python
from wanphys.fluid import HomeFreeDomain, HomeLbmModel

model = HomeLbmModel(...)
fluid = HomeFreeDomain(
    model,
    contact_angle_degrees=90.0,
    project_courant=True,
)
```

`HomeFreeDomain` 当前是 `HomeFreeGeometricDomain` 的公共默认名。它使用十矩 HOME、only-missing 气压边界、Weymouth-Yue 方向分裂 PLIC、质量加权动量输运、几何 topology 和可选 Hodge Courant 投影。默认投影迭代上限为 `max(160, 8*max(resolution))`，实际仍按严格残差提前停止。

需要配置驱动时使用工厂：

```python
from wanphys.fluid import HomeFreeBackend, create_home_free_domain

fluid = create_home_free_domain(
    model,
    backend=HomeFreeBackend.GEOMETRIC,
    contact_angle_degrees=90.0,
)
```

未知 backend 会直接失败，不会回落到 legacy。

## Legacy 回退

旧 link-wise HOME-Free 路径改用显式名称：

```python
from wanphys.fluid import HomeFreeLegacyDomain

fluid = HomeFreeLegacyDomain(model)
```

也可以使用 `backend=HomeFreeBackend.LEGACY`。该路径保留原 `mass/fill/excess` 链路输运和既有场景语义，用于回归、结果复现和迁移期二分；新功能不再默认接入它。它计划至少保留一个兼容发布周期，删除前必须另行公告并有结果迁移记录。

## 不能直接迁移的旧 LBM

`wanphys._src.fluid.fluid_grid.lbm.LbmDomain` 是 D3Q19 BGK/TRT、Shan-Chen 伪势多相模型，持久化 19 个 PDF 及宏观场。HOME-Free 是 D3Q27 十矩、显式锐界面 VOF 和气压边界。两者在状态、界面厚度、表面张力、润湿、单位和稳定参数上都不同，因此：

- `tau/G/sc_boundary_psi/rho_water/rho_air` 不能机械映射成 HOME-Free 参数；
- 旧 TRT 示例继续从 `.lbm` 导入，不会被 `HomeFreeDomain` 静默替换；
- 新自由表面场景需要重新定义 fill 几何、压力锚点、接触角、格子缩放和物理验收；
- 视觉相似不构成迁移通过，必须重新检查质量、动量、寄生流、界面位置和 FSI 作用反作用。

## 初始化要求

几何 fill 必须位于 `[0,1]`，并保持液体和气体之间至少一层 interface cell。平面静水使用 `initialize_planar_hydrostatic_lattice()`；一般几何使用 `initialize_anchored_hydrostatic_lattice()` 并给出有物理意义的压力参考坐标。界面接触固体或非周期域壁时必须显式给出接触角，不能依靠默认猜测。

可运行最小场景：

```bash
PYTHONPATH=. python -m wanphys.examples.lbm.fluid_grid_home_free_column \
  --device cuda:0 --steps 100
```

## 性能与诊断

统一基准入口：

```bash
PYTHONPATH=. python scripts/bench/bench_home_free_fsi.py \
  --backend geometric --project-courant --audit-quantization \
  --resolution 54x54x86 --warmup 10 --steps 50
```

`--backend legacy` 只用于兼容比较；`--no-project-courant` 用于确定投影成本下界，不代表移动自由表面默认验收配置。量化审计只检查候选编码范围，不改变 FP32 状态；任何实际低精度启用仍须重新通过全部物理回归。

## 迁移检查表

1. 把新 HOME-Free 场景导入改到 `wanphys.fluid`。
2. 明确选择 geometric；只有复现旧结果时选择 legacy。
3. 重新建立 fill、压力锚点、接触角和物理/格子缩放。
4. 先通过 CPU 小网格规范，再运行 CUDA 场景门槛。
5. 记录质量、总动量、界面误差、最大速度、投影散度和显存峰值。
6. 不以调小重力、表面张力或耦合力来掩盖失败。
