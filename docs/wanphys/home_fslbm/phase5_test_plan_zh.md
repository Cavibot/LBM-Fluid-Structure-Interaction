# HOME-FSLBM 阶段 5 测试计划

> **日期**: 2026-08-11  
> **范围**: 稳定性诊断、体积守恒、泡沫无聚并、管线序、剪切清单  
> **前置**: [阶段 5 设计](phase5_design_zh.md)

---

## 1. 诊断探针

模块：`stability_probe.py` / 测试内调用。

| 量 | 含义 |
|----|------|
| `bubble_volume[i]` | 几何体积 |
| `bubble_init_volume[i]` | 质量样量 |
| `bubble_rho[i]` | `init/V` |
| COM 距 | 两泡质心欧氏距 |
| `Σ|disjoin|` | 膜区分离压力总和 |

消融矩阵（短跑 ≤200 步）：

| 配置 | 预期 |
|------|------|
| 全开（默认） | 基线 |
| `enable_gas=False` | 去掉亨利→init 激励 |
| `enable_disjoin=False` | 去掉膜区斥力激励 |
| 单泡封闭箱 | 无反相约束，测净漂移 |

---

## 2. 回归门禁（`test_regression_stability.py`）

| ID | 测试 | 网格/步 | 验收 |
|----|------|---------|------|
| S1 | `test_pipeline_order_smoke` | 16³ / 1 | 单步后场有限；gas/disjoin 开关可关 |
| S2 | `test_foam_no_coalescence_500` | 32³~64³ / ≤500 | count=2、merge=0、间距 ≥ 0.5×初始；`max(V)−min(V)` 振幅有界 |
| S3a | `test_bubble_volume_conservation_smoke` | 32³ / 200，σ=0 | 相对漂移 < 1e-3 |
| S3b | `test_bubble_volume_conservation_1e4` | 64³ / 10⁴，σ=0 | `\|V_T−V_0\|/V_0 ≤ 1e-6`（`pytest.mark.slow`） |
| S4 | `test_shear_decay_listed` | 复用既有 shear 金标准 | 阶段五清单引用，不重复失败 |

无金标准文件时相关硬比 **skip**；功能门禁（S2/S3）不依赖导出器。

---

## 3. 可视化验收

| 示例 | 配置 | 人工检查 |
|------|------|----------|
| `foam_pair` | `enable_gas=False`；观察 ≤500 步等效 | count=2；stderr 打印 V/ρ；不要求长时稳态 |
| `rising_bubble_gas` | 默认气体开 | 上升；打印体积相对漂移；不发散 |
| `dambreak` | 既有 | 无崩溃 |

---

## 4. 成功条件

1. S3b 通过，或本地 slow 标记明确且冒烟 S3a 通过  
2. S2 泡沫窗内稳定  
3. `pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests` 无回归（slow 可选）
