# HOME-FSLBM 阶段 5 设计与实现报告

> **日期**: 2026-08-11  
> **范围**: 稳定性优先的集成与回归——体积守恒门禁、泡沫无聚并长程验收、管线/剪切清单化  
> **前置**: 阶段 1–4（流体 / 自由表面 / 气泡 CCL / 气体+泡沫）  
> **真理源**: 参考 CUDA（风险 R7），禁止为“压振荡”擅自改公式

---

## 1. 目标与门禁

| ID | 门禁 | 阈值 |
|----|------|------|
| S3 | 单泡体积长期守恒（**σ=0 簿记门禁**） | `10⁴` 步，`|V_T−V_0|/V_0 ≤ 1e-6`；排除 Laplace 驱动的物理缩胀 |
| S2 | 泡沫无聚并 | ≤500 步：`bubble_count==2`、`merge_flag==0`、间距不塌缩；体积反相振幅有界 |
| S1 | 两阶段管线序 | 与参考 `coupling` → `mrSolver3DGpu` 一致 |
| S4 | 剪切衰减 | 既有金标准路径仍绿 |

**本阶段不做**: 大规模 Warp graph 融合（R6，仅剖析证明瓶颈后再做）。

---

## 2. 稳定性威胁模型

### 2.1 体积账本（两条路径）

```
delta_phi  → bubble_volume_update_kernel     # 几何体积 V
delta_g    → bubble_volume_g_update_kernel   # init_volume（质量样量）→ rho = init/V
```

界面压力：`gas_pressure = rho − Laplace − disjoin_factor·P_disjoin`。

任一路径的符号错误、双缓冲漏拷或亨利在零初值气体下泵 `init_volume`，都会改变 `rho` 并反馈到界面，表现为体积漂移或振荡放大。

### 2.2 满液封闭双泡反相振荡

封闭箱 + 液体近不可压 ⇒ `ΔV₁ + ΔV₂ ≈ 0`（反相跷跷板）。  
`ρ = V_init/V` 提供气弹簧；`disjoin` / Laplace 噪声 / 亨利→`init_volume` 提供激励。  
长时零重力下振幅可发散——属物理约束 + 数值激励耦合，**不是**缺开敞上表面。

**决策**:

- 回归验收窗写死 ≤500 步（泡沫）/ 单泡 `10⁴`（体积）
- 体积守恒主门禁使用 **σ=0** 簿记场景（排除 Laplace 驱动的物理缩胀）；有表面张力时的缩胀属平衡过程，不作为 float64 账本门禁
- 可视化 `foam_pair` 默认关闭气体子系统（`enable_gas=False`），突出分离压力门禁；验收窗外可能出现满液反相振荡
- 禁止为压振荡修改参考压力公式；仅允许场景开关与验收窗

### 2.3 消融开关（Model）

| 字段 | 默认 | 作用 |
|------|------|------|
| `enable_gas` | `True` | `False` 时跳过整个 `g_handle` |
| `enable_disjoin` | `True` | `False` 时跳过 `calculate_disjoint`（力保持 0） |

用于诊断探针与泡沫演示；默认物理路径与阶段 4 一致。

---

## 3. 交付物

| 文件 | 内容 |
|------|------|
| [`model.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/model.py) | `enable_gas` / `enable_disjoin` |
| [`solver.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/solver.py) | 按开关门控 Phase1/2 |
| [`stability_probe.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/stability_probe.py) | 体积/ρ/间距/disjoin 探针 |
| [`tests/test_regression_stability.py`](../../wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_regression_stability.py) | S1–S4 |
| 样例 `foam_pair` / `rising_bubble_gas` | 稳定演示配置 + 体积日志 |
| 本文档 / [测试计划](phase5_test_plan_zh.md) / [金标准说明](phase5_stability_golden_zh.md) |

---

## 4. 风险

| ID | 处理 |
|----|------|
| R2 | 体积路径继续使用 `wp.float64` atomic；守恒门禁直接验证 |
| R4 | 曲率钳制保持与参考一致，不额外“改进” |
| R6 | 本阶段不阻塞于 graph 融合 |
| R7 | 任何公式改动必须先证明 Warp≠参考 |
