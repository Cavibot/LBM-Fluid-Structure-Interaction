# 中期答辩

**题目：** 基于 HOMELBM 的锐界面自由表面与刚体流固耦合  
**工程载体：** WanPhys / `LBM-Fluid-Structure-Interaction`  
**文档日期：** 2026-07-30

---

## 1. 起点与本阶段范围

### 1.1 已有基础

仓库开题时已有：来自 Edward Sun 的 **分布型 dist · D3Q19 · Shan-Chen** 及溃坝 / 双球等算例。该路径为扩散界面，与锐自由面、浮体吃水目标不完全对齐，且整场分布函数在三维高分辨率下带宽成本高。

### 1.2 本阶段重点：HOME-FREE VOF 线

在已有 WanPhys管线上，把**锐界面自由表面**做成可运行、可测、可接刚体的主线：

**HOME 矩步进 + VOF（L/G/I、质量交换）+ 双球 ME-FSI**

统一场景：溃坝；旗舰加挂重 / 轻双刚体球。

---

## 2. 当前技术路线

```
已有：dist · D3Q19 · Shan-Chen（对照）
        │
        └─ 本阶段主线
              HOME 矩步进（home_fp32_ref，D3Q27）
                    +
              VOF 锐界面（纯溃坝可验收）
                    +
              Eq.24 动壁 + 重构链路 ME → 双球 FSI
                    │
                    └── 旗舰：fluid_grid_lbm_dambreak_vof_two_spheres
```

### 2.1 默认 FSI 子步

每个流体–刚体子步大致为：

1. 刚体 SDF → `solid_phi` / 壁速；  
2. HOME-FREE VOF 推进（可选流中累积 ME）；  
3. 重构链路 **动量交换（ME）** → `body_f`；  
4. XPBD + 碰撞推进刚体。

动壁默认 **Eq.24**。经验浮力 / 拖曳仅 `--showcase-fsi`，**不作纯 ME 验收**。

### 2.2 开题前 vs 当前

| 维度 | 开题前已有 | 当前主线 |
|------|------------|----------|
| 界面 | Shan-Chen 扩散 | **VOF 锐界面** |
| 流体 | dist D3Q19 | **HOME-VOF（矩 + φ）** |
| FSI | 雏形 BB / 观感向 | **ME + Eq.24（默认）** |

---

## 3. 已完成工作

### 3.1 HOME-VOF 流体步进

- `home_fp32_ref`：D3Q27、矩编码步进、GPU fused 推进；  
- 无整场 \(f_i\) 作状态真源，需要时 Hermite 重构；  
- 动壁 / 流中 ME 所需的矩→链重构能力；  
- 可选 `--moment-quant`（矩持久化，显存叙事约 −25%）。

### 3.2 自由面与无球验收

- VOF：L / G / I、体积分数 \(\varphi\)、质量交换；  
- 纯溃坝：`fluid_grid_lbm_dambreak_vof.py --backend home`；  
- Martin–Moyce 前缘对照（L0）；  
- 质量 / 动量有界相关单测；  
- 可选 `--height-eq`（IF 上 \(\varphi\to\varphi^*\) 正则）；  
- 一格壁面液膜等 FSLBM 极限已文档化（避免无效调参）。

### 3.3 VOF 上的流固耦合

- 默认反馈 = **重构链路 ME**；`approx` 为诊断；  
- `recommended_me_force_scale`（刚体 \(g\) 与格子 \(g\) 分制下的标度）；  
- 可选 `--me-in-fused`；  
- 双球 / 单球算例；`--showcase-fsi` 与核心路径隔离；  
- 碰撞侧小优化（视觉 mesh 窄相、墙–墙 pair 剪枝等）。

---

## 4. 演示

```bash
# 纯溃坝 HOME-VOF
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof \
  --backend home --n 64 --viewer gl

# 旗舰双球（纯 ME）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64

# 观感对照（非研究默认）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64 --showcase-fsi
```

**演示要点：**

- 锐界面相对已有 Shan-Chen 更干净；  
- 重球（ρ>1）更易贴底冲刷，轻球（ρ<1）更易抬升（趋势）；  
- 默认路径日志应为 `feedback=ME`、`showcase_fsi=off`。

---

## 5. 存在问题与下阶段

| 问题 | 说明 |
|------|------|
| 轻球自由面持续弹跳 | 入水 ME 易过冲；腾空段淹没分数≈0，欠阻尼 |
| 重球后期可能被顶起 | ρ>1 仍可能被自由面 ME 噪声抬离底板 |
| 池面残余波纹 / 一格液膜 | FSLBM/IF 固有极限，不能单靠调参消掉 |
| 纯 ME 浮沉未定量验收 | 有趋势 ≠ 验收通过 |

**下阶段重点：**

- 纯 ME 入水冲量诊断、\(n=64/96\) 对比（离散 vs 标度）；  
- 目标观感：轻球冲击后衰减、重球保持下沉——仍走默认 ME，不用空中假阻尼冒充完成度；  
- 无球质量 / 前缘曲线出图；步进与耦合联调无回归。

---

## 6. 总结

1. **已有** dist / Shan-Chen；**本阶段**做成 HOME-FREE VOF 主线并可接刚体。  
2. **能演示：** 纯溃坝锐界面 + 双球 ME-FSI。  
3. **未完成：** 纯 ME 下浮沉定量可靠。  
4. **下阶段：** 收口 ME 自由面耗散与分辨率，保持研究默认与 showcase 分离。

---

## 附录 · 可能提问

| 问题 | 简答 |
|------|------|
| dist / Shan-Chen 是本阶段做的吗？ | 否，开题前已有；本阶段做的是 VOF 线。 |
| 为什么上矩编码？ | 降带宽；与 HOME / HOME-FREE 论文对齐。 |
| 浮球准吗？ | 轻重趋势可见；纯 ME 弹跳 / 误浮尚未收敛。 |
| height-eq 是否造质量？ | 否，是 IF 上 φ 的软正则。 |
