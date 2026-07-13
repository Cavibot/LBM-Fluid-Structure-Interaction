# Shan-Chen 溃坝观感与调参笔记（D3Q19 + TRT）

> 文档版本：2026-07-13  
> 范围：`wanphys/examples/lbm/fluid_grid_lbm_dambreak_trt.py` 及相关 LBM 参数  
> 关联：[LBM 扩展路线图](lbm_home_roadmap_zh.md)、[模块文件说明](lbm_module_files_zh.md)、[核心审计](lbm_core_audit_zh.md)

---

## 1. 背景与目的

在 G1 基建（`core/`、`phases/shan_chen`、`StepStats`、`benchmark/`）落地后，用无刚体的 TRT + Shan-Chen 溃坝算例检查：

1. 重构后仿真是否仍稳定可用；  
2. 观感上是否接近「底部先流出 → 撞墙大幅涌高 → 回落」的经典溃坝；  
3. 现象主要来自**参数**、**边界/可视化**，还是**当前 SC 实现能力**。

本文记录观察、试验与结论，避免后续重复无效拧旋钮。

---

## 2. G1 完成度（顺带结论）

| 任务 | 状态 |
|------|------|
| `LatticeSpec` / `StepStats` / `ShanChenPhase` / metrics+registry | 已完成 |
| Example 冒烟（two_spheres / dambreak_trt 可跑、无 NaN） | 已通过观察 |
| V0 JSON 数值基线 runner | **未做**（路线图 DoD 仍欠） |

**工程判断**：G1 代码基建可用；正式关闭 G1 仍差「产出 V0 基准 JSON」。观感问题不阻塞 G1 收尾，但影响对 V0「物理观感」的预期。

---

## 3. 主要观察现象

在 128³、Shan-Chen 扩散界面、封闭 bounce-back 盒下反复出现：

| 现象 | 描述 |
|------|------|
| 底板薄膜抢跑 | 主水体尚未到达时，底板已有一层「薄水」先滑出 |
| 后缘黏附 / 两角 | 溃坝柱后墙附近，尤其两侧角，像被黏住 |
| 两侧往中间塌 | 后缘不是立面后退，而是两侧向中线收拢 |
| 撞墙涌高低 | 击中对墙后爬升很矮，几乎没有大幅回落 |
| 阈值误判 | `ssfr-threshold` 提到 1.0 / 1.5 薄膜仍在 → **场内真有高密度前驱带**，非纯渲染假象 |
| 后壁不降、上表面内凹 | 理想溃坝后缘应持续下降成平滑楔；实际后墙一直贴着，只有前缘像楔 |

日志侧（基线附近）：`water` 质量大致守恒、COM 沿 x 推进再回落，**数值稳定**；问题在**形态与能量再分配**，不是整场发散。

### 3.1 密度场日志裁决（2026-07-13，分支重置后复跑）

命令（不依赖 OpenGL，直接读 `state.density`）：

```bash
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt \
  --viewer null --num-frames 180
```

判据：同一时刻比较 **底板层**（`k=1`）与 **中高层**（`k≈nz/4`）上 `ρ>thr` 的最大 `x`；若底板更靠前，则薄膜在求解器里。`thr` 取 0.5 / 0.7 / 1.0 / **1.5**（已接近水体 1.8）。

| t | thr=1.5 底板前缘 | thr=1.5 中高前缘 | lead | 后墙自由面 |
|---|------------------|------------------|------|------------|
| 0.5 s | x=121 | x=29 | **+92** | `z_rear≈104`，湿区已贴到 x=0 |
| 1.0 s | x=127 | x=23 | **+104** | `z_rear≈55` ≫ `z_front≈13` |
| 1.5 s | 已全宽 | 已全宽 | 0 | 后墙仍 `x=0` 附着 |
| 2.5–3.0 s | 仅有底板膜 | 中高无水 (`mid_z=-1`) | 极端 | 后墙高度再次抬起 |

**裁决：**

1. **底板薄层 = 求解器问题，不是 SSFR 渲染。** 即使用 `ρ>1.5`，底板仍领先主体约 90–100 格；渲染阈值再高也消不掉。  
2. **后缘不是「整体平滑楔形下降」。** 自由面从 `x=0` 起算一直湿；`z_rear` 长期偏高，形态是前楔 + 后墙黏附，符合「背后黏墙 → 上表面内凹」。  
3. 左壁 `ρ_max` 可达 3–4（`boundary rho x=0`），后墙附近有强密度堆积，与黏附一致。

诊断打印在 `fluid_grid_lbm_dambreak_trt.py` 的 `_print_density_shape_diagnostics`（每 30 帧）。

---

## 4. 已做试验与结果

算例入口：

```bash
cd LBM-Fluid-Structure-Interaction
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt --viewer gl
```

相关 CLI（试验中加入）：`--ssfr-threshold`、`--tau`、`--gravity-z`、`--lambda-trt`、`--sc-boundary-psi`、`--solid-walls`、`--sc-solid-psi-scale`、`--max-ueq-sq`。

### 4.1 可视化阈值

| 设置 | 结果 |
|------|------|
| `ssfr-threshold` 0.7 → 1.0 / 1.5 | 薄膜仍在 |

**结论**：前驱膜主要是密度场结构，不是 SSFR 切面过低。

### 4.2 降粘 + 加重力

| 设置 | 结果 |
|------|------|
| `tau` 0.55→0.52，`gz` -0.005→-0.008 | 观感更慢、反冲更弱 |

可能原因（与实现相关）：

- velocity-shift 路径存在 `|u_eq|²` **软钳**（原硬编码 0.16，即 `|u|≤0.4`）；加大驱动更容易顶满钳，**有效体力被削**；  
- SC 界面在更强驱动下混合/噪声耗散增加，主水体动能不一定升。

### 4.3 壁面润湿 `sc_boundary_psi`

| 设置 | 结果 |
|------|------|
| `-1`（mirror，历史默认） | 基线观感 |
| `1.0`（过强「水性」） | **淹场**：`water≈524288`（整格）、壁面 `ρ` 飙到 3–4+，画面变黑 |
| `0.1`（疏水，文档建议） | 减轻后墙黏附意图；仍可见两侧内塌、薄膜、低涌高 |

**重要**：六面 **bounce-back 已是固体壁**。`sc_boundary_psi` 只改 SC 对越界邻居的伪势润湿，**不是**「开/关固体壁」。

文档约定（`LbmModel.sc_boundary_psi`）：

- `~0.1` 疏水；`~0.61` 近中性；`~0.83` 亲水；  
- **不要用 `1.0`**（易诱发壁面凝结铺满）。

### 4.4 `--solid-walls`（`solid_phi` 壳）

在网格最外一层标 `solid_phi < 0`，走与刚体耦合相同的固体邻居逻辑。

| 结果 |
|------|
| 可跑；**不能**单独带来大涌高；薄膜与低反冲仍在 |

### 4.5 放宽速度钳 + 降粘 + 疏水壁（组合试验）

实现改动：

- `LbmModel.max_ueq_sq`（默认仍 `0.16`，其它算例不受影响）；  
- `apply_velocity_shift_kernel` 使用该上限；  
- `dambreak_trt` 试验默认曾设为：`tau=0.52`、`max_ueq_sq=0.25`（`|u|≤0.5`）、`sc_boundary_psi=0.1`。

| 观察 |
|------|
| 后缘两侧往中间塌（SC 毛细/接触线收缩，**非**理想溃坝立面）；涌高仍明显不够 |

---

## 5. 文献「SC 能做溃坝」与当前实现的关系

**文献结论成立**：改进 Shan-Chen（常配 TRT/更好 EOS 与 forcing）已有 broken-dam 验证，可对实验/标准解（如前沿位置）给出稳定结果。TRT 相对 BGK 有利于高 Re、界面稳定。

**不能直接推出**：

> 因此当前 `dambreak_trt` 拧 `tau/gz/psi` 就应出现自由表面式大涌高再回落。

| | 文献「改进 SC 溃坝」 | 当前 WanPhys SC 路径 |
|--|---------------------|----------------------|
| 验收重点 | 常为前沿轨迹、稳定性、密度比 | 三维观感 + 撞墙爬升 |
| 密度比 | 专门伪势/EOS 冲高比 | 约 **18:1**（1.8 / 0.1） |
| 维数 | 多为 2D 标定 | **128³ 三维** |
| Forcing | 仔细标定的 Guo/改进格式 | velocity-shift + 可调 Mach 钳 |
| 界面 | 仍是扩散界面，但参数体系完整 | 薄膜、角黏、低涌高 |

**表述修正**：

- 错：「Shan-Chen 做不出溃坝。」  
- 对：「**当前这版 SC + 这组工程参数**，观感达不到锐界面/自由表面溃坝；文献可行依赖改进模型与标定，不是 vanilla 旋钮即出。」

路线图立场仍成立：SC 作 **V0/V1 过渡对比**；锐界面大变形观感以 **G4 VOF / HOME-FREE** 为主目标。

---

## 6. 参数与实现检查清单（以后还想拧时）

| 旋钮 | 预期作用 | 风险 / 注意 |
|------|----------|-------------|
| `tau` ↓ | 降粘、增惯性 | 过低易炸；须同步缩 `lambda_trt` |
| `gz` ↑ | 加快塌落 | 易顶钳、界面碎 |
| `ρ_w / ρ_a` | 提高对比度 | SC 高比需换伪势/EOS，不能硬拉 |
| `G` | 界面张力/分离强度 | 过强假流/黏附，过弱混相 |
| `max_ueq_sq` ↑ | 少削冲击 | 负分布 / NaN |
| `sc_boundary_psi` | 润湿 | `1.0` 已证实淹场；用 0.1 / ~0.61 / ~0.83 |
| `--solid-walls` | `solid_phi` 固体壳 | 不替代锐界面 |
| `N` ↑ | 相对变薄界面 | 成本高 |
| `ssfr-threshold` | 仅可视化 | 已证明消不掉薄膜 |

内核相关位置：

- 速度钳：`kernels.apply_velocity_shift_kernel` ← `model.max_ueq_sq`  
- 壁伪势：`sc_boundary_psi` / `sc_solid_psi_scale`（见 `model.py` 注释）

---

## 7. 建议决策

1. **G1**：以稳定基线（历史上 `tau=0.55`、`gz=-0.005`、`psi=-1` 或文档疏水 `0.1`）作为 V0 **数值/性能**基线即可；不要用「是否像教科书溃坝」卡 G1。  
2. **观感**：接受当前 SC 三维溃坝的形态局限；若要对标实验前沿曲线，应开「改进 SC」专项（伪势/EOS/forcing/2D 标定），或等 G4 VOF。  
3. **试验默认**：`dambreak_trt` 上的激进默认（低粘、宽钳、疏水）仅作试验记录；合入主线前建议恢复保守默认，以免其它同学误以为是生产参数。  
4. **禁止再试**：`sc-boundary-psi 1.0` 作为「更固体/更爬升」——会淹场。

---

## 8. 相关命令速查

```bash
# 基线风格（建议对照）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt \
  --viewer gl --tau 0.55 --gravity-z -0.005 --sc-boundary-psi -1 --max-ueq-sq 0.16

# 试验组合（低粘 + 疏水 + 宽钳）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt \
  --viewer gl --tau 0.52 --sc-boundary-psi 0.1 --max-ueq-sq 0.25

# 固体壳（可选）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt \
  --viewer gl --solid-walls --sc-solid-psi-scale 1.2
```

带球 FSI 算例（无「去球」开关）：

```bash
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_two_spheres --viewer gl
```

---

## 9. 一句话总结

**当前 WanPhys 的 SC+TRT 溃坝能稳定跑，也符合扩散界面多相的常见形态；文献证明改进 SC 可以做好溃坝标定，但不等于本仓库现状拧参数就能得到锐界面式大涌高。后续要么做改进 SC 专项，要么把观感期望放到 G4 VOF。**

---

*维护：若默认算例参数回滚或 forcing/伪势升级，请更新 §4–§7。*
