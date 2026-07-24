# LATEST HOME-FREE VOF 双球：近期工作总结

> 文档版本：2026-07-24  
> 标记：`LATEST` — 当前主线进度快照（相对路线图 / 分专题文档）  
> 适用范围：`lbm_backend='home_fp32'` + `phase_mode=vof_sharp` + 双球 FSI  
> 主算例：`wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py`  
> 相关：[流固算法详解](lbm_home_vof_fsi_algorithm_zh.md)、[`height_eq` 总结](lbm_home_vof_height_eq_summary_zh.md)、[单格极限](lbm_home_fslbm_one_cell_limit_zh.md)、[路线图](lbm_home_roadmap_zh.md)、[模块文件](lbm_module_files_zh.md)

本文汇总截至写作日在 **溃坝双球** 上的工程结论：液面整形、球–液耦合抖动、16-bit 矩量化持久存储、刚体碰撞开销，以及与 `公式修订版` 论文的对齐关系。

---

## 1. 当前主线一句话

**HOME-FREE VOF（GPU fused）+ `solid_phi` 栅格 FSI + 可选 `--height-eq` + 可选 `--moment-quant`（混合精度持久矩）**；FPS 余量主要在 **流体步进** 与 **刚体 collide+XPBD**，不在论文式 8³ tile / 纯流体 50% 量化。

---

## 2. 已放弃 / 勿随意复活

| 路径 | 结果 |
|------|------|
| `--film-drain` 等贴壁薄膜启发式 | 质量泄漏、坑、气泡 |
| 连续 CSF / 全局弹簧当「物理找平」 | 受单相 FSLBM **一格 IF** 极限制约 |
| 墙 α 加强的 height-eq | A 墙波纹；已改为均匀 α + 近刚体 soft fade |

工作路径：`--height-eq` = 晚期池面 IF 的 **φ→φ\* 正则化**（非 NS CSF）。

---

## 3. 液面与球体观感

### 3.1 `--height-eq`

- Arms：`t ≥ 8` 后打开；默认 `rate=0.025`，`every=24`，`dh_cap=0.03`；GPU 路径 `u_damp=0`。
- 近刚体：曾用 `w=0` 死区 → 球边永久凹陷环。现改为地板 **`w_min=0.35`**（`height_eq_warp.py` / host 同逻辑），弯月面缓慢靠向 φ\*。
- 域墙：去掉 wall α boost；贴壁一格波纹在冲击后消退属预期。

### 3.2 球抖 / 挖坑

- 壳采样湿分数 chatter（如 light `sub` 0.28↔0.71）+ 强 `FLUID_PUSH` 会挖弯月面。
- 现：浸没 **EMA**（`SUB_EMA_ALPHA=0.05`）+ **`dsub_cap=0.015`**；`FLUID_PUSH_RATE=8`；height-eq 武装后推力 × **`LATE_POOL_PUSH_SCALE=0.12`**。
- 日志：`skipB` 在 `w_min` 后应接近 0（不再整圈跳过）。

### 3.3 试跑

```bash
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 48 --height-eq
```

---

## 4. 16-bit 矩量化（`--moment-quant`）

### 4.1 物理约定（必对）

HOME 存 **Hermite `S`**，平衡约 `u⊗u`（`c_s²` 已在投影子里）。neq：

\[
S_{\alpha\alpha}^{\mathrm{neq}} = S_{\alpha\alpha} - u_\alpha^2,\quad
S_{\alpha\beta}^{\mathrm{neq}} = S_{\alpha\beta} - u_\alpha u_\beta
\]

**不要再减 `c_s²`**（曾导致液体 `S≈0` → 钳位 → ρ 崩到 ~0.5）。

### 4.2 持久存储（已落地，论文 FSI 混合档）

| 项 | 内容 |
|----|------|
| 跨步 SoT | `moment_q`：5×uint32/格（ρ,u,S^neq 打包） |
| 工作集 | **一套** fp32 矩（不再分配矩双缓冲；`_b` 与 primary 别名） |
| fused | `use_quant_in`：从 `moment_q` 反量化读邻居 → 写 float → surface/solid 后 re-pack |
| 显存（仅矩） | 80 B/格（fp32 双缓冲）→ **60 B/格（−25%）** |
| 开关 | `LbmModel.vof_home_moment_quant` / CLI `--moment-quant` |

实现：`home_fp32_ref/quant.py`、`vof_warp.py`（fused + alloc）、`bridge._ensure_gpu(moment_quant=…)`。

### 4.3 与论文的边界

- 对齐：VN 范围、只存 S^neq、抖动、FSI **混合精度 ~25%** 叙事。
- **未做 / 与当前 VOF 线不合拍**：纯流体 50%（无整场 float）、8³ tile SoA、独立 `home_quant/` 双核固体校正。  
  那些默认单相 + mesh 固体；你们是 VOF 场 + `solid_phi` 栅格，硬搬性价比低。

```bash
# 可选叠加
... --height-eq --moment-quant
```

---

## 5. 性能剖面（n=48，12 substeps/frame）

典型拆分（ms/frame，无 viewer）：

| 段 | 约 | 说明 |
|----|-----|------|
| fluid（含耦合内 LBM） | ~11–12 | 仍最大单块 |
| collide | 修前 ~9–10 → 修后 **~7** | 见 §6 |
| XPBD | ~6 | `iterations=2`，**GPU Warp**，非 CPU 整段求解 |
| buoy / raster / feedback | ~1–2 合计 | 已 GPU 化 |

先前笼统「XPBD ~14ms」≈ **collide + XPBD**；其中 **碰撞 ≥ XPBD 求解**。

不减 LBM 子步的前提下：再砍 XPBD iter 收益有限；刚体侧优先碰撞管线。

---

## 6. 刚体碰撞优化

### 6.1 论文 vs 实测

`公式修订版` 中的「固体碰撞」多为：HMBB / 动量交换 / **分裂固体校正核** / 体素跳空。  
双球瓶颈是 **Newton/WanPhys 刚体对碰管线**（球–墙），不是 LBM 链路–三角形。

### 6.2 已做

1. **`has_meshes`**：仅当存在带 `ShapeFlags.COLLIDE_SHAPES` 的 mesh 才开三角形窄相。视觉 mesh（仅 VISIBLE）曾误开昂贵路径。  
   → `wanphys/_src/collision/rigid/pipeline.py`
2. **墙–墙 pair**：算例 finalize 后剪掉 static–static（28→13）。  
   → `_prune_static_static_contact_pairs` in two_spheres 算例

微基准：collide ~0.72 → **~0.49 ms/call**；刚体合计约 **16 → 13 ms/frame**。

### 6.3 可选后续（未做）

- LBM 仍 12 子步，刚体 collide/XPBD **隔步**（改耦合时序，需验收穿透）
- 墙改为平面约束 / 更轻宽相

---

## 7. 关键代码入口

| 主题 | 路径 |
|------|------|
| height-eq GPU | `.../home_fp32_ref/height_eq_warp.py` |
| 矩量化 | `.../home_fp32_ref/quant.py` |
| fused + 持久 quant | `.../home_fp32_ref/vof_warp.py` |
| 模型开关 | `lbm/model.py` → `vof_home_moment_quant*`、`vof_height_eq*` |
| 双球算例 | `examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py` |
| 碰撞 has_meshes | `collision/rigid/pipeline.py` |
| 浮力 EMA | `.../sphere_buoyancy_warp.py` |
| 质量/动量守恒测试 | `newton/tests/test_lbm_home_vof_conservation.py` |

目录树与各文件职责见 **[LBM 模块文件说明](lbm_module_files_zh.md)**（含 `home_fp32_ref/`）。

---

## 8. 正确性测试（水质量 / 动量）

文件：`newton/tests/test_lbm_home_vof_conservation.py`

**库存定义（HOME-FREE / Körner）**

| 量 | 定义 |
|----|------|
| 水质量 | `Σ mass`（液体 ≈ ρ；界面 = φρ；固体外） |
| 水体积代理 | `Σ φ` |
| 动量 | `Σ mass · u`（质量加权） |

闭墙 + 无 film-drain：水质量应近似守恒。墙面 BB / 重力下动量**不**守恒——对应用例只查静池 \|P\| 或「有重力时质量仍保」。

| 类 | 断言要点 |
|----|----------|
| `TestHomeVofMassNumpy` | 参考步进：闭墙 ± 重力，质量相对漂移 ≲ 2–3% |
| `TestHomeVofMassGpu` | fused GPU 同上；静池无外力 \|P\|/M 很小 |
| `TestHomeVofMomentQuantConservation` | quant 与 fp32 质量跟踪；液体 ρ 不塌到 ~0.5 |
| `TestHomeVofHeightEqMass` | 多次 φ→φ\* 后水质量仍保 |
| `TestHomeVofFiniteNoNan` | 长跑场有限 |

```bash
cd LBM-Fluid-Structure-Interaction
uv run --extra examples python -m unittest newton.tests.test_lbm_home_vof_conservation -v
```

相关冒烟：`test_lbm_home_vof.py`、`test_lbm_home_step.py`（周期 ρ）、`test_lbm_home_solver_hook.py`。

---

## 9. 建议下一步（按目标）

| 目标 | 建议 |
|------|------|
| 观感（球边坑、环波） | 继续调 `w_min` / push / EMA；先无 quant 对比 |
| FPS | 流体核与耦合 launch；碰撞隔步需单独决策 |
| 论文对齐（量化 HOME 全量） | 独立 `home_quant/` + 双核固体 — **新后端**，勿当成双球补丁 |

---

## 10. 变更清单（本周期）

- [x] height-eq 近刚体 `w_min`；晚期推力 / 浸没 EMA 收紧  
- [x] 16-bit 矩量化正确 neq + **持久混合存储（−25% 矩字节）**  
- [x] 碰撞：可视 mesh 不再强制 `has_meshes`；剪墙–墙 pair  
- [x] 剖面澄清：XPBD 在 GPU；collide 曾被算进「XPBD」桶  
- [x] 水质量 / 动量正确性测试套件 + 文档入口  
- [ ] 论文分裂固体核 / tile / 纯 50% quant（明确不在当前 VOF 双球主线）
