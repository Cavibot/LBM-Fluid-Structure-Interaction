# HOME-FREE VOF 动量交换与浮力：结论总结

> 文档版本：2026-08-07  
> 主算例：`wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py`  
> 算法全文：[lbm_home_vof_fsi_algorithm_zh.md](lbm_home_vof_fsi_algorithm_zh.md)  
> 论文对照：HOME-FREE §4.3（锐界面 + 切割单元链路力，式 32）

本文只总结**近期双球溃坝 FSI 实验的结论**：哪些是纯动量交换（ME），哪些是竖直补项，以及「轻球浮不起来」到底卡在哪。

---

## 1. 一句话结论

**水平冲击、推球、力矩：链路 ME 可用；竖直阿基米德起伏：在当前 Guo + 近均匀 \(\rho\approx 1\) 设定下，纯 ME 几乎给不出，所以默认加了 φ 体积 Archimedes 补项。**

这不是「VOF 不适合做动量交换」，而是 **体力实现让静水压差进不了链路力**。

---

## 2. 当前默认路径（混合）

每子步顺序：

```text
SDF 栅格化 → HOME Eq.24 动壁 → 重构链路 ME（式 32）
         → φ-volume Archimedes（竖直 Fz，默认开）
         → （可选）showcase / --me-drag
         → XPBD
```

| 项 | 来源 | 作用 |
|----|------|------|
| 链路 ME | `link_me_warp.py` / fused ME | 波击、水平推、转矩；论文主路径 |
| φ 体积 Archimedes | `phi_volume_buoyancy_warp.py` | \(F_z=\mathrm{scale}\cdot s\cdot\rho_L\cdot V\cdot\|g\|\)；竖直起伏 |
| showcase 浮力/推/拖 | `--showcase-fsi` | 更重的经验插件，**非**默认 |
| ME 路径拖曳 | `--me-drag` | 可选阻尼，**非**默认 |

开关：

- `--no-archimedes`：关掉竖直补项 → 基本只剩贴地滚动  
- `--archimedes-scale`：放大/缩小起伏（默认 `1.6`）  
- 水量与其它 VOF 溃坝一致：`DAM_X_FRAC=0.25`，`FILL_Z_FRAC=0.5`  
- 球**坝前干地**起步，等溃坝冲到后再谈浮沉（不预埋进水库）

---

## 3. 为什么纯 ME 浮不起来

论文式链路力（简化）：

\[
\Delta\hat{\mathbf{j}}
=
\varphi\,(f^*_{\bar{i}}+f_i-2w_i)\,\mathbf{c}_{\bar{i}}
-
\varphi\,(f^*_{\bar{i}}-f_i)\,\mathbf{u}^s
\]

本仓库对固体取 \(\mathbf{F}_{\mathrm{solid}}=-\sum\Delta\hat{\mathbf{j}}\)（Ladd 作用反作用；反号会把轻球往下按）。

在当前实现里：

1. **Guo 体力重力**把 \(g\) 加在碰撞/平衡里，**不是**靠静水 \(\rho(z)\) 在分布里形成压差。  
2. 液体侧近乎 **\(\rho\equiv 1\)** 时，\(f^*+f-2w_i\approx 0\)，链路第一项接近 0。  
3. 结果：ME 对**冲击、剪切、水平动量交换**仍敏感，对**静水阿基米德力**几乎失明。  
4. 溃坝残水相对球径偏浅时，竖直可浸没体积也小，观感更弱。

因此：关掉 Archimedes 后轻球「冲完只在地上滚」是**预期现象**，不是算例把球放错了那么简单。

曾试过「湿初始 + 静水 \(\rho(z)\)」让 ME 侧竖直略有抬升，但那是改 IC，且用户要求是**冲出后再起伏**，不是一开始泡在水里。

---

## 4. VOF 有没有问题？

| 判断 | 结论 |
|------|------|
| VOF + cut-cell 能不能做 ME？ | **能**。论文 §4.3 就是锐界面自由面 FSI；本路径水平推球也靠它。 |
| 竖直浮不起来是不是 VOF 不行？ | **不是。** 主因是 Guo + 均匀密度 → 静水压不进 \(\Delta\hat{\mathbf{j}}\)。 |
| 要不要退回纯经验 showcase？ | 不必。默认已是 **ME（水平）+ 窄竖直补项**；showcase 推/拖仍是可选演示。 |

φ 体积 Archimedes 与 showcase 的区别：

- 同一公式对重/轻球一视同仁：只靠 \(\rho_{\mathrm{sphere}}\)（质量）与浸没 \(s\) 分沉浮  
- **没有**对轻球单独加力、单独放大  
- 仍是壳采样 \(\varphi\) 的体积启发式，**不是**壁面压力积分，也**不是**论文 ME

---

## 5. 场景与密度（演示约定）

| 量 | 典型值 | 说明 |
|----|--------|------|
| 坝宽 / 填高 | \(0.25\times N\) / \(0.5\times N\) | 与 `dambreak_vof` / 单球一致（曾临时加深又改回） |
| 起步 | 坝前干地，\(z\approx R\) | 先冲后浮 |
| 重球密度 | \(1.8\) | \(>\rho_L\) → 沉 |
| 轻球密度 | \(0.30\) | \(<\rho_L\) → 浮（需 Archimedes 补项才明显） |
| 力单位 | \(\rho\,dh^4/\mathrm{dt}^2\)；\(g_{\mathrm{rigid}}=g_{\mathrm{lbm}}\,dh/\mathrm{dt}^2\) | ME 与刚体重力刻度对齐 |

接触刚度偏软是共用参数（方便任何浮力足以离地），不是轻球特化。

---

## 6. 实验观感对照

| 设定 | 大致观感 |
|------|----------|
| 纯 ME，`--no-archimedes`，干起步 | 冲过去 → 贴地滚；轻球几乎不浮 |
| ME + Archimedes（默认） | 冲过后轻球 \(z\) 可明显抬升并起伏；重球贴地 |
| 预埋进水库 / 加深坝 | 一开始就能「泡着」；不符合「冲出来再起伏」 |
| `--showcase-fsi` | 更强经验推/拖；观感更「演」，离论文更远 |

Headless 冒烟（水量改回后请以实跑为准）：`--viewer null --n 64`，看 status 里 light 的 \(z\) 与 `sub=`。

---

## 7. 若要坚持「更纯的 ME」

方向：

1. **Path A（已实现试跑）**：`--hydro-rho` 软维持静水 \(\rho(z)\)，并关闭 Archimedes。见 `hydrostatic_rho_warp.py`。  
2. **切胞压力积分**：在固体表面用 \(\varphi\)-切割单元压力。  
3. **接受论文验证方式**：原文重/轻体多为定性沉浮，并不声称零经验参数必出夸张起伏。

```bash
# Path A：纯 ME 竖直（靠维持 ρ(z)）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64 --hydro-rho
```

`--no-archimedes` 且无 `--hydro-rho` 时，轻球后期贴着重球略高多半是**接触顶起**，不是浮力。

在未做稳妥的 A/B 之前，**夸张竖直起伏**仍可用默认 Archimedes 补项。

---

## 8. 相关文件

| 角色 | 路径 |
|------|------|
| 双球算例 | `wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py` |
| 链路 ME | `.../home_fp32_ref/link_me_warp.py`（及 fused 内 ME） |
| φ 体积浮力 | `.../home_fp32_ref/phi_volume_buoyancy_warp.py` |
| 算法说明 | `docs/wanphys/lbm_home_vof_fsi_algorithm_zh.md` |

运行：

```bash
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64
```
