# HOME-FREE VOF 浮力：ME 与 Archimedes / Path B 结论

> 文档版本：2026-08-20  
> 主题：**浮力**（竖直沉浮）与水平冲击 ME 的分工  
> 主算例：`wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py`  
> 算法全文：[lbm_home_vof_fsi_algorithm_zh.md](lbm_home_vof_fsi_algorithm_zh.md)  
> 论文对照：HOME-FREE §4.3（锐界面 + 切割单元链路力，式 32）；Path B 参考 modified-pressure LBM（Liu；Guo et al. JCP 2025）

本文总结双球溃坝 FSI 上的**浮力**结论：哪些力负责水平冲击，哪些负责竖直浮力，以及「轻球浮不起来」卡在哪、有哪些可用开关。

---

## 1. 一句话结论（浮力）

**正常物理默认：液体 \(\rho\approx 1\) 均匀；水平冲击用链路 ME；竖直浮力用静水压差积分 \(F=\oint -p\,\mathbf{n}\,dA\)（\(p=\rho|g|h\)，`--archimedes`）。**

| 角色 | 默认 | 说明 |
|------|------|------|
| 冲击 / 力矩 | 式 32 ME | 波击、水平推；Guo+\(\rho\equiv1\) 下几乎无阿基米德浮力 |
| **浮力（默认）** | 压差 Archimedes | 刚体侧状态力；**无**流体反作用 |
| **浮力（Path B 稳）** | `--mod-pressure` → \(F_{\alpha,H}\)（默认只保留 \(F_z\)） | ME 内静水修正；仍配 Guo；关 Archimedes |
| **浮力（Path B 流体，实验）** | 再加 `--mod-pressure-fluid` | zero-Guo + 软化 \(\rho_G(z)\)；作用–反作用框架，浮力仍偏弱 |
| 勿当默认 | `--hydro-rho` | 用 \(\rho(z)\) 骗 ME；会改质量，**不是**不可压水 |

不要用静水 \(\rho(z)\) 冒充压差浮力。纯 ME 在 Guo+\(\rho\equiv1\) 下几乎无阿基米德力。

---

## 2. 当前默认路径（正常物理 + 浮力）

每子步顺序：

```text
SDF 栅格化 → HOME Eq.24 动壁 → 重构链路 ME（式 32）
         → 静水压差积分 Archimedes（竖直浮力，均匀 ρ_L）
         → （可选）showcase / --me-drag / --hydro-rho / --mod-pressure*
         → XPBD
```

| 项 | 来源 | 作用 |
|----|------|------|
| 链路 ME | `link_me_warp.py` / fused ME | 波击、水平推、转矩 |
| **压差浮力 Archimedes** | `coupling/archimedes_buoyancy.py`（核：`pressure_buoyancy_warp.py`） | \(F=\oint -p n\,dA\)；默认开 |
| **Path B 浮力 \(F_{\alpha,H}\)** | `hydro_me_warp.py` | `--mod-pressure`；默认只竖直分量 |
| Path B 流体 \(\rho_G\) | `vof_warp.py` FS BC | `--mod-pressure-fluid`；关 Guo；`fs_blend≈0.45` |
| `--hydro-rho` | `hydrostatic_rho_warp.py` | **实验**：\(\rho(z)\) 骗 ME；默认关 |
| showcase | `--showcase-fsi` | 更重经验插件（含经验浮力） |

默认：**均匀 \(\rho\)**、`archimedes=on`、`hydro_rho=off`、`mod_pressure=off`。水量 `DAM_X_FRAC=0.25`，`FILL_Z_FRAC=0.5`；球坝前干地起步。

---

## 3. 为什么纯 ME 浮力起不来

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
3. 结果：ME 对**冲击、剪切、水平动量交换**仍敏感，对**静水阿基米德浮力**几乎失明。  
4. 溃坝残水相对球径偏浅时，竖直可浸没体积也小，观感更弱。

因此：关掉 Archimedes 后轻球「冲完只在地上滚」是**预期现象**，不是算例把球放错了那么简单。

曾试过「湿初始 + 静水 \(\rho(z)\)」让 ME 侧竖直略有抬升，但那是改 IC，且演示约定是**冲出后再起伏**，不是一开始泡在水里。

---

## 4. VOF 有没有问题？

| 判断 | 结论 |
|------|------|
| VOF + cut-cell 能不能做 ME？ | **能**。论文 §4.3 就是锐界面自由面 FSI；本路径水平推球也靠它。 |
| 竖直**浮力**起不来是不是 VOF 不行？ | **不是。** 主因是 Guo + 均匀密度 → 静水压不进 \(\Delta\hat{\mathbf{j}}\)。 |
| 要不要退回纯经验 showcase？ | 不必。默认已是 **ME（水平）+ 压差浮力（竖直）**；showcase 推/拖仍是可选演示。 |

压差 Archimedes 浮力与 showcase 的区别：

- 同一压力公式对重/轻球一视同仁：只靠 \(\rho_{\mathrm{sphere}}\)（质量）与湿表面压差分沉浮  
- **没有**对轻球单独加力、单独放大  
- 默认是静水 \(p=\rho|g|h\) 的表面积分，**不是**显式排水体积；`method="volume"` 仍可切回旧 ρVg 路径；**不是**论文 ME

---

## 5. 场景与密度（演示约定）

| 量 | 典型值 | 说明 |
|----|--------|------|
| 坝宽 / 填高 | \(0.25\times N\) / \(0.5\times N\) | 与 `dambreak_vof` / 单球一致 |
| 起步 | 坝前干地，\(z\approx R\) | 先冲后浮 |
| 重球密度 | \(1.8\) | \(>\rho_L\) → 沉 |
| 轻球密度 | \(0.30\) | \(<\rho_L\) → 浮（需浮力路径才明显） |
| 力单位 | \(\rho\,dh^4/\mathrm{dt}^2\)；\(g_{\mathrm{rigid}}=g_{\mathrm{lbm}}\,dh/\mathrm{dt}^2\) | ME 与刚体重力刻度对齐 |

接触刚度偏软是共用参数（方便任何浮力足以离地），不是轻球特化。

---

## 6. 浮力路径观感对照

| 设定 | 大致观感 |
|------|----------|
| 纯 ME，`--no-archimedes`，干起步 | 冲过去 → 贴地滚；轻球几乎不浮 |
| ME + Archimedes（**默认浮力**） | 冲过后轻球 \(z\) 可明显抬升并起伏；重球贴地 |
| `--mod-pressure`（Guo + 竖直 \(F_{\alpha,H}\)） | 关 Archimedes；轻球可浮并随波右移（冒烟可用） |
| `--mod-pressure --mod-pressure-fluid` | zero-Guo + blend \(\rho_G\)；可右移，浮力偏弱/偏短暂 |
| 预埋进水库 / 加深坝 | 一开始就能「泡着」；不符合「冲出来再起伏」 |
| `--showcase-fsi` | 更强经验推/拖；观感更「演」，离论文更远 |

Headless 冒烟：`--viewer null --n 64`，看 status 里 light 的 \(z\) 与 `sub=`。

---

## 7. 更「纯」的浮力 / ME 实验路径

### 7.1 Path A：`--hydro-rho`（骗 ME 浮力）

软维持静水 \(\rho(z)\)，关 Archimedes。见 `hydrostatic_rho_warp.py`。会改湿质量刻度，**不是**正常不可压水。

### 7.2 Path B：`--mod-pressure`（推荐的 ME 侧浮力实验）

| 开关 | 流体 | ME 浮力 | Archimedes | 状态 |
|------|------|---------|------------|------|
| `--mod-pressure` | 保持 Guo | \(F_{\alpha,H}\)（默认只 \(F_z\)） | 关 | **稳**；轻球可浮 |
| 再加 `--mod-pressure-fluid` | zero-Guo + \(\rho_G(z)\) | 同上 | 关 | **实验**；右移 OK，浮力弱 |

要点：

- \(F_{\alpha,H}\)：按列自由面深度构造 \(\rho_{G,w}\)，在 ME 上加静水修正（`hydro_me_warp.py`）。  
- 默认 **`vof_mod_pressure_fh_vertical=True`**：只保留竖直浮力分量，去掉噪声水平力以免干扰溃坝冲击。  
- `--mod-pressure-fluid`：FS 用 \(\rho_G(z)=\rho_0-\rho_0 g_z(z-z_{\mathrm{ref}})/c_s^2\)，并用 **`vof_mod_pressure_fs_blend≈0.45`** 软化（全公式无球 COM 过猛）。  
- **禁止** Guo + \(\rho_G\) 叠用（双计重力，COM 过冲）。  
- 刚体-only 压差 Archimedes **无**流体反作用；fluid Path B 才是场内作用–反作用方向，但当前浮力仍弱于 Guo+\(F_H\) / Archimedes。

```bash
# 默认浮力：压差 Archimedes
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64

# Path B 浮力（稳）：F_α,H + Guo
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64 --mod-pressure

# Path B 流体浮力（实验）：zero Guo + ρ_G(z)
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64 --mod-pressure --mod-pressure-fluid

# Path A（骗 ME 浮力）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64 --hydro-rho
```

`--no-archimedes` 且无 `--mod-pressure` / `--hydro-rho` 时，轻球后期贴着重球略高多半是**接触顶起**，不是浮力。

日常夸张竖直起伏仍优先用默认 **Archimedes 浮力**；要 ME 内浮力演示用 `--mod-pressure`。

### 7.3 其他方向

- **切胞压力积分**：在固体表面用 \(\varphi\)-切割单元压力（真场压浮力，未做）。  
- **接受论文验证方式**：原文重/轻体多为定性沉浮，并不声称零经验参数必出夸张起伏。

---

## 8. 相关文件（浮力）

| 角色 | 路径 |
|------|------|
| 双球算例 | `wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py` |
| 链路 ME | `.../home_fp32_ref/link_me_warp.py`（及 fused 内 ME） |
| **压差浮力** | `.../home_fp32_ref/pressure_buoyancy_warp.py` + `coupling/archimedes_buoyancy.py` |
| **Path B ME 浮力** | `.../home_fp32_ref/hydro_me_warp.py`（`accumulate_hydro_me_correction`） |
| Path B 流体 \(\rho_G\) | `.../home_fp32_ref/vof_warp.py`（FS \(\rho_G\)；`vof_mod_pressure_*`） |
| （legacy）φ 体积浮力 | `.../home_fp32_ref/phi_volume_buoyancy_warp.py` |
| 算法说明 | `docs/wanphys/lbm_home_vof_fsi_algorithm_zh.md` |
