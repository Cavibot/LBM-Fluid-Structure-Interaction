# HOME-FREE VOF：界面 φ 液面正则化（`vof_height_eq`）总结

> 文档版本：2026-07-20  
> 适用范围：`wanphys` `lbm_backend='home_fp32'`（`home_fp32_ref`）  
> 实现：`wanphys/_src/fluid/fluid_grid/lbm/backends/moment/home_fp32_ref/height_eq.py`  
> 接线：`bridge.py` → `HomeFp32VofBridge.apply_height_equation`；`LbmModel.vof_height_eq*`  
> 算例：`wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof.py --backend home --n 48 --height-eq`  
> 相关文档：[近静置找平试验](lbm_home_vof_quiet_level_summary_zh.md)、[单格找平极限](lbm_home_fslbm_one_cell_limit_zh.md)

本文总结当前 **opt-in** 的主液面整形方法：在 sharp FS / VOF 步进之外，对**池面界面格**做质量守恒的 φ 松弛与拓扑清理，目标是溃坝后期主液面「大片高低消失、近似一层 IF」。

---

## 1. 一句话定位

`vof_height_eq` 是 **late-pool / 安静池面的自由面正则化（regularizer）**，不是 VOF 本构，也不是 PLIC 那种几何重建。

| 对比 | PLIC | `vof_height_eq` |
|------|------|-----------------|
| 角色 | 由局部 \(φ\) 重建与通量相容的界面几何 | 按「主液面应平坦」先验改写界面 |
| 物理地位 | 离散格式的几何闭合 | **目标驱动的投影 / 政策** |
| 关掉后 | 换一套通量几何，场仍自洽 | 液面可不平，但动力学更「原教旨」 |
| 默认 | 求解路径内常用 | **默认关**（`--height-eq` 才开） |

sharp 单相 FS **没有**连续高度 PDE；Körner 交换在近静时几乎停住。光学级水平面靠 LBM  alone 很难达到——这是引入正则化的工程动机，不是把它升格成界面物理。

---

## 2. 要解决的现象

稳定 dam-break（典型 `τ=0.51`、`γ=1.5e-3`、D3Q27、`home_fp32`）冲击过后常见：

- 主池面仍有大片左右高低（`hA ≠ hB`）或亚格子 φ 起伏  
- 空中断联液滴 / 碎 IF（高度图 `h_p2p*` 可到十余格）  
- 贴壁 / 角上灰尘 IF（`φ≈0`）在连续高度里读成「塌一格」  
- 众数平面下一层的台阶凹陷（`k = mode_k − 1`）

**刻意不做的：** 例程里一次性 host wipe、发明质量 topup、整列体积重写造空中水（早期失败路径）。

---

## 3. 当前算法（每调用一次）

入口：`apply_vof_height_equation(buf, …)`（host 抽 `cell/φ/mass/ρ/u`，写回 GPU）。

```text
池面 IF 集合
  = 每柱第一个「IF 且正下方为液体（或底）」的自由面格
  （跳过气隙上的爬升膜）

mode_k = 池面 IF 的 k 众数（主导平面）

① 空中液滴落回（可选）
   湿格且正下方为气 → 质量并入本柱池面 IF；不造悬空液

② 低一层台阶愈合
   池面 IF 在 mode_k−1，上方为气或薄 IF：
   从同平面邻格借质量 → 本格升为液体，上格为 IF
   优先处理水平贴壁接触多的柱（域边界或 solid_phi<0）
   —— 不写死某个角的 (i,j)

③ 薄 / 灰尘 IF 补满
   众数平面上 φ < φ_lo 的 IF（含 finder 因 dust 漏掉的）：
   向邻格借到安全带下沿

④ 平面 φ → φ*
   仅 mode_k 上池面 IF：
   Δφ = α_eff (φ* − φ)，|Δφ|≤dh_cap
   φ* = 平面质量加权均值，并夹在安全带内
   贴壁且低于 φ* 时 α 加大
   更新后 Σmass 在平面上重标定（避免截断丢质量）

⑤ 弱尖端速度衰减（可选，默认很弱）
```

### 安全带

\[
\phi \in [\phi_{\mathrm{lo}},\phi_{\mathrm{hi}}] \approx [0.18,\,0.82]
\]

避免算子把 φ 推到填/空阈值，减少「抹平 → 随机坑 → 圆形修复波」的循环。

### 质量

- 设计目标：调用内 \(\Delta m \approx 0\)（浮点误差量级）  
- 曾踩坑：落回后截断 φ、或平面 clip 后不重标定 → 一次丢几十质量；已修  

---

## 4. 开关与参数

| `LbmModel` 字段 | 默认 | 含义 |
|-----------------|------|------|
| `vof_height_eq` | `False` | 总开关 |
| `vof_height_eq_rate` | `0.05` | \(\alpha\)：\(\Delta\phi=\alpha(\phi^*-\phi)\) |
| `vof_height_eq_dh_cap` | `0.05` | 每格每调用 \(\lvert\Delta\phi\rvert\) 上限 |
| `vof_height_eq_u_max` | `0.05` | 池面 IF 平均 \(\lvert u\rvert\) 过大则跳过 φ 步（仍可 drop/heal） |
| `vof_height_eq_every` | `12` | 每 N 个格子步调用一次 |
| `vof_height_eq_sweeps` | `1` | 保留位（平面平移已关） |

算例：`--height-eq` 在 `t≥8` 后置 `vof_height_eq=True`。

```bash
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof \
  --backend home --n 48 --height-eq
```

日志字段（示意）：`H*` / `φ*` / `φσ` / `drop` / `heal` / `boost` / `Δm_heq`。

---

## 5. 走过的弯路（便于回滚认知）

| 路径 | 现象 | 结论 |
|------|------|------|
| 列高 \(h\) 整列 project / 改 L↔I↔G | 凭空生水下落、质量漂 | **废弃** |
| 全局 φ→φ* 过猛 + φ 贴 0/1 | 整体变平，但随机凹陷 + 固定点圆形低块反复 | 保留平面 φ*，加安全带与弱步进 |
| 仅邻域 Laplacian + 强 u 阻尼 | 空中球没了，但左右斜面冻住、「不平了」 | Laplacian 不作主手段；阻尼保持很弱 |
| 只动众数平面、不管 dust IF | 角上「塌一格」残存 | 补 thin-IF boost + 低台阶 heal |

与旧管线 `vof_quiet_fill` / `level_high_to_low` / topup（见 [quiet_level 总结](lbm_home_vof_quiet_level_summary_zh.md)）并列存在；**不要混称**。`height_eq` 强调：IF 上 φ、尽量不发明质量、可开关。

---

## 6. 适用域与非适用域

### 相对适用

- 重力朝 \(-\hat z\)、矩形池、**单一主导自由面**  
- 冲击已过后、希望主池面外观平坦的演示 / 交互  
- 规则不绑定固定角标；贴壁用「水平 wall contact 计数」泛化  

### 不适用或需隔离

| 场景 | 原因 |
|------|------|
| 忠实动力学验收 / `--home-faithful` | 正则化改变界面，非 Home 对照 |
| 强破碎、多坨游离液、射流 | `mode_k` 先验失败；drop/heal 误伤 |
| 封闭气泡 / H7 夹气 | 落回与抬台阶可能吞结构 |
| **双刚体球等 FSI** | 球周局部自由面、动壁、薄膜不是「该平的池面」；默认应关。若弱开：须排除 solid 邻域、只动远离固体的池面 IF，且更大 `every` / 更严 `u_max` |

**结论：** 对「溃坝后安静主液面」可复用；对「任意 VOF + FSI」**不通用**。流固耦合主物理仍走 `solid_phi` + HOME-FREE；本模块不是耦合的一部分。

---

## 7. 与求解器的关系（推荐表述）

文档与代码注释建议统一写成：

> Opt-in **pool free-surface regularizer** for HOME-FREE sharp FS (quiet / late-pool).  
> Not part of the core VOF update; disable for faithful / multiphase / violent FS / FSI-near-body runs.

若未来要物理内嵌抹平，方向是高度函数力、CSF、wetting 等，而不是继续加强投影启发式。

---

## 8. 文件索引

| 路径 | 内容 |
|------|------|
| `.../home_fp32_ref/height_eq.py` | 算法实现 |
| `.../home_fp32_ref/bridge.py` | `step` 内 every-N 调用、`apply_height_equation` |
| `.../lbm/model.py` | `vof_height_eq*` 字段 |
| `wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof.py` | `--height-eq`、arm、`t≥8` |

---

## 9. 当前观感验收（n=48，`--height-eq`）

经验性目标（非严格 CI）：

- `t≥8` 后主液面大片高低消失；`Δm_heq≈0`  
- 上空断联液滴明显减少（`drop` 先跳后近 0）  
- 角上灰尘塌陷可被 heal/boost 拉回众数平面附近  
- 不出现整列 project 式「天上掉水」  

细节仍随参数与网格变化；FSI 双球算例请保持本开关关闭，直至有固体邻域隔离策略。
