# HOME-FREE VOF：近静置液面找平试验总结

> 文档版本：2026-07-14  
> 相关极限说明：[单格找平极限](lbm_home_fslbm_one_cell_limit_zh.md)  
> 实现：`wanphys/_src/fluid/fluid_grid/lbm/backends/moment/home_fp32_ref/vof_warp.py`、`bridge.py`  
> 算例：`wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof.py --backend home --n 64`

本文记录溃坝 VOF 算例在**冲击结束后、宏观速度已很小**时，为把两侧液面高度（`hA/hB`）拉齐所做过的手段、取舍与当前默认策略。  
**大尺度冲击 / 晃荡**仍由 FSLBM 守恒步进负责；本文管线只作用于近静置的**后期外观整平**。

---

## 1. 问题是什么

稳定参数下（典型 `τ=0.51`、`γ=1.5e-3`、`g∝1/n`、no-slip、D3Q27），64³ 溃坝到 `t≳10` 时常出现：

| 观测量 | 典型值（找平前） |
|--------|------------------|
| `|u|` | ~0.009（已近静） |
| `hA` / `hB` | 约 7.6 / 6.0（差 ~1–2 格） |
| 高度图 `z_p2p` | 仍可有大 outlier；有效高低台差约 1 格 |
| `Σmass` 相对 `vol0` | `Δ ≈ −70 ~ −80`（缓慢平台） |

物理根因见[单格找平极限](lbm_home_fslbm_one_cell_limit_zh.md)：单相自由面在 **O(1) 格 Δh** 上静水驱动不足，接触线钉扎，单纯再扫几个 LBM 步无法保证光学级水平面。

---

## 2. 试过什么（按效果）

| 手段 | 做法摘要 | 结果 | 结论 |
|------|-----------|------|------|
| 强墙润湿 `κ` 偏置 | `vof_wall_wetting` 疏水 / 近角放大 | 液面离墙内缩 → **四棱台** | **默认关（0）** |
| 贴壁薄膜排液 | `vof_wall_film_drain` + `atomic` 重分配 | 连通弯液面无效；易漏质量 | **默认关** |
| 从下方“填洼” | 从池底抽质量垫高 | 底部空洞、剧烈振荡 | 废弃 |
| 全局高→低（旧：全局 max/min） | 只对邻柱 / 全场极值 | 细网格墙角爬升钉死 `z_p2p`，找平不停 → 落水感 + 质量连降 | **已改为主池带 + 爬升另剥** |
| 全局高→低 `level_high_to_low`（现） | 中位数±`climb_margin` 内找平；更高爬升柱单独剥到池面 | 大差可搬；1 格后停 | **保留为第一阶段** |
| 削高 `shave` | 高柱直接删到低平台 | `hA≈hB≈6`，但一次 **删 ~2600** 质量 | **废弃**（减太多） |
| 按 `−Δ` 预算补低 | 发明质量 ≤ `vol0−mass` | `Δ→0`，但低柱仍可能差 1 格（预算不够） | 不够齐 |
| **无预算补低（当前）** | 发明足够质量抬到高侧目标 | `hA=hB=7`，`invented~10³`，`Δ` 可变成正 | **采用** |

原则：

- **守恒找平**能搬则搬；不要再用删界面 / 削高把上千质量单位扔掉。
- **发明质量**可以，但要明确这是**演示用外观修正**，不是物理冷凝，也不进守恒验收。

---

## 3. 当前默认管线

算例里 `lvl` 标志：`-` 未启用 / `L` 守恒找平中 / `T` 已做 topup。

```text
t ≥ HEIGHT_MAP_START_T 且 |u| < vof_quiet_fill_u_max
        │
        ▼
  [level ON]  vof_quiet_fill = True
        │     每帧 level_high_to_low
        │     · 主池：median 附近 band 内高→低（dz_min=2）
        │     · 爬升：z > median+climb_margin 单独剥到池面
        │     · 新建气→界面格写零速矩，减轻下一步蒸发
        ▼
  高度图 robust_p2p = p95−p05 ≤ 1
  （或找平超时 ~300 frame）
        │     ※ 不用全场 z_p2p：爬升 outlier 会让它永不收敛（n=96 可见）
        ▼
  [level OFF]  停守恒找平
        │
        ▼
  [topup]  一次性 topup_with_budget(inf)
        │  · 目标高度 ≈ round(max(中位数, max(hA,hB)))
        │  · 低柱向上长满（可超过 vol0）
        ▼
  lvl=T ，此后不再找平 / 再 topup
```

### 关键 API

| 函数 / 方法 | 位置 | 作用 |
|-------------|------|------|
| `level_surface_high_to_low` | `vof_warp.py` | 守恒：剥高填低 |
| `topup_surface_with_budget` | `vof_warp.py` | 发明 ≤ `budget`（`inf` → 实际用大 cap） |
| `HomeFp32VofBridge.level_high_to_low` | `bridge.py` | 对外封装 |
| `HomeFp32VofBridge.topup_with_budget` | `bridge.py` | 对外封装；`budget=inf` 表示不截断 |

模型侧：`vof_quiet_fill`、`vof_quiet_fill_rate`、`vof_quiet_fill_u_max`。  
润湿 / 薄膜：`vof_wall_wetting=0`、`vof_wall_film_drain=False`。

---

## 4. 64³ 实测备忘（2026-07-14）

无预算 topup 一次跑典型日志：

```text
[level ON]  t=10.0s |u|≈0.009
[level OFF] t=10.5s z_p2p=1.00
[topup] invented≈1021 (unbounded)  mass≈32190  (Δ→+944)
此后 hA≈7.0  hB≈7.0  corn≈7  lvl=T
```

要点：

- 守恒阶段把大体高度差压到 **1 格**；
- 无预算补齐把两侧拉到同一整数层（此处为 7）；
- 代价是 **`Σmass` 相对初始 `vol0` 可到 +O(10³)**——日志里应看成“演示库存”，不要当守恒指标通过。

若只按 `budget=−Δ`（约 77）补，往往抬不满低侧，`hA/hB` 仍会差约 0.5–1 格。

若改削高，两侧也能齐到 6，但一次扔掉约 2660 质量，观感与库存都差，已否决。

---

## 5. 验收建议（分层）

| 层级 | 看什么 | 期望 |
|------|--------|------|
| **动力学 / 守恒**（topup **前**） | 冲击后 `Δ` 平台、无持续线性泄漏 | 允许 O(10²) 的早期漂移平台；忌启发式硬删界面 |
| **晃荡** | `hA/hB` 振荡后靠近 | 趋势正确即可 |
| **近静置物理极限** | 找平停在 `z_p2p≤1` | 记为 FSLBM 容差，不必逼 0 |
| **演示外观**（topup **后**） | `hA≈hB`、视觉齐平 | 允许 `Δ>0`；文档/报告须标明非守恒补齐 |

---

## 6. 一句话

> 近静置一格台阶是 FSLBM 的方法边界；工程上先做**守恒全局高→低**，卡住后再做**无预算发明补齐**把两侧拉齐。削高能齐但毁库存，按 `−Δ` 补往往不够齐——当前默认取后者的“超预算”版本，并把它标成外观修正而非物理守恒结果。
