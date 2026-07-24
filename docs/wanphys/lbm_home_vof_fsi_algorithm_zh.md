# HOME-FREE VOF 流固耦合算法说明

> 文档版本：2026-07-24  
> 适用范围：`lbm_backend='home_fp32'` + `phase_mode=vof_sharp` + 刚体栅格 FSI  
> 主算例：`wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py`  
> 相关：[刚体耦合导览（分布型路径）](lbm_rigid_coupling_guide_zh.md)、[LATEST 进度](LATEST_home_vof_two_spheres_progress_zh.md)、[模块文件](lbm_module_files_zh.md)

本文描述**当前双球主线**的流固算法：矩编码 HOME-FREE 自由面流体如何与 Newton/WanPhys 刚体互相作用。  
旧版分布函数 `f` + bounce-back 路径见 [lbm_rigid_coupling_guide_zh.md](lbm_rigid_coupling_guide_zh.md)；本页以 **无整场 `f`、用 `solid_phi` + 动壁拉流** 为准。

---

## 1. 一句话架构

**每子步：刚体栅格化 → HOME-FREE 把固体当动壁（拉流）并清空固体内流体 → 宏观相对法向通量近似流体力 + 经验浮力/拖曳 → XPBD 推进刚体。**

严格 Ladd 动量交换需要分布函数，在 `home_fp32` 上**不可用**（自动回退到宏观近似）。

```text
┌─────────────── 一帧 (1/60 s，默认 12 子步) ───────────────┐
│  for substep:                                              │
│    ① GridLbmRigidCoupling.step                             │
│         raster SDF → solid_phi / body_id                   │
│         embed MAC wall vel → vel_solid_*                   │
│         LbmDomain.step → HomeFp32VofBridge (流体+固壁)     │
│         clear body_f + approx feedback → body_f            │
│    ② 经验浮力 / 流体追赶 / 水中拖曳 → body_f (atomic add)  │
│    ③ RigidDomain.step → collide + XPBD                     │
│  （可选）t≥8 后每 N 步 height-eq 找平 IF φ                  │
└────────────────────────────────────────────────────────────┘
```

---

## 2. 物理与离散表示

### 2.1 流体（HOME-FREE VOF）

| 量 | 含义 |
|----|------|
| \((\rho, \mathbf{u}, S)\) | Hermite 矩；无整场分布 `f`，需要时现场重构 |
| \(\varphi\) | 体积分数；液体 \(\varphi=1\)，界面 \(0<\varphi<1\) |
| `cell_type` | GAS / INTERFACE / LIQUID（及短暂 IF/IG/GI） |
| `mass` | 液体库存：液格 \(\approx\rho\)，界面 \(=\varphi\rho\) |

格子：默认 **D3Q27**。碰撞为 HOME 矩松弛；自由面用 Körner 质量交换 + FS 边界。

### 2.2 固体（刚体）

| 量 | 含义 |
|----|------|
| `body_q` / `body_qd` | 位姿与速度（Newton `RigidState`） |
| `solid_phi` | 单元中心 SDF；**`< 0` = 固体内部** |
| `solid_body_id` | 该格归属的刚体索引 |
| `vel_solid_u/v/w` | MAC 面上的壁面速度（格子单位） |
| `solid_ux/uy/uz` | HOME 缓冲里的**单元中心**壁速（由 MAC 平均） |

双球算例：半径 \(R=0.08\)，密度不同的重球/轻球；刚体重力 \(g_z^{\mathrm{rigid}}=-1\)，LBM 体力 \(g_z\) 约 \(-0.002\times(48/N)\)（**两套单位刻意分开**）。

### 2.3 耦合层级（概念）

| 层级 | 本路径状态 |
|------|------------|
| 一向：刚体 → 流体几何 | ✅ SDF + 壁速 |
| 动壁：固体拖动流体 | ✅ 拉流时用 \(\mathbf{u}_{\mathrm{wall}}\) |
| 双向：流体 → 刚体力 | ⚠️ 宏观近似 + **经验**浮力/拖曳（非严格 ME） |

---

## 3. 子步时间序（权威）

算例 `_step_coupled`：

```python
self.coupling.step(self.sim_dt)           # ① 栅格化 + 流体 + 近似反馈
self._apply_empirical_buoyancy_and_drag() # ② 浮力/推/拖
self.rigid_domain.step(self.sim_dt)       # ③ 碰撞 + XPBD
```

配置要点（双球）：

- `set_rigid_dynamics_enabled(False)` — **不在** coupling 内推进刚体  
- `set_two_way_feedback_enabled(True, force_scale=6)`  
- `set_feedback_mode("approx")` — `home_fp32` 下即使选 ME 也会回退  
- `vof_home_wall_eq=True` — 固壁用 \(f^{\mathrm{eq}}(\rho,\mathbf{u}_w)\)（见 §5）

重力斜坡：`GRAVITY_RAMP_STEPS=40` 内同步抬高 LBM \(g_z\) 与刚体 \(g_z\)，避免冷启动冲击。

---

## 4. 刚体 → 流体：几何与壁速

### 4.1 SDF 栅格化

**核：** `rasterize_all_body_sdf_warp`（`coupling_kernels.py`）

- 单元中心 \(\mathbf{p}=(i+\tfrac12,j+\tfrac12,k+\tfrac12)\,dh\)
- 球：\(\mathrm{sdf}=\|\mathbf{p}-\mathbf{c}\|-R\)
- 多体取 **min SDF**；写入 `solid_body_id`
- 空流体默认 `solid_phi ≈ 1000`（HOME alloc 填远场）

### 4.2 MAC 壁面速度

**核：** `embed_all_solid_velocity_{u,v,w}`

- 仅对「至少一侧为固体」的面写速度，否则 0  
- 世界系表面速度：

\[
\mathbf{v}_{\mathrm{surf}} = \mathbf{v}_{\mathrm{lin}} + \boldsymbol{\omega}\times(\mathbf{x}_{\mathrm{face}}-\mathbf{x}_{\mathrm{com}})
\]

- 写入格子单位：

\[
\mathbf{u}_{\mathrm{wall}}^{\mathrm{lat}} = \mathbf{v}_{\mathrm{surf}}\cdot\frac{\mathrm{dt}}{dh}
\]

（`velocity_scale = dt/dh`）

### 4.3 同步进 HOME 缓冲

`HomeFp32VofBridge` → `sync_solids_from_lbm_state`：

1. 拷贝 `LbmState.solid_phi` → `buf.solid_phi`  
2. `home_vof_solid_mac_to_cell_kernel`：固体格上对相邻 MAC 面做平均 → `solid_ux/uy/uz`

### 4.4 固体内清空（TYPE_S）

fused 内核：若 `solid_phi < 0`，该格**不参与**质量交换 / 流体重构。  
步进末尾 `home_vof_apply_solid_mask_kernel`：固体格 \(\rho,u,S,\mathrm{mass},\varphi\) 清零，`cell_type=GAS`。

含义：球**内部不存水**；浮力采样必须在壳外（约 \(1.25R\) 等偏移）。

---

## 5. 固体 → 流体：动壁边界（无 bounce-back）

本后端没有经典「撞墙反弹 `f`」。在 **pull-stream** 中，若邻居落在固体（或域墙）：

### 5.1 `vof_home_wall_eq=True`（双球默认）

用流体侧密度 \(\rho_c\) 与壁速 \(\mathbf{u}^p\) 写平衡分布：

\[
f_i^{\mathrm{wall}} = f_i^{\mathrm{eq}}\bigl(w_i,\rho_c,\mathbf{u}_{\mathrm{wall}},\mathbf{c}_i\bigr)
\]

实现：`_feq_w`（`vof_warp.py`）。偏 Home-FSLBM 风格的动壁近似。

### 5.2 `vof_home_wall_eq=False`（HOME Eq. 24）

保留非平衡应力，只换速度：

\[
S^p_{\alpha\beta} = u^p_\alpha u^p_\beta + \bigl(S^x_{\alpha\beta}-u^x_\alpha u^x_\beta\bigr),\quad
\rho^p=\rho^x,\ \mathbf{u}^p=\mathbf{u}_{\mathrm{wall}}
\]

再按 HOME 重构 \(f_i\)（`_solid_f_eq24` / `solid_moments_eq24`）。

### 5.3 与 VOF 质量交换

固墙链**不进入** Körner 质量交换分支；域墙同理。液体质量只在流体–流体 / 流体–界面邻居间交换。

---

## 6. 流体 → 刚体：力与力矩

力写在 `RigidState.body_f`（`spatial_vector`：力 + 力矩）。顺序：

1. coupling 内 `clear_forces`  
2. **近似边界反馈**（§6.1）  
3. 算例侧 **经验浮力/推/拖** atomic add（§6.2）  
4. 刚体模型重力等由 `RigidDomain.step` 一并处理  

### 6.1 宏观相对法向通量（`feedback_mode="approx"`）

**核：** `accumulate_lbm_boundary_feedback_all_bodies` → `_accumulate_lbm_face_feedback`

对每个与固体相邻的流体单元面，外法向 \(\mathbf{n}\)（流体 → 固体）：

\[
u_{\mathrm{rel},n} = \mathbf{u}_f\cdot\mathbf{n} - u_{\mathrm{wall},n}
\]

仅当 \(u_{\mathrm{rel},n}>0\)（流体压向壁）贡献：

\[
\mathbf{F} = s\cdot\rho\, u_{\mathrm{rel},n}\, A\,\mathbf{n},\quad
A=dh^2,\quad
s=\texttt{feedback\_force\_scale}
\]

\[
\boldsymbol{\tau} = (\mathbf{x}_{\mathrm{face}}-\mathbf{x}_{\mathrm{com}})\times\mathbf{F}
\]

双球默认 \(s=6\)。这是**格子单位 × 经验缩放**的早期双向近似，不是压力面积分，也不是 Ladd ME。

### 6.2 严格动量交换（本路径不可用）

`accumulate_lbm_momentum_exchange_all_bodies` 依赖 live 分布 \(f\)。  
`home_fp32` 无整场 `f` → coupling **强制回退 approx** 并告警一次。

### 6.3 经验浮力 / 流体追赶 / 拖曳

**文件：** `sphere_buoyancy_warp.py`；算例 `_apply_empirical_buoyancy_and_drag`。

**浸没分数 \(s_b\)**

- 在球心周围固定偏移点采样（`BUOYANCY_SAMPLE_OFFSETS`）  
- 跳过 `solid_phi < 0`；有效点：`cell ≠ GAS` 且 \(\varphi > 0.25\) 计为湿  
- \(s_{\mathrm{raw}}=N_{\mathrm{wet}}/N_{\mathrm{valid}}\)  
- EMA + 变化率帽：\(\alpha=0.05\)，`dsub_cap=0.015`（抑 chatter）

**力（\(m=\rho_{\mathrm{sphere}}V\)，\(V=\frac43\pi R^3\)）**

\[
F_z^{\mathrm{buoy}} = s_b\,\rho_L\,V\,|g_{\mathrm{rigid}}|\,s_{\mathrm{buoy}}
\]

\[
\mathrm{push}=k_{\mathrm{push}}\,m\,s_b,\quad
\mathbf{u}_f^{\mathrm{world}}=\langle\mathbf{u}\rangle_{\mathrm{wet}}\cdot\frac{dh}{\mathrm{dt}}
\]

\[
\begin{aligned}
F_x &= \mathrm{push}\,(u_{f,x}-v_x) - k_{xy}\,m\,s_b\,v_x \\
F_y &= \mathrm{push}\,(u_{f,y}-v_y) - k_{xy}\,m\,s_b\,v_y \\
F_z &= F_z^{\mathrm{buoy}} + 0.35\,\mathrm{push}\,(u_{f,z}-v_z) - k_z\,m\,s_b\,v_z
\end{aligned}
\]

默认常数（算例）：

| 符号 | 值 | 作用 |
|------|-----|------|
| `BUOYANCY_FORCE_SCALE` | 1 | 浮力缩放 |
| `FLUID_PUSH_RATE` | 8 | 追赶流体速度 |
| `WATER_HORIZONTAL_DRAG_RATE` | 4 | 水平线性拖曳 |
| `WATER_VERTICAL_DRAG_RATE` | 12 | 竖直拖曳（更强） |
| `LATE_POOL_PUSH_SCALE` | 0.12 | height-eq 武装后压低 push |

**不是**阿基米德 SDF 体积积分；是壳采样启发式，用于稳住球–液观感。

---

## 7. 流体步进内部（与 FSI 相关的片段）

`HomeFp32VofBridge.step` → `step_home_vof_gpu` 概要：

1. 同步固体场（§4.3）  
2. （可选）PLIC \(\kappa\)、气泡准备  
3. **`home_vof_fused_kernel`**：pull + 质量交换 + FS + collide；固体格早退；邻居固体走 §5  
4. surface1/2/3 + 可选 seal  
5. **`home_vof_apply_solid_mask_kernel`**  
6. （可选）矩量化 re-pack  
7. （可选）`apply_vof_height_equation`（§8）  
8. `sync_to_state`；固体场 `state_in → state_out` 拷贝  

---

## 8. 与液面正则化（height-eq）的交叉

`--height-eq` **不是** FSI 物理，而是晚期池面 IF 的 \(\varphi\to\varphi^*\) 正则化。

与刚体相关的设计：

| 点 | 行为 |
|----|------|
| 武装时间 | \(t\ge 8\) |
| 固体足迹 | 足迹上 \(w=0\)；邻域 soft fade，地板 **`w_min=0.35`**（避免球边永久坑） |
| push | 武装后 `FLUID_PUSH × 0.12`，减轻弯月面被「追赶力」挖坏 |

详见 [lbm_home_vof_height_eq_summary_zh.md](lbm_home_vof_height_eq_summary_zh.md)。

---

## 9. 数据流图

```text
RigidState (q, qd)
        │
        ▼
 rasterize_all_body_sdf_warp ──► solid_phi, solid_body_id
 embed_all_solid_velocity_* ───► vel_solid_u/v/w
        │
        ▼
 HomeFp32VofBridge
   sync_solids + MAC→cell
   fused HOME-FREE VOF (动壁拉流)
   solid_mask → GAS
        │
        ▼
 LbmState (ρ,u,φ,…) + solid_*
        │
        ├─► approx face feedback ──┐
        │                          ├─► body_f
        └─► empirical buoyancy ────┘
                    │
                    ▼
            RigidDomain.step (collide + XPBD)
                    │
                    ▼
              更新 body_q / body_qd
```

---

## 10. 关键代码入口

| 主题 | 路径 | 符号 |
|------|------|------|
| 子步编排 | `examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py` | `_step_coupled`, `_apply_empirical_buoyancy_and_drag` |
| 耦合器 | `fluid_grid/coupling/grid_lbm_rigid_coupling.py` | `GridLbmRigidCoupling.step` |
| SDF / MAC / 反馈 | `fluid_grid/coupling/coupling_kernels.py` | `rasterize_all_body_sdf_warp`, `embed_all_solid_velocity_*`, `accumulate_lbm_boundary_feedback_all_bodies` |
| 桥接 | `home_fp32_ref/bridge.py` | `HomeFp32VofBridge.step` |
| fused + 固壁 | `home_fp32_ref/vof_warp.py` | `home_vof_fused_kernel`, `_feq_w`, `_solid_f_eq24`, `home_vof_apply_solid_mask_kernel` |
| Eq.24 | `home_fp32_ref/bc.py` | `solid_moments_eq24` |
| 浮力 | `home_fp32_ref/sphere_buoyancy_warp.py` | `apply_sphere_buoyancy_forces_gpu` |
| 模型开关 | `lbm/model.py` | `vof_home_wall_eq`, `vof_height_eq*` |

---

## 11. 已知近似与局限

1. **无严格 Ladd ME**（无整场 `f`）。  
2. **Approx 反馈单侧且粗糙**：只计 \(u_{\mathrm{rel},n}>0\)；经验 `force_scale`。  
3. **浮力/推/拖为壳启发式**，非压力或体积阿基米德。  
4. **刚体重力与 LBM 体力单位刻意不同**；push 用 \(dh/\mathrm{dt}\) 把格子速度拉回世界系。  
5. **固体为二元 SDF**，无部分体积固体；固体推进时内部液体被 mask 掉；two-way 开启时「揭盖修复」关闭。  
6. 默认 `vof_home_wall_eq=True` 用 \(f^{\mathrm{eq}}\)，**不是**完整 Eq.24 应力保留。  
7. height-eq / 经验 push 会互相打架，需靠 `LATE_POOL_PUSH_SCALE` 与 `w_min` 折中。  
8. 论文式分裂固体校正核 / 8³ tile / 纯流体 50% 量化 **不在**本 VOF 主线。

---

## 12. 和旧文档的关系

| 文档 | 讲什么 |
|------|--------|
| 本文 | **HOME-FREE 矩路径** 当前双球 FSI 算法 |
| [lbm_rigid_coupling_guide_zh.md](lbm_rigid_coupling_guide_zh.md) | 分布型 BB、MAC、早期 two-way 导览（仍有效，但是另一后端） |
| [lbm_rigid_coupling_experiments_zh.md](lbm_rigid_coupling_experiments_zh.md) | 观察实验手册 |
| [LATEST_…](LATEST_home_vof_two_spheres_progress_zh.md) | 工程调参与性能快照 |

---

## 13. 试跑

```bash
cd LBM-Fluid-Structure-Interaction
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 48 --height-eq
# 可选：--moment-quant
```

守恒回归（流体侧质量，不含刚体动量严格守恒）：

```bash
uv run --extra examples python -m unittest newton.tests.test_lbm_home_vof_conservation -v
```
