# HOME-FREE VOF 流固耦合算法说明

> 文档版本：2026-07-28  
> 适用范围：`lbm_backend='home_fp32'` + `phase_mode=vof_sharp` + 刚体栅格 FSI  
> 主算例：`wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py`  
> 相关：[刚体耦合导览（分布型路径）](lbm_rigid_coupling_guide_zh.md)、[LATEST 进度](LATEST_home_vof_two_spheres_progress_zh.md)、[模块文件](lbm_module_files_zh.md)  
> 对照参考：同级目录外的 `OpenHOMELBM/`（Li et al. HOME 官方开源；**GPL-3.0**，只对照算法不拷贝代码）

本文描述**当前双球主线**的流固算法：矩编码 HOME-FREE 自由面流体如何与 Newton/WanPhys 刚体互相作用。  
旧版分布函数 `f` + bounce-back 路径见 [lbm_rigid_coupling_guide_zh.md](lbm_rigid_coupling_guide_zh.md)；本页以 **无整场 `f`、用 `solid_phi` + 动壁拉流** 为准。

---

## 1. 一句话架构

**默认每子步：刚体栅格化 → HOME Eq.24 动壁拉流并清空固体内流体 → 矩重构链路 ME → XPBD 推进刚体。**

经验浮力 / push / 拖曳 **仅** `--showcase-fsi`（双球）或 `--empirical-fsi`（单球）开启，不进 coupling 默认路径。  
宏观 `approx` 面通量保留为 **legacy / 诊断**。默认漂浮幅度弱于旧 showcase，属预期。

```text
┌─────────────── 一帧 (1/60 s，默认 12 子步) ───────────────┐
│  for substep:                                              │
│    ① GridLbmRigidCoupling.step                             │
│         raster SDF → solid_phi / body_id                   │
│         embed MAC wall vel → vel_solid_*                   │
│         LbmDomain.step → HomeFp32Bridge (流体+Eq.24固壁) │
│         clear body_f + reconstructed-link ME → body_f      │
│    ② （仅 showcase）经验浮力 / 追赶 / 拖曳 → body_f        │
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
| 动壁：固体拖动流体 | ✅ 默认 Eq.24；showcase 可开 `vof_home_wall_eq` |
| 双向：流体 → 刚体力 | ✅ 默认 **重构链路 ME**；`approx` = legacy；经验插件 = showcase only |

---

## 3. 子步时间序（权威）

算例 `_step_coupled`：

```python
self.coupling.step(self.sim_dt)           # ① 栅格化 + 流体 + ME
# only if --showcase-fsi / --empirical-fsi:
self._apply_empirical_buoyancy_and_drag() # ② 浮力/推/拖（可选）
self.rigid_domain.step(self.sim_dt)       # ③ 碰撞 + XPBD
```

配置要点（双球**默认**严谨路径）：

- `set_rigid_dynamics_enabled(False)` — **不在** coupling 内推进刚体  
- `set_two_way_feedback_enabled(True, force_scale=recommended_me_force_scale(dh, dt))`  
- `set_feedback_mode("momentum_exchange")` — `home_fp32` 用矩重构链路 ME  
- `vof_home_wall_eq=False` — HOME Eq.24（showcase 可开 eq-wall）  
- 经验插件默认 **关**；`--showcase-fsi` 打开

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

`HomeFp32Bridge` → `sync_solids_from_lbm_state`：

1. 拷贝 `LbmState.solid_phi` → `buf.solid_phi`  
2. `home_vof_solid_mac_to_cell_kernel`：固体格上对相邻 MAC 面做平均 → `solid_ux/uy/uz`

### 4.4 固体内清空（TYPE_S）

fused 内核：若 `solid_phi < 0`，该格**不参与**质量交换 / 流体重构。  
步进末尾 `home_vof_apply_solid_mask_kernel`：固体格 \(\rho,u,S,\mathrm{mass},\varphi\) 清零，`cell_type=GAS`。

含义：球**内部不存水**；浮力采样必须在壳外（约 \(1.25R\) 等偏移）。

---

## 5. 固体 → 流体：动壁边界（无 bounce-back）

本后端没有经典「撞墙反弹 `f`」。在 **pull-stream** 中，若邻居落在固体（或域墙）：

### 5.1 `vof_home_wall_eq=True`（showcase 偏好）

用流体侧密度 \(\rho_c\) 与壁速 \(\mathbf{u}^p\) 写平衡分布：

\[
f_i^{\mathrm{wall}} = f_i^{\mathrm{eq}}\bigl(w_i,\rho_c,\mathbf{u}_{\mathrm{wall}},\mathbf{c}_i\bigr)
\]

实现：`_feq_w`（`vof_warp.py`）。偏 Home-FSLBM 风格的动壁近似；双球/单球仅在 `--showcase-fsi` / `--empirical-fsi` 时默认打开。

### 5.2 `vof_home_wall_eq=False`（研究默认：HOME Eq. 24）

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
2. **默认：重构链路 ME**（§6.1）  
3. （仅 showcase）算例侧 **经验浮力/推/拖** atomic add（§6.3）  
4. 刚体模型重力等由 `RigidDomain.step` 一并处理  

### 6.1 动量交换（核心，`feedback_mode="momentum_exchange"`）

- **分布路径**（`lbm_backend=dist`）：`accumulate_lbm_momentum_exchange_all_bodies` 用 live \(f\)。  
- **`home_fp32`**：无整场 `f`；走 `link_me_warp`——用与 fused 固壁相同的矩重构 \(f_{\mathrm{wall}}\)、\(f_{\mathrm{opp}}\)，再按 Ladd 链力累加到 `body_f`。  
- `force_scale` 起点：`recommended_me_force_scale(dh, dt) = (dh/\mathrm{dt})^2`（核内已乘 \(dh^3\)）。

本轮**不**把 ME 原子累加并进 `home_vof_fused_kernel`；post-step reconstructed ME 即为耦合核心。

### 6.2 宏观相对法向通量（legacy / 诊断，`feedback_mode="approx"`）

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

旧 showcase 曾用 \(s=6\)。这是**格子单位 × 经验缩放**的早期双向近似，**不是**物理主路径。

### 6.3 经验浮力 / 流体追赶 / 拖曳（showcase-only）

**文件：** `sphere_buoyancy_warp.py`；算例 `_apply_empirical_buoyancy_and_drag`（仅 `--showcase-fsi` / `--empirical-fsi`）。

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

**不是**阿基米德 SDF 体积积分；是壳采样启发式，用于稳住球–液观感。φ-体积浮力插件仍为可选实验，不接入双球默认。

---

## 7. 流体步进内部（与 FSI 相关的片段）

`HomeFp32Bridge.step` → `step_home_vof_gpu` 概要：

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
 HomeFp32Bridge
   sync_solids + MAC→cell
   fused HOME-FREE VOF (Eq.24 动壁拉流)
   solid_mask → GAS
        │
        ▼
 LbmState (ρ,u,φ,…) + solid_*
        │
        ├─► reconstructed-link ME ──┐
        │                           ├─► body_f
        └─► (showcase) empirical ───┘
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
| 桥接 | `home_fp32_ref/bridge.py` | `HomeFp32Bridge.step` |
| fused + 固壁 | `home_fp32_ref/vof_warp.py` | `home_vof_fused_kernel`, `_feq_w`, `_solid_f_eq24`, `home_vof_apply_solid_mask_kernel` |
| Eq.24 | `home_fp32_ref/bc.py` | `solid_moments_eq24` |
| 反馈 / ME | `home_fp32_ref/link_me_warp.py`；`recommended_me_force_scale` | `accumulate_home_reconstructed_link_me_kernel` |
| 浮力（showcase） | `home_fp32_ref/sphere_buoyancy_warp.py` | `apply_sphere_buoyancy_forces_gpu` |
| 模型开关 | `lbm/model.py` | `vof_home_wall_eq`, `vof_height_eq*` |

---

## 11. 已知近似与局限

1. **链路 ME 为矩重构近似**（无整场 `f`；与 fused 固壁 BC 一致）；尚未并进 fused 核。  
2. **Approx 反馈**为 legacy：单侧、粗糙；经验 `force_scale`。  
3. **浮力/推/拖为壳启发式（showcase-only）**；可选 φ 体积浮力插件仍是壳采样阿基米德，非压力积分。  
4. **刚体重力与 LBM 体力单位刻意不同**；ME 用 `recommended_me_force_scale` 作量纲起点。  
5. **固体为二元 SDF**，无部分体积固体；固体推进时内部液体被 mask 掉；two-way 开启时「揭盖修复」关闭。  
6. 研究默认 `vof_home_wall_eq=False`（Eq.24）；showcase 可开 \(f^{\mathrm{eq}}\)。  
7. height-eq / 经验 push 会互相打架，需靠 `LATE_POOL_PUSH_SCALE` 与 `w_min` 折中。  
8. 论文式分裂固体校正核 / 8³ tile / 纯流体 50% 量化 **不在**本 VOF 主线。

细节对照见下节 OpenHOMELBM。

---

## 12. 与 OpenHOMELBM 对照（算法参考）

仓库位置（本 monorepo 旁）：`DALAB-Newton/OpenHOMELBM/`  
来源：SIGGRAPH 2026 课程配套、Li et al. (2023) HOME-LBM 开源实现。  
许可：**GPL-3.0-or-later** — 可对照公式与结构，**勿**把其源码并入本 Apache 工程。

### 12.1 问题域差异（先分清）

| | 本路径（双球） | OpenHOMELBM |
|--|----------------|-------------|
| 流体 | HOME-FREE **VOF** 自由面 | 单相 HOME（无 φ / 无溃坝） |
| 固体 | 球 **SDF** `solid_phi < 0` | **三角网格** ray cut-cell |
| 刚体 | Newton XPBD | MuJoCo-Warp |
| 典型场景 | 溃坝 + 浮沉球 | 卡门涡街、鳗鱼游动（全浸没） |

因此：**不能**把他们的 mesh FSI 整套搬来当 VOF 答案；有价值的是 **动壁公式、链路力、量纲、工程习惯**。

### 12.2 流体 → 刚体力（核心对照）

**本路径（两层叠力）**

1. **Approx 面通量**（`accumulate_lbm_boundary_feedback_all_bodies`），步进**之后**扫邻固流体面：

\[
u_{\mathrm{rel},n}=\mathbf{u}_f\cdot\mathbf{n}-u_{\mathrm{wall},n},\quad
\mathbf{F}=s\,\rho\,u_{\mathrm{rel},n}\,A\,\mathbf{n}\ \ (u_{\mathrm{rel},n}>0)
\]

默认 \(s=6\)；单侧、宏观、格子单位 × 经验缩放。

2. **经验浮力 / push / 拖曳**（壳采样 \(s_b\)），再 atomic 到 `body_f`。

**OpenHOMELBM（链路 ME，与 stream 同核）**

在 cut-cell 链路上用 Eq.24 改壁侧分布后，按动量交换累加（示意）：

\[
\mathbf{F}_i
\sim
f_{\bar{\imath}}\,(\mathbf{c}_{\bar{\imath}}-\mathbf{u}_p)
-
f_i\,(\mathbf{c}_i-\mathbf{u}_p)
\]

\[
\boldsymbol{\tau}=\mathbf{r}\times\mathbf{F},\quad
\mathbf{r}=\mathbf{x}_{\mathrm{hit}}-\mathbf{x}_{\mathrm{com}}
\]

其中 \(f_i\) 为壁侧重构分布，\(f_{\bar{\imath}}\) 为反向链；\(u_p\) 优先用**帧间网格点位移**（否则 \(\mathbf{v}+\boldsymbol{\omega}\times\mathbf{r}\)）。力在 `stream_and_collide_*` 内 `atomic_add`，无「事后宏观扫面」。

| 维度 | 本路径 | OpenHOMELBM |
|------|--------|-------------|
| 时机 | 流体步**后**两段后处理 | 拉流 / 碰壁**当时** |
| 分辨率 | 单元面宏观通量 | **每条离散速度链** |
| 物理叙事 | 早期 two-way 近似 + 观感启发式 | 接近 HOME/Ladd 边界动量交换 |
| 对 VOF | 与 `solid_mask→GAS`、无整场 `f` 兼容 | 假设浸没单相 + mesh 求交 |
| 已知缺口 | 需 `force_scale`；浮力另挂 | 无自由面质量 / 弯月面问题 |

**可吸收方向（未做）：** 在 fused 已重构壁侧 \(f_i\) 的分支上，按上式累加 \(\mathbf{F}\)（仍用 `solid_phi`，不必上 mesh ray），再逐步降权 `FEEDBACK_FORCE_SCALE` / 经验 push。许可上只复述公式，不复制 GPL 内核。

### 12.3 固体 → 流体（动壁）

| | 本路径（双球默认） | OpenHOMELBM |
|--|-------------------|-------------|
| 壁侧分布 | \(f^{\mathrm{eq}}(\rho_c,\mathbf{u}_w)\)（`vof_home_wall_eq=True`） | **Eq.24**：\(S^p=u_p⊗u_p+(S-uu)\)，再重构 |
| 可选 | `vof_home_wall_eq=False` → 本仓 `_solid_f_eq24` / `solid_moments_eq24` | （默认即 Eq.24） |
| 壁速来源 | MAC `vel_solid_*` → 单元中心平均 | 网格命中点帧间 \(\Delta\mathbf{x}\) 或刚体速度 |

对照实验建议：双球关 wall_eq、开 Eq.24，看剪切与球边弯月面（不动反馈公式也能先比）。

### 12.4 量纲（标定时可参考）

OpenHOMELBM 文档给出的格子 → 物理换算：

\[
F_{\mathrm{phys}}
=
F_{\mathrm{lbm}}\,\rho_{\mathrm{fluid}}\,\frac{dx^4}{dt^2},\qquad
\tau_{\mathrm{phys}}
=
\tau_{\mathrm{lbm}}\,\rho_{\mathrm{fluid}}\,\frac{dx^5}{dt^2}
\]

本路径刻意拆开：刚体 \(|g^{\mathrm{rigid}}|=1\)，LBM \(g_z\sim\mathcal{O}(10^{-3})\)，反馈靠经验 `force_scale`。若收紧双向耦合，可用上式估计「合理量级」，再调 \(s\)，而不是盲目加大浮力系数。

### 12.5 工程习惯（可选借鉴）

| 点 | OpenHOMELBM | 对本路径含义 |
|----|-------------|--------------|
| 窄带裁剪 | 包围球 `R+2` 跳过远场 mesh ray；有 ON/OFF 等价脚本 | SDF 球可做邻域跳过；重点学「裁剪 + 回归证明」 |
| CUDA graph | `ScopedCapture` 包 stream→BC→swap | 流体核稳定后可 capture；见仓库 CUDA graph roadmap |
| 多 world | Gym 批仿真 / SAC | 当前 GP 非目标 |
| 应力 SoT | 对角常存 \(\Pi/\rho-c_s^2\) | 与本仓 Hermite「\(S\approx u⊗u\)、neq **勿再减** \(c_s^2\)」不同，对照时勿混 |

### 12.6 一句话

OpenHOMELBM = **浸没网格上的论文式 HOME 链路 FSI**；本路径默认 = **自由面 SDF 球 + 重构链路 ME + Eq.24**（经验浮力为 showcase）。对照价值在 **Eq.24、链路 \(\mathbf{F}\)、量纲与核性能**，不在 VOF/溃坝本身。

---

## 13. 和旧文档的关系

| 文档 | 讲什么 |
|------|--------|
| 本文 | **HOME-FREE 矩路径** 当前双球 FSI 算法（含 §12 对照） |
| [lbm_rigid_coupling_guide_zh.md](lbm_rigid_coupling_guide_zh.md) | 分布型 BB、MAC、早期 two-way 导览（仍有效，但是另一后端） |
| [lbm_rigid_coupling_experiments_zh.md](lbm_rigid_coupling_experiments_zh.md) | 观察实验手册 |
| [LATEST_…](LATEST_home_vof_two_spheres_progress_zh.md) | 工程调参与性能快照 |

---

## 14. P0：通用性基线（启发式默认关）

目标：双球观感常数不再冒充「默认 VOF 物理」。

| 项 | 状态 |
|----|------|
| `make_home_vof_model` / `generic_home_vof_flags` | `home_fp32_ref/generic.py` — height-eq / film / orphan / bubble / quiet **默认关** |
| `vof_orphan_reabsorb` 模型默认 | **False**（需 late-pool 显式开） |
| 经验浮力 / push / 拖 | 算例插件 `examples/lbm/_home_vof_empirical_sphere_fsi.py`；**不进** bridge / coupling；双球 `--showcase-fsi` / 单球 `--empirical-fsi` |
| 无刚体溃坝 | `fluid_grid_lbm_dambreak_vof.py --backend home` 走 generic factory |
| 单球 | `fluid_grid_lbm_dambreak_vof_single_sphere.py`（默认 ME+Eq.24；`--empirical-fsi` = showcase） |
| 双球 | 默认 ME+Eq.24；`--showcase-fsi` 打开经验插件 + eq-wall |

推荐配置：

```bash
# 通用无刚体
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof --backend home --n 48

# 单球：核心 FSI（ME，默认）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_single_sphere \
  --viewer gl --n 48

# 双球：核心 FSI（ME，默认无经验推力；漂浮弱于旧 showcase）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 48

# 双球 showcase（经验浮力插件）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 48 --showcase-fsi
```

---

## 15. P1：IC / BC / 圆柱几何 / 反馈枚举

| 项 | 路径 |
|----|------|
| IC 工厂 | `home_fp32_ref/ic.py`：`seed_pool`、`seed_droplet`、`seed_dam_break_column`、`mark_liquid_interfaces` |
| Bridge 播种 | `HomeFp32Bridge.seed_pool` / `seed_droplet` / `seed_host_state` |
| BC 预设 | `HomeDomainBC.open_top()`、已有 `channel_x()`；`apply_home_domain_bc_to_model` + `refresh_domain_bc()` |
| 圆柱→`solid_phi` | `GridLbmRigidCoupling.add_body_cylinder`（`coupling_kernels.sdf_cylinder`）；mesh 仍用 `add_body_mesh` |
| 反馈策略 | `LbmFeedbackMode`: `none` / `approx` / `momentum_exchange`（核外枚举） |

```bash
# 池 + 静圆柱（反馈 none）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_home_vof_pool_cylinder --n 32 --steps 40

uv run --extra examples python -m unittest newton.tests.test_lbm_home_vof_p1_generic -v
```

---

## 16. P2：可选物理与性能

| 项 | 路径 |
|----|------|
| 重构链路 ME | `home_fp32_ref/link_me_warp.py`；`HomeFp32Bridge.accumulate_reconstructed_link_me`；`LbmFeedbackMode.MOMENTUM_EXCHANGE` 在 `home_fp32` 上走此路径（不再回退 approx） |
| φ 体积浮力 | `phi_volume_buoyancy_warp.py` + 算例插件 `examples/lbm/_home_vof_phi_volume_buoyancy.py`（默认无 push/drag） |
| SDF 窄带 | `GridLbmRigidCoupling.set_solid_narrowband(cells)` → `rasterize_all_body_sdf_warp_narrowband` |
| CUDA graph | 模型旗标 `vof_home_cuda_graph`（默认关）；无 bubble/film/κ 时 capture fused+surface+mask；Python swap 在图外 |

```bash
uv run --extra examples python -m unittest newton.tests.test_lbm_home_vof_p2_optional -v
```

---

## 17. L0 基线：纯流体溃坝（Martin & Moyce）

优化顺序上 **先做 L0**（fused VOF 核 + 闭墙 + 重力），**不开** height-eq / orphan / 经验 FSI。验收用 Martin & Moyce (1952) **n²=2** 浪头 \(Z(T)\)。

| 项 | 做法 |
|----|------|
| 几何 | `dam_x:fill_z = 1:2`（`n²=2`） |
| 浪头检测 | 床层窄带 `bed_surge_front_cells`（忌整场 max-i 抓飞溅） |
| 时间对齐 | `align_gate`：过 `Z≈1.44` 对齐实验锚点（门释放偏移） |
| 标定报告 | `MartinMoyceScale`：`τ`、`ν`、`Re_column≈√(g H³)/ν` |
| `g` 随分辨率 | `recommended_g(n)` 保持 `g·H` 大致恒定 |

```bash
# CLI：表 + CSV
uv run --extra examples python -m wanphys.examples.lbm.run_martin_moyce_compare --n 48
uv run --extra examples python -m unittest newton.tests.test_lbm_home_vof_martin_moyce -v
```

路径：`lbm/benchmark/martin_moyce.py`、`home_vof_dambreak_front.py`。  
期望：**定性贴合**；粗网格 3D 粘性 LBM 中段 `mean_rel` 仍可 ~15–28%（实验近无粘）。

---

## 18. 试跑

```bash
cd LBM-Fluid-Structure-Interaction
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 48 --height-eq
# 可选：--moment-quant
```

守恒回归（流体侧质量，不含刚体动量严格守恒）：

```bash
uv run --extra examples python -m unittest newton.tests.test_lbm_home_vof_conservation -v
uv run --extra examples python -m unittest newton.tests.test_lbm_home_vof_sphere_momentum_energy -v
# L0：Martin & Moyce 浪头
uv run --extra examples python -m unittest newton.tests.test_lbm_home_vof_martin_moyce -v
```
