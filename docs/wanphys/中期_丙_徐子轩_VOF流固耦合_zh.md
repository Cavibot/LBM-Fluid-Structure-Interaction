# 成员丙（徐子轩）详细说明：VOF 上的流固耦合（FSI）

> 角色：中期答辩 **丙**  
> 领域：**VOF 上的流固耦合**（Eq.24、ME、单球/双球、力标度与观感诊断）  
> 对应总稿：[中期答辩_LBM自由表面流固耦合_zh.md](中期答辩_LBM自由表面流固耦合_zh.md)  
> 算法权威：[lbm_home_vof_fsi_algorithm_zh.md](lbm_home_vof_fsi_algorithm_zh.md)  
> 进度快照：[LATEST_home_vof_two_spheres_progress_zh.md](LATEST_home_vof_two_spheres_progress_zh.md)  
> 日期：2026-07-30

---

## 1. 职责边界

### 1.1 负责什么

在甲的 HOME 步进 + 乙的 VOF 自由面之上，完成 **刚体 ↔ 流体** 双向耦合，并交付旗舰演示：

| 项 | 内容 |
|----|------|
| 刚体 → 流体 | SDF → `solid_phi` / `body_id`，MAC 壁速嵌入 |
| 动壁 | 默认 **Eq.24**（非整场 \(f\) 经典 BB）；showcase 可改 eq-wall |
| 流体 → 刚体 | 默认 **重构链路动量交换（ME）** |
| 标度 | `recommended_me_force_scale`（刚体 \(g\) 与格子 \(g\) 分制） |
| 算例 | 双球旗舰、单球简化；轨迹日志等 |
| 隔离 | `--showcase-fsi` 经验浮力/推/拖 = **观感 only**，不作纯 ME 验收 |
| 诊断 | 轻球弹跳、重球误浮、贴墙粘滞等观感问题的主责分析 |

一句话：**VOF 线上「球怎么跟水耦」由丙负责；水本身由乙验收，步进由甲托底。**

### 1.2 不负责什么

| 不主责 | 归属 |
|--------|------|
| 无球纯溃坝通过标准、Martin–Moyce 出图 | **乙（张弋洋）** |
| fused 内核结构、矩量化 SoT 实现细节 | **甲（杨宇峰）** |
| 用空中假阻尼 / 经验拖曳冒充「纯 ME 已验证」 | **禁止**（丙自己也遵守） |

可协同：弹跳排查时请甲做 `--me-in-fused` A/B、重构链路检查；请乙确认无球时液面是否已静到可对比。

### 1.3 与甲、乙的接口契约

```
丙需要甲：Eq.24 + 矩重构 ME 原语、可选 in-fused ME
丙需要乙：同几何纯溃坝行为可对照（无球基线）
丙产出：双球可演示；FSI 默认策略文档；已知问题表

子步（算例）：
  coupling.step  → 栅格 + HOME-VOF + ME     （甲乙流体已在内）
  [可选 showcase] 经验力
  rigid_domain.step → collide + XPBD
```

**S2：** 乙纯溃坝 + 丙双球同 \(n\) 可演示。

---

## 2. 主改目录与关键文件

| 路径 | 丙侧职责 |
|------|----------|
| `fluid_grid/coupling/grid_lbm_rigid_coupling.py` | 耦合步进、反馈模式、`recommended_me_force_scale` |
| `fluid_grid/coupling/coupling_kernels.py` | SDF 栅格、壁速嵌入等 |
| `home_fp32_ref/link_me_warp.py` / fused ME 接线 | 与甲协同；**默认开关联丙** |
| `examples/lbm/fluid_grid_lbm_dambreak_vof_two_spheres.py` | **旗舰** |
| `examples/lbm/fluid_grid_lbm_dambreak_vof_single_sphere.py` | 简化 |
| `examples/lbm/_home_vof_empirical_sphere_fsi.py` | showcase 插件（隔离） |
| `docs/wanphys/lbm_home_vof_fsi_algorithm_zh.md` | FSI 算法说明维护 |
| 刚体 collide 小优化 | 视觉 mesh 窄相、墙–墙 pair 剪枝等演示向工程 |

**少改：** 乙的 height-eq 物理默认；甲的矩碰撞核心（除非 ME 联调需要）。

---

## 3. 中期已完成工作（详细）

### 3.1 默认研究路径：ME + Eq.24

- [x] `feedback_mode = momentum_exchange` 为默认；`approx` 降级为诊断；  
- [x] 动壁默认 Eq.24（`vof_home_wall_eq=False`）；  
- [x] coupling 内可关闭刚体推进，由算例外侧 `RigidDomain.step` 统一顺序；  
- [x] 算法文档写清子步顺序与 showcase 边界。

### 3.2 力标度

- [x] `recommended_me_force_scale(dh, dt, …)`：抑制原始 \((dh/dt)^2\) 在分制重力下过猛；  
- [x] CLI `--feedback-force-scale` 可覆盖；  
- [x] 文档声明：标度是方法学问题，不是「再加经验阻尼」了事。

### 3.3 算例与隔离

- [x] 双球：重 ρ≈1.35、轻 ρ≈0.45，溃坝冲击；  
- [x] 单球简化路径；  
- [x] `--showcase-fsi` / 经验插件默认关；打开时日志明示；  
- [x] 可选 `--me-in-fused`（与甲实现协同）。

### 3.4 演示向工程

- [x] 碰撞：避免视觉 mesh 误开昂贵窄相；剪 static–static 墙对；  
- [x] 球摩擦等接触参数可调（减轻骑球/粘墙，**不替代** ME 物理验收）。

### 3.5 进行中 / 未完成（答辩必须诚实）

| 状态 | 项 | 说明 |
|------|-----|------|
| 进行中 | 轻球自由面弹跳 | 腾空时 sub≈0；入水 ME 易过冲；欠阻尼 |
| 进行中 | 重球后期被顶起 | ρ>1 仍可能被 ME 噪声抬离底板 |
| 未完成 | 纯 ME 浮沉**定量**验收 | 趋势（轻重差）可见 ≠ 验收通过 |
| 未完成 | 与 cut-cell / OpenHOMELBM 级 ME 对标 | 当前为栅格 SDF + 重构链路 ME |

**禁止叙事：** 「加了 me-drag / 空中阻尼所以纯 ME 成功了」。观感向选项须标明 showcase。

---

## 4. 答辩时丙怎么讲（建议 3–4 分钟 · 主段 3）

1. **承接乙：** 无球 VOF 已演示；现在挂球。  
2. **默认路径：** 栅格 → Eq.24 → ME → XPBD；**不是** showcase。  
3. **演示：** 双球 `--n 64`；指出重球偏沉、轻球易抬（趋势）。  
4. **诚实：** 轻球可能持续弹；重球久了可能被顶起——中期未收敛。  
5. **可选 30 s：** `--showcase-fsi` 对比观感，并强调 ≠ 研究默认。  
6. **协同：** 弹跳与甲（ME/步进）、乙（界面）继续联调。

```bash
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64

# 观感对照（非验收）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64 --showcase-fsi
```

日志应看到类似：`feedback=ME`，`showcase_fsi=off`（主演示时）。

---

## 5. 下阶段（丙 · 与甲乙并行）

| 优先级 | 任务 | 验收 |
|--------|------|------|
| P0 | 入水瞬间 ME 力/冲量日志；`--me-in-fused` A/B | 判断过冲来源 |
| P0 | \(n=64\) vs \(96\)：弹跳是否随 \(R/dh\) 减轻 | 离散 vs 标度 |
| P1 | 重球长时间 \(z\) 是否保持贴底统计 | 误浮是否复现 |
| P1 | 栅格 ME vs 参考 cut-cell 的文字边界 | 写进算法文档 |
| P2 | 性能：substeps 与甲 fused 耗时一起砍 | 演示更稳 |

目标观感（结题）：轻球冲击后**衰减**；重球**保持下沉**——且仍走纯 ME 默认。

---

## 6. 问题排查清单（丙主责）

| 现象 | 先查 | 再协同 |
|------|------|--------|
| 轻球永弹 | force_scale、ME 符号/相位、\(R/dh\)、sub 日志 | 甲 in-fused；乙无球是否也狂抖 |
| 重球飞起 | 晚期 ME \(F_z\)、淹没分数、回流 | 甲力累加；乙池面 |
| 贴墙粘住 | 接触 μ、墙 pair、速度是否被接触锁死 | 刚体侧，次要怪流体 |
| 骑在另一球上 | 球–球摩擦、干接触无拖曳 | 接触参数，非 ME「浮力」 |

---

## 7. 自测与命令（丙常用）

```bash
uv run python -m unittest \
  newton.tests.test_lbm_home_vof_sphere_momentum_energy \
  newton.tests.test_lbm_home_vof_p2_optional -v

uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof_two_spheres \
  --viewer gl --n 64
```

关注：`test_me_density_trend_light_vs_heavy`（趋势）；长跑观感另记 CSV / 笔记。

---

## 8. 文档与代码索引

| 资源 | 用途 |
|------|------|
| 本页 | 丙个人范围、答辩口径、排查表 |
| [lbm_home_vof_fsi_algorithm_zh.md](lbm_home_vof_fsi_algorithm_zh.md) | 子步/ME/Eq.24 权威 |
| [LATEST_home_vof_two_spheres_progress_zh.md](LATEST_home_vof_two_spheres_progress_zh.md) | 双球进度快照 |
| 甲文档 | 步进与 ME 原语 |
| 乙文档 | 无球基线 |

---

*丙 = VOF-FSI 与双球旗舰负责人。并行于甲、乙；纯 ME 未收敛须在中期如实汇报。*
