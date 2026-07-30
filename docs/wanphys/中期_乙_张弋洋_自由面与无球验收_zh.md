# 成员乙（张弋洋）详细说明：自由面物理与无球验收

> 角色：中期答辩 **乙**  
> 领域：**自由面物理与无球验收**（VOF 界面/质量交换、纯溃坝、Martin–Moyce、height-eq）  
> 对应总稿：[中期答辩_LBM自由表面流固耦合_zh.md](中期答辩_LBM自由表面流固耦合_zh.md)  
> 专题：[lbm_home_vof_height_eq_summary_zh.md](lbm_home_vof_height_eq_summary_zh.md)、[lbm_home_fslbm_one_cell_limit_zh.md](lbm_home_fslbm_one_cell_limit_zh.md)、[lbm_home_vof_quiet_level_summary_zh.md](lbm_home_vof_quiet_level_summary_zh.md)  
> 日期：2026-07-30

---

## 1. 职责边界

### 1.1 负责什么

在甲提供的 **HOME 步进**之上，把 **锐界面自由表面**做成「无球也可说清、可测、可演示」：

| 项 | 内容 |
|----|------|
| 界面表示 | \(\varphi\)、L / G / I（及短暂过渡型）、液体质量库存 |
| 物理过程 | Körner 型质量交换、FS 边界行为（与 fused/步进协同） |
| 验收场景 | **纯溃坝**（无刚体球） |
| 基准 | Martin–Moyce 前缘等（L0 级） |
| 晚期整形 | 可选 `--height-eq`（IF 上 \(\varphi\to\varphi^*\)），并说清**不是**随意造质量 |
| 极限说明 | 一格厚壁面液膜等 FSLBM/IF 固有限制 |

一句话：**VOF 线的「水本身」是否靠谱，由乙验收；有球是丙的事。**

### 1.2 不负责什么

| 不主责 | 归属 |
|--------|------|
| 矩碰撞细节、fused 核结构重构 | **甲（杨宇峰）** |
| 双球 ME、力标度、showcase 浮力 | **丙（徐子轩）** |
| Shan-Chen 伪势当主交付 | **已有基础** |

可协同：丙反馈「球边弯月面坑」时，乙查 height-eq 近刚体权重、质量交换是否在球周异常；**不**用经验拖曳替丙修弹跳。

### 1.3 与甲、丙的接口契约

```
乙需要甲：稳定 HOME-VOF step，φ/mass/cell 语义一致
乙交给丙：无球验收通过的流体行为作 FSI 基线
乙不改：coupling 默认 ME 策略、双球 CLI 默认

S2 联调：乙纯溃坝可演示（同 n）+ 丙双球可演示
```

---

## 2. 主改目录与关键文件

| 路径 | 乙侧职责 |
|------|----------|
| `lbm/phases/vof_*.py`、`vof_plic.py` 等 | 界面几何 / κ 等相关 |
| `home_fp32_ref/vof_warp.py` 中 **自由面/质量** 逻辑（与甲协商改核） | 质量交换、类型更新 |
| `home_fp32_ref/height_eq*.py` | 晚期池面 φ 正则 |
| `examples/lbm/fluid_grid_lbm_dambreak_vof.py` | **主演示：纯溃坝** |
| `examples/lbm/run_martin_moyce_compare.py` | 前缘对照 |
| `newton/tests/test_lbm_home_vof_conservation.py` 等 | 质量/动量有界 |
| 专题 md | height-eq、quiet-level、一格极限 |

**少改：** `coupling/`、双球算例默认 FSI 旗标（丙）。

---

## 3. 中期已完成工作（详细）

### 3.1 VOF 锐界面主路径

- [x] L / G / I 单元与体积分数 \(\varphi\)；  
- [x] 与 HOME-FREE 一致的「只推进液相动力学」叙事；  
- [x] 质量交换与界面演化可在 GPU fused 路径上跑通。

### 3.2 纯溃坝验收

- [x] 算例：`fluid_grid_lbm_dambreak_vof.py --backend home`；  
- [x] 可 GL 演示锐界面（对比开题前 SC 扩散界面——仅口述对照）；  
- [x] 与甲联调：backend home 下完整步进。

### 3.3 基准与单测

- [x] Martin–Moyce 前缘对照（L0）；  
- [x] 水质量库存 \(\sum\mathrm{mass}\)、动量有界等单测（守恒套件）；  
- [x] 文档中明确容差与「何谓通过」。

### 3.4 height-eq 与极限认知

- [x] 可选 `--height-eq`：\(t\) 达阈值后对 IF 做 \(\varphi\to\varphi^*\) 软正则；  
- [x] 近刚体权重避免「球周永久死区坑」（与丙场景相关的流体侧）；  
- [x] 书面说明：**一格液膜 / 完全镜面找平**受方法极限约束，禁止无效调参叙事（见单格极限文档）；  
- [x] 记录已放弃的有害启发式（如乱造质量的 film-drain 等，见 quiet-level / LATEST）。

### 3.5 进行中 / 未完成

| 状态 | 项 |
|------|-----|
| 进行中 | \(n=64/96/128\) 质量漂移与界面粗糙度曲线 |
| 未完成 | 更多自由面专题出图（泼溅细节、闭气泡等——按结题范围裁） |

---

## 4. 答辩时乙怎么讲（建议 2–3 分钟 · 主段 1）

1. **范围：** 本阶段 VOF 线；先讲**无球**水。  
2. **对比一句：** 已有 SC 是扩散界面；现在是锐界面 \(\varphi\)。  
3. **演示：** 纯溃坝 HOME-VOF，指出波前、液面。  
4. **证据：** Martin–Moyce / 质量单测（口头或一页图）。  
5. **诚实：** 池面不一定「绝对平」；一格极限；height-eq 是正则不是魔法。  
6. **交接：** 无球基线交给丙挂双球。

```bash
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof \
  --backend home --n 64 --viewer gl
# 可选
# ... --height-eq
```

---

## 5. 下阶段（乙 · 与甲丙并行）

| 优先级 | 任务 | 验收 |
|--------|------|------|
| P0 | 分辨率扫描：质量相对漂移、粗糙度指标出图 | 可放进结题材料 |
| P0 | 纯溃坝「通过标准」一页纸（与甲约定数值旗标） | S2 可重复 |
| P1 | Martin–Moyce 图版完善 | 答辩/论文图 |
| P2 | 扩展自由面专题 | 不阻塞丙 ME 诊断 |

**原则：** 不把「球弹跳」揽成乙用 height-eq 硬抹平；球的问题主责在丙。

---

## 6. 自测与命令（乙常用）

```bash
uv run python -m unittest newton.tests.test_lbm_home_vof_conservation -v

uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof \
  --backend home --n 64 --viewer gl

# Martin–Moyce（以仓库入口为准）
uv run --extra examples python -m wanphys.examples.lbm.run_martin_moyce_compare
```

观察要点（无球）：

- 溃坝后大尺度晃动是否随时间耗散；  
- 质量日志是否缓慢漂移而非崩塌；  
- 开 height-eq 后晚期 IF 是否更「池面状」，且无离谱质量跳变。

---

## 7. 文档与代码索引

| 资源 | 用途 |
|------|------|
| 本页 | 乙个人范围与答辩口径 |
| [lbm_home_vof_height_eq_summary_zh.md](lbm_home_vof_height_eq_summary_zh.md) | height-eq 细节 |
| [lbm_home_fslbm_one_cell_limit_zh.md](lbm_home_fslbm_one_cell_limit_zh.md) | 一格极限（答辩必知） |
| [lbm_home_vof_quiet_level_summary_zh.md](lbm_home_vof_quiet_level_summary_zh.md) | 找平试验与废弃路径 |
| 甲文档 | 步进——乙依赖其稳定 |
| 丙文档 | FSI——乙提供无球基线 |

---

*乙 = VOF 自由面与无球验收负责人。并行于甲、丙；旗舰双球演示由丙主讲。*
