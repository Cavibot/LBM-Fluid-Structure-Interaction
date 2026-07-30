# 成员甲（杨宇峰）详细说明：HOME-VOF 流体步进

> 角色：中期答辩 **甲**  
> 领域：**HOME-VOF 流体步进**（矩后端、fused GPU 推进、与 VOF / ME 的数据接口）  
> 对应总稿：[中期答辩_LBM自由表面流固耦合_zh.md](中期答辩_LBM自由表面流固耦合_zh.md)  
> 技术深文：[lbm_home_vof_fsi_algorithm_zh.md](lbm_home_vof_fsi_algorithm_zh.md)、[lbm_module_files_zh.md](lbm_module_files_zh.md)  
> 日期：2026-07-30

---

## 1. 职责边界

### 1.1 负责什么

在**已有** dist / Shan-Chen 工程之上，为本阶段 **VOF 线**提供可用的 **HOME 矩编码流体步进**：

| 项 | 内容 |
|----|------|
| 状态表示 | 以 Hermite 矩 \((\rho,\mathbf{u},S)\) 为主，**无整场**分布函数 \(f_i\) 作 SoT |
| 格子 | 主线 **D3Q27** |
| 运行时 | GPU fused 推进（`vof_warp` / bridge），numpy 参考路径可对照 |
| 对外接口 | 乙拿 φ / cell / mass 做自由面；丙拿矩重构链路做动壁与 ME |

一句话：**让「HOME 一步」在 GPU 上稳定可调，并暴露乙、丙需要的缓冲与钩子。**

### 1.2 不负责什么（避免抢活）

| 不主责 | 归属 |
|--------|------|
| VOF 界面物理口径、纯溃坝验收曲线、height-eq 策略 | **乙（张弋洋）** |
| `GridLbmRigidCoupling` 默认策略、双球场景、ME 力标度调参结论 | **丙（徐子轩）** |
| 开题前 dist / Shan-Chen 内核当课题主体 | **已有基础**（Edward Sun 等），仅对照 |

可协同：丙查 ME 相位/力异常时，甲查**矩重构与 fused 内动壁分支**是否写对。

### 1.3 与乙、丙的接口契约

```
甲产出                          乙使用                    丙使用
─────────                       ──────                    ──────
矩场 ρ,u,S（及可选 moment_q）    宏观量参与质量/φ          重构 f_i 做 ME
fused 一步写回                   cell_type / mass / φ      solid 拉流后的矩
Eq.24 / 动壁拉流钩子             （不改耦合策略）           壁速 → 流体
prepare_fused_link_me 等       —                         流中 ME 累加 body_f
```

**S1 联调：** 甲保证 `LbmDomain.step` + `home_fp32` 在「无球」下可跑；乙挂 VOF；丙挂 `solid_phi` 后仍能 step。

---

## 2. 主改目录与关键文件

根路径：`wanphys/_src/fluid/fluid_grid/lbm/backends/moment/home_fp32_ref/`

| 文件 | 甲侧职责 |
|------|----------|
| `bridge.py` | `HomeFp32Bridge`：与 `LbmSolver` 挂钩、分配缓冲、调用步进 / ME 准备 |
| `vof_warp.py` | fused GPU 核：碰撞、拉流、固壁分支、可选流中 ME 累加 |
| `vof_step.py` | numpy 参考步进（对照与小测） |
| `bc.py` / 相关 | Eq.24 等动壁公式实现（策略开关可由丙配置） |
| `link_me_warp.py` | 矩重构链路 ME 核（**实现**在甲侧；**是否默认、标度**由丙定） |
| `quant.py` | 可选矩量化 pack/unpack（`--moment-quant`；与步进持久化相关） |
| `generic.py` | `make_home_vof_model` 等工厂 |

相关上层：`lbm/model.py`（`lbm_backend=home_fp32`、`vof_home_me_in_fused` 等旗标）、`lbm/solver.py` 挂钩。

**少改：** `phases/` 里乙主导的 VOF 策略文档与 height-eq 参数默认；`coupling/` 里丙主导的反馈模式与算例 CLI。

---

## 3. 中期已完成工作（详细）

### 3.1 HOME 矩步进骨架

- [x] `home_fp32_ref`：每格约 10 个矩量级表示 + 需要时 Hermite 重建分布；  
- [x] 与 WanPhys `LbmModel` / `LbmDomain` 通过 bridge 接入，后端可切换而不拆掉已有 dist；  
- [x] D3Q27 主线服务 VOF（相对开题前 D3Q19 dist）。

### 3.2 与 VOF fused 对接

- [x] fused 核内完成「矩工作集」读写，供乙的 φ / 质量交换同核或同子步使用；  
- [x] 固体内流体清空、动壁拉流分支可运行（几何由丙栅格化写入 `solid_phi`）；  
- [x] 保证乙的纯溃坝（`--backend home`）能走到完整流体步。

### 3.3 动壁与 ME 所需的矩→链能力

- [x] 从矩重构链分布，支持 Eq.24（及可选 eq-wall）拉流；  
- [x] 重构链路 ME（post-step 与可选 in-fused）所需的 \(f\) 重构与累加原语；  
- [x] `prepare_fused_link_me` 等臂装接口，供丙打开 `vof_home_me_in_fused`。

### 3.4 进行中 / 未完成

| 状态 | 项 | 说明 |
|------|-----|------|
| 进行中 | 步进稳定性 | 大 \(n\)、多子步下无 NaN、矩不爆 |
| 进行中 | 配合丙查 ME 相位 | 流中 ME vs 步后 ME 差异 |
| 未完成 | 完整量化论文栈后端 | 双核固体校正、8³ tile 等——中期可不展开，结题视范围 |

---

## 4. 答辩时甲怎么讲（建议 1–2 分钟）

1. **问题：** 整场 \(f_i\) 贵；本线用 HOME 矩推进。  
2. **做法：** `home_fp32_ref` + fused；无整场 \(f\) SoT。  
3. **交接：** 乙在其上做 VOF；丙用重构链路做动壁/ME。  
4. **演示一句：** 可提 fused / 可选 `--moment-quant`（若现场开启）。  
5. **诚实：** 量化完整论文栈未做完；当前保证 VOF 线能跑。

---

## 5. 下阶段（甲 · 与乙丙并行）

| 优先级 | 任务 | 验收 |
|--------|------|------|
| P0 | 双球长跑无步进回归（配合丙） | 同命令下无崩溃、矩有界 |
| P0 | ME 异常时提供重构/fused 分支对照开关说明 | 丙能 A/B `--me-in-fused` |
| P1 | 步进耗时剖面（fused 占比） | 给丙调 substeps 的依据 |
| P2 | 量化后端按结题范围推进 | 单独里程碑，不阻塞乙验收 |

**联调：** 每周短会；接口变更先在 S 类节点三人对齐。

---

## 6. 自测与命令（甲常用）

```bash
# 求解器挂钩冒烟（示例）
uv run python -m unittest newton.tests.test_lbm_home_solver_hook -v

# 无球流体（确认步进+乙 VOF 合金）
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_vof \
  --backend home --n 48 --viewer gl
```

相关单测还可涉及：`test_lbm_home_vof_p2_optional`（ME in-fused）、守恒套件中带 `--moment-quant` 的用例。

---

## 7. 文档与代码索引

| 资源 | 用途 |
|------|------|
| 本页 | 甲个人范围与答辩口径 |
| [lbm_home_vof_fsi_algorithm_zh.md](lbm_home_vof_fsi_algorithm_zh.md) | Eq.24 / ME 公式级说明（实现细节） |
| [lbm_module_files_zh.md](lbm_module_files_zh.md) | 文件树 |
| 乙文档 | 自由面验收——甲勿越界改口径 |
| 丙文档 | FSI——甲提供原语，丙定默认策略 |

---

*甲 = HOME-VOF 步进负责人。并行于乙、丙；总稿只保留分工表，细节以本页为准。*
