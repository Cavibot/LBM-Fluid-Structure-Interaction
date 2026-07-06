# WanPhys LBM 示例一览

## 标签维度

| 维度 | 标签 | 含义 |
|------|------|------|
| **相态** (Phase) | `phase:single` | 单相流 |
| | `phase:multi` | 多相流 (Shan-Chen) |
| **碰撞** (Collision) | `collision:bgk` | BGK 单松弛 |
| | `collision:trt` | TRT 双松弛 |
| | `collision:reg` | 正则化 |
| **耦合** (Coupling) | `coupling:none` | 纯流体 |
| | `coupling:oneway` | 单向 (刚体→流体) |
| | `coupling:twoway` | 双向 (流体↔刚体) |
| | `rigid:static` | 刚体静止 |
| | `rigid:moving` | 刚体运动 |
| **场景** (Scene) | `scene:dambreak` | 溃坝 / 水箱坍塌 |
| | `scene:cavity` | 顶盖驱动腔体 |
| | `scene:cylinder` | 圆柱绕流 |
| | `scene:droplet` | 液滴振荡 |
| | `scene:container` | 刚性容器 |
| | `scene:channel` | 通道流动 |
| | `scene:fsi` | 通用流固耦合 |
| | `scene:generic` | 通用/无指定场景 |
| **模式** (Mode) | `mode:headless` | 无头命令行 |
| | `mode:visual` | 带 OpenGL 可视化 |
| **用途** (Purpose) | `purpose:demo` | 功能演示 |
| | `purpose:validate` | 验证/测试 |
| | `purpose:debug` | 调试工具 |

## 示例文件清单

| 文件名 | 相态 | 碰撞 | 耦合 | 场景 | 模式 | 用途 |
|--------|------|------|------|------|------|------|
| `fluid_grid_lbm.py` | single | bgk | none | droplet | headless | demo |
| `fluid_grid_lbm_cavity.py` | single | bgk | none | cavity | headless | demo |
| `fluid_grid_lbm_cylinder.py` | single | bgk | none | cylinder | headless | demo |
| `fluid_grid_lbm_dambreak.py` | single | bgk | none | dambreak | headless | demo |
| `fluid_grid_lbm_dambreak_reg.py` | single | bgk, reg | none | dambreak | headless | demo |
| `fluid_grid_lbm_dambreak_trt.py` | single | trt | none | dambreak | headless | demo |
| `fluid_grid_lbm_dambreak_sc.py` | multi | bgk | none | dambreak | headless | demo |
| `fluid_grid_lbm_twophase.py` | multi | bgk | none | generic | headless | demo |
| `fluid_grid_lbm_twophase_gravity.py` | multi | bgk | none | generic | headless | demo |
| `fluid_grid_lbm_container.py` | multi | bgk | oneway, rigid:static | container | headless | demo |
| `fluid_grid_lbm_dambreak_sc_rigid.py` | multi | bgk | oneway, rigid:moving | dambreak | headless | demo |
| `fluid_grid_lbm_oneway_moving_rigid_visual.py` | multi | bgk | oneway, rigid:moving | fsi | visual | demo |
| `fluid_grid_lbm_oneway_moving_rigid_visual_trt.py` | multi | trt | oneway, rigid:moving | fsi | visual | demo |
| `fluid_grid_lbm_fsi.py` | single | bgk | twoway, rigid:static | fsi | headless | demo |
| `fluid_grid_lbm_twoway_fsi.py` | single | bgk | twoway, rigid:moving | fsi | headless | demo |
| `fluid_grid_lbm_twoway_fsi_visual.py` | single | bgk | twoway, rigid:moving | fsi | visual | demo |
| `fluid_grid_lbm_debug.py` | single | bgk | none | generic | visual | debug |
| `fluid_grid_lbm_validate.py` | single | bgk | none | generic | headless | validate |

## 运行示例

```bash
# 运行任意示例
uv run python -m wanphys.examples.<example_name>

# 例如：运行溃坝示例
uv run python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak

# 带 OpenGL 可视化
uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi_visual --viewer gl
```

## 按标签筛选

想要快速筛选特定类型的示例，可以用 grep 或编辑器搜索标签：

```bash
# 找所有可视化示例
grep "mode:visual" README.md

# 找所有双向耦合示例
grep "twoway" README.md

# 找所有 TRT 碰撞示例
grep "collision:trt" README.md
```

## 辅助模块

- `_lbm_twoway_scene.py` — 双向耦合场景的共享配置和工具函数（非独立示例）

## 双相流稳定性改动说明

这次 P0～P1 改动可以理解为：**没有改变双相流默认玩法，而是把原来写死在代码里的稳定性/性能小技巧变成可控参数，并加了测试，防止以后调坏。**

### 原来有什么问题

- 双相流里的 Shan-Chen 力负责让液相和气相分开。
- 为了跑得快，代码原来做了两个优化：
  - 不每一步都重新算 Shan-Chen 力，而是每 2 步算一次。
  - 如果某个格子周围看起来很均匀，就直接认为力是 0，跳过完整计算。
- 这两个优化是合理的，但之前是写死的，不能关。
- 另外代码注释里有一处说速度修正应该用 `τ₊`，但实际代码用的是 `τ`。这会让后续维护者误以为代码写错了，然后可能改回一个不稳定版本。

### 现在改了什么

- 新增了 3 个模型参数：
  - `sc_force_stride=2`：几步重新算一次 Shan-Chen 力；设成 `1` 就是每步都算，更准确。
  - `sc_homogeneous_early_out=True`：是否启用“均匀区域跳过计算”。
  - `sc_homogeneous_rel_tol=0.15`：判断“足够均匀”的密度容忍度。
- 默认值和原来行为一样，所以现有示例不会突然变慢或变样。
- 把注释改清楚：当前速度修正用 `τ` 是有意为之，是为了强界面双相流更稳定。
- 增加了参数检查：
  - `psi_type` 只能是 0 或 1。
  - `psi_ref` 必须大于 0。
  - `sc_force_stride` 至少是 1。
  - `sc_homogeneous_rel_tol` 不能小于 0。

### 为什么这很重要

- 以后调试双相流时，可以更清楚地区分问题来源：
  - 想要最准确：设 `sc_force_stride=1`，关掉 early-out。
  - 想要性能：保持默认 `2` 和 early-out。
- 如果 dam-break、液滴、刚体附近界面出问题，可以先关优化验证是不是优化造成的。
- 测试会保护这些行为，防止之后有人无意中改坏。

### 新增测试在测什么

- 壁面/固体虚拟密度方向是否正确。
- early-out 是否真的可以关闭。
- `sc_force_stride=1` 和 `2` 在小规模双相场里都不会爆炸。
- 非法参数是否会立刻报错，而不是跑到一半才崩。

### 当前鲁棒性结论

- 比之前更鲁棒了，因为关键稳定性开关可控了。
- 但还不是完整物理验证。现在主要保证“不容易被调坏、不容易静默崩、可诊断”。
- 后续 P2 适合做更物理的验证，比如液滴压力差、接触角、长期质量守恒、强 dam-break 对比。


