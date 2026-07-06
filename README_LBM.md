# WanPhys LBM 示例运行指南

## 环境准备

```bash
# 安装示例依赖
uv sync --extra examples
```

> **注意**：可视化（`--viewer gl`）需要 OpenGL 支持。在 headless 环境或 SSH 会话中请去掉 `--viewer gl`，示例会自动以无头模式运行。

---

## 1. 溃坝 (Dam-Break)

| 示例 | 命令 |
|------|------|
| 溃坝 (trt) | `uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt --viewer gl` |
| 溃坝 + 双球 FSI | `uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_two_spheres --viewer gl` |


## 2. 液滴 (Droplet)

| 示例 | 命令 |
|------|------|
| 液滴自由落体入池 | `uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_pool_drop --viewer gl` |
| 液滴撞干地板 | `uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_droplet_floor_trt --viewer gl` |
| 液滴撞液膜飞溅 | `uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_droplet_splash_trt --viewer gl` |
| 液滴碰撞合并 | `uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_droplet_coalescence_trt --viewer gl` |
| 液滴下落 (自由落体) | `uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_droplet_fall_trt --viewer gl` |

## 3. 相分离 / 自旋odal 分解

| 示例 | 命令 |
|------|------|
| Spinodal 相分离 | `uv run --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_spinodal_trt --viewer gl` |

---

## 运行说明

- **可视化模式**：添加 `--viewer gl` 启用 OpenGL 交互窗口（支持 [Space] 暂停/继续、[R] 重置、鼠标拖拽旋转视角、滚轮缩放）。
- **无头模式**：去掉 `--viewer gl`，示例以命令行模式运行，适合服务器或批量测试。
- **首次运行**：Warp (CUDA) 内核会在首次执行时编译，可能需要几秒到几十秒，之后运行会明显加快。

### 键盘快捷键（可视化窗口）

| 按键 | 功能 |
|------|------|
| `Space` | 暂停 / 继续 |
| `R` | 重置场景 |
| 鼠标拖拽 | 旋转视角 |
| 滚轮 | 缩放 |

---

## 完整示例列表

```
wanphys/examples/lbm/
├── fluid_grid_lbm_dambreak_trt.py            # 溃坝 (TRT)
├── fluid_grid_lbm_dambreak_two_spheres.py     # 溃坝 + 双球 (BGK FSI)
├── fluid_grid_lbm_pool_drop.py                # 液滴落池 (TRT + SC)
├── fluid_grid_lbm_droplet_fall_trt.py         # 液滴自由落体 (TRT + SC)
├── fluid_grid_lbm_droplet_floor_trt.py        # 液滴撞地板 (TRT + SC)
├── fluid_grid_lbm_droplet_splash_trt.py       # 液滴撞液膜 (TRT + SC)
├── fluid_grid_lbm_droplet_coalescence_trt.py  # 液滴碰撞合并 (TRT + SC)
├── fluid_grid_lbm_spinodal_trt.py             # Spinodal 相分离 (TRT + SC)
```

> 更多细节（碰撞模型、相态、耦合方式等标签）参见 `wanphys/examples/lbm/README.md`。
