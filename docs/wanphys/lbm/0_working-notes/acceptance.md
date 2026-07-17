# LBM Phase 0 核心重构验收入口

> 更新：2026-07-17  
> 本文件是 HOME/FullF 核心重构阶段的合并前验收清单。  
> 详细历史见 `execution-plan.md`；测试语义见 `test-explanation.md`；收尾记录见 `maybe-todo.md`。  
> Raw MRT、FullF NOCM、外力、边界、D3QN 等属于后续能力路线图，不属于 Phase 0 合并阻塞项。

## 当前结论

```text
架构已经接通
公式单元测试已覆盖
一步运行路径已覆盖
定向 streaming 测试已补
多步 shear-wave 测试已补
HOME-NOCM + Shan-Chen 会 fail-fast
历史可视化测试在缺失示例时 skip
CUDA smoke 脚本与记录已补
lint / format / type / unit 仍需本地或 CI 最终确认
```

## Phase 0 完成项

| 项 | 状态 | 说明 |
|---|---|---|
| HOME-NOCM + Shan-Chen 静默忽略 | 已完成 | `LbmModel` 构造时 `NotImplementedError` |
| 四条定向 streaming 测试 | 已完成 | `test_lbm_directional_streaming.py` |
| 多步 shear-wave 测试 | 已完成 | FullF SRT/TRT 对理论衰减；HOME 路径有限/守恒/衰减 |
| backend 文档契约 | 已完成 | backend 只提供元数据；kernel 由 Solver 启动 |
| 历史可视化测试 | 已处理 | 缺失示例时显式 skip，注明恢复任务 |
| CUDA smoke | 已记录 | 见 `cuda-acceptance-record.md` |
| 自动化质量检查 | 待确认 | 本地或 CI 执行 lint / type / unit |

## Phase 0 明确不覆盖

以下内容不是“永不实现”，而是不属于本阶段验收范围：

- `backend.collide()` 包装层；
- FullF Raw MRT；
- FullF NOCM MRT；
- HOME / NOCM 外力闭合；
- Shan-Chen 与 MRT/NOCM 的外力注入；
- Zou-He inlet、convective outlet、moving-wall/cut-link 边界迁移；
- MEM / LinkImpulse 重构；
- D3Q27 / D3QN。

## 核心回归命令

```bash
uv run python -m unittest \
  newton.tests.test_lbm_state_encoding \
  newton.tests.test_lbm_streaming_matrix \
  newton.tests.test_lbm_collision_backends \
  newton.tests.test_lbm_home_nocm \
  newton.tests.test_lbm_shan_chen_wall_force \
  newton.tests.test_lbm_rigid_coupling \
  newton.tests.test_lbm_directional_streaming -v
```

历史可视化测试（允许 skip）：

```bash
uv run python -m unittest \
  newton.tests.test_lbm_dambreak_examples \
  newton.tests.test_lbm_dambreak_falling_sphere_visualization \
  newton.tests.test_lbm_falling_sphere_visualization \
  newton.tests.test_lbm_rigid_visualization -v
```

## CollisionBackend 契约

```text
CollisionBackend / SrtCollision / TrtCollision / HomeNocmMrtCollision
= 碰撞选择与参数描述对象

LbmSolver
= 真正选择并启动 Warp kernel 的调度器
```

第一版不提供统一 `backend.collide(...)`。这是有意决定，不是半成品遗漏。

## 历史测试策略

默认方案：

```text
缺失示例/helper 时显式 SkipTest
并注明：恢复示例或删除测试属于独立清理任务
```

当前缺失模块包括但不限于：

- `fluid_grid_lbm_dambreak`
- `fluid_grid_lbm_dambreak_falling_sphere_visual`
- `fluid_grid_lbm_falling_sphere_twophase_visual`
- `fluid_grid_lbm_density_sphere_twophase_visual`
- `_lbm_falling_sphere_scene`
- `_lbm_twoway_scene`
- `fluid_grid_lbm_twoway_fsi_visual`

`dambreak_mem` 包仍在仓库中，但依赖上述缺失 helper，因此相关测试也会 skip。

## CUDA 验收命令与记录模板

### 命令

```bash
# 设备信息
uv run python -c "import warp as wp; print(wp.get_devices()); wp.init(); print(wp.get_cuda_devices())"

# 六种组合一步 + HOME scratch 检查 + FullF SC smoke + 小规模多步
uv run python scripts/lbm_cuda_acceptance.py
```

记录见 `cuda-acceptance-record.md`。

### 记录字段

```text
日期:
GPU 型号:
CUDA Driver / Toolkit:
Warp 版本:
测试网格:
六种组合一步:
HOME reconstruction/NOCM 编译:
FullF Shan-Chen smoke:
小规模多步有限性:
HOME-NOCM population scratch:
显存占用:
结论: pass / fail
备注:
```

## 合并门槛

```text
[x] 修复 FullF+HOME-NOCM+SC 静默忽略
[x] backend 文档与代码契约一致
[x] 非均匀定向 streaming 测试
[x] 多步 shear-wave 物理验收
[x] 历史测试 skip 处理
[x] CUDA smoke 脚本与记录
[ ] lint / format / type / unit CI 全绿
```
