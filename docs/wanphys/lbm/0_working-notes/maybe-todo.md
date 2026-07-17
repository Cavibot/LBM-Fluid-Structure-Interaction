# LBM Phase 0 收尾记录

> 更新：2026-07-17  
> 本文件记录 HOME/FullF 核心重构阶段的合并前质量门槛。  
> 当前验收入口见 `acceptance.md`。  
> Raw MRT、FullF NOCM、外力、边界、D3QN 等后续能力不再放在本文件追踪。

## 状态总览

```text
[x] 修复 FullF+HOME-NOCM+SC 静默忽略
[x] backend 文档与代码契约一致
[x] 非均匀定向 streaming 测试已补
[x] 多步 shear-wave 物理验收测试已补
[x] 历史可视化测试缺示例时显式 skip
[x] CUDA smoke 脚本与记录已补
[ ] lint / format / type / unit 由本地或 CI 最终确认
```

## 1. FullF + HOME-NOCM + Shan-Chen fail-fast

原问题：

```text
FullF + home_nocm + G != 0
```

会构造成功，但 Shan-Chen 力不会进入 EMC collision，存在静默忽略风险。

当前处理：

- `LbmModel` 构造时直接抛 `NotImplementedError`；
- `test_lbm_state_encoding.py` 覆盖该组合；
- 后续若要支持，应进入“外力注入 / NOCM forcing closure”路线，而不是取消 fail-fast。

状态：已完成。

## 2. CollisionBackend 契约

原问题：

计划文字暗示 backend 可能拥有统一：

```python
backend.collide(...)
```

但实现中真正启动 Warp kernel 的仍然是 `LbmSolver`。

当前处理：

- `collisions.py` docstring 明确 backend 是 collision selection metadata；
- `execution-plan.md` 说明第一版不实现 `backend.collide()`；
- `acceptance.md` 固化该契约。

状态：已完成。

## 3. 非均匀 directional streaming 测试

原问题：

均匀平衡场不能检查方向、source/target、periodic wrap、HOME reconstruction/provider、EMC collector 是否接错。

当前处理：

- 新增 `test_lbm_directional_streaming.py`；
- 覆盖：
  - FullF → populations；
  - HOME → populations；
  - FullF → moments；
  - HOME → moments；
  - 单方向 packet 移动；
  - 周期回绕。

状态：已完成。

## 4. 多步 shear-wave 物理验收

原问题：

一步、均匀、公式一致不足以证明新时间层多步稳定。

当前处理：

- 在 `test_lbm_directional_streaming.py` 中增加 shear-wave decay；
- FullF SRT/TRT 对理论黏性衰减做误差窗口检查；
- HOME 相关路径检查有限性、质量守恒和衰减方向。

状态：已完成。

## 5. 历史可视化测试红项

原问题：

若干历史测试引用仓库中已经不存在的 LBM 示例/helper，导致 `ModuleNotFoundError` 红掉。

当前处理：

- 缺失历史示例/helper 时显式 `SkipTest`；
- skip 信息说明“恢复示例或删除测试”是独立清理任务；
- 不在本阶段恢复已删除示例。

状态：已完成。

## 6. CUDA smoke 记录

当前处理：

- 新增 `scripts/lbm_cuda_acceptance.py`；
- 新增 `cuda-acceptance-record.md`；
- 记录 2026-07-16 本机 CUDA smoke：RTX 4070 SUPER / Warp 1.12 / CUDA 12.9。

状态：已完成。

## 7. 自动化质量检查

仍需由本地或 CI 最终确认：

```text
ruff format --check
ruff check
basedpyright 或项目指定类型检查
相关 unittest
```

本次收尾不声称这些检查已经重新执行通过。

状态：待 CI / 本地验证确认。

## 后续能力不属于本文件

以下内容是下一阶段路线图，不是 Phase 0 收尾项：

- FullF Raw MRT；
- FullF NOCM MRT；
- HOME/NOCM 外力闭合；
- Shan-Chen 与 MRT/NOCM 的外力注入；
- Zou-He / convective / moving-wall 等边界迁移；
- D3Q27 / D3QN；
- MEM / LinkImpulse 重构。
