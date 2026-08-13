# FIX2 工程计划：孤立界面闭包与零接收者 residual

状态：`IN_PROGRESS / PARTIALLY_RESOLVED`

## 1. 问题复现

RTX 4070 SUPER、`64×64×64`、`gravity_z=-2e-4`、device profile 在
epoch 832 稳定复现：

```text
cell=(21,5,41)
pending=-4.45738478e-05
center=INTERFACE>INTERFACE>INTERFACE
q4=INTERFACE>GAS>GAS
other 17 neighbors=GAS>GAS>GAS
stored_receivers=actual_receivers=0
```

这排除了 CUDA race 和 receiver-count 写入错误。中心单元没有跨过
`-epsilon=-1e-4`，唯一界面邻居却在同一步退出，最终形成无 LIQUID 支撑的孤立
INTERFACE。

## 2. 参考实现差异

`[Code-Ref]Home-FSLBM/inc/3D/gpu/mrLbmSolverGpu3D.cu` 除质量阈值外还使用：

```text
no GAS neighbor    -> INTERFACE to LIQUID
no LIQUID neighbor -> INTERFACE to GAS
```

其零接收者 fallback 把 excess 加回本地 mass。WanPhys 不能直接照搬后者，因为
authoritative resident mass 与几何 `phi` 必须保持 canonical bounded；因此零接收者
质量继续保存在显式 `pending_excess`，由下一步本地重试。

## 3. 冻结语义

### 3.1 D3Q19 邻域分类

只对 old INTERFACE，以 old committed D3Q19 topology 统计 LIQUID/GAS：

```text
if phi_pre >= 1+epsilon or no GAS neighbor:
    proposed = LIQUID
elif phi_pre <= -epsilon or no LIQUID neighbor:
    proposed = GAS
else:
    proposed = INTERFACE
```

保持 LIQUID 优先级，与 Home-FSLBM 的 `if/else if` 顺序一致。周期边界使用同一
D3Q19 映射。随后仍由既有 topology resolver 消除直接 LIQUID-GAS 邻接。

### 3.2 零接收者 residual

```text
abs(pending)>mass_tolerance and receiver_count==0 and actual_count==0
```

是合法的 retained residual，不再是 invalid route。下一步 P2 将其加回同一 sender
的 provisional mass；P4 再次 canonicalize：若新 final topology 出现 INTERFACE
receiver，则写入新的非零 count；否则继续保留。

以下仍 fail-fast：

```text
stored_receiver_count != committed actual_receiver_count
non-finite pending
resident+pending ledger 不守恒
```

## 4. 代码范围

- `vof/transition_kernels.py`：proposal 增加 no-LIQUID/no-GAS；
- `vof/transition.py`：允许 material zero receiver；
- `vof/advection.py`：允许 count=actual=0 的输入；
- `vof/advection_kernels.py`：冻结 material local retry 注释；
- `vof/runtime.py`、`vof/diagnostics.py`：zero receiver 改为观测指标，mismatch
  继续作为错误；
- P2/P4/FIX1 测试：独立 oracle、孤立界面、material retry、诊断分类；
- FIX2 Plan、总结和架构改动记录。

## 5. 验收目标

```text
[x] Home-compatible no-LIQUID/no-GAS D3Q19 proposal oracle
[x] epoch-832 topology fixture: isolated center proposes/finalizes GAS
[x] material zero-receiver residual survives at least two transactions
[x] phi remains in [0,1]
[x] sum(mass)+sum(pending) remains conserved
[x] zero receiver is observable but does not fail
[x] stored/actual count mismatch still fails
[x] P2/P4/FIX1/P7/P8 targeted regression
[x] Ruff F/I and git diff --check
[x] CUDA device-profile rerun beyond epoch 832 (`128^3` to epoch 7980)
[ ] CUDA strict/headless acceptance and CPU/CUDA differential
```

## 6. 明确未完全解决

本 FIX2 当前只解决局部拓扑闭包和 residual 生命周期合法性，仍不声明：

- residual 最大滞留步数或 age/stall 上限；
- 永久孤立 GAS sender 的 residual 最终去向；
- D3Q19 与 Home-FSLBM D3Q27/26-neighbor 的完全等价；
- same-step capacity-constrained redistribution；
- CUDA 长跑通过（必须由修复后的实际日志验收）；
- solid、bubble、foam、open-boundary VOF。

因此完成报告必须保持 `PARTIALLY_RESOLVED`，不得标为完全解决。
