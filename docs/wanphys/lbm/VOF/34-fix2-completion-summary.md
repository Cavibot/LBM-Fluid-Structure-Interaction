# FIX2 完成总结：孤立界面闭包与 retained residual

状态：`PARTIALLY_RESOLVED / CPU_TARGETED_ACCEPTED / CUDA_DEVICE_LONGRUN_OBSERVED`

## 1. 结论

本轮已修复两项可证明的代码缺失：

1. P4 proposal 增加 Home-FSLBM 风格的 no-LIQUID/no-GAS 邻域分类；
2. material zero-receiver excess 不再作为非法路由，而是留在显式
   `pending_excess` 中跨步本地重试。

几何 `phi` 和 resident mass 继续 canonical bounded；守恒权威仍为：

```text
sum(resident mass) + sum(pending_excess)
```

receiver-count mismatch、非有限状态、拓扑非法和 ledger 误差仍 fail-fast。

## 2. epoch-832 问题闭包

原 CUDA trace：

```text
center I>I>I, pending=-4.45738478e-05
q4 I>G>G
other 17 G>G>G
```

FIX2 独立 fixture 使用相同局部模式。由于中心和 q4 均没有 old LIQUID 邻居，二者
都 proposal/finalize 为 GAS；负 excess 留在 count=0 的 pending 通道，resident
mass/phi 保持 `0/0`。

## 3. residual 生命周期

P2 对 count=0 sender 将 pending 加入同一格点 provisional mass。P4 随新拓扑重新
canonicalize：

```text
仍无 receiver -> pending/count=0 继续保留
出现 receiver -> 写入 final INTERFACE count，下一步等分 gather
```

两步 transaction 回归证明 material residual 不丢失、不写入几何 mass，也不会触发
invalid route。device/host diagnostics 把 zero receiver 作为观测计数；只有 stored 与
actual count 不一致才报错。

## 4. 验收证据

目标测试：

```text
P2 + P4 + FIX1 + FIX2
Ran 30 tests
OK (skipped=1 CUDA)

P1-P8 + FIX1 + FIX2
Ran 99 tests
OK (skipped=2 CUDA)

P0 core + directional
Ran 45 tests
OK
```

VOF P1-P8 + FIX1/FIX2 在冻结默认参数下通过。用户工作树仍保留独立的
`128^3 + HOME/NOCM-MRT + gravity=-2e-4` 默认参数实验，该实验不纳入 FIX2 提交。

额外 CPU 长跑：

```text
grid=12^3, gravity_z=-2e-4, profile=device, epoch=900
relative mass error=5.376e-8
phi=[0,1]
invalid pending=0
zero receiver at final epoch=0
```

代码质量：

```text
Ruff F/I: passed
git diff --check: passed
```

修复后用户侧 CUDA device-profile 长跑：

```text
RTX 4070 SUPER, grid=128^3, HOME/NOCM-MRT, gravity_z=-2e-4
crossed old failure epoch 832 and reached epoch 7980 without exception
relative mass error approximately 3.2e-8 to 4.1e-8
phi=[0,1]
zero_rx became persistently 4 from epoch 5580 to 7980
```

这证明 CUDA kernel 可以编译运行且原崩溃路径已被跨越，但持续 `zero_rx=4` 同时证明
residual stall 尚未闭合。该日志不是 strict/headless 或 CPU/CUDA differential 验收。

## 5. 为什么仍标记“未完全解决”

本修复没有为 retained residual 增加 age。若 sender 永久与任何 INTERFACE 隔离，
pending 可以永久留在 conservative ledger 中而不参与几何。当前 `zero_rx` 运行指标
只能观测数量，不能证明最终消散。

另外 Home-FSLBM 使用26邻域及 FLUID/INTERFACE/转换态 receiver；WanPhys 使用
D3Q19 final-INTERFACE receiver，因此不是逐行等价移植。CUDA device profile 已实际
越过 epoch 832，但 strict/headless 与 CPU/CUDA differential 仍未验收。

明确未完成：

- residual age/stall policy；
- 永久孤立 residual 的最终 local/nonlocal closure；
- same-step receiver capacity solver；
- D3Q19 与参考 D3Q27 的等价证明；
- CUDA strict/headless 与 CPU/CUDA differential acceptance；
- solid、bubble、foam、open boundary。

因此本文件不得被引用为“完整解决”或 `CUDA_ACCEPTED` 证据。
