# P2 固定拓扑质量平流完成总结

## 1. 结论

P2 已达到：

```text
CPU_ACCEPTED
```

本阶段完成了 FullF authoritative VOF 的独立、固定拓扑、无副作用质量 transport。
完整 `LbmDomain.step()` 仍在所有写入前 fail-fast，直到 P3 完成零表面张力自由面
population reconstruction。

## 2. 冻结的三项核心决策

### 2.1 质量交换 scheme

```text
vof_mass_scheme = fslbm_neighbor
```

采用用户指定首选参考库 Home-FSLBM 的经典邻居类型 scheme：

```text
center LIQUID + active neighbor -> weight 1
center INTERFACE + LIQUID       -> weight 1
center INTERFACE + INTERFACE    -> average(phi)
any active center + GAS         -> weight 0
```

该方案明确标为经典 FSLBM/Home-FSLBM 适配，不声称等于目标论文印刷 Eq. (10) 的
中心类型分段。

### 2.2 density 时间层

```text
populations = n
mass/phi/type = n
density = n
phi_tmp = mass_tmp / density_n
```

P2 不等待 streamed collector `rho*`，也不读取未来 `state_out.density`。

### 2.3 interface-gas logical population

```text
interface-gas mass flux = 0
```

因此 P2 不读取 gas persistent populations。gas-to-interface LBM boundary 的 Eq. (11)
由 P3 单独实现。

## 3. 实现结果

新增：

- `VofMassScheme.FSLBM_NEIGHBOR`；
- `LbmModel.vof_mass_scheme` 规范化与未知值拒绝；
- `VofMassTransport` solver-owned scratch；
- FullF D3Q19 fixed-topology Warp kernel；
- `LbmSolver.compute_vof_mass_transport()` 独立调度；
- P2 独立 NumPy oracle 和针对性测试。

transport 输入只读：

```text
f_post
density
vof.mass
vof.phi
vof.cell_type
periodicity
```

输出仅为：

```text
mass_tmp
phi_tmp
mass_delta
```

没有新增持久 VOF 字段。

## 4. 物理与数值不变量

已经由测试锁定：

- pull 来源为 `x-c_q`；
- outgoing population 使用 `opposite(q)`；
- rest population 不参与质量交换；
- active-active 格链权重两端对称；
- 周期域全局 `sum(mass_delta)` 为零；
- non-periodic out-of-domain mass flux 为零；
- gas populations 任意改变不影响 active mass transport；
- `mass_tmp = mass_n + mass_delta`；
- `phi_tmp = mass_tmp / density_n`；
- transport 不修改任何输入数组。

P2 不执行：

- clamp；
- cell type 变化；
- topology repair；
- redistribution；
- surface completion；
- geometry 或 surface tension。

## 5. 验收记录

验收日期：2026-07-30。

基础 HEAD：

```text
12d74bc — docs(lbm): update formula format
```

P2 targeted：

```text
Ran 8 tests
OK
```

P1：

```text
Ran 18 tests
OK
```

当前 P0/核心与 directional 回归命令实际覆盖：

```text
45 core tests
6 directional tests
```

合并运行：

```text
Ran 77 tests
OK
```

代码质量：

```text
Ruff F/I: passed
git diff --check: passed
```

设备：

```text
CPU: accepted
CUDA: not accepted（当前 Warp 构建没有 CUDA）
```

## 6. 主要测试证据

`newton/tests/test_lbm_vof_p2.py` 覆盖：

- scheme 默认值和非法值；
- HOME P2 fail-fast；
- 完整 VOF step 仍在写入前 fail-fast；
- L-L、L-I、I-L、I-I、I-G、G-I 手算；
- rest direction；
- non-periodic impermeable link；
- 周期随机场与独立 NumPy oracle；
- 全局质量守恒；
- gas storage independence；
- 输入无副作用；
- provisional phi 使用 `density^n`。

## 7. 已知边界

P2 不能用于声称：

```text
自由表面 LBM 边界已经正确
authoritative VOF 可以完整 step
界面可以跨网格移动
mass 越界会被修复
HOME transport 已支持
CUDA 已验收
```

`phi_tmp` 是 provisional scratch。最终 `phi^{n+1}` 在 P3/P4 完整调度中必须和选定
的最终 density 时间语义重新闭合。

## 8. P3 交接

P3 可以依赖：

```text
P1 canonical mass/phi/type
P2 fslbm_neighbor mass scratch
P2 interface-gas zero mass flux
现有 FullF pull streaming
```

P3 必须新增：

```text
fixed atmospheric pressure
gamma = 0
Eq.11 gas-to-interface reconstruction
surface link completion before collector
fixed-topology full step
```

P3 不得修改 P2 mass scheme，也不得为方便 reconstruction 重新读取 gas kinetic
storage。

## 9. Phase 提交

P2 的代码、测试、Plan、完成总结和架构差异说明位于同一个 P2 单一提交中。提交
身份以包含本文件的 `git log` 记录为准；不在提交内容中嵌入不可自引用的自身 SHA。
