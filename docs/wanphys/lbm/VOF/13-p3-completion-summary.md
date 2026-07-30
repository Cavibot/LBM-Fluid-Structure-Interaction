# P3 零表面张力自由面边界完成总结

## 1. 结论

P3 已达到：

```text
CPU_ACCEPTED
```

FullF authoritative VOF 现在可以在固定拓扑、固定大气压力和零表面张力条件下执行
完整 `LbmDomain.step()`。P2 质量输运、Eq. (11) surface completion、普通 LBM
collector/collision 和最终 VOF 提交已经形成一条明确的时间步流水线。

当 provisional mass 要求 `INTERFACE` 变为 `LIQUID/GAS` 时，P3 不做 clamp，也不
静默丢弃质量，而是在交换 domain 当前缓冲区前失败；类型转换从 P4 开始。

## 2. 冻结的物理与时间语义

### 2.1 固定气相压力

```text
vof_atmosphere_pressure = c_s^2 = 1/3
vof_surface_tension = 0
rho_g = p_atmos / c_s^2 = 1
```

`vof_atmosphere_pressure` 必须有限且大于零。非零表面张力被明确拒绝到 P6。

### 2.2 Eq. (11) 的 pull 方向

对目标 interface 格点 `x` 和方向 `q`：

```text
source = x - c_q
source type = GAS
write f_star[q, x]
read f_post[opposite(q), x]

f_star[q, x]
  = feq[q](rho_g, u_n[x])
  + feq[opposite(q)](rho_g, u_n[x])
  - f_post[opposite(q), x]
```

surface stage 读取 `state_in(n)` 的 type、velocity 和对向 population；它不读取
gas persistent kinetic storage，也不重复施加 half-force velocity correction。

### 2.3 P3 step 顺序

```text
1. P2 mass transport -> solver scratch
2. ordinary FullF pull streaming -> f_star
3. Eq. (11) overwrite gas-to-interface links
4. ordinary domain boundary / collector / force / collision
5. write state_out macroscopic fields
6. restore semantically invalid GAS storage
7. commit mass_tmp and derive phi using density_out
8. validate fixed topology
9. successful return lets LbmDomain swap buffers
```

P2 的 provisional `phi_tmp=mass_tmp/density_n` 仍用于独立 transport 观察；P3 的
持久 `phi_{n+1}` 使用最终 `density_out` 闭合。

## 3. 实现结果

新增：

- `VofSurfaceBoundary`：P3 surface stage 的调度与 gas storage 隔离；
- FullF gas-to-interface Eq. (11) Warp kernel；
- FullF GAS kinetic/macroscopic storage restore kernel；
- fixed-topology VOF finalization kernel；
- `validate_p3_fixed_topology_state()`；
- `vof_atmosphere_pressure` 和 `vof_surface_tension` 配置契约；
- P3 独立公式、隔离和集成测试。

限制被显式编码：

```text
encoding = FullF only
surface_tension = 0
domain boundary = periodic or static bounce-back
cell_type = fixed
HOME = fail-fast until P7
```

## 4. 已锁定的不变量

- 只有目标为 `INTERFACE`、来源为 `GAS` 的 moving link 被 Eq. (11) 覆盖；
- rest population 不被 surface stage 改写；
- 18 个 moving direction 和 opposite 映射与独立 NumPy 公式一致；
- 修改所有 gas populations 不改变 active streamed/collided 结果；
- surface stage 不修改 `state_in`；
- generic all-cell LBM kernel 不会让 gas storage 漂移；
- P2 `mass_tmp` 被原样提交到 active `state_out.vof.mass`；
- `phi_{n+1}=mass_{n+1}/density_{n+1}`；
- GAS 保持 `mass=phi=0`；
- LIQUID 保持 `phi≈1`，INTERFACE 必须保持 `0<phi<1`；
- P3 不改变 `cell_type`；
- 验证失败不会交换 domain 当前缓冲区。

## 5. 验收记录

验收日期：2026-07-30。

基础 HEAD：

```text
f30f1ad — feat(lbm): complete P2 VOF mass transport
```

P3 targeted：

```text
Ran 9 tests
OK
```

组合回归：

```text
P0 core:       39
directional:    6
P1:            18
P2:             8
P3:             9
total:         80

Ran 80 tests
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

平面静止自由面运行 12 步后满足预冻结的 float32 验收范围：

```text
phi absolute tolerance = 1e-5
mass comparison places = 5
max active velocity <= 1e-6
cell_type exactly unchanged
```

## 6. 已知边界

P3 不能用于声称：

```text
界面可以跨网格移动
phi 越界会被修复
excess/deficit mass 会被重分配
新 interface kinetic 已初始化
PLIC/curvature/surface tension 已支持
HOME 或 CUDA 已验收
VOF 开口 domain boundary 已支持
```

固定拓扑限制意味着带有足够界面运动的真实场景会按设计失败；这是 P4 的入口条件，
不是可长期运行的 moving-interface solver。

## 7. P4 交接

P4 可以依赖：

```text
P1 canonical authoritative state
P2 conservative mass_tmp
P3 complete FullF active kinetic step
P3 density_out-based phi finalization
P3 transaction boundary before domain buffer swap
```

P4 必须在编码前冻结：

```text
transition thresholds and exact comparison semantics
topology repair neighborhood and deterministic conflict rules
positive/negative excess definition
redistribution receiver set and weights
zero-receiver behavior
new-interface list contract handed to P5
```

## 8. Phase 提交

P3 的代码、测试、Plan、完成总结和架构差异说明位于同一个 P3 单一提交中。提交
身份以包含本文件的 `git log` 记录为准；不在提交内容中嵌入不可自引用的自身 SHA。
