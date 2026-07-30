# P5 新界面 kinetic 初始化完成总结

## 1. 结论

P5 已达到：

```text
CPU_ACCEPTED
```

FullF authoritative VOF 现在可以提交包含 GAS→INTERFACE 的 gamma=0 moving-interface
step。P4 `new_interface` mask 被 P5 消费，每个新界面在 VOF commit 前获得合法的
D3Q19 equilibrium populations、density、velocity、gravity force 和重新闭合的
phi。

P5 不增加质量：`mass_final` 完全只读。kinetic 初始化改变新 active 格点的 density
时，只通过 `phi=mass/density` 更新派生占据率。

## 2. 冻结的 donor 契约

对新界面 `x`，D3Q19 邻格 `y` 只有同时满足：

```text
type_n[y] != GAS
final_type[y] != GAS
```

才可成为 donor。

因此 donor 包含：

- unchanged LIQUID/INTERFACE；
- INTERFACE→LIQUID；
- LIQUID→INTERFACE。

明确排除：

- old GAS；
- 同批 GAS→INTERFACE；
- INTERFACE→GAS。

该资格矩阵锁定了 gas storage 无效语义，并避免同批新格点沿 GPU 调度顺序传播未初始
化值。

## 3. 时间层、权重与 kinetic

读取 P3 collision 后的候选 `state_out`：

```text
rho_init = arithmetic mean(donor density_out)
u_init   = arithmetic mean(donor velocity_out)
```

使用统一 D3Q19 二阶 equilibrium：

```text
f_post_out[q] = feq_q(rho_init, u_init), q=0..18
```

同时写：

```text
density_out = rho_init
velocity_out = u_init
force_out = rho_init * gravity
phi_final = mass_final / rho_init
```

P5 不额外加入 half-force correction，也不使用 normal weighting。前者与现有 uniform
equilibrium 初始化和 P3 velocity 语义一致；重力场的初始化动量误差由 P7 综合基准
量化。

## 4. 两段式事务边界

### prepare

只写 solver scratch：

```text
donor_count
rho_init
ux_init / uy_init / uz_init
```

在 kinetic 写入前验证：

- `new_interface` 必须精确等于 old GAS→final INTERFACE；
- donor count 非零；
- rho finite/positive；
- velocity finite 且满足 low-Mach limit；
- equilibrium finite；
- positivity 开启时满足 population floor。

### initialize

prepare 成功后只写 new-interface kinetic/macro、gravity force 和 P4 `phi_final`。
随后验证 19 populations、宏观矩、force 和 mass/phi 闭合，再提交
`state_out.vof`。

任何失败都发生在 domain current-buffer swap 之前。

## 5. 实现结果

新增：

- `VofKineticInitializer`；
- `VofKineticInitializationResult`；
- donor averaging Warp kernel；
- FullF equilibrium initialization Warp kernel；
- prepare/output host validators；
- P5 donor、equilibrium、隔离与 moving-interface 集成测试。

solver 更新：

```text
P4 transition
-> P5 prepare
-> P5 initialize/validate
-> P4 VOF commit
-> refresh MAC face velocities
-> domain swap
```

P4 的 P5 gate 已被安全移除。

## 6. 验收记录

验收日期：2026-07-30。

基础 HEAD：

```text
1f4134d — feat(lbm): complete P4 VOF topology transitions
```

P5 targeted：

```text
Ran 10 tests
OK
```

组合回归：

```text
P0 core:       39
directional:    6
P1:            18
P2:             8
P3:             9
P4:            10
P5:            10
total:        100

Ran 100 tests
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

## 7. 测试证据

`newton/tests/test_lbm_vof_p5.py` 覆盖：

- old/final type donor 资格矩阵；
- uniform rho/u mean 手算；
- periodic seam；
- 同批 new interface 与 gas macro 垃圾隔离；
- zero donor 在 kinetic write 前失败；
- P5 失败不交换或修改 current state；
- 19 个 equilibrium population 独立 NumPy 公式；
- density、momentum、force 与 phi 闭合；
- non-new kinetic/macro bitwise unchanged；
- positive front commit；
- new interface 下一步继续 stream/collide；
- create/retire 序列 finite 且质量闭合。

P0-P4 的 90 项既有验收继续全部通过。

## 8. 已知边界

P5 不能用于声称：

```text
surface tension 已支持
PLIC/curvature 已支持
重力下新界面动量严格守恒
HOME kinetic 初始化已支持
CUDA 已验收
open VOF boundary 已支持
```

P5 采用 equilibrium projection，会抹去新格点未定义的非平衡应力；这是首版稳定、可
审计策略。FullF/HOME 差分和高阶 stress 初始化留给 P7。

## 9. P6 交接

P6 可以依赖每个 final active cell 均具有合法 kinetic，并从最终
`mass/phi/cell_type` 构造 authoritative geometry。

P6 编码前必须冻结：

```text
normal direction and stencil
PLIC plane equation and offset convention
volume inversion algorithm/tolerances
curvature estimator and sign
geometry storage/epoch semantics
rho_g positivity policy for p_atmos-2*gamma*kappa
surface pressure timing
```

## 10. Phase 提交

P5 的代码、测试、Plan、完成总结和架构差异说明位于同一个 P5 单一提交中。提交
身份以包含本文件的 `git log` 记录为准；不在提交内容中嵌入不可自引用的自身 SHA。
