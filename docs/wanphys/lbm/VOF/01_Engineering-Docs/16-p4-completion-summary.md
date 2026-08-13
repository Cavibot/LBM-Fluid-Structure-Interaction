# P4 类型转换与守恒质量重分配完成总结

## 1. 结论

P4 已达到：

```text
CPU_ACCEPTED
```

本阶段实现了 FullF authoritative VOF 的类型候选、确定性界面层修复、clamp、
positive/negative excess 等权重分配和最终质量闭合。transition 的所有中间数组由
solver 持有，不进入 persistent state。

P4 已能完整计算包含 GAS→INTERFACE 的最终 VOF scratch，并给出精确
`new_interface` mask；但只要该 mask 非空，domain step 会在 current-buffer swap
前明确等待 P5。P4 不会把 gas 中无效 kinetic 假装成新界面的合法状态。

## 2. 冻结的决策

### 2.1 阈值

```text
vof_transition_epsilon = 1e-4

phi_pre >= 1 + epsilon -> proposed LIQUID
phi_pre <= 0 - epsilon -> proposed GAS
otherwise              -> proposed INTERFACE
```

比较符与论文 PDF 第 6 页一致，等号包含在转换分支内。

### 2.2 拓扑优先级

采用 Home-FSLBM `surface_1` 先于 `surface_2` 的语义，并改写为只读 gather：

```text
I->L 邻接旧 GAS -> GAS 提升为 INTERFACE
I->G 邻接旧 LIQUID -> LIQUID 降为 INTERFACE
相邻 I->L / I->G -> I->L 优先，I->G 取消为 INTERFACE
```

输出通过完整 D3Q19 topology validator，不能存在 LIQUID–GAS 直接格链。

### 2.3 mass clamp

```text
final LIQUID:   mass_base = density_out
final GAS:      mass_base = 0
final INTERFACE mass_base = clamp(mass_pre, 0, density_out)
excess = mass_pre - mass_base
```

最终 LIQUID 若只有 `2e-6` 内 float32 roundoff，会保留原 mass 并由
`phi=mass/density` 闭合，避免静态自由面每步制造虚假 redistribution。真正的类型
转换至少越过 `epsilon=1e-4`，不会被该 roundoff 规则吞掉。

### 2.4 redistribution

```text
receivers = final INTERFACE cells on D3Q19 moving links
share = sender excess / receiver_count
receiver gathers neighbor shares in q=1..18 order
```

不使用 scatter、atomic 或邻居原地写。正 excess 与负 deficit 使用同一有符号公式。

若 `abs(excess)>2e-6` 且没有最终 INTERFACE receiver，显式失败；不丢弃质量、不写
隐藏 side buffer，也不改用全域非局部分配。

接收后的 INTERFACE 可暂时越过 `[0,1]`，下一步由相同 threshold 继续转换。这对应
参考算法“本步产生 excess、后续接收后再分类”的离散时序，同时保持
`mass_final=density_out*phi_final`。

## 3. 实现结果

新增：

- `vof_transition_epsilon` 配置与校验；
- `VofTopologyTransition` 和 `VofTransitionResult`；
- 四个明确 pass：
  1. proposal；
  2. topology gather；
  3. clamp/excess/share；
  4. redistribution gather；
- `proposed_type/final_type/mass_base/excess/share/receiver_count` 等
  solver-owned scratch；
- `new_interface/retired_active/changed` 三类 mask；
- host invariant validator；
- 独立 NumPy oracle 与 P4 定向测试。

P2 输入 validator 被扩展为接受 P4 合法的 transient INTERFACE overshoot，但仍要求：

```text
all mass/phi finite
mass ~= density*phi
GAS mass=phi=0
LIQUID phi≈1
no direct L-G link
```

## 4. Solver 集成与事务性

P4 后的 FullF step：

```text
P2 mass_pre
-> P3 stream/surface/collision
-> restore old GAS storage
-> P4 transition scratch
-> validate topology/masks/mass
-> if new_interface: P5 gate, no domain swap
-> otherwise commit VOF state_out and swap
```

以下情形现在可直接提交：

- 无类型变化；
- 不创建新 active kinetic 的类型变化；
- 例如全 interface 域中的 `I→L`，或 `I→G` 加旧 active `L→I` halo。

产生 `GAS→INTERFACE` 时，VOF scratch 已完成但 kinetic 尚无权提交。

## 5. 验收记录

验收日期：2026-07-30。

基础 HEAD：

```text
65408fb — feat(lbm): complete P3 VOF surface boundary
```

P4 targeted：

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
total:         90

Ran 90 tests
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

## 6. 测试证据

`newton/tests/test_lbm_vof_p4.py` 锁定：

- epsilon 默认值、非法值和阈值等号；
- `I→L` gas halo；
- `I→G` liquid halo；
- opposing candidate 的 `I→L` 优先级；
- periodic seam；
- positive/negative 单 sender 手算；
- 多 sender/multi receiver 独立 NumPy oracle；
- 生产输入数组无副作用；
- 重复运行逐位一致；
- global mass closure；
- zero-receiver fail-fast；
- 无新界面时 commit/swap；
- 有新界面时 P5 gate/no-swap。

P3 平面静止 12 步固定点在 P4 roundoff 规则下继续通过。

## 7. 已知边界

P4 不能用于声称：

```text
GAS->INTERFACE kinetic 已合法
任意 moving interface domain step 都能提交
receiver capacity constrained redistribution 已实现
PLIC/curvature/surface tension 已支持
HOME 或 CUDA 已验收
```

equal-share 接收可能使 interface 再次越界；这是显式的下一步转换输入，不是 silent
clamp。P7 长时间 benchmark 必须监测 phi extrema、transition count 和质量误差。

## 8. P5 交接

P5 可以直接读取同一 solver-owned transition result：

```text
final_type
mass_final
phi_final
new_interface
retired_active
changed
```

P5 必须冻结：

```text
new-interface neighbor eligibility
rho/u averaging weights and time layer
FullF equilibrium initialization
zero-valid-neighbor behavior
force/macroscopic field initialization
retired GAS invalidation policy
kinetic write ordering before VOF commit
```

## 9. Phase 提交

P4 的代码、测试、Plan、完成总结和架构差异说明位于同一个 P4 单一提交中。提交
身份以包含本文件的 `git log` 记录为准；不在提交内容中嵌入不可自引用的自身 SHA。
