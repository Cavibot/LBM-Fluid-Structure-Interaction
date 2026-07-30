# P6 PLIC、曲率与表面张力完成总结

## 1. 结论

P6 已达到：

```text
CPU_ACCEPTED
```

FullF authoritative VOF 现在持有与 `mass/phi/cell_type` 同 epoch 的 normal、PLIC
offset 和 mean curvature，并在下一步自由面 population completion 中按论文
Eq. (12) 使用常数 atmosphere pressure 与 surface tension。

P6 没有改变 P2 的格链质量输运器。PLIC 是只读派生几何，不是第二套质量平流。

## 2. 冻结的几何契约

### 2.1 normal

Parker–Youngs 3×3×3 weighted Sobel：

```text
normal = -grad(phi) / |grad(phi)|
```

方向为液体到气体。periodic 轴 wrap，static domain 边缘 clamp；退化梯度为零向量。

### 2.2 PLIC

以 cell center 为原点的单位立方体：

```text
r in [-0.5,0.5]^3
liquid side: dot(normal,r) <= plic_offset
```

固定轮数二分反演精确 unit-cube inclusion–exclusion volume。轴对齐时：

```text
plic_offset = phi - 0.5
```

近零 normal component 采用 `1e-4` 几何退化阈值，避免 float32 inclusion–exclusion
在近降维平面上的灾难性消减。该阈值在回归前冻结；随机法向的 volume closure 容差
为 `2e-5`。

### 2.3 curvature

首版 estimator：

```text
kappa = -0.5 * div(normal)
kappa in [-1,1] lattice units
```

normal 是 PLIC 平面的定向 normal。因此：

```text
plane: kappa=0
convex liquid sphere: kappa=-1/R
```

半径 4、8、12 lattice cell 的球面测试验证 mean-curvature 误差随半径下降。

## 3. Eq. (12)

对 INTERFACE target：

```text
rho_g = (p_atmos - 2*gamma*kappa) / cs2
cs2 = 1/3
```

凸液滴 `kappa=-1/R`，所以边界 density 增量为 `+6*gamma/R`，符号对应正 Laplace
压差。

gamma 精确为零时走 P3 独立分支：

```text
rho_g = p_atmos/cs2
```

修改曲率不会改变 gamma=0 的 completed populations，回归为 bitwise equal。

`rho_g` 若非有限或不大于零，在 surface kernel 写入前失败；不做 limiter/clamp。
surface tension 没有作为 CSF force 再施加一次。

## 4. 状态与时序

`VofGridState` 现在分为：

```text
conservative: mass / phi / cell_type
derived:      normal / plic_offset / curvature
metadata:     epoch / geometry_epoch
```

生命周期：

```text
construct/clear -> (-1,-1)
initialize      -> (0,0)
each commit     -> (n+1,-1)
geometry build -> (n+1,n+1)
```

Eq.12 preflight 要求 `geometry_epoch == epoch`。clone/copy 复制数值与 epoch，但数组
存储独立。

调度固定为：

```text
state_in geometry
-> Eq.11/Eq.12 surface completion
-> stream/collide
-> P2 mass
-> P4 topology
-> P5 new kinetic
-> VOF commit/epoch++
-> P6 final geometry
-> domain swap
```

## 5. 验收证据

验收日期：2026-07-30。

基础 HEAD：

```text
729513f — feat(lbm): complete P5 VOF kinetic initialization
```

P6 targeted：

```text
Ran 11 tests
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
P6:            11
total:        111

Ran 111 tests
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

## 6. P6 测试覆盖

`newton/tests/test_lbm_vof_p6.py` 覆盖：

- geometry state/clone/clear/epoch lifecycle；
- stale geometry transaction rejection；
- axis plane normal、解析 PLIC offset 与零 curvature；
- isolated degenerate interface canonical zero；
- 96 组随机 normal/fill unit-cube volume closure；
- convex sphere curvature 负号和 lattice-radius convergence；
- Eq.12 单链接独立手算；
- convex Laplace density increment；
- gamma=0 bitwise regression；
- non-positive `rho_g` pre-write failure；
- nonzero gamma 平面多步质量守恒。

P4/P5 的 synthetic transition fixture 现在在修改 `phi` 后显式重建 geometry，避免
以过期派生状态测试后续阶段。

## 7. 已知边界

P6 不能用于声称：

```text
HOME VOF 已支持
CUDA 已验收
open VOF boundary 已支持
静态液滴长期伪流已达到最终阈值
bubble pressure / foam / solid coupling 已支持
```

当前 curvature 是 Parker–Youngs normal divergence 首版，不是参考库的五参数
Monge-patch least-squares estimator。球面几何和由 Eq.12 得到的 Laplace boundary
density 已验证收敛；完整静态液滴动力学、伪流与长时质量账本进入 P7 综合验收。

## 8. P7 交接

P7 可以依赖：

```text
每个 committed state 的 geometry current
FullF gamma>=0 surface completion
P0-P6 111-test CPU baseline
```

P7 编码前仍需冻结：

```text
HOME logical post-collision population provider
HOME new-interface kinetic encoding
cross-encoding tolerances
scenario mass/pressure/velocity thresholds
CUDA availability and skip/accept policy
```

## 9. Phase 提交

P6 的代码、测试、Plan、完成总结和架构差异说明位于同一个 P6 唯一提交中。提交身份
以包含本文件的 `git log` 记录为准。
