# LBM 核心模块审计报告：优化空间、改进方向与潜在 Bug

> 审计日期：2026-06-28
> 审计范围：`wanphys/_src/fluid/fluid_grid/lbm/` 基础模块（排除流固耦合 `coupling/` 与多相流耦合交互）
> 审计依据：对 `solver.py`、`kernels.py`、`model.py`、`state.py`、`constants.py`、`domain.py`、`base.py` 的逐行阅读

---

## A. 确认的 Bug（基本模块内）

### Bug 1 严重：Guo 体力在碰撞**前**施加 — 符号反转、量级错误

**位置**：`solver.py:216` → `kernels.py:953-1013` (`apply_guo_force_kernel`)

**问题链路**：

```
step() 流水线：
  ① compute_moments   → 从 f 计算 ρ, u
  ② apply_guo_force   → 对 state_in.f 原位加 Δf = (1-ω/2)·w·3·(c·F)   ← 碰撞前
  ③ collide_stream    → 从 state_in.f（已加力）碰撞-迁移，写出 state_out.f
```

Guo 格式的 `(1-ω/2)` 系数是为**碰撞后**（post-collision）添加设计的。当前在碰撞前加到 `f` 上，碰撞算子会将该扰动向平衡态松弛，净效果：

$$\Delta f_{\text{有效}} = (1-\omega)\cdot(1-\tfrac{\omega}{2})\cdot w_d \cdot 3\,(c_d \cdot F)$$

正确值应为：

$$\Delta f_{\text{正确}} = (1-\tfrac{\omega}{2})\cdot w_d \cdot 3\,(c_d \cdot F)$$

对于典型参数 $\tau=0.55$（$\omega \approx 1.818$）：
- $(1-\omega) = -0.818$ → **符号反转**
- $(1-\omega/2) = 0.091$
- 有效系数 $= -0.818 \times 0.091 = -0.0745$（应为 $+0.091$）

**后果**：gravity-only 路径（`G=0` 且 `gravity≠0`）下，重力以**反方向**施加。`_diag_guo.py` 脚本的存在说明此问题已被怀疑但未修复。

**修复方向**：
1. 将 `apply_guo_force_kernel` 改为对 `state_out.f`（post-collision）施加，或
2. 统一为 velocity-shift 方案（与 Shan-Chen 路径一致）：`u_eq = u + τ·g`，消除路径分裂。

---

### Bug 2 中等：Shan-Chen 速度偏移覆盖了输出速度场

**位置**：`solver.py:195`（`apply_velocity_shift_kernel`）→ `solver.py:283-291`（copy to state_out）

**问题链路**：

```python
# step() 中 Shan-Chen 路径：
apply_velocity_shift_kernel  # ux[i,j,k] += τ·(F/ρ + g)   ← 原位修改 self._ux
collide_stream_bounceback    # 使用偏移后的 ux 做平衡态   ← 正确
wp.copy(state_out.velocity_x, self._ux)  # 输出偏移后速度 ← 错误
moments_to_mac_u_kernel(self._ux, ...)   # MAC 面速度      ← 错误
```

`self._ux` 在 `apply_velocity_shift_kernel` 后存储的是**平衡态速度** $u_{eq} = u + \tau(F/\rho + g)$，而非物理速度 $u$。step 6 和 7 直接将其拷贝到 `state_out.velocity_*` 和 MAC 面速度。

**后果**：
- 输出速度场偏大：包含 $\tau \cdot F/\rho$ 附加项
- 对可视化、刚体耦合、边界条件判断产生误差
- 强 Shan-Chen 力或大重力下偏差显著

**修复**：在 velocity shift 前保存物理速度，或在 step 6 之前从 `self._ux` 减回偏移量：
```python
# 在 collide_stream 之后、copy 之前：
u_eq_shift = tau * (F/rho + g)
state_out.velocity_x = self._ux - shift  # 恢复物理速度
```

---

### Bug 3 中等：TRT 模式下 velocity shift 使用 τ⁻ 而非 τ⁺

**位置**：`solver.py:195` → `kernels.py:1442`

`apply_velocity_shift_kernel` 使用 `self.model.tau`（= τ₋ = τ）做偏移。在 BGK 模式（`lambda_trt=0`）下 τ₊ = τ₋ = τ，无误差。但在 TRT 模式下，平衡态速度偏移应使用 τ₊（能量弛豫时间）：

$$u_{eq} = u + \tau_+ \cdot (F/\rho + g)$$

当前代码：
```python
ux[i,j,k] = ux[i,j,k] + tau * (fx[i,j,k] * inv_rho + gx)  # tau = tau_minus
```

**后果**：TRT 模式下体力施加精度下降，可能影响 Poiseuille 流等精确解的二阶精度。

**修复**：传入 `omega_plus` 对应的 τ₊，或在 BGK 模式下退化为 τ。

---

### Bug 4 中等：MAC 壁面速度用单元中心值而非壁面速度

**位置**：`kernels.py:1452-1506` (`moments_to_mac_u/v/w_kernel`)

```python
if i == 0:
    vel_u[i, j, k] = ux[0, j, k]   # 直接复制边界单元中心速度
elif i == nx:
    vel_u[i, j, k] = ux[nx - 1, j, k]
```

对于 bounce-back no-slip 壁面，MAC 面速度（位于壁面上）应为 **0**（或壁面运动速度）。当前代码复制的是边界单元中心速度——该值在 halfway bounce-back 下约为来流速度的一半，而非 0。

**后果**：刚体-流体耦合时，壁面附近的速度场不准确，可能导致耦合力计算偏差。

---

### Bug 5 低：Zou-He 边界不更新 ρ/u 宏观场

**位置**：`kernels.py:1571-1928` (`apply_boundary_conditions_kernel`)

Zou-He 入口计算了 `rho_w` 并修正了分布函数 `f`，但**没有回写** `rho[i,j,k]`、`ux[i,j,k]` 等宏观场数组。step 6 输出的 `state_out.density/velocity_*` 是 step 1 从 `state_in.f` 计算的旧值，与 Zou-He 修正后的 `f` 不一致。

**后果**：边界单元的输出宏观量与实际分布函数不一致。下一步的 `compute_moments` 会从修正后的 `f` 重新计算，因此不影响模拟稳定性，但影响可视化/分析/耦合的瞬时准确性。

---

### Bug 6 低：负密度保护不完善

**位置**：`kernels.py:201`

```python
r = wp.max(r, 1.0e-12)
```

当密度为负时（强非物理扰动下可能发生），`wp.max(负数, 1e-12)` 返回 `1e-12`，但动量 `mx` 未修正，导致 `ux = mx / 1e-12` 可能极大，引发数值爆炸。

**修复**：`r = wp.max(wp.abs(r), 1.0e-12)` 或对负密度做更安全的处理（如标记为固体、跳过碰撞）。

---

### Bug 7 低：gravity-only 路径不输出力场

**位置**：`solver.py:285-288`

```python
if G_sc != 0.0 or sc_wall_G != 0.0:
    wp.copy(state_out.force_x, self._fx)  # 仅 Shan-Chen 路径输出
```

gravity-only 路径下 `self._fx` 等保持为零，`state_out.force_*` 未被赋值（保持上一步的值或初始零），用户无法检查实际施加的体力。

---

## B. 优化与提升机会

### OPT-1：融合 compute_moments + collide_stream（减半显存带宽）

`compute_moments_kernel` 和 `collide_stream_bounceback_kernel` 都需要读取全部 19 个分布函数。当前分两步，每个 cell 读 19 次 `f` × 2 pass = 38 次全局存储访问。

**方案**：改为 **push scheme** — 在源 cell 计算宏观量 + 碰撞，然后 push 到邻居。消除独立的 moments pass。或保留 pull scheme 但用 shared memory 预加载 block 内的 `f` 数据。

### OPT-2：共享内存加速 collide-stream

`collide_stream_bounceback_kernel` 每个线程从 19 个邻居 cell 读取 `f`，全局存储访问极度分散。用 CUDA shared memory 加载一个 tile 的数据可显著减少全局存储流量。对 64³ 网格，预期 2-3× 加速。

### OPT-3：对流出口的 `for d in range(19)` 循环展开

**位置**：`kernels.py:1677, 1728, 1777, 1827, 1877`

```python
for d in range(19):
    f[d * stride + idx] = f[d * stride + src_idx]
```

Warp 可能无法完全展开动态循环。改为 19 条显式赋值语句（如其他 kernel 的写法）可确保编译器优化。

### OPT-4：减少 kernel launch 开销

每步 launch 5-8 个 kernel。`compute_pressure_kernel` 和 3 个 MAC 插值 kernel 可以合并为一个 fused post-process kernel，减少 3 次 launch。

### OPT-5：MRT kernel 已定义但未接入

**位置**：`kernels.py:2207-2387` (`mrt_collide_kernel`)

MRT 碰撞算子代码完整存在（19×19 矩阵变换 + 松弛 + 逆变换），但 `solver.py` 从未调用。这是**死代码**——未测试、未使用。应：
1. 接入 solver pipeline（作为 BGK/TRT 之外的第三种碰撞模式），或
2. 删除以减少维护负担。

MRT 对多孔介质流、非牛顿流体、高 Re 流的稳定性优于 BGK/TRT，接入有实际价值。

### OPT-6：双精度选项

所有数组用 `dtype=float`（Warp 中通常为 float32）。长时间运行 LBM 模拟在单精度下会产生质量漂移。添加 `dtype=wp.float64` 选项用于长时模拟/高精度验证场景。

### OPT-7：质量守恒监控与校正

无质量守恒诊断。LBM 在 Zou-He 边界、Shan-Chen 多相下容易发生质量漂移。建议：
- 每步/每 N 步计算全局 ∑ρ 并记录
- 提供密度重缩放选项（mass correction）

### OPT-8：内存优化 — 单缓冲 f

当前双缓冲 `state_in.f` + `state_out.f` 各需 `19 × N × 4` 字节。对 128³ 网格，`f` 数组占 `2 × 19 × 128³ × 4 ≈ 317 MB`。可用原地碰撞-推流方案（careful dependency management）减半 `f` 内存。

### OPT-9：Checkpoint/重启机制

`LbmState` 无序列化/反序列化接口。长模拟需要断点续算能力。建议添加 `save(path)` / `load(path)` 方法。

### OPT-10：子网格湍流模型（LES-LBM）

当前 BGK/TRT 用常数弛豫时间，无法处理高 Re 湍流。Smagorinsky-LBM（根据局部应变率自适应调整 τ）可将适用 Re 范围提升 1-2 个量级。

---

## C. 研究方向

### RES-1：Entropic LBM (ELBM)

用 H 函数自适应调整弛豫参数，可在任意低粘度下保持稳定。对高 Re 自由剪切流、空气动力学场景有重要价值。

### RES-2：Cumulant 碰撞算子

Cumulant LBM 在矩空间用累积量代替矩，对 Galilean 不变性更好，适合高 Re + 曲线边界场景。性能优于 MRT，稳定性优于 BGK/TRT。

### RES-3：曲线边界插值 bounce-back (BMF/IBM)

当前 halfway bounce-back 对曲线边界仅一阶精度。Bouzidi-Minexaou-Firdaouss (BMF) 插值 bounce-back 或 Immersed Boundary Method 可提升到二阶，显著改善球体/复杂几何绕流精度。

### RES-4：非牛顿流体模型

当前假设牛顿流体（常数 τ）。支持 power-law / Bingham / Carreau 模型只需将 τ 改为空间变化的 τ(γ̇)，在碰撞前根据局部应变率计算有效粘度。

### RES-5：热 LBM

当前为等温模型。双分布函数 (DDF) 热 LBM 或 Guo 热模型可扩展到温度场耦合，支撑自然对流、热传导等场景。

### RES-6：多块/自适应网格细化 (AMR)

均匀网格对边界层分辨率不足。多块 LBM（overlapping grids with buffer zone）或嵌套网格细化可将边界层计算成本降低一个量级。

### RES-7：D3Q27 格式

当前 D3Q19 缺少面对角方向 (±1,±1,±1)，在强各向异性场景（如高 Mach 数、复杂几何）下精度不足。D3Q27 提供更好的各向同性，但需 42% 更多内存和计算量。可通过条件编译支持两种格式切换。

---

## D. 问题优先级矩阵

| 编号 | 类型 | 严重性 | 影响场景 | 修复难度 |
|------|------|--------|----------|----------|
| Bug 1 | Guo力符号反转 | **严重** | 所有gravity-only单相流 | 低 |
| Bug 2 | 输出速度=平衡速度 | 中等 | 所有SC多相流 | 低 |
| Bug 3 | TRT velocity shift用τ⁻ | 中等 | TRT模式+体力 | 低 |
| Bug 4 | MAC壁面速度错误 | 中等 | 耦合/壁面分析 | 低 |
| Bug 5 | Zou-He不更新宏观量 | 低 | 可视化/耦合 | 低 |
| Bug 6 | 负密度保护不完善 | 低 | 极端扰动 | 低 |
| Bug 7 | gravity-only不输出力场 | 低 | 调试/可视化 | 低 |
| OPT-1 | 融合moments+collide | — | 性能2-3× | 中 |
| OPT-2 | 共享内存加速 | — | 性能2-3× | 中 |
| OPT-5 | MRT死代码 | — | 功能/维护 | 中 |
| RES-3 | 曲线边界IBM | — | 精度 | 高 |

---

## E. 关键结论

最关键的问题是 **Bug 1（Guo 体力符号反转，使 gravity-only 路径重力方向相反）** 和 **Bug 2（Shan-Chen 路径输出速度包含力偏移）**。两者都出现在 forcing 路径分裂的设计中：

- gravity-only 路径：Guo 力法（pre-collision，且当前实现错误）
- Shan-Chen 路径：velocity-shift 法（post-moments）

**长期建议**：统一为单一 velocity-shift forcing scheme，消除路径分裂。这样：
1. 修复 Bug 1（不再有 Guo pre-collision 路径）
2. 简化 Bug 2 的修复（统一保存/恢复物理速度的逻辑）
3. 修复 Bug 3（统一使用 τ₊）
4. 减少 solver.step() 中的条件分支，降低维护复杂度

短期可先做最小修复：将 `apply_guo_force_kernel` 改为对 `state_out.f`（post-collision）施加，并修正 Bug 2 的速度回写逻辑。
