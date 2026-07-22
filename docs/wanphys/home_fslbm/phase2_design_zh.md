# HOME-FSLBM 阶段 2 自由表面实施方案

> **日期**: 2026-07-20
> **范围**: VOF 锐界面自由表面追踪（`kernels_surface.py`）及 `stream_collide_bvh` 自由表面逻辑补全
> **前置**: 阶段 1（流体核心 17/17 测试通过）

---

## 1. 架构概览

阶段 2 完成自由表面子系统，包含两大部分：

| 部分 | 文件 | 内容 |
|------|------|------|
| **A. 表面辅助函数** | `kernels_surface.py` | `calculate_phi`, `calculate_normal`, `plic_cube`, `plic_cube_reduced`, `lu_solve`, `calculate_curvature` |
| **B. 表面标记 kernel** | `kernels_surface.py` | `surface_1_kernel`, `surface_2_kernel`, `surface_3_kernel` |
| **C. 流体内核补全** | `kernels_fluid.py` | `stream_collide_bvh_kernel` Phase C（曲率/气体压力）+ Phase D（质量交换）骨架→完整 |
| **D. Solver 集成** | `solver.py` | 在 Phase 2 管线中启动 surface_1/2/3 |

### 1.1 参考管线时序

```
mrSolver3DGpu() 每步调用顺序:
  1. calculate_disjoint          (Phase 3 — 分离压力)
  2. [条件] clear_inlet          (Phase 3 — 入口清除)
  3. atmosphere_rho_update       (Phase 3 — 大气压力)
  4. atmosphere_volme_update     (Phase 3 — 大气体积)
  5. stream_collide_bvh ★        (阶段 1 + 阶段 2 补全)
  6. ResetDisjoinForce           (Phase 3)
  7. surface_1 → surface_2 → surface_3  (★ 阶段 2 实现)
  8. mrSolver3D_step2Kernel      (阶段 1 — fMom 交换)
```

### 1.2 三个表面 Kernel 的职责

| Kernel | 参考行号 | 功能 |
|--------|---------|------|
| `surface_1` | `mrLbmSolverGpu3D.cu:444-477` | **标记传播**：TYPE_IF 单元阻止邻居 TYPE_IG，将邻居 TYPE_G 设为 TYPE_GI |
| `surface_2` | `mrLbmSolverGpu3D.cu:479-602` | **GI 初始化**：TYPE_GI 单元从邻居 TYPE_F/TYPE_I 平均 (ρ, u)，设为平衡态分布；TYPE_IG 单元将邻居 TYPE_F/IF 转为 TYPE_I |
| `surface_3` | `mrLbmSolverGpu3D.cu:604-701` | **质量重分配**：按 flag 类型处理 mass/phi 转换、标记过渡（IF→F、IG→G、GI→I）、多余质量分发到邻居 |

---

## 2. 逐模块详细设计

### 2.1 `kernels_surface.py` — 辅助函数

#### 2.1.1 `calculate_phi(rho, mass, flag) → float`
[SRC: `mrUtilFuncGpu3D.h:351-354`]

```python
@wp.func
def calculate_phi(rhon: float, massn: float, flagsn: int) -> float:
    if flagsn & TYPE_F:
        return 1.0
    elif flagsn & TYPE_I:
        return wp.clamp(massn / rhon, 0.0, 1.0) if rhon > 0.0 else 0.5
    else:
        return 0.0
```

#### 2.1.2 `calculate_normal(phij) → wp.vec3`
[SRC: `mrUtilFuncGpu3D.h:357-369`]

27 点加权有限差分计算 ∇φ（梯度）。D3Q27 权重：面邻居=4，边邻居=2，角邻居=1。

```
bx.x = 4*(phi[2]-phi[1]) + 2*(phi[8]-phi[7]+phi[10]-phi[9]+...) + 1*(phi[20]-phi[19]+...)
bx.y = 4*(phi[4]-phi[3]) + 2*(...)
bx.z = 4*(phi[6]-phi[5]) + 2*(...)
return normalize(bx)
```

**注**：参考代码在 `calculate_normal` 中先将 `phit` 通过 `index3dInv_gpu`（即 `OPPOSITE`）重排——因为调用方传入的 phij 数组是按邻居方向排列的，而法向计算需要按"从中心指向邻居"的方向。Warp 实现中，调用方负责按正确顺序传入 phij。

#### 2.1.3 `plic_cube_reduced(V, n1, n2, n3) → float`
[SRC: `mrUtilFuncGpu3D.h:104-119`]

SZ & Kawano 2022 方法。对称化简后的 PLIC 解析解，处理 5 种情形：
- Case (5): n12 ≤ 2·n3·V → d = n3·V + 0.5·n12
- Case (1): n3·V < n1²/(6·n2) → d = cbrt(6·n1·n2·n3·V)
- Case (2): v1 ≤ n3·V < v1 + 0.5·(n2−n1) → d = 0.5·(n1 + sqrt(n1² + 8·n2·(n3·V−v1)))
- Case (3)/(4): 三次方程 → d = c − 2·t·sin(asin(...)/3)

#### 2.1.4 `plic_cube(V0, n) → float`
[SRC: `mrUtilFuncGpu3D.h:142-149`]

单位立方体与平面相交：体积 V0∈[0,1]，法向量 n → 平面偏移 d0。
1. 对称化简：ax=|n.x|, ay=|n.y|, az=|n.z|, V = 0.5−|V0−0.5|
2. L1 归一化：l = ax+ay+az; n1=min(ax,ay,az)/l; n3=max(ax,ay,az)/l; n2=max(0,1−n1−n3)
3. 调用 plic_cube_reduced(V, n1, n2, n3) → d
4. 缩放回原坐标系：d0 = l·copysign(0.5−d, V0−0.5)

#### 2.1.5 `lu_solve(M, x, b, N, Nsol)`
[SRC: `mrUtilFuncGpu3D.h:124-140`]

就地 LU 分解求解小线性系统（5×5）。用于曲率拟合。
1. 对 i=0..Nsol-1：分解 M=L·U（就地）
2. 前代求解 L·y = b
3. 回代求解 U·x = y

#### 2.1.6 `calculate_curvature(phij) → float`
[SRC: `mrUtilFuncGpu3D.h:371-420`]

Monge 曲面片二次拟合求平均曲率 κ。

```
步骤 1: 计算法向 n = normalize(∇φ)（27 点模板）
步骤 2: 建立局部正交坐标系 {bx, by, bz=n}
步骤 3: 收集邻居接口点（0<φ<1），用 PLIC 偏移投影到局部坐标
步骤 4: 5×5 最小二乘拟合 f(x,y) = Ax²+By²+Cxy+Hx+Iy
步骤 5: 平均曲率 K = (A(1+I²)+B(1+H²)−CHI) / (1+H²+I²)^(3/2)
步骤 6: 钳制到 [-1, 1]
```

需要至少 5 个邻居接口点，否则返回 0.0。

---

### 2.2 `kernels_surface.py` — Surface Kernel

#### 2.2.1 `surface_1_kernel` — 标记传播
[SRC: `mrLbmSolverGpu3D.cu:444-477`]

```
对每个 TYPE_IF 单元:
  for di in 1..26:
    邻居 = (i-cx[di], j-cy[di], k-cz[di])
    if 邻居.flag_su == TYPE_IG:
      邻居.flag = (flag & ~TYPE_SU) | TYPE_I   // 阻止接口邻居变气体
    elif 邻居.flag_su == TYPE_G:
      邻居.flag = (flag & ~TYPE_SU) | TYPE_GI  // 气体邻居变接口
```

#### 2.2.2 `surface_2_kernel` — GI 初始化 + IG 处理
[SRC: `mrLbmSolverGpu3D.cu:479-602`]

**GI 分支**（lines 494-567）：
```
对每个 TYPE_GI 单元:
  counter=0; rhot=0; uxt,uyt,uzt=0; rho_gt=0; c_k=0
  for di in 1..26:
    邻居 = ...
    if 邻居是 TYPE_F/TYPE_I/TYPE_IF:
      counter+=1
      rhot += 邻居.ρ; uxt += 邻居.ux; ...
      if di < 7:  // D3Q7 面邻居
        rho_gt += 邻居.c_value; c_k+=1
  rhon = rhot/counter (默认 1.0)
  uxn = uxt/counter (默认 0.0)
  
  // 计算平衡态分布 → 通过求和 feq 的加权矩来重建 HOME 矩
  feq[0..26] = calculate_f_eq(rhon, uxn, uyn, uzn, ...)
  feq[i] += w3d[i]  // 回加权重偏置
  
  // 由 feq 求和计算应力张量
  pixx = Σ feq[i]·cx[i]²  (对所有 i=1..26 含 x 分量)
  pixy = Σ feq[i]·cx[i]·cy[i]
  ...
  
  // 存储 HOME 矩
  f_mom_post[0] = rhon
  f_mom_post[1..3] = uxn, uyn, uzn
  f_mom_post[4..9] = (pixx/rhon − cs²), pixy/rhon, ... 
  
  // 初始化气体浓度
  c_value = rho_gt (平均邻居 c_value)
  g_mom[0..6] = calculate_g_eq(rho_gt, uxn, uyn, uzn)
```

**IG 分支**（lines 569-600）：
```
对每个 TYPE_IG 单元:
  for di in 1..26:
    邻居 = ...
    if 邻居是 TYPE_F 或 TYPE_IF:
      if 邻居.islet == 0:
        邻居.flag = TYPE_I  // 阻止邻居变纯流体
        邻居.merge_detector = 1  // 标记需要合并检测
      else:
        当前单元.flag = TYPE_I  // 孤立气泡保留标记
```

#### 2.2.3 `surface_3_kernel` — VOF 质量重分配
[SRC: `mrLbmSolverGpu3D.cu:604-701`]

```
对每个非固体单元:
  flagsn_sus = flag & (TYPE_SU | TYPE_S)
  if flagsn_sus & TYPE_S: return
  if islet == 1:  // 孤立气泡
    previous_tag = tag; tag = -1; return
  
  rhon = f_mom_post[0]
  massn = mass[cur]; massexn = 0; phin = 0
  
  # Phase A: 按 flag 类型处理(参考 lines 631-668)
  if   TYPE_F:  massexn=massn-rhon; massn=rhon; phin=1.0; tag→previous_tag; tag=-1
  elif TYPE_I:  massexn=(massn>rhon?massn-rhon:massn<0?massn:0); massn=clamp(massn,0,rhon); phin=calculate_phi
  elif TYPE_G:  massexn=massn; massn=0; phin=0
  elif TYPE_IF: flag→TYPE_F; report_split(); massexn=massn-rhon; massn=rhon; phin=1.0
  elif TYPE_IG: flag→TYPE_G; massexn=massn; massn=0; phin=0
  elif TYPE_GI: flag→TYPE_I; massexn=(massn>rhon?massn-rhon:...); massn=clamp(...); phin=calculate_phi
  
  # Phase B: 计数流体/接口邻居(lines 669-683)
  counter = 对 27 邻居计数(TYPE_F, TYPE_I, TYPE_IF, TYPE_GI)
  
  # Phase C: 多余质量分发(lines 684-685)
  if counter == 0: massn += massexn  // 无邻居可分发时自我吸收
  massexn /= max(counter, 1)  // 均分给邻居
  
  # Phase D: 存储(lines 686-699)
  mass = massn; massex = massexn
  delta_phi = phin − phi_old
  
  # 气体浓度修正(TYPE_I)
  if (flag & TYPE_SU) == TYPE_I:
    rhon_g = Σ g_mom[cur+0..6]
    delta_g[cur] -= rhon_g * delta_phi
  
  phi = phin
```

**关键差异与 stream_collide_bvh 的 Phase C/D**：
- `surface_3` 只做 mass/phi 的 flag-based 重分配和标记过渡
- `stream_collide_bvh` 的 Phase C/D 做的是**流传输驱动的质量通量**（基于 fhn−fon 差值的 Δmass 计算）和**气体边界填充**
- 两者互补：`stream_collide_bvh` 计算通量并更新 mass，`surface_3` 校正 mass 并更新 phi/flag

---

### 2.3 `kernels_fluid.py` 补全

阶段 1 的 `stream_collide_bvh_kernel` 有完整的 Phase A/B/E/F，但 Phase C/D 有骨架占位符。

#### Phase C 补全项：
1. **曲率计算**（替换 `curv = 0.0`）：
   - 收集 27 邻居 phi 值
   - 调用 `calculate_curvature(phij)` 从 `kernels_surface`
   - 注：需要处理邻居索引（按 OPPOSITE 方向获取 phi）

2. **phi 计算**（替换 `phi[i,j,k] = 0.5`）：
   - 调用 `calculate_phi(rho_new, massn, TYPE_I)`

#### Phase D 补全项 — 质量交换：
```
for di in 1..26:
  ni = i − cx[di]; nj = j − cy[di]; nk = k − cz[di]
  nflag_su = flag[ni,nj,nk] & TYPE_SU
  if nflag_su == TYPE_F or TYPE_I:
    nphi = phi[ni,nj,nk]
    inv_di = OPPOSITE[di]
    dflux = f_streamed[di] − fon[inv_di]
    if nflag_su == TYPE_F:
      massn += dflux
    else:  # TYPE_I
      massn += 0.5 * (nphi + phi_center) * dflux
```

---

### 2.4 Solver 集成

在 `solver.py` 的 `step()` 方法中，`stream_collide_bvh_kernel` 启动之后、`swap_moments_kernel` 之前，依次启动：

```python
# ---- Surface marker propagation (Phase 2) ----
wp.launch(kernels_surface.surface_1_kernel, dim=(nx,ny,nz), inputs=[...])
wp.launch(kernels_surface.surface_2_kernel, dim=(nx,ny,nz), inputs=[...])
wp.launch(kernels_surface.surface_3_kernel, dim=(nx,ny,nz), inputs=[...])
```

---

## 3. 阶段 2 测试计划

目标：以"相同输入→相同输出"为标准，验证 Warp 实现与参考代码在自由表面组件上的一致。

### 3.1 已实现

| 测试 | 文件 | 说明 |
|------|------|------|
| `test_calculate_phi` (3 个子) | `test_kernels_surface.py` | TYPE_F→1.0, TYPE_I→mass/rho, TYPE_G→0.0 |
| `test_plic_cube_known_cases` (3 个) | `test_kernels_surface.py` | V=0.5 n=(1,0,0)→d=0; V=0.1 n=(1,0,0)→d=-0.4; V=0.5 n=(1,1,1)→d≈0 |
| `test_surface_1_flag_propagation` (G→GI) | `test_kernels_surface.py` | TYPE_IF 邻居 TYPE_G → TYPE_GI |
| `test_surface_3_type_f_mass_rho` | `test_kernels_surface.py` | mass→rho, φ→1.0, massex 均分 |
| `test_surface_3_type_i_mass_clamp` | `test_kernels_surface.py` | mass 钳制, φ=1.0 |

### 3.2 金标准回归测试

在 `export_solver_golden.cpp` 新增场景，运行若干步 `mrSolver3DGpu()`（完整管线），导出 flag/phi/mass/fMom 作为金标准。Warp 端用相同初始条件运行相同步数，逐单元对比。

#### 3.2.1 球体液滴系列（覆盖曲率计算全部半径区间）

| 场景 | 网格 | R | 步数 | 验证点 |
|------|------|---|------|--------|
| `droplet_r4` | 16³ | 4 | 5 | 小球面（曲率含 PLIC 偏置，Warp 与参考一致即可） |
| `droplet_r8` | 32³ | 8 | 10 | 中球面，充分覆盖 TYPE_I 环带 |
| `droplet_r12` | 32³ | 12 | 10 | 大球面，低曲率 |

#### 3.2.2 平面界面

| 场景 | 网格 | 初始 | 步数 | 验证点 |
|------|------|------|------|--------|
| `flat_interface` | 32³ | z<16 为 TYPE_F，其余 TYPE_G | 10 | κ≈0，界面不漂移 |

#### 3.2.3 近接双液滴

| 场景 | 网格 | 初始 | 步数 | 验证点 |
|------|------|------|------|--------|
| `near_droplets` | 32³ | 两 R=6 液滴中心距 14 | 20 | surface_1 G→GI 传播，不误合并 |

#### 3.2.4 椭球振荡

| 场景 | 网格 | 初始 | 步数 | 验证点 |
|------|------|------|------|--------|
| `ellipsoid` | 32³ | rx≠ry≠rz 椭球体液滴 | 50 | 曲率驱动向球形松弛，质量守恒 |

#### 3.2.5 重力下落液滴

| 场景 | 网格 | 初始 | 步数 | 验证点 |
|------|------|------|------|--------|
| `falling_droplet` | 32³ | R=6 液滴，gz=-0.001 | 30 | 重心下移，质量守恒 |

#### 3.2.6 物理参数变化

| 场景 | 网格 | R | ω | 步数 | 验证点 |
|------|------|---|-----|------|--------|
| `droplet_tau055` | 32³ | 8 | 1/0.55 | 20 | 低粘性振荡更持久 |
| `droplet_tau2` | 32³ | 8 | 0.5 | 20 | 高粘性快速阻尼 |

#### 3.2.7 壁面接触

| 场景 | 网格 | 初始 | 步数 | 验证点 |
|------|------|------|------|--------|
| `droplet_wall` | 32³ | R=6 液滴邻接 z=0 壁面 | 10 | 壁面反弹 + surface_1 在固体边界行为 |

### 3.3 分析解 / 手动设置验证

| 测试 | 说明 |
|------|------|
| `test_plic_cube_known_cases` 补全到 6 个 | 新增 V=0.9/n=(1,0,0), V=0.1/n=(1,1,1), V=0.9/n=(1,1,1) |
| `test_calculate_normal_from_phi` | 平面 φ 场 → 法向 = 已知 n |
| `test_surface_1_flag_propagation` IG→I | TYPE_IF 中心 + TYPE_IG 邻居 → 保留 I |
| `test_surface_2_gi_initialization` | TYPE_GI + 已知 ρ/u 的 TYPE_F 邻居 → fMom/c_value/g_mom |
| `test_surface_3_mass_exchange_conservation` | 混合 TYPE_F/I/G，跨步 Σmass 守恒 |
| `test_surface_3_gas_boundary_fill` | TYPE_I 邻 TYPE_G → f_streamed = feq(ρ_gas,u) |
| `test_surface_3_counter_zero` | 孤立 TYPE_I 无流体邻居 → mass 自我吸收 |
| `test_surface_3_flag_transitions` | IF→F, IG→G, GI→I 逐个验证 |
| `test_stream_collide_bvh_curvature_sphere` | 球面 φ 场 → κ≈2/R |
| `test_stream_collide_bvh_curvature_plane` | 平面 φ 场 → κ≈0 |
| `test_stream_collide_bvh_mass_conservation` | 随机 φ 场 + stream_collide_bvh + surface_3 → Σmass 守恒 |

### 3.4 回归：阶段 1 测试

阶段 1 的 17 个测试全部通过（`test_kernels_fluid.py`）。

---

## 4. 文件清单

```
wanphys/_src/fluid/fluid_grid/home_fslbm/
├── kernels_surface.py          # ★ 新增 ~600 行
├── kernels_fluid.py            # 修改（Phase C/D 补全）
├── solver.py                   # 修改（接入 surface kernel）
├── __init__.py                 # 修改（导出新模块）
└── tests/
    ├── conftest.py             # 不变
    ├── test_kernels_surface.py # ★ 新增 ~300 行
    ├── test_kernels_fluid.py   # 不变（Phase 1 测试）
    └── ...
```

---

## 5. 风险与注意事项

1. **Warp `@wp.func` 限制**：`calculate_curvature` 包含 5×5 LU 分解和循环收集邻居点，需要在 GPU 线程中高效执行。Warp 不支持动态数组，必须使用固定大小数组（24 点）。
2. **`copysign` / `cbrt` / `sin`/`asin`**：Warp 提供 `wp.copysign`、`wp.cbrt`、`wp.sin`、`wp.asin`。
3. **`fdimf`**：C 标准库 `fdim(x,y) = max(x-y, 0)`。Warp 使用 `max(x-y, 0.0)`。
4. **原子操作**：`surface_3` 当前无原子操作（只在邻居分发 massex 时有隐式依赖——但参考代码中 massex[i] 的读取在 `stream_collide_bvh` 下一步才发生，因此无需原子操作）。但 `surface_2` 和 `surface_1` 会修改邻居 flag，存在并行写冲突——参考代码依赖 CUDA 的宽松内存模型（先写者胜），Warp 同理。
5. **Phase C 曲率性能**：`calculate_curvature` 是每 TYPE_I 单元最昂贵的操作（PLIC × N_interface_neighbors + 5×5 LU）。仅在接口单元调用。
