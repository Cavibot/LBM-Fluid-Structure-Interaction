# Home-FSLBM 迁移执行手册

> **来源**: 提取自 migration_plan_zh.md §6–§10 及附录 A/B/C
> **前置阅读**: 请先阅读 [迁移计划](migration_plan_zh.md) §1–§5 了解架构与算法设计
> **关联文档**: [审核意见书](migration_audit_zh.md)

---
## 6. 单元测试方案

### 6.1 测试策略与执行框架

测试分为三个层级：

1. **单元测试**（`test_*.py`）：每个 `@wp.func` 和 `@wp.kernel` 独立验证，对照参考 C++ 函数输出（通过预先导出的金标准数据或解析解）
2. **集成测试**（`test_solver.py`、`test_domain.py`）：验证 solver 管线 kernel 调用顺序、缓冲区交换、质量/动量守恒
3. **回归测试**（`test_regression.py`）：端到端场景对照参考代码完整仿真输出

所有测试使用 `pytest` 运行，`conftest.py` 提供 `default_model`（32³ 网格）、`default_domain` 等 fixture。金标准数据存储在 `tests/golden_data/` 下，格式为参考代码二进制转储。

> **编译注意事项**：`kernels_fluid.py` 包含大型单体 kernel（`stream_collide_bvh_kernel`），
> 其多循环嵌套、条件 break 及 27 方向条件返回的 `@wp.func` 超出 Warp 1.12
> adjoint 代码生成器的处理能力。由于 HOME-FSLBM 阶段 1 无需微分，模块顶部已通过
> `wp.set_module_options({"enable_backward": False})` 禁用 backward 代码生成，
> 与 WanPhys 和 Newton 项目的标准做法一致。详见
> [迁移计划 §1.4](migration_plan_zh.md#14-warp-微分需求说明)。

> **编码规范**：测试代码及 kernel 源码中的注释、文档字符串、断言消息
> **禁止使用 Unicode 特殊字符**（如 Sigma、rho、middle-dot、箭头等）。
> 请使用纯 ASCII 替代写法（sum、rho、*、-> 等）。含 Unicode 的字符串
> 会导致自动化编辑工具匹配失败。

### 6.2 `test_constants.py` — 格子常数与标记枚举

| 测试名称 | 验证内容 | 输入 | 预期结果 |
|---------|---------|------|---------|
| `test_d3q27_directions_bit_exact` | CX、CY、CZ 与 `ex3d_gpu`/`ey3d_gpu`/`ez3d_gpu` 逐位精确匹配 | 硬编码方向列表 | `CX[1]=1, CX[2]=-1, ...` 共 27 方向完全一致 |
| `test_d3q27_weights` | 权重分布正确 | 权重数组 | W[0]=8/27, W[1-6]=2/27, W[7-18]=1/54, W[19-26]=1/216 |
| `test_opposite_direction_symmetry` | OPPOSITE[i] 始终指向反向 | 遍历 0..26 | `CX[i] + CX[OPPOSITE[i]] == 0`（对所有 i） |
| `test_flag_composite_masks` | 组合标记位域正确 | CellFlag 类 | `TYPE_IF == TYPE_I \| TYPE_F`、`TYPE_SU_MASK == 0x38`、`TYPE_BO_MASK == 0x03` |
| `test_d3q7_gas_directions` | D3Q7 方向为 D3Q27 的前 7 个方向 | 方向数组 | D3Q7 的 CX[0..6] == D3Q27 的 CX[0..6] |
| `test_gas_cmr_s_rates` | CMR-MRT 松弛率向量 | `GAS_CMR_S` | `s[0]=1.0, s[1]=s[2]=s[3]=1/0.9, s[4]=s[5]=s[6]=1.5` |

### 6.3 `test_model.py` — 模型参数验证

| 测试名称 | 验证内容 | 输入 | 预期结果 |
|---------|---------|------|---------|
| `test_default_construction` | 默认参数构造成功 | 无参数 | 所有字段为默认值，`__post_init__` 通过 |
| `test_tau_stability_guard` | ω ≤ 0 触发 ValueError | `omega=0.0` 或 `omega=-0.1` | `ValueError`（τ = 1/ω ≤ 0.5 不稳定） |
| `test_grid_res_properties` | 网格分辨率属性正确派生 | `fluid_grid_res=(64,32,16)` | `nx=64, ny=32, nz=16` |
| `test_turbulence_radius_default` | 默认值为 3（审核项 S6） | 默认构造 | `model.turbulence_radius == 3` |
| `test_device_selection` | device 参数正确解析 | `device="cpu"` | `model._device` 为 CPU 设备 |
| `test_periodic_bc_consistency` | bc_types 与 bc_periodic 一致性检查 | bc_types 含 type 3 但 bc_periodic 为 False | `ValueError` |

### 6.4 `test_state.py` — 状态分配与生命周期

| 测试名称 | 验证内容 | 输入 | 预期结果 |
|---------|---------|------|---------|
| `test_allocation_shapes` | 所有 warp 数组形状正确 | 32³ model | f_mom 长度 = 10×32768，3D 数组 = (32,32,32)，气泡数组 = (65536,) |
| `test_default_initial_values` | 默认初始值正确 | 新分配 state | `solid_phi` 全部 = 1000.0，`tag_matrix` 全部 = -1，`bubble_count` = 0 |
| `test_clear_resets_all` | clear() 归零全部数组 | 填充随机数据后 clear | 全部数组为零（solid_phi=1000.0, solid_body_id=-1 除外） |
| `test_clone_deep_copy` | clone() 生成独立深拷贝 | 修改原 state 后比较 | 克隆不受原 state 修改影响，所有 wp.array 内容相等 |
| `test_wp_float64_bubble_arrays` | 气泡数组为 double 精度 | 检查 dtype | `bubble_volume.dtype == wp.float64` |
| `test_domain_state_protocol` | 实现 DomainState 协议 | `isinstance(state, DomainState)` | `True`；`clear_forces()` 可调用（空操作） |

### 6.5 `test_kernels_fluid.py` — HOME-LBM 碰撞与流传输

| 测试名称 | 验证内容 | 测试设置 | 参考对比 |
|---------|---------|---------|---------|
| `test_f_eq_d3q27_all_27_directions` | D3Q27 Maxwell-Boltzmann 平衡态 | (ρ=1.0, u=(0.02, 0.01, 0.0)) | 逐方向比较 `calculate_f_eq` 输出（`mrUtilFuncGpu3D.h:292-320`），容差 1e-12 |
| `test_reconstruct_distribution_from_moments` | Hermite 展开矩→27 分布 | (ρ=1.0, u=(0.1,0,0), S=0) | 逐方向比较 `mlCalDistributionFourthOrderD3Q27AtIndex` 输出（`mrUtilFuncGpu3D.h:153-266`） |
| `test_compute_rho_u_from_distributions` | 27 分布→宏观矩 | 已知 f_i 分布 | 比较 `calculate_rho_u` 输出（`mrUtilFuncGpu3D.h:272-287`），ρ 容差 1e-12 |
| `test_nocm_mrt_collision_output` | NOCM-MRT 碰撞更新 | (ρ=1.0, u=(0.1,0,0), F=0, ω=1.0) | 逐分量比较 `mlGetPIAfterCollision` 输出的 6 个应力分量（`mrUtilFuncGpu3D.h:424-471`） |
| `test_stream_collide_bvh_uniform_equilibrium` | 均匀平衡态碰撞+流传输后不变 | 32³ 网格，ρ=1.0, u=0, ω=1.0, 无表面/气泡 | 1 步后 ρ'=1.0, u'=0（所有单元），容差 1e-10 |
| `test_stream_collide_bvh_shear_decay` | 剪切流粘性衰减速率 | 32³，初始 `u_x = sin(2π·z/L_z)`，A=0.01, ω=1.0, 100 步 | 与参考代码输出比较速度剖面（审核项 S3）。衰减速率直接依赖 NOCM-MRT，可区分不同碰撞算子 |
| `test_stream_collide_bvh_solid_bounce_back` | 固体壁面反弹产生零速度 | 32³，z=0 和 z=31 为 TYPE_S，其余为 TYPE_F，重力 g_z=-0.001, 500 步 | 收敛后壁面附近 |u| < 1e-6 |
| `test_mass_conservation_through_collision` | 碰撞前后总质量守恒 | 随机初始 ρ∈[0.5,1.5], u=0 | `|Σρ_after - Σρ_before| / Σρ_before < 1e-12` |
| `test_turbulence_omega_modification` | 涡粘性修改 Ω 的行为 | 32³，一个 TYPE_I 单元邻近小气泡（V<5e6），其余为 TYPE_F | 流经气泡附近单元的 Ω 被修改（≠ 1.0），纯流体单元 Ω 不变 |

### 6.6 `test_kernels_surface.py` — VOF 表面与 PLIC

| 测试名称 | 验证内容 | 测试设置 | 参考对比 |
|---------|---------|---------|---------|
| `test_calculate_phi` | 填充水平计算 | `(ρ=1.0, mass=0.5, flag=TYPE_I)` → φ=0.5；`flag=TYPE_F` → φ=1.0；`flag=TYPE_G` → φ=0.0 | 解析验证 |
| `test_plic_cube_known_cases` | PLIC 体积-平面相交 | (V=0.5, n=(1,0,0)) → d=0.0；(V=0.1, n=(1,0,0)) → d=-0.4 等 6 个解析解 | `plic_cube` 对照参考代码（`mrUtilFuncGpu3D.h:104-149`），容差 1e-12 |
| `test_calculate_normal_from_phi` | 接口法向 27 点模板 | 已知平面 φ 场（φ = clamp(0.5 - n·x, 0, 1)） | 计算法向与已知 n 夹角 < 1° |
| `test_calculate_curvature_sphere` | 球面曲率 | 半径 R=10 格点的球面 φ 场 | 计算 κ ≈ 2/R（容差 5%） |
| `test_surface_1_flag_propagation` | IF 邻居阻止 IG，G 邻居设 GI | 8³，中心为 TYPE_IF，周围为 TYPE_G 和 TYPE_I 混合 | 检查标记转换：IG→保留 I，G→GI，I/F 不变 |
| `test_surface_2_gi_initialization` | GI 单元从流体邻居初始化 | 8³，中心 TYPE_GI，周围 2 个 TYPE_F | fMom 为邻居平均 ρ、u 的平衡态 |
| `test_surface_3_type_f_mass_rho` | TYPE_F 单元 mass 设为 ρ | 8³，TYPE_F 单元 mass=1.2, ρ=1.0 | mass'=1.0, φ'=1.0, massex=0.2 |
| `test_surface_3_type_i_mass_clamp` | TYPE_I 单元 mass 钳制到 [0,ρ] | 8³，TYPE_I 单元 mass=1.5, ρ=1.0 | mass'=1.0, φ'=1.0 |
| `test_surface_3_mass_exchange_conservation` | 质量交换前后全局 Σmass 守恒 | 32³，含 TYPE_F/TYPE_I/TYPE_G 混合 | `|Σmass_after - Σmass_before| < 1e-10 × Σmass_before` |
| `test_surface_3_gas_boundary_fill` | 接口单元气体侧用气体平衡态填充 | 16³，TYPE_I 邻居为 TYPE_G | 气体侧 f_i 匹配 `calculate_f_eq(ρ_gas, u)` 输出 |

### 6.7 `test_kernels_bubble.py` — CCL 与气泡追踪

| 测试名称 | 验证内容 | 测试设置 | 参考对比 |
|---------|---------|---------|---------|
| `test_init_tag_boundary_fluid` | 边界和流体单元 tag=-1 | 16³，含 TYPE_S 壁面和 TYPE_F 流体 | tag_matrix = -1 |
| `test_convert_flag_to_input` | TYPE_I/TYPE_G → 255，其他→0 | 16³ 混合标记 | input_matrix 精确逐位匹配 |
| `test_ccl_golden_16cube_3_spheres` | **CCL 金标准验证**（审核项 S1） | 16³ 网格内含 3 个半径=3 格点的不重叠球体（TYPE_I），其余为 TYPE_G | 与参考 YACCLAB CCL 输出逐单元比较 label_matrix。允许标签重编号置换（1↔2↔3 六种排列之一），但**拓扑必须一致**：连通域数量 = 3，每个球的标签集合不重叠 |
| `test_ccl_deterministic` | 多次运行结果相同 | 相同输入运行 5 次 | label_matrix 逐位相同 |
| `test_parse_label_volume_accumulation` | atomicAdd(double) 气泡体积累加 | 已知 2 气泡各占 φ=0.3 的 100 格点 | bubble_volume[bubble_id] = 100×(1-0.3) = 70.0，容差 1e-9 |
| `test_create_bubble_label_init` | 气泡属性从标签初始化 | label_num=2, label_volume=[70,50] | bubble_volume=[70,50], bubble_init_volume=[70,50], bubble_rho=[1.0,1.0] |
| `test_bubble_rho_gas_law` | PV=const 气体定律 | V_init=100, V_new=80, ρ_old=1.0 | ρ_new = 1.0×100/80 = 1.25 |
| `test_merge_detection_two_old_one_new` | 合并检测（二旧→一新） | 64³，2 气泡（tag 1,2）在模拟中合并 | merge_flag=1，合并后 V·ρ 守恒 |
| `test_split_detection_one_old_two_new` | 分裂检测（一旧→二新） | 64³，1 气泡分裂为 2 个 | split_flag=1 |
| `test_assign_tag_resolve_conflict` | assign_tag_kernel 恢复并行标签冲突 | 模拟并行竞争：二线程同时写不同 tag 到同单元 | previous_merge_tag 正确恢复旧标签 |
| `test_recheck_merge_distinguish_motion` | 区分真合并与气泡移动 | 气泡 A 移动但未接触气泡 B | merge_flag=0（不误报） |
| `test_clear_detector_reset` | clear_detector 清零 merge_detector 和标志位 | 设置 merge_detector 后调用 | 全为 0/false |
| `test_reset_label_volume` | 新一轮 CCL 前清零临时缓冲 | 填充后调用 | label_volume 和 label_init_volume 全为零 |

### 6.8 `test_kernels_gas.py` — D3Q7 CMR-MRT 气体

| 测试名称 | 验证内容 | 测试设置 | 参考对比 |
|---------|---------|---------|---------|
| `test_g_eq_d3q7` | D3Q7 平衡态 | (c=0.5, u=0) 和 (c=0.5, u=(0.1,0,0)) | 逐方向比较 `calculate_g_eq` 输出，容差 1e-12 |
| `test_convert_to_central_moment_d3q7` | 正变换：7 分布→7 中心矩 | 给定 7 个已知分布 `g_i` 和 u=(0.1,0,0) | 逐分量比较 `mlConvertCmrMoment_d3q7` 输出（`mrUtilFuncGpu3D.h:474-496`） |
| `test_convert_from_central_moment_d3q7` | 反变换：7 中心矩→7 分布 | 给定 7 个已知中心矩和 u=(0.1,0,0) | 逐分量比较 `mlConvertCmrF_d3q7` 输出（`mrUtilFuncGpu3D.h:499-518`） |
| `test_cmr_mrt_roundtrip` | 正+反变换恒等 | 随机分布 → 正变换 → 反变换 | 输出 = 输入（容差 1e-10） |
| `test_g_stream_collide_relaxation` | CMR-MRT 碰撞松弛至平衡态 | 16³，非平衡初始 g，无边界，100 步 | ρ_g 收敛至守恒值，高阶矩衰减至平衡态 |
| `test_g_reconstruction_henry_law` | 亨利定律接口源项 | 16³，TYPE_I 单元邻接 TYPE_F，ρ_bubble=1.0 | 接口处 c_value = K_h · ρ_bubble / 4 = 2.5e-4 |
| `test_bubble_volume_g_update` | 气体通量→气泡体积 | 已知 delta_g=0.01, phi=0.5, factor=1.0 | `init_volume += (1/4) × 1.0 × 0.01 × 0.5`，atomicAdd(double) 精度验证 |

### 6.9 `test_kernels_foam.py` — 分离压力与大气

| 测试名称 | 验证内容 | 测试设置 | 预期结果 |
|---------|---------|---------|---------|
| `test_disjoint_raycast_hit` | 射线沿 -n 命中其他接口 | 16³，两个 TYPE_I 球体间距 2 格点 | `disjoin_force > 0`（在重叠区域附近） |
| `test_disjoint_no_neighbor` | 无其他接口 | 16³，单个 TYPE_I 球体 | 全部 `disjoin_force = 0` |
| `test_disjoint_distance_scaling` | d=0→force=1, d=4→force=0 | 人工构造已知距离的两个接口 | `P_disjoin = max(0, 1 - d/4)` 精确匹配 |
| `test_reset_disjoin_force` | 清零累加器 | 填充随机值后调用 | 全部 `disjoin_force=0`, `massex=0` |
| `test_atmosphere_rho_update` | 水箱大气压力更新 | atmosphere_open=True, 大气泡 V>1e6 | `bubble_rho[atmosphere_tag] = 1.0` |
| `test_atmosphere_volme_update` | 大气体积修正 | atmosphere_open=True, rho=1.0, volume=1000 | `init_volume = 1000` |

### 6.10 `test_solver.py` — 管线集成

| 测试名称 | 验证内容 | 测试设置 | 预期结果 |
|---------|---------|---------|---------|
| `test_solver_kernel_launch_order` | kernel 按正确顺序启动 | 单步执行，记录启动顺序 | Phase1: get_tag→assign_tag→recheck_merge→bubble_update→g_recon→g_stream→g_update→g_swap→bubble_rho；Phase2: disjoint→atmosphere→stream_collide_bvh→reset_disjoint→surface_1→surface_2→surface_3→fMom_swap |
| `test_solver_mass_conservation` | 跨步全局质量守恒 | 32³，100 步 | `\|Σρ_after - Σρ_before\| / Σρ_before < 1e-10` |
| `test_solver_bubble_ccl_periodic` | 气泡 CCL 按周期执行 | 设置 bubble_stride=5，运行 12 步 | CCL 在第 5 和 10 步执行（`_step_count % 5 == 0`） |
| `test_solver_buffer_swap` | 步后 fMom 和 gMom 交换 | 单步执行，比较 state_in 和 state_out | 步后 `state.f_mom == state_out.f_mom_post`（原值），`state.g_mom == state_out.g_mom_post` |
| `test_solver_conditional_clear_inlet` | clear_inlet 仅在 step==57595 触发 | 运行 57594→57596 步 | `clear_inlet` kernel 仅在第 57595 步启动 |
| `test_solver_handle_merge_split_trigger` | merge_flag 触发完整 CCL 管线 | 设置 merge_flag=1 | _handle_merge_split 执行全部 8 个子步骤 |

### 6.11 `test_domain.py` — Domain 编排

| 测试名称 | 验证内容 | 测试设置 | 预期结果 |
|---------|---------|---------|---------|
| `test_name_identifier` | domain.name 正确 | 创建 domain | `"fluid_grid_home_fslbm"` |
| `test_lazy_state_creation` | create_state() 惰性分配 | 构造 domain 后不调用 create_state，访问 state 属性 | 自动触发 create_state，state 非 None |
| `test_double_buffer_isolation` | step 后双缓冲正确交换 | 初始 state_in.f_mom != state_out.f_mom | 步后原 state_in 变为 state_out |
| `test_pre_post_step_hooks` | pre_step/post_step 钩子可重写 | 子类重写钩子 | 钩子在 step 前后被调用 |

### 6.12 `test_regression.py` — 端到端回归

每个回归测试生成金标准数据的流程见第 10 章。所有测试的相对容差为浮点场 1e-4、整数字段精确匹配（允许标签重编号置换）。

| 测试名称 | 场景描述 | 网格 | 步数 | 检查点步数 | 关键验证项 | 参考构建方式 |
|---------|---------|------|------|-----------|-----------|------------|
| `test_regression_shear_decay` | 剪切流粘性衰减 | 32³, τ=0.55 | 100 | 50, 100 | u_x(z) 剖面逐点比较，最大相对误差 | 参考代码 `stream_collide_bvh` 相同初始条件输出 |
| `test_regression_droplet` | 静止球形液滴表面张力平衡 | 64³, 半径=12 格点球体 | 200 | 100, 200 | φ 场、曲率 κ、Laplace 压力 ΔP = 2σ/R | 参考代码完整仿真输出 |
| `test_regression_dambreak` | 溃坝自由表面演化 | 128×64×64, 初始水柱 40×30×30 格点 | 500 | 100, 300, 500 | 前缘位置、φ=0.5 等值面 | 参考代码完整仿真输出 |
| `test_regression_rising_bubble` | 单气泡浮力上升 | 64³, 气泡半径=8 格点, g_z=-0.0001 | 300 | 100, 200, 300 | 气泡质心位置、体积变化率 | 参考代码完整仿真输出 |
| `test_regression_two_bubbles_merge` | 两气泡合并 | 64³, 两气泡半径=6 格点，间距=4 格点 | 500 | 200, 400, 500 | 合并时间步、合并前后 V·ρ 守恒 | 参考代码完整仿真输出 |
| `test_regression_foam_no_coalescence` | 泡沫分离压力阻止聚并（审核项 S2） | 64³, 两气泡半径=6, 间距=3 格点, disjoin_factor=0.032 | 500 | 250, 500 | Σdisjoin_force > 0 且气泡间最小距离 > 0（始终不发生合并） | 参考代码完整仿真输出 |
| `test_regression_bubble_volume_conservation` | 双精度气泡体积长期守恒（审核项 B2） | 128³, 1 个大气泡半径=16 格点 | 10⁴ | 1000, 5000, 10000 | `|V_T - V_0| / V_0 ≤ 1×10⁻⁶` | 参考代码 `bubble_volume_update` 输出对比 |

**`conftest.py` 加载金标准数据的伪代码**：

```python
@pytest.fixture
def golden_data(request):
    test_name = request.node.name
    step = request.node.get_closest_marker("step").args[0]
    path = Path(__file__).parent / "golden_data" / f"{test_name}_step{step}.bin"
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.float32)
    # 重塑为对应场的形状进行逐分量比较
    return raw
```

---

## 7. 文件清单

```
wanphys/_src/fluid/fluid_grid/home_fslbm/
├── __init__.py                 # ~20 行  [P0]
├── constants.py                # ~120 行 [P0]  (+CMR-MRT 松弛率向量)
├── model.py                    # ~150 行 [P0]  (turbulence_radius=3)
├── state.py                    # ~250 行 [P0]  (+4 字段, DomainState 基类, wp.float64)
├── solver.py                   # ~300 行 [P0]  (两阶段结构)
├── domain.py                   # ~100 行 [P0]
├── kernels_fluid.py            # ~1000 行 [P0] (单一 stream_collide_bvh + 内联湍流)
├── kernels_surface.py          # ~600 行 [P0]
├── kernels_bubble.py           # ~900 行 [P1]  (+YACCLAB CCL, +9 kernel)
├── kernels_gas.py              # ~350 行 [P1]  (+CMR-MRT 变换, +bubble_volume_g_update)
├── kernels_foam.py             # ~200 行 [P2]  (+atmosphere_volme_update)
└── tests/
    ├── test_constants.py       # ~100 行
    ├── test_model.py           # ~80 行
    ├── test_state.py           # ~120 行
    ├── test_kernels_fluid.py   # ~350 行 (+剪切流衰减)
    ├── test_kernels_surface.py # ~300 行
    ├── test_kernels_bubble.py  # ~350 行 (+CCL 金标准)
    ├── test_kernels_gas.py     # ~200 行
    ├── test_kernels_foam.py    # ~150 行
    ├── test_solver.py          # ~250 行
    ├── test_domain.py          # ~80 行
    └── test_regression.py      # ~300 行 (+泡沫, +体积守恒, +剪切流衰减)

合计: ~4,200 行生产代码 + ~2,280 行测试代码
（按审核更正后 +700/+420）
```

---

## 8. 迁移路线图

### 第一阶段：核心 LBM（P0）—— 预估 5-7 天

| 步骤 | 任务 | 涉及文件 |
|------|------|----------|
| 1.1 | 创建目录结构和 `__init__.py` | `__init__.py` |
| 1.2 | 实现格子常数、标记枚举、CMR-MRT 松弛率向量 | `constants.py` |
| 1.3 | 实现 `HomeFslbmModel` dataclass（turbulence_radius=3） | `model.py` |
| 1.4 | 实现 `HomeFslbmState`（继承 DomainState，全部字段含新增 4 项，wp.float64） | `state.py` |
| 1.5 | 实现 `@wp.func` 辅助函数：`f_eq_d3q27`、`reconstruct_distribution`、`compute_pi_from_moments` | `kernels_fluid.py` |
| 1.6 | 实现 `stream_collide_bvh_kernel`（含内联湍流模型，第 703-1057 行全部逻辑） | `kernels_fluid.py` |
| 1.7 | 实现 `HomeFslbmSolver` 两阶段管线框架（先实现 Phase 2 流体部分） | `solver.py` |
| 1.8 | 实现 `HomeFslbmDomain` | `domain.py` |
| 1.9 | 第一阶段单元测试 | `tests/test_kernels_fluid.py` |
| **门禁** | **剪切流衰减测试与参考代码一致；NOCM-MRT 碰撞输出逐分量匹配** | |

### 第二阶段：自由表面（P0）—— 预估 4-5 天

| 步骤 | 任务 | 涉及文件 |
|------|------|----------|
| 2.1 | 实现 PLIC 辅助函数：`plic_cube`、`calculate_normal`、`calculate_curvature`、`calculate_phi` | `kernels_surface.py` |
| 2.2 | 实现 `surface_1_kernel`（标记传播，第 444-477 行） | `kernels_surface.py` |
| 2.3 | 实现 `surface_2_kernel`（GI 初始化，第 479-603 行） | `kernels_surface.py` |
| 2.4 | 实现 `surface_3_kernel`（质量重分配+标记转换，第 601-937 行） | `kernels_surface.py` |
| 2.5 | 将表面追踪三阶段集成至 solver 管线（Phase 2 内，stream_collide_bvh 之后） | `solver.py` |
| 2.6 | 第二阶段单元测试 | `tests/test_kernels_surface.py` |
| **门禁** | **溃坝仿真表面位置与参考代码一致；质量交换前后 Σmass 守恒** | |

### 第三阶段：YACCLAB CCL + 气泡模型（P1）—— 预估 6-8 天（审核项 B1 +2-3 天）

| 步骤 | 任务 | 涉及文件 |
|------|------|----------|
| **3.0** | **CCL 原型验证**：16³ 网格 3 分离球体，对比参考代码 label_matrix 输出，验证标签拓扑一致性 | `kernels_bubble.py` |
| 3.1 | 实现 InitLabeling + MergeMasks（2×2×2 block-level，Warp atomic_min） | `kernels_bubble.py` |
| 3.2 | 实现 LabelAnalysis + LabelReduction（全局 merge chain resolution） | `kernels_bubble.py` |
| 3.3 | 实现 FinalLabeling（密集 0..N-1 输出） | `kernels_bubble.py` |
| 3.4 | 实现 clear_detector、report_split、clear_inlet、ResetLabelVolume | `kernels_bubble.py` |
| 3.5 | 实现 assign_tag_kernel、recheck_merge_kernel（并行标签冲突解决） | `kernels_bubble.py` |
| 3.6 | 实现 MergeSplitDetectorKernel（全局合并/分裂检测） | `kernels_bubble.py` |
| 3.7 | 实现 bubble_volume_update_kernel、bubble_rho_update_kernel（wp.float64 atomic） | `kernels_bubble.py` |
| 3.8 | 第三阶段单元测试 —— CCL 金标准对比 | `tests/test_kernels_bubble.py` |
| **门禁** | **CCL 标签在 16³ 测试上与参考代码一致；两气泡合并后 V·ρ = V₁·ρ₁+V₂·ρ₂** | |

### 第四阶段：CMR-MRT 气体 + 泡沫（P1-P2）—— 预估 4-5 天（审核项 B3 +1 天）

| 步骤 | 任务 | 涉及文件 |
|------|------|----------|
| 4.1 | 实现 `convert_to_central_moment_d3q7`（7 分布→7 中心矩，第 474-496 行） | `kernels_gas.py` |
| 4.2 | 实现 `convert_from_central_moment_d3q7`（7 中心矩→7 分布，第 499-518 行） | `kernels_gas.py` |
| 4.3 | 实现 `g_stream_collide_kernel`（CMR-MRT 7 松弛率，第 1750-1848 行） | `kernels_gas.py` |
| 4.4 | 实现 `g_reconstruction_kernel`（亨利定律边界交换，第 1644-1748 行） | `kernels_gas.py` |
| 4.5 | 实现 `bubble_volume_g_update_kernel`（气体通量→气泡体积，第 1853-1879 行） | `kernels_gas.py` |
| 4.6 | 实现分离压力 kernel（calculate_disjoint、ResetDisjoinForce） | `kernels_foam.py` |
| 4.7 | 实现大气 kernel（atmosphere_rho_update、atmosphere_volme_update） | `kernels_foam.py` |
| 4.8 | 第四阶段单元测试 | `tests/test_kernels_gas.py`，`tests/test_kernels_foam.py` |
| **门禁** | **泡沫膜阻止气泡聚并；CMR-MRT 变换对照参考精确匹配** | |

### 第五阶段：集成 + 回归（P0）—— 预估 4-5 天（审核项 S2/S3/S5 +1 天）

| 步骤 | 任务 | 涉及文件 |
|------|------|----------|
| 5.1 | Domain 级集成测试——验证两阶段管线全部 kernel 正确顺序执行 | `tests/test_solver.py` |
| 5.2 | 剪切流衰减端到端回归（32³, 100 步，替代区分度为零的 uniform_flow） | `tests/test_regression.py` |
| 5.3 | 泡沫无聚并端到端回归（64³, 500 步，验证分离压力功能） | `tests/test_regression.py` |
| 5.4 | 气泡体积守恒测试（128³, 10⁴ 步，验证 wp.float64 精度） | `tests/test_regression.py` |
| 5.5 | 可视化验证（viewer 集成） | viewer 模块 |
| 5.6 | 性能剖析与优化 | 全部 kernel |
| 5.7 | 金标准数据生成流程建立、文档定稿 | docs |
| **门禁** | **全部回归测试在容差范围内通过；10⁴ 步气泡体积漂移 ≤ 1e-6** | |

**预估总工时: 23-30 天**（较原估算 19-25 天增加，按审核更正调整）。

---

## 9. 风险登记

| ID | 风险 | 严重程度 | 可能性 | 影响 | 检测难度 | 缓解措施 | 应急方案 |
|----|------|---------|--------|------|---------|---------|---------|
| R1 | **YACCLAB CCL 在 Warp 中无直接等价实现**。参考代码依赖 OpenCV `GpuMat3`（pitched 3D memory）和 CUDA `__syncthreads()` block 级同步。Warp 无等价 pitched memory API 且 barrier 模型不同。 | 高 | 高 | 气泡追踪完全失效，CCL 结果是后续所有气泡属性计算的基础 | 高（仅端到端可见） | Phase 3.0 原型验证子阶段：16³ 网格 3 分离球体逐单元对比 label_matrix。CUDA `__syncthreads()` 处拆分为独立 kernel 启动（Warp 的 kernel 间有隐式全局同步）。`wp.atomic_min` 替代 CUDA `atomicMin` 进行 block 间合并。在 Phase 3.1 前必须通过原型验证门禁。 | 若 Warp CCL 无法在合理时间内（< 参考代码 2× 时间）达到拓扑等价：回退至 CPU 端 `scipy.ndimage.label`，接受气泡更新步骤的 GPU→CPU 传输开销（~10ms/帧于 128³）。 |
| R2 | **wp.float64 原子操作不可用**。Warp 的 float64 支持有限，`wp.atomic_add(wp.float64)` 可能在当前后端不可用。气泡体积累加依赖 double 精度原子操作（`mrLbmSolverGpu3D.cu:1083/1873`），float 精度在 10⁵ 步累积后漂移 > 1%。 | 高 | 中 | 气泡体积长期漂移导致气体定律失效，泡沫压力错误 | 中（需 10⁴ 步方可检测） | 首先测试 Warp float64 atomic 可用性。若不可用：实现定点化方案——将 double 值缩放为 int64（scale = 1e9），使用 `wp.atomic_add(wp.int64)` 累加，读取时还原。提供蒙特卡洛定量误差分析证明 10⁴ 步累积误差 ≤ double 参考的 2 倍。在 `test_regression_bubble_volume_conservation` 中验证。 | 若定点化方案误差超限：每个气泡体积归约改用 CPU 端归约（`wp.copy` → numpy → `np.sum`），牺牲 GPU 归约性能。 |
| R3 | **CMR-MRT 变换函数逐行精确复现失败**。`mlConvertCmrMoment_d3q7`（`mrUtilFuncGpu3D.h:474-496`）和 `mlConvertCmrF_d3q7`（`mrUtilFuncGpu3D.h:499-518`）包含二次速度展开和高阶修正项，任一符号错误导致 CMR-MRT 碰撞数值完全偏离。 | 高 | 低 | D3Q7 碰撞完全错误，溶解气体浓度场不可用 | 低（金标准测试向量立即检测） | 逐行对照参考代码复现两个函数。在 `test_kernels_gas.py` 中实现：正反变换恒等测试（roundtrip 容差 1e-10）、金标准向量测试（对照参考代码输出）。Phase 4 实现前必须先通过门禁。 | 若无法精确复现：保留参考 C++ 函数的调用（通过 ctypes/pybind11 包装）作为 fallback，在 Phase 5 优化阶段再替换为 Warp 实现。 |
| R4 | **PLIC 曲率在尖角处发散**。Monge 曲面片二次拟合在界面曲率半径 < 2 格点时可能产生 κ 的极端值（> 100），导致 Laplace 压力爆炸和数值崩溃。 | 中 | 中 | 尖角/薄片处仿真发散 | 中（可见于溃坝飞溅场景） | 实现三级曲率保护：(1) 钳制 κ ∈ [-1,1]（与参考代码一致）；(2) 当拟合点数 < 5 时回退至法向差分曲率（`∇·n`）；(3) 当 κ 的局部变化率 > 10/格点时使用局部平均。在 `test_regression_dambreak` 的飞溅区域验证稳定性。 | 若三级保护仍不足：在该格点临时禁用表面张力（κ=0），依赖数值耗散平滑界面。 |
| R5 | **气泡合并/分裂并行竞态条件**。`merge_detector` 标记、`tag_matrix` 更新和 `previous_merge_tag` 恢复涉及多个 kernel 间的数据依赖和原子操作竞争，可能导致标签不一致（气泡"消失"或"复制"）。 | 高 | 中 | 气泡计数错误、体积不守恒 | 高（仅在特定并行调度下触发） | 参考代码已通过 `assign_tag_kernel` + `recheck_merge_kernel` + `atomicExch` 三重机制提供确定性标签冲突解决。迁移时严格保持此机制不变，禁止简化。增加确定性测试：同一初始条件下运行 10 次，标签结果完全相同。 | 若确定性仍无法保证：将合并/分裂检测改为 CPU 端顺序处理（仅在 `merge_flag/split_flag` 置位时触发，频率低，不影响性能）。 |
| R6 | **Warp kernel 启动开销累积**。两阶段 solver 管线涉及 15+ 次独立的 `wp.launch` 调用（Phase 1 约 8 次，Phase 2 约 7 次）。Warp 的 kernel 调度开销（~10-50μs/次）可能成为小网格（≤ 64³）的主要耗时。 | 中 | 中 | 小网格下性能劣于参考 CUDA 实现 | 低（profiler 直接可见） | 早期性能剖析（Phase 1.9 后即开始）。若开销显著：(1) 合并不依赖中间结果的相邻 kernel（如 `atmosphere_rho_update` + `atmosphere_volme_update` → 单 kernel）；(2) 使用 Warp graph capture 预录制固定管线。 | 若融合仍不足：对小网格（< 64³）提供"快速模式"管线变体，将 surface_1/2/3 合并、跳过大气步骤。 |
| R7 | **论文公式与参考代码的细微不一致**。论文四的公式推导和参考代码实现之间存在已知差异（如 D3Q27 vs D3Q19、BGK vs CMR-MRT、涡粘性模型的硬编码因子 4）。审核已暴露多项此类差异，可能存在未发现的差异。 | 中 | 低 | 单个模块数值偏离 | 中（需对照测试方可见） | 以参考代码为真理源，非论文公式。所有模块的对照测试必须使用参考代码的实际输出作为金标准，而非论文公式的计算结果。在附录 A 中标注"参考代码行号"而非仅"论文公式编号"。 | 若发现新差异：优先匹配参考代码行为，在文档中标注"偏离论文 Eq.X 但匹配参考实现"。 |

**风险等级说明**：
- **高**：阻塞性，不缓解无法通过该阶段门禁
- **中**：需在对应阶段实施缓解措施，缓解后可接受
- **低**：已知但计划内可容忍

**检测难度说明**：
- **低**：单元测试或 profiler 可直接检测
- **中**：需特定测试场景或较长时间运行方可暴露
- **高**：仅在特定并行调度或极端边界条件下触发

---

## 10. 金标准数据生成流程

### 10.1 参考代码编译环境

| 组件 | 版本/规格 | 来源 |
|------|----------|------|
| CUDA Toolkit | 12.8 | `CMakeSettings.json` 中 `cmakeCommandArgs: "-DCMAKE_CUDA_ARCHITECTURES=86"` |
| C++ 编译器 | MSVC 2022 (Windows) / GCC 7+ (Linux) | `CMakeLists.txt` 要求 C++17 |
| CMake | ≥ 3.10 | `CMakeLists.txt` 中 `cmake_minimum_required(VERSION 3.10)` |
| 构建系统 | Ninja (Windows: VS 集成) / Make (Linux) | `CMakeSettings.json` 指定 `"generator": "Ninja"` |
| OpenCV | ≥ 4.x | YACCLAB CCL 和可视化依赖 |
| Eigen | 3rdParty 捆绑版本 | `CMakeLists.txt` 中 `include_directories(${CMAKE_SOURCE_DIR}/eigen)` |
| GPU 硬件 | NVIDIA compute capability ≥ 6.0 (sm_86 target) | `CMakeLists.txt` 中 `set(CMAKE_CUDA_ARCHITECTURES 86)` |
| 编译产物 | `lbm_flow_proj` 可执行文件 + `lbm_flow_proj-cpu.lib` + `lbm_flow_proj-cuda.lib` | `CMakeLists.txt` 中 `add_executable` + `add_library` |

### 10.2 金标准数据文件格式

输出文件命名：`{test_name}_step{N}.bin`

**二进制布局**（little-endian，与参考代码平台一致）：

| 偏移（字节） | 字段 | 元素类型 | 元素数量 | 总字节数 |
|-------------|------|---------|---------|---------|
| 0 | `fMom[10*N]` | float32 | 10×N | 40×N |
| 40×N | `flag[N]` | uint8 | N | N |
| 41×N | `mass[N]` | float32 | N | 4×N |
| 45×N | `phi[N]` | float32 | N | 4×N |
| 49×N | `force_x[N]` | float32 | N | 4×N |
| 53×N | `force_y[N]` | float32 | N | 4×N |
| 57×N | `force_z[N]` | float32 | N | 4×N |
| 61×N | `gMom[7*N]` | float32 | 7×N | 28×N |
| 89×N | `delta_g[N]` | float32 | N | 4×N |
| 93×N | `c_value[N]` | float32 | N | 4×N |
| 97×N | `tag_matrix[N]` | int32 | N | 4×N |
| 101×N | `label_matrix[N]` | int32 | N | 4×N |
| 105×N | `disjoin_force[N]` | float32 | N | 4×N |
| 109×N | `bubble_count` | int32 | 1 | 4 |
| 109×N+4 | `bubble_volume[bubble_count]` | float64 | bubble_count | 8×bubble_count |
| 109×N+4+8×bc | `bubble_rho[bubble_count]` | float64 | bubble_count | 8×bubble_count |

其中 N = nx × ny × nz，bc = bubble_count。

### 10.3 参考代码输出补丁

在 `testMrLBM3D_bubble.cpp` 的仿真循环中插入以下输出逻辑：

```cpp
// 在每个检查点步数后调用
void dump_golden_data(mrFlow3D* flow, const char* filename, int step) {
    char path[256];
    snprintf(path, sizeof(path), "golden_data/%s_step%d.bin", filename, step);
    FILE* f = fopen(path, "wb");
    int N = flow->param->validCount;
    // 按 §10.2 布局顺序写入
    fwrite(flow->fMom, sizeof(REAL), 10 * N, f);          // 40N bytes
    fwrite(flow->flag, sizeof(MLLATTICENODE_SURFACE_FLAG), N, f);  // N bytes
    fwrite(flow->mass, sizeof(REAL), N, f);               // 4N bytes
    fwrite(flow->phi, sizeof(REAL), N, f);                // 4N bytes
    fwrite(flow->forcex, sizeof(REAL), N, f);             // 4N bytes
    fwrite(flow->forcey, sizeof(REAL), N, f);             // 4N bytes
    fwrite(flow->forcez, sizeof(REAL), N, f);             // 4N bytes
    fwrite(flow->gMom, sizeof(float), 7 * N, f);          // 28N bytes
    fwrite(flow->delta_g, sizeof(float), N, f);           // 4N bytes
    fwrite(flow->c_value, sizeof(float), N, f);           // 4N bytes
    fwrite(flow->tag_matrix, sizeof(int), N, f);          // 4N bytes
    fwrite(flow->label_matrix, sizeof(int), N, f);        // 4N bytes
    fwrite(flow->disjoin_force, sizeof(float), N, f);     // 4N bytes
    fwrite(&flow->bubble.bubble_count, sizeof(int), 1, f); // 4 bytes
    fwrite(flow->bubble.volume, sizeof(double), flow->bubble.bubble_count, f);
    fwrite(flow->bubble.rho, sizeof(double), flow->bubble.bubble_count, f);
    fclose(f);
}
```

### 10.4 Python 端数据加载

```python
# conftest.py
import numpy as np
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / "golden_data"

def load_golden(filename: str) -> dict:
    """加载金标准二进制文件，返回各字段的 numpy 数组字典。"""
    with open(GOLDEN_DIR / filename, "rb") as f:
        data = f.read()
    offset = 0
    N = None  # 需要从文件名或头部确定 N

    def read_field(dtype, count):
        nonlocal offset
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        offset += arr.nbytes
        return arr

    # N 从文件大小反推（对于给定网格尺寸是固定的）
    # fMom: 10*N float32, flag: N uint8, mass/phi: 各 N float32, force_xyz: 各 N float32
    # gMom: 7*N float32, delta_g/c_value: 各 N float32
    # tag/label: 各 N int32, disjoin: N float32
    # 固定部分: N * (40 + 1 + 4 + 4 + 4 + 4 + 4 + 28 + 4 + 4 + 4 + 4 + 4) = N * 109 bytes
    # 可变部分: 4 (bubble_count) + 8*bc (volume) + 8*bc (rho)
    return {"f_mom": read_field(np.float32, 10 * N),
            "flag": read_field(np.uint8, N),
            "mass": read_field(np.float32, N),
            # ... 依此类推
            }
```

### 10.5 CI 集成方案

金标准数据生成和验证的 CI 流程：

1. **构建阶段**：在 NVIDIA GPU CI runner 上编译参考代码（`cmake .. -G Ninja && ninja`）
2. **生成阶段**：运行 `lbm_flow_proj` 为每个回归测试场景生成金标准 `.bin` 文件。参数通过命令行传入（网格尺寸、初始条件、检查点步数）
3. **存储阶段**：金标准文件上传至 CI 持久化存储（Git LFS 或 S3 bucket），按 `{test_name}/step{N}.bin` 组织
4. **验证阶段**：Warp 实现的 `test_regression.py` 自动从存储拉取对应金标准文件，使用 `conftest.py` 的 `load_golden` 函数加载，逐分量比较
5. **更新流程**：金标准文件随参考代码的版本锁定（commit SHA），仅在参考代码更新或新增测试场景时重新生成

### 10.6 对比逻辑详细规则

**浮点场量（f_mom, mass, phi, force_*, g_mom, delta_g, c_value, disjoin_force, bubble_volume, bubble_rho）**：
- 相对容差：`|a - b| / max(|a|, |b|, 1e-10) ≤ 1e-4`
- 当参考值接近零（< 1e-8）时使用绝对容差：`|a - b| ≤ 1e-10`

**整数字段在允许置换后的匹配**：
- `flag`：逐位精确匹配（不允许差异）
- `tag_matrix`：允许标签重编号置换。算法：分别提取参考和 Warp 输出的唯一标签集合，尝试所有双射（bijection）映射，选择使不一致率最小的映射。若最小不一致率 > 0（即存在拓扑差异——某气泡在参考中连通但在 Warp 中不连通），测试失败
- `label_matrix`：同 tag_matrix 的置换规则

**气泡属性（bubble_count, bubble_volume, bubble_rho）**：
- `bubble_count`：必须精确相等（连通域数量一致）
- `bubble_volume/rho`：按标签重编号置换后逐气泡比较，容差同浮点规则
- **体积守恒**（仅 `test_regression_bubble_volume_conservation`）：`|ΣV(T) - ΣV(0)| / ΣV(0) ≤ 1e-6`

---

## 附录 A：论文公式到代码映射表（完整）

| 论文章节 | 算法 | 参考代码文件 | 行号 | 目标文件 |
|---------|------|-------------|------|---------|
| 论文五 Eq.(17) | Hermite 矩展开 | `mrUtilFuncGpu3D.h` | 153-266 | `kernels_fluid.py` |
| 论文五 Eq.(21-23) | NOCM-MRT 碰撞 | `mrUtilFuncGpu3D.h` | 424-471 | `kernels_fluid.py` |
| 论文四 Eq.(9-10) | VOF 质量平流 | `mrLbmSolverGpu3D.cu` | 601-937 | `kernels_surface.py` |
| 论文四 Eq.(11) | 接口压力边界条件 | `mrLbmSolverGpu3D.cu` | 840-910 | `kernels_surface.py` |
| 论文四 Eq.(12) | PLIC 曲率 | `mrUtilFuncGpu3D.h` | 320-420 | `kernels_surface.py` |
| 论文四 Eq.(16-17) | 分布重建 | `mrUtilFuncGpu3D.h` | 153-266 | `kernels_fluid.py` |
| 论文四 Eq.(18-21) | 碰撞更新 | `mrUtilFuncGpu3D.h` | 424-471 | `kernels_fluid.py` |
| 论文四 Eq.(22-25) | 气泡气体定律 | `mrLbmSolverGpu3D.cu` | 1300-1365 | `kernels_bubble.py` |
| 论文四 Eq.(33-36) | 气体 CMR-MRT 对流扩散 | `mrLbmSolverGpu3D.cu` | 1750-1848 | `kernels_gas.py` |
| 论文四 Eq.(38) | 亨利定律 | `mrLbmSolverGpu3D.cu` | 1711-1714 | `kernels_gas.py` |
| 论文四 Eq.(40-41) | 分离压力 | `mrLbmSolverGpu3D.cu` | 162-272 | `kernels_foam.py` |
| 论文四 Sec.4.1 | 表面追踪 | `mrLbmSolverGpu3D.cu` | 444-937 | `kernels_surface.py` |
| 论文四 Sec.5.1 | 湍流模型（内联） | `mrLbmSolverGpu3D.cu` | 1001-1028 | `kernels_fluid.py` |
| 论文四 Sec.5.2 | YACCLAB CCL | `tDCCL.cu` | 1-578 | `kernels_bubble.py` |
| — | CMR-MRT 正变换 | `mrUtilFuncGpu3D.h` | 474-496 | `kernels_gas.py` |
| — | CMR-MRT 反变换 | `mrUtilFuncGpu3D.h` | 499-518 | `kernels_gas.py` |
| — | 大气体积更新 | `mrLbmSolverGpu3D.cu` | 1286-1298 | `kernels_foam.py` |
| — | 气体通量→气泡体积 | `mrLbmSolverGpu3D.cu` | 1853-1879 | `kernels_gas.py` |
| — | 清零探测器 | `mrLbmSolverGpu3D.cu` | 64-107 | `kernels_bubble.py` |
| — | 清零入口 | `mrLbmSolverGpu3D.cu` | 110-142 | `kernels_bubble.py` |
| — | 报告分裂 | `mrLbmSolverGpu3D.cu` | 53-61 | `kernels_bubble.py` |
| — | 分配标签（并行冲突解决） | `mrLbmSolverGpu3D.cu` | 1424-1450 | `kernels_bubble.py` |
| — | 二次验证合并 | `mrLbmSolverGpu3D.cu` | 1452-1475 | `kernels_bubble.py` |
| — | MergeSplitDetector | `mrLbmSolverGpu3D.cu` | 1366-1374 | `kernels_bubble.py` |
| — | ResetLabelVolume | `mrLbmSolverGpu3D.cu` | 1185-1193 | `kernels_bubble.py` |

## 附录 B：NOCM-MRT 碰撞 — 完整伪代码

```
[ALGO: mlGetPIAfterCollision — NOCM-MRT 碰撞更新]
模块: kernels_fluid.py
源文件: mrUtilFuncGpu3D.h:424-471

输入:
  R     : 密度 ρ
  U,V,W : 速度分量 (u_x, u_y, u_z)
  Fx,Fy,Fz : 力分量
  omega : 松弛频率
  pixx_t45, piyy_t45, pizz_t45 : 碰撞前对角应力 Π_αα（×45 缩放）
  pixy_t90, pixz_t90, piyz_t90 : 碰撞前非对角应力 Π_αβ（×90 缩放）

算法:
  // 1. 将对角应力分解为无迹部分
  pixx_part = (2·pixx_t45 - piyy_t45 - pizz_t45) / 3
  piyy_part = (2·piyy_t45 - pixx_t45 - pizz_t45) / 3
  pizz_part = (2·pizz_t45 - pixx_t45 - piyy_t45) / 3

  // 2. 计算 R·u² 项
  RU2 = R·U·U
  RV2 = R·V·V
  RW2 = R·W·W
  RUVW2 = (RU2 + RV2 + RW2) / 3

  // 3. 更新对角应力（闭式 NOCM-MRT）
  pixx_t45 = R/3 + pixx_part·(1-ω) + RUVW2 + ω·(2RU2-RV2-RW2)/3 + Fx·U
  piyy_t45 = R/3 + piyy_part·(1-ω) + RUVW2 + ω·(2RV2-RU2-RW2)/3 + Fy·V
  pizz_t45 = R/3 + pizz_part·(1-ω) + RUVW2 + ω·(2RW2-RU2-RV2)/3 + Fz·W

  // 4. 更新非对角应力
  pixy_t90 = pixy_t90·(1-ω) + ω·R·U·V + (Fy·U + Fx·V)/2
  pixz_t90 = pixz_t90·(1-ω) + ω·R·U·W + (Fz·U + Fx·W)/2
  piyz_t90 = piyz_t90·(1-ω) + ω·R·V·W + (Fz·V + Fy·W)/2

输出: pixx_t45, pixy_t90, pixz_t90, piyy_t45, piyz_t90, pizz_t90

// 5. 碰撞后矩存储（源自 stream_collide_bvh, mrLbmSolverGpu3D.cu:1046-1055）
  invRho = 1.0 / R
  fMomPost[0] = R
  fMomPost[1] = U + Fx·invRho·0.5
  fMomPost[2] = V + Fy·invRho·0.5
  fMomPost[3] = W + Fz·invRho·0.5
  fMomPost[4] = pixx_t45·invRho - c_s²
  fMomPost[5] = pixy_t90·invRho
  fMomPost[6] = pixz_t90·invRho
  fMomPost[7] = piyy_t45·invRho - c_s²
  fMomPost[8] = piyz_t90·invRho
  fMomPost[9] = pizz_t45·invRho - c_s²
```

## 附录 C：PLIC Cube — 完整伪代码

```
[ALGO: plic_cube — 单位立方体-平面相交]
模块: kernels_surface.py
源文件: mrUtilFuncGpu3D.h:104-149

输入: V0 ∈ [0,1] : 体积分数
       n = (nx, ny, nz) : 接口法向

算法:
  1. 对称化简:
     ax = |nx|, ay = |ny|, az = |nz|
     V = 0.5 - |V0 - 0.5|     // 映射到 [0, 0.5]
     L = ax + ay + az
     n1 = min(ax, ay, az) / L
     n3 = max(ax, ay, az) / L
     n2 = max(0.0, 1.0 - n1 - n3)  // 确保 n2 >= 0

  2. 简化问题 (plic_cube_reduced, mrUtilFuncGpu3D.h:104-119):
     // SZ & Kawano 2022 方法，采用简化对称情形
     if n1 + n2 <= 2.0·n3·V:
         return n3·V + 0.5·(n1 + n2)              // 情形 (5)
     v1 = n1² / (6.0·n2)
     if v1 <= n3·V < v1 + 0.5·(n2-n1):
         return 0.5·(n1 + √(n1² + 8.0·n2·(n3·V - v1)))  // 情形 (2)
     V6 = n1·6.0·n2·n3·V
     if n3·V < v1:
         return ³√(V6)                                // 情形 (1)
     // 情形 (3)/(4): 三次方根公式
     sqn12 = n1² + n2²
     V6cbn12 = V6 - n1³ - n2³
     case34 = (n3·V < v3)  // v3 =（详细公式见源文件）
     a = case34 ? V6cbn12 : 0.5·(V6cbn12 - n3³)
     b = case34 ? sqn12 : 0.5·(sqn12 + n3²)
     c_val = case34 ? n12 : 0.5
     t = √(c_val² - b)
     d = c_val - 2.0·t·sin(0.33333334·asin((c_val³ - 0.5·a - 1.5·b·c_val) / t³))
     return d

  3. 还原:
     return L · copysign(0.5 - d, V0 - 0.5)
```

---

*本计划已按审核意见 [初次审核意见书](migration_audit_zh.md) 及 [二次审核意见书](migration_reaudit_zh.md) 修订。全部 10 个阻塞项（B1-B10）和 6 个强建议项（S1-S6）均已处理。关键更正：YACCLAB CCL（禁止简化）、wp.float64 气泡数组、CMR-MRT 气体碰撞（非 BGK）、单一 stream_collide_bvh kernel 内联湍流、两阶段求解器管线、补充 4 个 State 字段、补充 9 个遗漏 kernel、State 继承 DomainState（非 FluidGridStateBase）、turbulence_radius=3、新增金标准数据生成流程章节。*
