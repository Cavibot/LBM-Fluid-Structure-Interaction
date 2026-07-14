# HOME-FSLBM DamBreak — 开发状态

> **日期**: 2026-07-14 | **当前阶段**: 阶段 3 完成

---

## 阶段 3 概要

阶段 3 实现了自由表面三步（surface_1/2/3），对标 [REF] `mrLbmSolverGpu3D.cu` L444-701。支持 VOF 追踪、界面过渡标记、新生界面初始化、质量交换与重标记。

### 新增文件/变更

| 文件 | 变更 |
|------|------|
| `kernels.py` | surface_1/2/3 kernel + _calculate_phi @wp.func；stream_collide 增加 TYPE_G 跳过、TYPE_I φ加权质量通量、massex 收集 |
| `solver.py` | step 流水线集成 surface_1/2/3 + calculate_disjoint_kernel；动态 flag 同步 |
| `boundary.py` | calculate_phi + calculate_curvature 骨架（σ=0） |
| `tests/test_home_fslbm_free_surface.py` | 阶段 3 验收测试套件 |

### 验收命令
```
# 快速测试（跳过长时间运行的静态水柱和排水测试）
uv run --extra examples python wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_home_fslbm_free_surface.py --quick

# 完整测试（含 500 步静态水柱测试）
uv run --extra examples python wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_home_fslbm_free_surface.py
```

### 验收状态

| 编号 | 测试 | 状态 | 备注 |
|------|------|------|------|
| 1 | VOF 有界性 (0 ≤ φ ≤ 1) | PASS | |
| 2 | 静态水柱 φ 不变 (500步) | PASS | |
| 3 | 封闭域质量守恒 | PASS | |
| 4 | TYPE_F/TYPE_G 不直接相邻 | PASS | |
| 5 | GI 初始化精度 | PASS | rho=0.833 (多邻居平均) |
| 6 | 过渡比率 < 5% | PASS | |
| 7 | 排水：200步无 NaN | **FAIL** | 重力 (`g_y=-5e-5`) 下第 ~53 步出现 NaN，需完整 gas-side closure |
| 8 | 排水：质量下降 | **FAIL** | NaN 导致无法检测，根因同 #7 |
| 9 | 阶段 1+2 回归 | PASS | 10+5+15=30 全部通过 |

### 已知限制

1. **界面→气体质量泄漏**: 静态界面单元因简化 gas-side closure（偶数/奇数方向近似替代参考代码的 se 判定），缓慢向气体单元泄漏质量。全流体封闭域质量守恒正常；存在气体域时每步 ~0.5% 损失。计划阶段 6 修复。

2. **重力驱动排水 NaN**: 重力下 (`g_y=-5e-5`) 排水测试第 ~53 步出现 NaN，因完整 gas-side closure 未实现。当前 FAIL，阶段 6 修复。

---

## 阶段 2 概要

阶段 2 实现了 `stream_collide_kernel`（单核融合迁移 + 中心矩 MRT 碰撞），对标 [REF] `stream_collide_bvh`。

### 已修复的 Bug

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 1 | `constants.py` | 系数表角方向 21-26 行与方向向量不匹配，致重建速度缩放 8/9 | 对齐至参考代码 switch case 顺序 |
| 2 | `constants.py` | 非对角应力系数 (Axy_t9, Axz_t9, Ayz_t9) 为正确值的 2 倍 | 列 5,6,8 全部 ÷2 |
| 3 | `solver.py` | numpy reshape→Warp 3D array 拷贝致 flag 错位（一格偏移） | 改为 Warp kernel 拷贝 |
| 4 | `kernels.py` | 反弹使用纯平衡态 feq(rho,0,0,0,d) 丢失非平衡应力 | 改为标准半程反弹 f_on[OPPOSITE[d]] |
| 5 | `kernels.py` | Guo 体力速度位移后存储应力未同步（du² 缺失） | 存储应力加 du_a·du_b 修正 |
| 6 | `kernels.py` | 速度限制在位移前，位移后可超 max_vel | 限制移到 U+du 之后 |
| 7 | `domain.py` | MomSwap 未正确交换 f_mom↔f_mom_post | 在 step 内联指针交换 |

### 算法实现

- **碰撞**: 中心矩 MRT [计划文档 §3.2, P4 Eq.(18-21)]
- **重建**: 三阶 Hermite 展开 [P4 Eq.(16-17)]
- **体力**: Guo 方案，碰撞后位移 F/(2ρ) + 应力 du² 修正
- **边界**: 半程反弹 + 周期 wrap
- **单核融合**: stream + collide + write 在一个 Warp kernel 内

---

## 验收状态

### 阶段 1 回归（unittest）
```
uv run --extra examples python -m unittest wanphys._src.fluid.fluid_grid.home_fslbm.tests.test_home_fslbm_reconstruct -v
uv run --extra examples python -m unittest wanphys._src.fluid.fluid_grid.home_fslbm.tests.test_home_fslbm_e2e_periodic -v
```
**15/15 PASS**

### 阶段 2 验收（独立脚本）
```
uv run --extra examples python wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_home_fslbm_stage2.py --quick
uv run --extra examples python wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_home_fslbm_stage2.py
```

| 编号 | 测试 | 状态 | 备注 |
|------|------|------|------|
| A.1 | 质量守恒 | PASS | |
| A.3 | 静止稳定性 | PASS | |
| A.6 | 重力驱动 | PASS | |
| A.7 | 速度限制 | PASS | |
| B.8 | 壁面无穿透 | PASS | |
| B.9 | 壁面无滑移 | PASS | |
| B.9 | 抛物线形状 | PASS | |
| B.9 | **中心线速度** | **FAIL** | 见已知问题 #1 |
| B.10 | Courant | PASS | |
| B.11 | **单调衰减** | **PASS** (2000步) | 8000步时数值抖动偶现失败，见 #2 |
| B.11 | **终态动能** | **FAIL** | 见已知问题 #1 |
| B.11 | 无 NaN/Inf | PASS | |
| C.13 | 阶段 1 回归 | PASS | |
| C.15 | 无中文注释 | PASS | |

---

## 已知问题

### 1. 剪切流速度偏低 / 能量不衰减（B.9·B.11 根因）

**现象**: Poiseuille 中心线速度为理论的 ~12%；腔体衰减动能冻结在 ~0.145。

**根因**: 单弛豫时间 ω=1.82 (τ=0.55) 致非对角应力过弛豫（|1-ω|=0.82）。碰撞后 S_xy 反转符号并衰减到 82%，streaming 从速度梯度再生应力。稳态平衡下时间平均应力仅约理论值的 12%。

**计划修复**: 阶段 6b 涡粘性模型 [P4 §4.2; REF stream_collide_bvh:1021-1023]。该模型根据局部 Frobenius 范数动态调整 ω：高速剪切区 ω 趋近 2（更强耗散），静止区 ω 退化。当前阶段 2 的固定 ω 无此能力。

### 2. B.11 单调性在 8000 步偶现失败

**现象**: 2000 步时 PASS，8000 步时可能出现 FAIL。

**成因**: 动能冻结在 0.1447，float32 精度下多采样点累积 ULP 级波动触发 `1e-10` 容差。非物理非单调，但暴露了动能无法进一步衰减的根因（同 #1）。

### 3. Warp 1.12.0 import bug

`broad_phase_hash.py` 的 `@wp.kernel` 与 Python 3.11 tokenizer 冲突。绕过方式：独立脚本通过 `sys.modules` 预注册空壳模块阻断碰撞模块导入链。不影响核心功能。

---

## 代码文件

| 文件 | 变更 |
|------|------|
| `constants.py` | 系数表角方向重新排序 + 非对角列 ÷2 |
| `kernels.py` | 中心矩 MRT 碰撞；半程反弹；du² 修正；速度限制位移后；质量通量 |
| `solver.py` | Warp kernel flag 拷贝；`_flag_flat` 管理；MRT 系数表 |
| `domain.py` | MomSwap 修复 |
| `model.py` | 无变更 |
| `boundary.py` | 无变更 |
| `tests/test_home_fslbm_stage2.py` | 阶段 2 验收套件 |
| `tests/_safe_import.py` | Warp import bug 绕过 |

---

## 下一步：阶段 4

DamBreak 样例脚本 `wanphys/examples/home_fslbm_dambreak.py`。配置：200×100×50 网格，τ=0.55，重力 (0, -5e-5, 0)，8000 步。

可同步推进阶段 6b（涡粘性）修复 B.9/B.11。

参考：`docs/wanphys/home_fslbm_dambreak_plan_zh.md` 第 7、9.2 节。
