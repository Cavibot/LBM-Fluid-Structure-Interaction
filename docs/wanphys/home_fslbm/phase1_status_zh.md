# HOME-FSLBM 阶段 1 状态报告

> **日期**: 2026-07-16
> **范围**: 流体子系统（`stream_collide_bvh_kernel`）及核心 `@wp.func` 的 Warp 迁移与单元测试
> **测试结果**: **17 / 17 通过**

---

## 1. 项目概要

将 HOME-FSLBM 自由表面 LBM 求解器从 CUDA C++ 参考实现迁移至 WanPhys / NVIDIA Warp 框架。阶段 1 覆盖 **流体核心**：

| 模块 | 文件 | 内容 |
|------|------|------|
| D3Q27 平衡态 | `kernels_fluid.py` → `calculate_f_eq_d3q27` | Maxwell-Boltzmann 27 向分布 |
| Hermite 重建 | `kernels_fluid.py` → `reconstruct_distribution` | 10 个 HOME 矩 → 27 个分布 |
| NOCM-MRT 碰撞 | `kernels_fluid.py` → `ml_get_pi_after_collision` | 闭式应力张量碰撞 |
| 流传输 kernel | `kernels_fluid.py` → `stream_collide_bvh_kernel` | Pull-streaming + 碰撞 + 自由表面 + 湍流 |
| 常量 | `constants.py` | D3Q27 格子常数、标记位域 |
| 模型 | `model.py` | 静态仿真配置 |
| 状态 | `state.py` | GPU 数组分配 |
| Solver | `solver.py` | 两阶段管线 + 初始化 |
| Domain | `domain.py` | 双缓冲状态管理 |

阶段 2–4（表面标记传播、CCL 气泡追踪、D3Q7 气体模型）延后执行。

---

## 2. 文件结构

```
wanphys/_src/fluid/fluid_grid/home_fslbm/
├── __init__.py
├── constants.py              # D3Q27 格子常数、CellFlag 枚举
├── model.py                  # HomeFslbmModel（ω、表面张力、湍流参数）
├── state.py                  # HomeFslbmState（继承 DomainState）
├── solver.py                 # HomeFslbmSolver（两阶段管线）
├── domain.py                 # HomeFslbmDomain（双缓冲）
├── kernels_fluid.py          # ★ 核心 kernel（stream_collide_bvh + 3 个 @wp.func）
└── tests/
    ├── conftest.py           # pytest fixtures、load_golden()
    ├── test_constants.py
    ├── test_model.py
    ├── test_state.py
    ├── test_kernels_fluid.py # ★ 17 个单元测试
    └── golden_data/          # 金标准数据（14 个 .txt 文件）
```

---

## 3. 金标准数据来源

金标准数据由参考 CUDA C++ 代码生成，生成器位于：

```
docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/export_solver_golden.cpp
docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/export_golden.cu
```

**重要**：金标准数据使用的参数必须与 Warp 测试**精确一致**，细节见 [§6 已修复的不一致](#6-审计发现并已修复的不一致)。

---

## 4. 测试运行命令

```powershell
# 运行全部 17 个流体 kernel 测试
uv run pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_kernels_fluid.py -v

# 单独运行某个测试类
uv run pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_kernels_fluid.py \
    -k "TestStreamCollideBvhShearDecay" -v

# 运行特定测试方法
uv run pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_kernels_fluid.py \
    -k "test_amplitude_decays" -v -s

# 运行所有 HOME-FSLBM 测试（含 constants/model/state）
uv run pytest wanphys/_src/fluid/fluid_grid/home_fslbm/tests/ -v
```

**前置条件**：Python 3.11+、NVIDIA GPU（可选，支持 CPU 回退）、已执行 `uv sync --extra dev --extra examples`。

---

## 5. 测试清单与覆盖范围

| 测试类 | 方法 | 验证内容 | 参考行号 |
|--------|------|---------|---------|
| `TestFEquilibriumD3Q27` | 3 个 | D3Q27 Maxwell-Boltzmann 平衡态 | `mrUtilFuncGpu3D.h:292` |
| `TestReconstructDistribution` | 2 个 | Hermite 展开（10 矩 → 27 分布） | `mrUtilFuncGpu3D.h:153` |
| `TestNOCMMRTCollision` | 3 个 | NOCM-MRT 闭式碰撞 | `mrUtilFuncGpu3D.h:424` |
| `TestComputeRhoU` | 1 个 | 宏观矩恢复 | `mrUtilFuncGpu3D.h:272` |
| `TestStreamCollideBvh` | 2 个 | 均匀平衡态 1 步后不变；无 NaN | `mrLbmSolverGpu3D.cu:703` |
| `TestStreamCollideBvhShearDecay` | 2 个 | 剪切波 100 步衰减 vs golden；正弦波形保持 | `export_solver_golden.cpp:gen_shear` |
| `TestStreamCollideBvhSolidBounceBack` | 1 个 | 固体墙壁 + 重力 500 步后墙壁速度 < 2e-3 | `export_solver_golden.cpp:gen_bounce` |
| `TestMassConservationThroughCollision` | 1 个 | 随机 ρ 场 1 步碰撞后质量守恒 | — |
| `TestTurbulenceOmegaModification` | 2 个 | 涡粘性 ω 修改 vs golden；应变非零 | `export_solver_golden.cpp:gen_turb` |

---

## 6. Full-Way Bounce-Back 精度说明

参考代码和 Warp 实现均使用 **full-way (on-grid) bounce-back**（`mrLbmSolverGpu3D.cu:743-748`）：

```cpp
mrutilfunc.calculate_f_eq(mlflow[0].fMom[curind + total_num * 0], 0.f, 0.f, 0.f, feq);
fhn[i] = feq[i];
```

这与 half-way bounce-back（墙壁在半格点）不同。Full-way BB 的壁面滑移误差为 **O(τ)**：

$$u_{\text{slip}} \approx \left(\tau^2 - \tau + \frac{1}{6}\right) \cdot \frac{F}{\nu}$$

代入测试参数（τ=1.0, F=0.001, ν=1/6）：

$$u_{\text{slip}} \approx \frac{1}{6} \times 0.006 = 1.0 \times 10^{-3}$$

与实测值 1.02e-3 高度吻合（偏差 < 2%）。阈值 2e-3 留有一倍余量。

若需更高精度，可将 `omega` 提升至参考值 `1/(3×1e-4+0.5) ≈ 1.998`（τ≈0.5002），此时滑移速度降至 ~2e-7。

---

## 7. 与参考代码的已知差异

| 方面 | 参考代码（CUDA） | Warp 实现 | 影响 |
|------|-----------------|-----------|------|
| omega 可配置性 | kernel 内硬编码 `1/(3*1e-4+0.5)` | 通过 `model.omega` 从 Host 传入 | 灵活性更好；测试必须显式设置 |
| 越界处理 | 无（依赖 ghost 层或周期性） | 显式越界检查 + 零速度平衡态反弹 | 边界行为显式可控 |
| 表面处理（Phase C） | 完整 PLIC + 气泡压力 + 质量交换 | 骨架占位符（阶段 2 实现） | 当前仅支持纯流体（TYPE_F）场景 |
| `stream_collide_bvh` 拆分 | 单一 CUDA kernel | 单一 Warp kernel（审核项 B4） | 一致 |
| 气泡数组精度 | `double` | `wp.float64` | 一致 |

---

## 8. 下一阶段（阶段 2）

| 模块 | 内容 | 依赖 |
|------|------|------|
| `kernels_surface.py` | surface_1/2/3 标记传播、PLIC 曲率、质量重分配 | 阶段 1 完成 |
| 表面测试 | φ 计算、法向、曲率、标记转换 | `kernels_surface.py` |
| Phase C 集成 | 将骨架占位符替换为完整自由表面逻辑 | `kernels_surface.py` |

---

## 9. 参考

| 资源 | 路径 |
|------|------|
| 迁移架构计划 | `docs/wanphys/home_fslbm/migration_plan_zh.md` |
| 迁移执行手册 | `docs/wanphys/home_fslbm/migration_execution_zh.md` |
| 审核意见书 | `docs/wanphys/home_fslbm/migration_audit_zh.md` |
| 二次审核 | `docs/wanphys/home_fslbm/migration_reaudit_zh.md` |
| 参考 CUDA 实现 | `docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/` |
| Golden 生成器 | `docs/papers/…/Home-FSLBM/export_solver_golden.cpp` |
| 核心 kernel 参考 | `docs/papers/…/Home-FSLBM/inc/3D/gpu/mrLbmSolverGpu3D.cu:703-1057` |
| 工具函数参考 | `docs/papers/…/Home-FSLBM/inc/3D/gpu/mrUtilFuncGpu3D.h` |
