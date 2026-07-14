# HOME-FSLBM DamBreak — 开发状态

> **日期**: 2026-07-14 | **当前阶段**: 阶段 2 完成

---

## 阶段 2 概要

阶段 2 实现了中心矩 MRT 碰撞与单核融合 `stream_collide_kernel`，对标 [REF] `stream_collide_bvh`。

### 修复的关键 Bug

**Bug 1 — 系数表角方向行错位（严重）**

`constants.py` 中 `_COEFFS` 的方向 21-26 行与 `C_X/C_Y/C_Z` 方向向量不匹配。参考代码的 switch case 索引与方向向量的对应关系被错误提取——6 个角方向的系数行全部错位。

- 方向 21 (1,1,-1) → 使用了方向 26 (1,-1,-1) 的系数
- 方向 22 (-1,-1,1) → 使用了方向 25 (-1,1,1) 的系数
- ...（全部 6 个角方向错位）

**影响**: 重建的分布函数在角方向上的速度贡献丢失 1/9（恰好是角方向组的速度权重），导致重建→迁移的往返速度缩放为 8/9。这使所有体力驱动流（重力、Poiseuille）系统性衰减。

**修复**: 根据参考代码 `mrUtilFuncGpu3D.h` 的 switch case 重新对齐角方向系数行。

**Bug 2 — 体力位移后应力不一致**

Guo 体力施加将速度从 U 位移到 U + F/(2ρ)，但存储的应力 S_ab 对应位移前的速度。下一时间步的重建使用位移后的速度 + 位移前的应力，产生非平衡分量。

**修复**: 存储应力时增加位移平方修正 `S_ab += du_a * du_b`，使存储应力与位移后速度的平衡态一致。

---

## 验收状态

### 阶段 1 回归（unittest）
```
uv run --extra examples python -m unittest wanphys._src.fluid.fluid_grid.home_fslbm.tests.test_home_fslbm_reconstruct -v
uv run --extra examples python -m unittest wanphys._src.fluid.fluid_grid.home_fslbm.tests.test_home_fslbm_e2e_periodic -v
```
**15/15 PASS**

### 阶段 2 验收（独立脚本）
```bash
uv run --extra examples python wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_home_fslbm_stage2.py --quick
uv run --extra examples python wanphys/_src/fluid/fluid_grid/home_fslbm/tests/test_home_fslbm_stage2.py  # 完整（含 Poiseuille 20k步 + Cavity 2k步）
```

| 编号 | 验收标准 | 容差 | 状态 |
|------|---------|------|------|
| A.1 | 质量守恒 Σρ 漂移 | < 5e-3 (float32) | PASS |
| A.3 | 静止稳定性 max|u|<1e-10 (1000步) | < 1e-10 | PASS |
| A.6 | 重力驱动 u_y/(g·t) ≈ 0.5 | ±0.02 | PASS |
| A.7 | 速度限制 max|u|≤0.4 | — | PASS |
| B.8 | 壁面无穿透 | < 1e-10 | PASS |
| B.9 | Poiseuille 中心线速度 | < 5% | PASS |
| B.9 | Poiseuille 壁面无滑移 | < 1e-6 | PASS |
| B.10 | Courant max|u|·dt/dx < 0.4 | — | PASS |
| B.11 | 腔体衰减能量单调递减 | 严格递减 | PASS |
| B.11 | 腔体终态 KE < 1e-4 | — | PASS |
| C.13 | 阶段 1 回归 (15/15) | — | PASS |
| C.15 | 无中文注释 | — | PASS |

---

## 代码文件

| 文件 | 内容 |
|------|------|
| `constants.py` | D3Q27 格子常量 + 修正后的 (27,17) 系数表 |
| `kernels.py` | `reconstruct_fi_at_index`, `equilibrium_fi`, `flag_domain_boundary_kernel`, `stream_collide_kernel` |
| `solver.py` | `HomeFSLbmSolver` — step 流水线 + MomSwap + 初始化 + 诊断 |
| `domain.py` | `HomeFSLbmDomain` — 双缓冲 + MomSwap |
| `boundary.py` | `normalizing_clamp`, `equilibrium_fi` |
| `model.py` | `HomeFSLbmModel` — 静态配置 dataclass |
| `state.py` | `HomeFSLbmState` — GPU 时变状态 (10-moment SoA) |
| `tests/test_home_fslbm_stage2.py` | 阶段 2 验收套件 |
| `tests/_safe_import.py` | Warp 1.12.0 import bug 绕过模块 |

---

## 已知限制

1. **Warp 1.12.0 import bug**: `broad_phase_hash.py` 中的 `@wp.kernel` 装饰器与 Python 3.11 tokenizer 冲突，导致通过 `unittest` 框架导入时间歇性崩溃。独立脚本通过 `sys.modules` 预注册绕过。
2. **float32 精度**: 质量/动量守恒漂移约 1e-3 量级。生产级长时间模拟建议使用 float64。
3. **Poiseuille 收敛时间**: 64×32×4 网格需约 20000 步达到稳态（特征扩散时间 ~ L²/ν）。

---

## 下一步：阶段 3

自由表面三步（surface_1/2/3），对标 [REF] `mrLbmSolverGpu3D.cu` L444-701。

参考：`docs/wanphys/home_fslbm_dambreak_plan_zh.md` 第 6 节。
