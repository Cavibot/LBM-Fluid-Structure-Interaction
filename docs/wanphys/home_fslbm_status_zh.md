# HOME-FSLBM DamBreak — 开发状态

> **日期**: 2026-07-13 | **当前阶段**: 阶段 1 完成

---

## 当前进度

阶段 1（基础设施与分布重建）已完成。模块位于：

```
wanphys/_src/fluid/fluid_grid/home_fslbm/
```

已实现：
- D3Q27 格子常量与 (27,17) 系数表 (`constants.py`)
- HomeFSLbmModel / HomeFSLbmState / HomeFSLbmDomain / HomeFSLbmSolver
- `reconstruct_fi_at_index` — 三阶 Hermite 分布重建 (`kernels.py`)
- 反弹边界辅助函数 (`boundary.py`)

---

## 测试用例

阶段 1 验收标准共 15 项，当前全部通过。

运行命令：
```bash
uv run --extra examples python -m unittest wanphys._src.fluid.fluid_grid.home_fslbm.tests.test_home_fslbm_reconstruct -v
uv run --extra examples python -m unittest wanphys._src.fluid.fluid_grid.home_fslbm.tests.test_home_fslbm_e2e_periodic -v
```

| 编号 | 验收标准 | 容差 | 状态 |
|------|---------|------|------|
| A.1 | D3Q27 权重归一化 `sum(W) = 1.0` | < 1e-15 | PASS |
| A.2 | 反向索引 `C[i] + C[OPPOSITE[i]] = (0,0,0)` | 精确 | PASS |
| A.3 | 静止平衡态 `ρ=1, u=0, S=0` → `f_i = w_i` | < 1e-10 | PASS |
| A.4 | 匀速流矩守恒 `ρ=1, u=(0.1,0,0)` 重建后 Σf_i=1, Σc_i f_i = u | < 1e-6 / < 1e-4 | PASS |
| A.5 | GPU 重建与 NumPy 参考实现逐方向一致 | < 1e-5 | PASS |
| A.6 | Model 参数 `tau > 0.5`, `bc_types` 长度 = 6 | — | PASS |
| A.7 | State 内存 `12 × nx × ny × nz × 4` bytes | — | PASS |
| B.8 | `(8,8,8)` 周期网格 100 步无崩溃、无 NaN、max|u| < 1e-10 | — | PASS |
| B.9 | Domain 双缓冲翻转一致性 | — | PASS |
| B.10 | `create_state()` 返回独立 state | — | PASS |
| B.11 | `solver.step()` 被 `domain.step()` 正确调用 | — | PASS |
| C.12 | 反弹边界静止壁面 `f_hn[d] = feq(rho, 0,0,0, d)` | < 1e-12 | PASS |
| C.13 | 反弹无穿透 `f_hn[d] = f_on[OPPOSITE[d]]` | < 1e-12 | PASS |
| C.14 | 域边界 6 个面 `TYPE_S` 标记正确 | — | PASS |
| C.15 | 无中文注释 | — | PASS |

---

## 原始要求与禁令

1. **零侵入**: 不修改 `wanphys/_src/fluid/fluid_grid/lbm/` 下任何文件
2. **禁止降级**: D3Q27（非 D3Q19）、三阶 Hermite 展开、中心矩 MRT 碰撞、单核融合、完整自由表面三步
3. **无中文注释**: 所有代码注释和 docstring 为英文
4. **公式可追溯**: 所有数据模型和伪代码标注论文公式编号或参考代码行号
5. **禁止冒烟即可**: 每个阶段必须通过全部列出的严格验收标准，必须可运行、行为正确、边界处理得当
6. **独立 oracle**: 测试不能自己验证自己
7. **参考代码**: `docs/papers/锐界面动力学自由表面流与泡沫/Home-FSLBM/`
8. **主论文**: Wang et al. 2025 — *Kinetic Free-Surface Flows and Foams with Sharp Interfaces*
