# P0 基线冻结记录

## 1. 冻结结论

| 项目 | 记录 |
|---|---|
| 状态 | `CPU_ACCEPTED` |
| 测试日期 | 2026-07-29 |
| 冻结代码 SHA | `2e05a8ed348f2612401ce419120fb9938449172b` |
| 配置/状态改造提交 | `a90385a` |
| 单次 hydrodynamic closure 提交 | `430f005` |
| CUDA | `CUDA_NOT_ACCEPTED`；当前 Warp 构建未启用 CUDA |

P0 冻结的是既有 LBM 与 SC-to-VOF 调试观察行为，不代表 authoritative VOF 已实现。

## 2. 环境

```text
OS       macOS 15.3.1, arm64
Python   3.11.14
Warp     1.12.0
NumPy    2.4.2
device   cpu (arm)
CUDA     no devices; CUDA not enabled in this Warp build
storage  Warp float32 (density/populations/VOF-shaped debug arrays)
```

## 3. 固定回归命令

核心配置、碰撞、力、streaming 与观察：

```bash
uv run python -m unittest \
  newton.tests.lbm_part123.test_part0_contracts \
  newton.tests.lbm_part123.test_part1_collision_spaces \
  newton.tests.lbm_part123.test_part2_force_pipeline \
  newton.tests.test_lbm_collision_backends \
  newton.tests.test_lbm_home_nocm \
  newton.tests.test_lbm_streaming_matrix \
  newton.tests.test_lbm_shan_chen_wall_force \
  newton.tests.test_lbm_debug_mock_sc_to_vof -v
```

结果：`Ran 39 tests ... OK`。

方向 streaming 与剪切波：

```bash
uv run python -m unittest newton.tests.test_lbm_directional_streaming -v
```

结果：`Ran 6 tests ... OK`。

静态检查：

```bash
uvx ruff check --select F,I <P0 changed Python files>
```

结果：`All checks passed!`。完整仓库仍有本次改造前已存在的 line-length 与其他 lint
债务，不纳入 P0 物理验收。

## 4. 已冻结证据

- 配置矩阵覆盖 `off/shan_chen/vof`、gravity、`G` 和 observation 的合法/非法组合。
- `force_model` 不能由调用者配置，只能由 `interface_model + gravity` 生成。
- positivity 开关两侧及 FullF/HOME 已声明碰撞路径均满足第一步
  `force_x=g`、`velocity_x=0.5g`、碰撞后动量 `rho*g`。
- observation 开关前后 populations、density、velocity 与 force 数组逐元素相等。
- 调试状态的 clone/copy/clear、density 分类、周期法向、固体掩码和 dam-break 界面已覆盖。
- Visualizer 前后 density、`phi/cell_type/normal` 逐元素不变。
- FullF/HOME streaming、碰撞 reference、静态 bounce-back、SC wall force 与剪切波回归通过。

## 5. 写入权限冻结

| 数据 | P0 权威写入者 | 可视化权限 |
|---|---|---|
| Shan-Chen density | LBM/SC pipeline | 只读 |
| `state.debug_mock_sc_to_vof.phi/cell_type` | `DebugMockScToVofObserver` | 只读 |
| `.normal` | `InterfaceGeometry` 调试计算 | 只读 |
| `state.vof.mass/phi/cell_type` | 无；P1 才引入 VOF stepper | P0 不存在 |
| 渲染点、坐标、颜色、normal line | `VofInterfaceVisualizer` | 可计算渲染数据 |

Visualizer 不得计算或修改 `phi/cell_type/normal/curvature`。

## 6. 已知限制

- authoritative `state.vof`、`mass`、质量平流、自由面 population 补全、类型转换、
  PLIC、curvature 与表面张力均未开始；`interface_model="vof"` 会 fail-fast。
- `DebugMockScToVofState` 完全由 density 派生，不守恒，也不能反馈给 LBM。
- 零力静止平衡的 27-cell float32 总质量观察到确定性误差
  `9.65595245e-6`。独立测试提交 `2e05a8e` 将边界设为每格低于四个 float32
  epsilon；它不改变生产算法。
- 当前无 CUDA 设备和 CUDA-enabled Warp，不能声称 CPU/CUDA 一致性已验收。

P1 可以在此基线上开始 `mass/phi/cell_type/config/initialization`，但不得复用
`state.debug_mock_sc_to_vof` 作为正式状态。
