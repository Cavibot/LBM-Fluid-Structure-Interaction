# WanPhys LBM 刚体展示层验收清单

本文用于验收 LBM 刚体耦合的展示层入口，不替代 kernel 级 regression tests。这里关注的是示例能否启动、viewer 路径是否接通、基础 telemetry 是否有限，以及操作者是否能看到合理的视觉信号。

重要边界：

- `one-way` 路径验证刚体边界能驱动 LBM 速度扰动。
- `two-way` 路径验证 LBM boundary feedback 能进入刚体 telemetry。
- two-way 场景使用 `xmin` 速度入口和 `xmax` 出口维持持续来流；只设置初始速度会在几十到几百帧内耗散成近似静止场。
- two-way GL 默认使用 GPU tracer points 展示流动。它们默认播撒在 y/z 中心流道，随 LBM 速度场平流；越界或靠近固体到安全间隙内后会从入口重新播撒，用于肉眼确认流动正在推进，避免贴墙边界层或刚体表面造成误判。
- two-way GL 的 SSFR 标量场现在是可选诊断层，通过 `--fluid-render-mode field` 或 `--fluid-render-mode both` 打开。它显示刚体附近流体区域的速度扰动场 `|u - u0|`，不是自由液面，也不是整域绝对速度 `|u|`。低分辨率网格下这个诊断层会有 voxel 棱角，不能作为真实水面验收。
- 当前 `two-way force magnitude` 只作为 diagnostic early feedback。它可以帮助确认路径接通、数值有限、趋势可观察，但目前不应作为物理正确性阈值或稳定标定指标。

## 1. One-way GL 可视化

命令：

```bash
uv run --python 3.11 --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_fsi --viewer gl
```

预期信号：

- OpenGL viewer 能打开，不出现 traceback。
- 控制台打印 `FSI:`、网格尺寸、球半径、球速等初始化信息。
- 按 Space 取消暂停后，刚体球沿 +x 方向运动。
- 速度场可视化出现围绕球体的扰动或尾迹，`speed [min, max]` 中的 max 保持有限值。
- 该入口是 one-way：不要求看到刚体被流体反馈推动，也不要求输出 `body_f`。

如果本机没有可用 OpenGL，上述命令可以失败；这属于环境验收问题，不代表 headless coupling 路径失败。

## 2. Two-way GL 可视化

命令：

```bash
uv run --python 3.11 --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi_visual --viewer gl
```

如果需要同时查看 SSFR 诊断场，建议使用更高分辨率、相同物理尺寸的网格：

```bash
uv run --python 3.11 --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi_visual --viewer gl --fluid-render-mode both --grid-res 32 24 24 --cell-size 0.05 --tracer-count 768
```

预期信号：

- OpenGL viewer 能打开，不出现 traceback。
- 场景中能看到 LBM 流场和刚体球；球体附近速度扰动持续更新，tracer points 沿 +x 来流方向移动。
- 控制台或 overlay 能看到 two-way telemetry，例如 `force_norm`、`torque_norm`、`display=[min,max]`、`flow=[min,max]`。
- `force_norm`、`torque_norm`、`display` 和 `flow` 数值均为 finite。
- `flow` 是真实流体速度范围；`display` 是给 SSFR 的扰动标量范围。验收“水有没有动”时优先看 `flow` 和 tracer points。
- `force_norm` 可以作为早期诊断信号观察，但不要求达到固定大小；目前只证明反馈路径接通，不证明力模型已经完成物理标定。

建议输出格式保持机器可读，例如 helper 返回 `force_norm / torque_norm / speed_max / flow_speed_max` 字段，CLI 再把它们打印为清晰的 key-value 文本。

## 3. Two-way null smoke

命令：

```bash
uv run --python 3.11 --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi_visual --viewer null --num-frames 3 --quiet
```

预期信号：

- 不创建 OpenGL 窗口。
- 示例完成有限帧数后退出，退出码为 0。
- 输出或可测试 helper 返回 finite 的 `force_norm`、`torque_norm`、`speed_max`、`flow_speed_max`。
- `flow_speed_max` 应能反映 LBM 速度场已推进；它不应为 NaN、inf 或在持续来流场景中衰减到接近 0。
- `force_norm` 和 `torque_norm` 的 magnitude 仍然只作为 diagnostic early feedback，不作为验收阈值。

该路径适合本地快速检查和 CI focused smoke。对应测试在 `newton/tests/test_lbm_rigid_visualization.py` 中只导入模块并调用 smoke helper，不要求 OpenGL。

## 4. 可选 USD 导出

命令：

```bash
uv run --python 3.11 --extra examples python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi_visual --viewer usd --output-path .docs/lbm_rigid_twoway_fsi.usd --num-frames 60 --quiet
```

预期信号：

- 命令退出码为 0。
- `.docs/lbm_rigid_twoway_fsi.usd` 被写出且文件非空。
- 用 Omniverse、Blender 或 USD 工具打开时，能看到刚体球随帧更新；如果当前 viewer 支持流体场几何或体渲染导出，也应看到与 GL 路径一致的流场提示。
- CLI telemetry 仍应给出 finite 的 `force_norm`、`torque_norm`、`display`、`flow`，方便和 null smoke 对齐。

USD 路径是可选展示验收，不应阻塞 headless two-way smoke；它主要用于录屏、离线检查和跨机器复现视觉结果。

## 5. 验收结论模板

记录一次验收时建议保存以下信息：

- Git commit 或工作树说明。
- 运行设备，例如 `cpu`、`cuda:0`。
- one-way GL 是否能打开，是否看到球体运动和速度扰动。
- two-way GL 是否能打开，是否看到 finite telemetry。
- null smoke 的 `force_norm / torque_norm / speed_max / flow_speed_max`。
- USD 文件路径、大小，以及是否能被外部工具打开。

再次强调：在 two-way 反馈公式和尺度完成标定前，`force_norm` 的具体大小只说明早期诊断状态，不是物理验收标准。


