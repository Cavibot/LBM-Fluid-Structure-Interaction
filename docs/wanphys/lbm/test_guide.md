# LBM 性能测试操作说明

本文档用于指导如何测试 LBM 求解器和 LBM 示例的运行性能。测试内容主要包括：

- 纯 LBM 计算性能
- 不同网格分辨率下的缩放性能
- 真实示例在 `null` viewer 和 `gl` viewer 下的性能差异
- GPU 显存占用
- 测试结果出图

## 1. 测试前准备

所有命令都在项目根目录执行：

```bash
cd ....../LBM-Fluid-Structure-Interaction
```

确认当前机器可以使用 NVIDIA GPU：

```bash
nvidia-smi
```

如果 `nvidia-smi` 能正常显示显卡信息，就可以继续测试。推荐先从小网格开始，确认环境可运行后再跑大网格。

## 2. 快速冒烟测试

先跑一个很小的纯 LBM 测试，确认 Warp、CUDA、LBM kernel 都能正常工作：

```bash
uv run python scripts/bench/bench_lbm.py -n 64 -f 60 -w 10
```

参数含义：

- `-n 64`：网格为 `64^3`
- `-f 60`：正式统计 60 帧
- `-w 10`：先预热 10 帧，不计入统计

如果能看到类似下面的信息，说明基础测试通过：

```text
LBM Benchmark Results
Grid         : 64^3
Average     : ... ms -> ... FPS
GPU Memory  : ...
```

## 3. 纯 LBM 分辨率扫描

用于测试不同网格规模下，纯 LBM 求解器的性能变化。

```bash
uv run python scripts/bench/bench_lbm.py \
  --scan 64,96,128,160,192 \
  -f 100 -w 20 \
  --csv scripts/bench/bench_results/dambreak_scale/aggregate.csv
```

这个测试不打开 viewer，不测 OpenGL 渲染，只看 LBM 计算本身。

推荐观察：

- `Avg ms`：平均每帧耗时
- `Avg FPS`：平均帧率
- `P99 ms`：99% 帧都不超过的耗时，用来看卡顿
- `Warp Mem`：Warp mempool 显存占用
- `Process Mem`：`nvidia-smi` 看到的 GPU 显存占用参考值

一般判断标准：

- `128^3` 仍能高于 60 FPS：适合实时交互
- `160^3` 接近或低于 60 FPS：开始进入压力区
- `192^3` 低于 30 FPS：适合压力测试或离线录制

## 4. 测试真实 LBM 示例

使用 `bench_lbm_run.py` 可以运行现有示例，并拆分统计模拟和渲染耗时。

### 4.1 只测模拟性能

使用 `null` viewer，不打开真实 OpenGL 渲染：

```bash
uv run python scripts/bench/bench_lbm_run.py \
  -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt \
  -f 120 -w 20 \
  --viewer null \
  --csv scripts/bench/bench_results/substeps2/aggregate.csv
```

`null` viewer 的结果主要反映模拟本身的耗时。

### 4.2 测试真实 GL 渲染性能

使用 `gl` viewer，会统计模拟加 OpenGL/SSFR 渲染的总耗时：

```bash
uv run python scripts/bench/bench_lbm_run.py \
  -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt \
  -f 120 -w 20 \
  --viewer gl \
  --csv scripts/bench/bench_results/substeps2/aggregate.csv
```

对比 `null` 和 `gl` 的结果，可以判断渲染开销有多大。

重点看这三项：

- `Total Frame`：模拟 + 渲染总耗时
- `Step only`：只包含模拟耗时
- `Render only`：只包含渲染耗时

如果 `Step only` 接近 `Total Frame`，说明瓶颈主要是模拟；如果 `Render only` 很高，说明瓶颈在渲染。

## 5. 覆盖子步数和网格大小

有时需要临时改变示例的 `SIM_SUBSTEPS` 或 `N`，可以使用：

```bash
uv run python scripts/bench/bench_lbm_run.py \
  -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt \
  -f 120 -w 20 \
  --viewer null \
  --substeps 3 \
  --grid-res 128
```

注意：

- `--substeps` 会影响每帧模拟计算量
- `--grid-res` 会改变 LBM 网格规模
- 不同子步数的 FPS 不能直接比较，需要换算每个 substep 的耗时

例如：

```text
3 substeps, 11.05 ms/frame -> 3.68 ms/substep
5 substeps, 17.87 ms/frame -> 3.57 ms/substep
```

这说明两组结果虽然 FPS 不同，但单个 LBM step 的性能基本一致。

## 6. 生成逐帧 CSV

如果需要分析卡顿、尖峰、P99，可以保存逐帧数据：

```bash
uv run python scripts/bench/bench_lbm.py \
  -n 128 -f 200 -w 20 \
  --csv-per-frame scripts/bench/bench_results/frames_128.csv
```

真实示例也可以保存逐帧数据：

```bash
uv run python scripts/bench/bench_lbm_run.py \
  -m wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt \
  -f 200 -w 20 \
  --viewer gl \
  --csv-per-frame scripts/bench/bench_results/dambreak_gl_frames.csv
```

逐帧 CSV 适合排查：

- 哪些帧突然变慢
- P99 是否由少数尖峰造成
- GL viewer 是否带来额外波动

## 7. 生成性能图

当已经生成汇总 CSV 后，可以用绘图脚本输出 PNG：

```bash
uv run python scripts/bench/plot_bench_results.py \
  --example-csv scripts/bench/bench_results/substeps2/aggregate.csv \
  --scale-csv scripts/bench/bench_results/dambreak_scale/aggregate.csv \
  --output-dir scripts/bench/bench_results
```

生成文件：

```text
scripts/bench/bench_results/example_comparison.png
scripts/bench/bench_results/resolution_scaling.png
```

这两个图分别用于：

- `example_comparison.png`：比较不同 LBM 示例的 FPS 和显存
- `resolution_scaling.png`：比较不同网格分辨率下的性能变化

## 8. 如何解读结果
~~交给AI~~
### 8.1 Avg ms 和 Avg FPS

`Avg ms` 是平均每帧耗时，越低越好。

`Avg FPS` 是平均帧率，越高越好。

换算关系：

```text
FPS = 1000 / Avg ms
```

例如：

```text
17.87 ms -> 1000 / 17.87 = 56 FPS
```

### 8.2 P50、P99、Min、Max

- `P50`：一半帧比它快，一半帧比它慢
- `P99`：99% 帧都不超过这个耗时
- `Max`：最慢的一帧

如果 `Avg` 很好但 `P99` 很高，说明偶尔有卡顿尖峰。

### 8.3 Warp Mem 和 Process Mem

`Warp Mem` 更接近 LBM 数据结构本身使用的显存。

`Process Mem` 来自 `nvidia-smi`，包含 CUDA 上下文、驱动、JIT、库加载等开销，也可能受其他 GPU 进程影响。因此它适合作为运行时参考，不适合作为精确的 LBM 数组占用。

## 9. 推荐测试流程

完整测试建议按这个顺序执行：

1. 先跑 `64^3` 冒烟测试，确认环境正常。
2. 跑纯 LBM 分辨率扫描，得到核心求解器性能曲线。
3. 跑真实示例的 `null` viewer，确认模拟本身性能。
4. 跑真实示例的 `gl` viewer，确认渲染额外开销。
5. 保存 CSV。
6. 使用 `plot_bench_results.py` 出图。
7. 根据 `Avg`、`P99`、显存判断推荐网格规模。

## 10. 常见问题

### 10.1 第一次运行有编译输出

第一次运行会看到类似：

```text
Module ... load on device 'cuda:0' took ... ms (compiled)
```

这是 Warp 编译 kernel 的时间。正式测试会通过 warm-up 排除启动阶段的不稳定。

### 10.2 `gl` 比 `null` 慢多少算正常

如果 `gl` 比 `null` 慢 5% 到 15%，通常是正常的。  
如果慢很多，需要检查 SSFR 参数、窗口大小、GPU 同步和显存压力。

### 10.3 P99 明显高于平均值

可能原因：

- 示例内部有周期性诊断输出
- GPU 到 CPU 的数据读取
- 第一次 kernel 调用或缓存抖动
- 系统中有其他 GPU 任务

可以通过增加帧数、关闭诊断输出、查看逐帧 CSV 来确认。

### 10.4 不同 substeps 的结果怎么比较

不要只看 FPS，要换算每个 substep 的耗时：

```text
ms_per_substep = Avg ms / substeps
```

这样才能判断求解器本身是不是变快或变慢。

