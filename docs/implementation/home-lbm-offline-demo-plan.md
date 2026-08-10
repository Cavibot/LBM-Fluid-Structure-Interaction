# HOME-LBM 最终场景离线演示计划

本文定义 HOME-LBM 重构完成后的最终场景演示工作。目标不是制作实时交互程序，而是在 RTX 5090 上完成可复现的高质量仿真、液面重建和逐帧离线渲染，再把图片及诊断结果传回本地；需要视频时只把已经验收的图片序列编码为视频。

本阶段不修改旧 D3Q19/TRT 示例，不以 VNC 或 ViewerGL 帧率作为交付指标，也不通过降低物理门槛换取画面。仿真核心继续使用当前默认的几何 HOME-Free 后端。

## 1. 最终交付

最终至少交付三套场景：

1. **纯流体 Dam-break**：展示重力、sharp VOF、PLIC 界面、拓扑变化、铺展和撞壁。
2. **Dam-break 撞击可移动球体**：作为主场景，展示自由表面、动态 cut-link、切格、作用反作用和刚体运动。
3. **球体入水、回升与漂浮**：展示 entry/exit、fresh/dead cell、动态润湿、浮力和水面恢复。

每套场景必须包含：

- 固定且版本化的物理配置；
- 小网格自动化测试配置；
- 5090 正式分辨率无渲染预验收记录；
- 可恢复的仿真 checkpoint 和逐帧快照；
- 干净版 PNG 图片序列；
- 带诊断信息的审计版 PNG 图片序列；
- `manifest.json`、`metrics.csv`、运行日志和环境清单；
- 三个关键时刻的高质量静帧；
- 可选的本地视频，不把视频作为唯一原始成果。

推荐最终图片为 2560x1440、16-bit 或高质量 8-bit PNG；关键帧另保留 multi-layer EXR。目标播放速率为 24 fps，每个场景约 8-12 秒，即 192-288 个真实仿真时刻。最终帧数由物理时间尺度决定，不通过重复帧或视觉插帧凑时长。

## 2. 不可退让原则

1. 测试、无渲染预跑和最终渲染共用同一场景构建器，禁止复制后再手工调出三套参数。
2. 渲染只读取提交态，不得回写 `fill`、HOME moments、刚体状态、压力缓存或 topology scratch。
3. 每张最终图片对应真实提交时刻；需要更平滑的运动时增加仿真输出频率，不做帧插值。
4. 先通过物理验收，再导出表面，再渲染。画面好看不能覆盖质量、动量、散度或穿透失败。
5. 物理状态使用 FP32 生产路径；此前未获准的 int16 状态不能为了扩大演示网格而启用。
6. 表面平滑只能属于显示层，并记录算法和参数；不能改变仿真场，也不能被描述为求解器输出。
7. 所有长任务必须原子写文件、支持恢复，并能识别配置或代码哈希变化，避免混合两次不兼容运行。
8. 5090 同一时间只承担一个重 GPU 阶段。仿真、表面重建和 Cycles 渲染串行调度，避免显存争用改变性能和稳定性。

## 3. 建议代码边界

新增目录，不修改旧示例：

```text
wanphys/examples/lbm/home_free_offline/
├── __init__.py
├── config.py                 # 可序列化的公共场景和输出配置
├── scene.py                  # 生命周期、步进和诊断协议
├── capture.py                # 提交态快照与原子写入
├── checkpoint.py             # 完整恢复状态
├── validation.py             # 场景级物理门槛
├── surface.py                # PLIC 感知液面重建入口
└── scenes/
    ├── dambreak.py
    ├── dambreak_sphere.py
    └── sphere_entry.py

scripts/render/home_free/
├── simulate.py               # 5090 无渲染仿真
├── extract_surfaces.py       # 快照到网格
├── render_blender.py         # 调用 Blender background worker
├── blender_scene.py          # 在 Blender Python 中建场景和材质
├── annotate_frames.py        # 生成审计版图片
├── verify_frames.py          # 黑帧、尺寸、编号和统计检查
└── encode_video.sh           # 可选；最终可在 Mac 执行

newton/tests/
├── test_home_free_offline_scene_config.py
├── test_home_free_offline_dambreak.py
├── test_home_free_offline_dambreak_sphere.py
├── test_home_free_offline_sphere_entry.py
├── test_home_free_offline_capture.py
└── test_home_free_offline_surface.py
```

场景模块只编排已经验收的 HOME-Free 和刚体 API。表面重建、渲染和图片后处理不能进入 `home_lbm/` 求解器目录。

## 4. 场景配置层级

每个场景定义三个层级，保持物理无量纲参数和几何比例一致：

| 层级 | 用途 | 要求 |
|---|---|---|
| L0 | CPU/CUDA pytest | 小网格、短时间，覆盖初始化、至少一次关键事件和硬错误 |
| L1 | 5090 预演 | 中等网格，走完整物理阶段，用于收敛、镜头和输出频率设计 |
| L2 | 5090 最终 | 目标分辨率和完整时长，先无渲染验收，再固定结果渲染 |

不能只把格点数增加而保持所有格子参数不变。升级 L0/L1/L2 时必须重新计算并记录：

- `dx`、`dt` 和参考密度；
- 黏度和松弛率；
- 最大格子速度及 Mach 上限；
- Reynolds、Froude、Weber 数；
- 表面张力、重力和接触角；
- 球体直径格点数、密度比、质量和惯量；
- 刚体子步、强耦合容差和最大迭代数；
- Courant 投影容差和迭代上限。

L2 候选分辨率不提前写死。以 L1 与两个相邻分辨率的界面位置、刚体轨迹、质量和动量误差比较为依据，选择达到结果稳定且 5090 资源可承受的最小 L2。初始资源预算可按约 0.8-1.5 百万格开始探测，实际配置必须由显存和收敛数据决定。

## 5. 三套场景定义

### 5.1 纯流体 Dam-break

物理内容：矩形液柱在封闭水箱内受重力坍塌，沿底板铺展并撞击端壁。跨厚度方向必须具有真实有限宽度；不能沿用只为测试节省计算的 4 格薄层作为最终画面。

必须记录：

- 初始和逐帧代表质量；
- 液体质心；
- 前沿位置和最高液面；
- 最大速度、密度范围和 Mach 数；
- interface 数量和 topology 转换数量；
- Courant 投影前后最大散度；
- PLIC 几何修正计数和最大修正。

L0 门槛在当前公共几何液柱示例上扩展，补齐 geometric column-collapse 的前沿、质心、拓扑变化和速度区间测试。L1/L2 必须出现明确坍塌和铺展；静止、仅扩散、质量守恒但动力学错误均不通过。

### 5.2 Dam-break 撞击球体

物理内容：液柱前沿撞击可移动球体，流体冲量推动球体并产生绕流和自由表面变形。这是最终主镜头。

必须记录：

- 场景 5.1 的全部流体指标；
- 球体位置、线速度、角速度和接触状态；
- wet/dry cut-link 数量、fresh/dead cell 数量；
- remap 和 queue 的质量、动量账本；
- cut-link 作用反作用残差；
- 流体加刚体总动量及外力积分后的闭合残差；
- 强耦合迭代数、残差、失败和回滚计数；
- 球体与水箱的最小间距以及穿透计数。

L0 直接建立在 `test_home_free_column_impact.py` 的 geometric CPU/CUDA 路径上。L1 确定球体位置和镜头，但不允许为了得到更戏剧化的运动随意放大耦合力；只能通过有物理含义的液柱高度、密度比、球径和重力改变无量纲条件。

### 5.3 球体入水、回升与漂浮

物理内容：密度低于液体的球体从液面上方以重力或明确初速度入水，经历浸没、减速、回升和阻尼漂浮。最终状态应与排液体积和重量平衡一致。

必须记录：

- 球体首次接触、最大浸没、首次回升和进入稳定漂浮区间的物理时间；
- 球体竖直速度、最大横向漂移和最大角速度；
- 浮力、重量、动态压力和黏性载荷；
- 位移液体体积和静态平衡误差；
- 入水/出水阶段 fresh/dead cell、queue 和 cut-link 账本；
- 总质量、总动量和投影散度；
- 自由表面恢复后的寄生速度。

L0 组合现有 geometric entry/exit、floating 和 rigid-transition 测试的契约，但最终动态场景必须有自己的端到端测试。若球体最终应漂浮，密度比、初始能量和阻尼必须在 manifest 中明确，不能通过额外隐藏力把球固定在水面。

## 6. 阶段计划与验收门槛

### P0：冻结基线

1. 记录本地分支、Git diff 摘要、CodeGraph 状态和 5090 代码清单哈希。
2. 在 5090 重跑 HOME 聚焦测试和最小几何液柱示例。
3. 保存 GPU、驱动、Warp、Python 和 CUDA 信息。
4. 建立 `/root/autodl-tmp/lbm-offline-demo/`，代码仍位于仓库，产物只写数据盘。

通过条件：代码哈希一致；现有回归无新增失败；公共默认后端确认为 geometric。

### P1：共享场景工厂

1. 实现不可变、可 JSON 序列化的 scene config。
2. 所有派生格子参数由一个函数生成，manifest 保存输入和派生值。
3. 三个 scene builder 返回统一的 fluid、rigid、stepper、diagnostics 和 camera hint。
4. pytest 与演示 CLI 调用相同 builder。
5. 对非法分辨率、Mach、松弛率、接触角、密度比和输出时间硬失败。

通过条件：L0 CPU/CUDA 场景均通过；序列化往返不改变配置；相同 seed 和配置重复运行得到既有确定性契约内一致的结果。

### P2：场景级数值验收

1. 补 geometric dam-break 端到端测试。
2. 固化撞球和入水场景的 scene-level metrics。
3. 每个场景先跑 L0，再跑 L1 的完整物理时长。
4. 对至少两个空间分辨率和两个输出无关的时间步设置比较关键轨迹。
5. 门槛来自现有物理测试和 L1 收敛数据，不能在看到最终结果后只为让失败变绿而修改。

通过条件：质量、动量、散度、速度、拓扑和几何诊断全部通过；关键轨迹随加密趋于稳定；没有 NaN、穿透、缺失 cut-link 或事务回滚泄漏。

### P3：快照与恢复

每个输出帧保存紧凑提交态：

- frame index、step index、物理时间；
- active/interface cell 索引；
- fill、flags、PLIC 法线和偏移；
- 必要的密度或颜色标量；
- 刚体位姿、速度和几何引用；
- 本帧诊断。

完整 checkpoint 还必须保存恢复仿真所需的 HOME/VOF 双缓冲、excess queue、刚体状态、split parity、Courant pressure warm start 和强耦合事务状态。checkpoint 不等同于渲染快照。

文件先写 `*.tmp`，校验后原子 rename。manifest 记录 schema version、代码哈希、config hash 和每个文件 SHA-256。恢复时任何不匹配都硬失败。

通过条件：从中间 checkpoint 恢复后，后续 CUDA 结果满足项目确定性契约；故意截断文件、修改配置或混用代码版本会被拒绝；渲染快照不影响下一仿真步。

### P4：PLIC 感知表面重建

实现两种输出：

1. **审计表面**：直接裁剪每个 interface cell 的 PLIC 平面，作为几何真值和调试输出。
2. **展示表面**：从 PLIC 窄带构建连续 signed-distance 近似，再提取连续三角网格；法线平滑与几何重建参数明确记录。

展示表面不得简单把未经说明的 `fill=0.5` marching cubes 当作 PLIC 输出。需要处理液体与水箱壁、刚体表面的相交和封口；刚体本身使用原几何，不从 `solid_phi` 重建展示模型。

逐帧检查：

- 顶点、法线和索引有限且索引合法；
- 三角形非退化，法线方向一致；
- 非接触区域没有明显裂缝和孤立碎片；
- 表面位于 PLIC 窄带允许范围内；
- 闭合测试几何的网格体积与 fill 体积一致；
- 动态场景的位移不超过物理速度允许的帧间范围；
- 相同快照重复导出逐字节或在规定浮点容差内一致。

通过条件：解析平面和球面测试通过；三套 L1 场景关键帧通过几何检查；审计表面与展示表面的差异被量化而不是目测接受。

### P5：5090 离线渲染环境

当前 5090 环境已有 Warp 1.12.0 和 Pillow，但没有 Blender、`scikit-image`、`trimesh`、OpenEXR Python 包或 ffmpeg。计划如下：

1. 把固定版本的 Blender LTS 独立包安装到 `/root/autodl-tmp/tools/`，不占系统盘。
2. 仿真 venv 只增加表面导出所需的固定依赖；Blender 使用自带 Python 执行渲染脚本。
3. 以 background 模式启动，不需要 X server、VNC、pyglet 或 ImGui。
4. 启动时枚举 Cycles 设备并要求 RTX 5090 OptiX 可用；先做 64x64 单帧 smoke，再做 512x288 水材质 smoke。
5. 记录 Blender build、Cycles/OptiX 设备、驱动和渲染耗时。

Blender 官方支持命令行 background 渲染和 `--cycles-device OPTIX`。最终 worker 采用等价形式：

```bash
blender --background scene.blend \
  --python scripts/render/home_free/blender_scene.py \
  -- --frame-manifest path/to/frame.json --cycles-device OPTIX
```

通过条件：无 DISPLAY 环境可成功渲染；日志确认使用 RTX 5090 OptiX 而不是 CPU；输出图片非黑、尺寸和颜色空间正确；重复渲染关键帧没有随机材质或相机变化。

### P6：画面设计与关键帧

所有场景使用一致视觉语言，但相机和构图按物理事件确定：

- 固定世界尺度，Z 轴向上；
- 水材质使用物理折射率约 1.333、适度吸收和低粗糙度；
- 水箱壁采用可读但不遮挡液面的薄玻璃或边框表达；
- 刚体使用与水体有明显明度和色相差的哑光材质；
- 使用大面积主光、柔和补光和中性环境，不用装饰性粒子替代流体；
- 相机不穿过水箱，整个物理域和关键运动始终在安全画幅内；
- 关闭伪造流体运动的后期 motion blur；若需要真实模糊，应计算实际子帧。

先为每个场景渲染起始、关键事件、结束三个 1440p 静帧。确定材质和镜头后冻结 render config，之后不逐帧手调。

通过条件：九张关键帧均无裁切、穿模、黑面、异常折射或界面碎片；相机能读出主要物理事件；审计版能够对应到同一 frame manifest。

### P7：L2 正式仿真

1. 根据 P2 的收敛结果冻结 L2 配置和输出物理时间序列。
2. 在 `screen` 中运行无渲染仿真，stdout/stderr 同时写日志。
3. 每个输出帧写快照和 metrics，每固定间隔写完整 checkpoint。
4. 监控磁盘、显存、温度和错误标志；磁盘低于安全余量时停止提交新帧。
5. 任务完成后运行全序列物理审计，生成 `acceptance.json`。

通过条件：整段场景的所有硬错误为零；场景特定质量、动量、散度和轨迹门槛通过；帧时间严格单调且编号连续；从最后一个 checkpoint 可恢复并额外推进若干步。

L2 未通过时返回 P1/P2 修改有物理依据的配置或实现缺陷。禁止跳过失败帧直接渲染，也禁止只渲染看起来正常的时间段冒充完整结果。

### P8：批量渲染、检查和回传

1. 每帧作为独立 Blender job，完成后写 `.done`；重启时跳过哈希匹配的已完成帧。
2. 先渲染低采样 preview 全序列，检查镜头和拓扑连续性。
3. preview 通过后渲染最终 Cycles 图片，启用固定的采样、bounce 和 denoise 配置。
4. 保存 clean PNG；关键帧保存 EXR 的 beauty、depth、normal 和 object-ID pass。
5. Pillow 读取 `metrics.csv` 生成独立 overlay 版本，不污染 clean 原图。
6. 自动检查图片解码、尺寸、通道、均值/方差、黑白像素占比、编号连续性和文件哈希。
7. 使用 `rsync --partial --append-verify` 把 manifest、metrics、acceptance、关键 checkpoint、mesh 和图片传回 Mac。
8. 本地抽查关键帧和时间连续性；需要时再用 ffmpeg 编码 H.264/H.265 视频。

通过条件：图片序列完整且可解码；关键事件与 metrics 时间一致；没有黑帧、跳号或混入旧配置；远端和本地清单哈希一致。

## 7. 远端目录与资源管理

```text
/root/autodl-tmp/
├── LBM-Fluid-Structure-Interaction/   # 与本地同步的代码
├── tools/
│   └── blender-<pinned-lts>/
└── lbm-offline-demo/
    ├── env/
    ├── runs/
    │   └── <scene>/<run-id>/
    │       ├── manifest.json
    │       ├── acceptance.json
    │       ├── metrics.csv
    │       ├── checkpoints/
    │       ├── snapshots/
    │       ├── meshes/
    │       ├── frames-preview/
    │       ├── frames-clean/
    │       ├── frames-overlay/
    │       └── logs/
    └── cache/
```

数据盘目前约有 47 GB 可用。正式运行前必须根据一帧 snapshot、mesh、PNG 和 EXR 的实测大小估算整段占用，并至少保留 10 GB 安全余量。普通 checkpoint、preview 和中间 mesh 在本地验收并回传后可以按 manifest 清理；最终 clean frames、metrics、acceptance 和关键 checkpoint 不在自动清理范围内。

## 8. 自动化命令目标

最终 CLI 应形成下面的稳定工作流，具体参数由实现后的 `--help` 固化：

```bash
# L0 测试
pytest -q newton/tests/test_home_free_offline_*.py

# L1/L2 无渲染仿真
PYTHONPATH=. python scripts/render/home_free/simulate.py \
  --scene dambreak-sphere --level L2 --device cuda:0 \
  --output /root/autodl-tmp/lbm-offline-demo/runs

# 全序列物理审计
PYTHONPATH=. python scripts/render/home_free/simulate.py \
  --audit-only /path/to/run

# PLIC 表面导出
PYTHONPATH=. python scripts/render/home_free/extract_surfaces.py \
  --run /path/to/run --resume

# preview 与 final 渲染
PYTHONPATH=. python scripts/render/home_free/render_blender.py \
  --run /path/to/run --quality preview --resume
PYTHONPATH=. python scripts/render/home_free/render_blender.py \
  --run /path/to/run --quality final --resume

# 图片审计
PYTHONPATH=. python scripts/render/home_free/verify_frames.py \
  --run /path/to/run --quality final
```

每条命令必须支持 `--dry-run` 显示解析后的配置、预计帧数、目录和空间预算，不启动 GPU 任务。

## 9. 测试矩阵

| 类别 | CPU | CUDA 5090 | 关键检查 |
|---|---:|---:|---|
| config/manifest | 是 | 是 | round-trip、hash、非法配置 |
| L0 三场景 | 是 | 是 | 物理门槛和跨设备误差 |
| checkpoint restore | 可选小网格 | 是 | 中断恢复、parity、pressure cache |
| PLIC 审计表面 | 是 | 是 | 解析平面/球面、有限性、方向 |
| 展示表面 | 是 | 是 | 连续性、体积/窄带误差 |
| Blender smoke | 否 | 是 | headless、OptiX、非黑帧 |
| L1 预演 | 否 | 是 | 完整事件、收敛和镜头 |
| L2 最终仿真 | 否 | 是 | 全序列 acceptance |
| preview/final 渲染 | 否 | 是 | 连续帧、画质、恢复 |
| 本地回传 | Mac | 远端清单 | SHA-256 一致 |

## 10. 最终完成定义

只有同时满足以下条件，本演示阶段才算完成：

1. 三套 L0 测试和现有全部 HOME 回归通过。
2. 三套 L2 场景都在 RTX 5090 上完成无渲染全时段验收。
3. 每套场景都有通过几何检查的 PLIC 感知表面序列。
4. Blender background 日志确认使用 RTX 5090 OptiX。
5. 每套场景都有完整 clean/overlay PNG 序列和三张关键 EXR。
6. 图片自动审计和人工关键帧审查均通过。
7. manifest、metrics、acceptance、环境信息和文件哈希完整。
8. 所有最终成果已传回本地，远端与本地哈希一致。
9. 项目文档记录最终配置、实测耗时、显存、误差和已知视觉限制。

任何单一漂亮截图、短时间 smoke、旧 TRT 可视化或 VNC 实时窗口都不能替代上述完成定义。

## 11. 实施顺序

严格按以下顺序推进，每一步通过后再进入下一步：

1. P0 基线冻结。
2. P1 共享场景工厂。
3. P2 geometric dam-break 及三场景 L0/L1 物理验收。
4. P3 快照、checkpoint 和恢复。
5. P4 PLIC 感知表面重建。
6. P5 5090 Blender/OptiX 环境。
7. P6 九张关键帧和视觉配置冻结。
8. P7 三套 L2 正式仿真。
9. P8 preview、final 渲染、回传和最终文档。

优先贯通一条窄而完整的路径：先用纯流体 geometric dam-break 完成 P0-P6 的单帧和短序列闭环；确认数据契约、表面和渲染无误后，再并行扩展撞球和入水场景，最后才运行昂贵的三套 L2。

## 12. 外部工具依据

- [Blender 命令行渲染](https://docs.blender.org/manual/zh-hans/5.0/advanced/command_line/arguments.html)：background 渲染、输出格式和 `--cycles-device OPTIX`。
- [Blender Cycles GPU 配置](https://docs.blender.org/manual/zh-hans/5.0/editors/preferences/system.html)：NVIDIA CUDA/OptiX 设备选择。

实际安装版本在 P5 固定，并把下载来源、SHA-256 和 `blender --version` 写入环境清单；文档示例中的版本号不替代远端实测。
