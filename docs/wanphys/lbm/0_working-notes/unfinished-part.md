目前未实现内容分成三类。

### 1. 明确预留的碰撞算子

以下配置名称已经保留，但选择时直接抛出 `NotImplementedError`：

- `raw_mrt`：常规 FullF Raw MRT。
- `nocm_mrt`：FullF 显式 NOCM-MRT。

目前没有对应数学 kernel，也没有伪实现。这符合计划。

### 2. 尚未支持的组合能力

外力：

- HOME + Shan–Chen。
- HOME + SRT/TRT + prescribed gravity。
- HOME-NOCM 的中心矩外力闭合。
- FullF + HOME-NOCM + gravity/SC。

边界：

- 新 post-collision 流水线只支持 periodic 和 static halfway bounce-back。
- Zou–He inlet 尚未迁移。
- Convective outlet 尚未迁移。
- Moving-wall/cut-link 尚未迁移到新流水线。

耦合：

- HOME rigid coupling。
- HOME 的 Galilean-invariant momentum exchange。
- 新时间层下的 `LinkImpulse/MEM` 重构。
- 当前 FullF moving-wall/MEM 仍走旧融合兼容路径。

研究扩展：

- D3Q27 或通用 D3QN。
- HOME quantization。
- Free-surface population completion。
- HOME + 强密度梯度/多相物理验证。
- HOME 的 CUDA 性能和融合 kernel。

### 3. 有意没有创建的架构类型

这些不是遗漏，而是计划中明确不需要：

- 独立 `ExecutionPlan` 类。
- `CollisionResult` 数据类。
- `output_encoding` 配置。
- 两套 Domain/Model/Solver。
- 大型 `codec/transport/collision/execution` 子包结构。

### 当前存在的两个实现缺口

第一，CollisionBackend 目前还比较“轻”。

现在：

- `SrtCollision`、`TrtCollision` 只提供 `input_kind` 和 relaxation rates；
- `HomeNocmMrtCollision` 主要提供类型标识；
- 真正的 Warp kernel 仍由 Solver 直接启动。

也就是说，完整的：

```python
backend.collide(input, output)
```

接口还没有实现。数值功能完整，但 collision backend 尚未完全封装。这与计划中提到的统一 `collide()` 调用约定还有一点距离。

第二，存在一个 fail-fast 漏洞：

```text
FullF + home_nocm + G != 0
```

理论上应明确拒绝，但目前不会在 Model/Solver 构造时拒绝，SC 力也不会真正进入 EMC collision。这需要补成 `NotImplementedError`，避免用户误以为该组合受支持。

因此，最值得立即补的不是新物理功能，而是：

1. 补齐 `FullF + HOME-NOCM + SC` 的 fail-fast。
2. 决定是否把 kernel 启动真正封装进 `CollisionBackend.collide()`。

其余均属于已经明确推迟的后续研究范围。