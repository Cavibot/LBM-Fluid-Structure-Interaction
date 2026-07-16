合并到 `main` 前，至少还有下面 6 项必须完成。Raw MRT、HOME+SC、HOME coupling 等明确延期内容不需要现在实现。

## 1. 修复已知 fail-fast 漏洞

当前：

```text
FullF + home_nocm + G != 0
```

没有被拒绝，但 SC 力实际上也没有进入 EMC collision。这可能产生“程序正常运行，但物理力被静默忽略”的危险结果。

必须改成构造时直接抛 `NotImplementedError`，并补测试。

## 2. 完成 CollisionBackend 契约，或者修改计划声明

目前 backend 类主要保存：

```text
input_kind
relaxation_rates
```

真正的 kernel 仍由 Solver 直接选择和启动，没有计划中暗示的统一：

```python
backend.collide(...)
```

合并前必须二选一：

- 实现真实的 backend 调用接口，由 Solver 调 backend；
- 明确修改 execution plan，说明第一版 backend 只是选择元数据，kernel 调度仍属于 Solver。

不能让文档声称一种架构，而代码实际是另一种。建议实现最小 `launch_collision()`，不需要引入复杂 Protocol。

## 3. 增加非均匀 streaming 测试

现在六种组合测试使用均匀平衡场。均匀场无法充分检查：

- `+x/-x` 方向是否写反；
- source/target 索引是否颠倒；
- periodic wrap 是否偏移一格；
- HOME provider 是否从正确邻居读取；
- EMC collector 是否在正确格点累积矩。

至少增加一个明确的方向传播测试：

```text
在单个格点设置一个已知方向 population
执行一次 streaming
检查它准确移动到 x+c_i
检查周期边界准确回绕
```

需要覆盖：

- FullF → populations；
- HOME → populations；
- FullF → moments；
- HOME → moments。

这是合并前最重要的新增正确性测试。

## 4. 增加一个多步非平衡物理验证

现在主要验证“一步、均匀、公式一致”，还没有证明新时间层经过多步不会漂移。

最低要求可以是周期域上的 shear-wave decay：

\[
u_x(y,0)=u_0\sin(2\pi y/L)
\]

理论上：

\[
u_x(t)=u_0 e^{-\nu k^2t}\sin(ky).
\]

它可以同时验证：

- streaming/collision 顺序；
- SRT/TRT 黏性；
- post-collision 双缓冲；
- 多步质量与动量；
- FullF 和 HOME 的宏观趋势。

至少验证 FullF SRT/TRT。HOME-NOCM 如果暂时达不到严格解析误差要求，也应验证有限性、质量守恒和衰减方向。

## 5. 解决全量测试中的 14 个红项

目前有 14 个历史测试引用已经不存在的示例模块。虽然不是本次重构造成的，但不能把一个明确红掉的 LBM 测试集合直接合并进 `main`。

必须选择一种处理方式：

- 恢复对应示例；
- 删除已经失效的测试；
- 将其明确标记为 skip，并写明缺少的资产或迁移任务；
- 更新测试名称，使其指向现存示例。

最低成本做法是合理地 `skip`，不能继续保留 `ModuleNotFoundError`。

## 6. 在 CUDA 环境完成一次 GPU 验证

当前只在 Warp CPU 后端验证。这个项目的主要运行目标是 GPU，因此合并前至少需要在 CUDA 上完成：

- 六种编码/碰撞组合各一步；
- HOME reconstruction/NOCM kernel 编译；
- FullF SC smoke；
- 检查 NaN/Inf；
- 检查 HOME-NOCM 没有 population scratch；
- 一个小规模多步运行。

不要求现在完成正式性能调优，但至少要记录：

```text
GPU 型号
Warp/CUDA 版本
测试网格
每个组合是否通过
显存占用
```

## 7. 自动化质量检查

最后应运行仓库正式工具，而不只是 `py_compile`：

```text
ruff format --check
ruff check
basedpyright 或项目指定类型检查
相关 unittest
```

当前环境没有可直接调用的 Ruff，因此这一项尚未完成。若 CI 会自动执行，也应在 PR 上确认全部通过。

---

最低合并门槛可以总结为：

```text
[ ] 修复 FullF+HOME-NOCM+SC 静默忽略
[ ] backend 文档与代码契约一致
[ ] 非均匀定向 streaming 测试
[ ] 一个多步 shear-wave/等价物理测试
[ ] 处理 14 个陈旧测试
[ ] CUDA smoke 通过
[ ] lint / format / type / unit CI 全绿
```

GPU性能优化、Raw MRT、FullF NOCM、HOME 外力和新 coupling 都可以留到后续，不属于本次合并阻塞项。