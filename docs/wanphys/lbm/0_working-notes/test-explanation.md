> 详细参考文档。当前合并门槛与命令入口见 `acceptance.md`。

可以把这些测试理解成四层防线：先检查“对象有没有搭对”，再检查“公式有没有抄对”，然后检查“流水线能不能走通”，最后检查“旧功能有没有被破坏”。

新增验收层：

- `test_lbm_directional_streaming.py`：非均匀定向 packet / 周期回绕；
- 同文件 shear-wave：多步黏性衰减与质量守恒。

## 1. State 与配置测试：检查架构契约

文件：[test_lbm_state_encoding.py](/Users/Alexandrite/Workspace/LBM-Fluid-Structure-Interaction/newton/tests/test_lbm_state_encoding.py)

它检查的不是流体结果，而是架构规则：

- 默认创建 `FullFLbmState`。
- `encoding="home"` 创建 `HomeLbmState`。
- HOME 只持久化 10 个矩，不偷偷保存 19 个 population。
- `state_in/state_out` 类型一致。
- FullF 的 `.f` 是 `.f_post` 兼容别名。
- `raw_mrt`、`nocm_mrt` 明确报未实现。
- HOME+SC、HOME coupling 等非法组合提前报错。
- HOME-NOCM 不分配 `f_star/f_post` population scratch。
- 尚未迁移的边界类型不能被静默接受。

可以把它理解成：

> 检查零件型号和接线方式，不检查发动机性能。

## 2. Streaming 测试：检查 2×2 路由

文件：[test_lbm_streaming_matrix.py](/Users/Alexandrite/Workspace/LBM-Fluid-Structure-Interaction/newton/tests/test_lbm_streaming_matrix.py)

测试了六个具体组合：

```text
FullF + SRT
FullF + TRT
FullF + HOME-NOCM
HOME  + SRT
HOME  + TRT
HOME  + HOME-NOCM
```

给所有格点相同的平衡状态，然后走一步。

因为均匀、周期场 streaming 前后应该完全一样，所以如果结果改变，通常说明：

- FullF/HOME provider 读错；
- streaming 方向或索引错误；
- encoding/decoding 丢失质量或动量；
- Solver 选错 collector/collision；
- collision 破坏平衡态。

另一个测试检查 static bounce-back 前后总质量不变，主要捕获：

- `opposite_direction` 配错；
- pull-streaming 来源格点错误；
- 壁面反弹读取了错误 population。

但它只证明基本守恒，不证明 bounce-back 的二阶精度或复杂几何精度。

## 3. Collision 测试：检查碰撞基本性质

文件：[test_lbm_collision_backends.py](/Users/Alexandrite/Workspace/LBM-Fluid-Structure-Interaction/newton/tests/test_lbm_collision_backends.py)

两个核心性质：

第一，平衡态是 SRT 的固定点：

```text
f* = fEq
collision(f*) = fEq
```

如果失败，说明平衡分布、方向权重或松弛公式有问题。

第二，当 TRT 两个松弛率相等时，TRT 必须退化为 SRT：

```text
omegaEven = omegaOdd
TRT == SRT
```

这验证了偶/奇分解以及 opposite pairing。

它还没有验证：

- TRT magic parameter 对 Poiseuille 边界误差的改善；
- 给定 `tau` 是否恢复正确黏性；
- 非平衡流动的长期稳定性。

这些需要专门的物理 benchmark。

## 4. HOME 公式测试：检查论文公式翻译

文件：[test_lbm_home_nocm.py](/Users/Alexandrite/Workspace/LBM-Fluid-Structure-Interaction/newton/tests/test_lbm_home_nocm.py)

这是最重要的数学单元测试。

### HOME reconstruction

先在 NumPy 中按公式计算：

```text
rho, rho*u, rho*S
    ↓
19 个 D3Q19 population
```

再让 Warp kernel 计算同样结果，逐方向比较。

然后执行闭合检查：

```text
HOME moments
    ↓ reconstruct
populations
    ↓ extract moments
HOME moments
```

最终恢复的 `rho`、`rho*u`、`rho*S` 应与输入一致。

这主要检查：

- 密度加权和归一化约定没有混淆；
- D3Q19 权重和方向没有写错；
- 对角/非对角二阶矩没有错位；
- 三阶闭式项的符号和系数正确；
- encode/reconstruct 能在保留矩空间内闭合。

### HOME-NOCM

同样先用 NumPy 写闭式二阶矩碰撞，再与 Warp 输出比较：

```text
NumPy closed form ≈ Warp HOME-NOCM kernel
```

它证明的是：

> Warp 实现与我们确定的数学公式一致。

但需要注意：NumPy 参考公式也是根据论文和 Home-FSLBM 参考代码翻译出来的。因此它能很好地发现 GPU 编码错误，却不能单独证明我们对论文的理解绝对正确。论文公式、参考仓库和测试公式三者交叉核对才构成完整依据。

## 5. 旧功能回归：检查没有破坏现有行为

文件：

- [test_lbm_shan_chen_wall_force.py](/Users/Alexandrite/Workspace/LBM-Fluid-Structure-Interaction/newton/tests/test_lbm_shan_chen_wall_force.py)
- [test_lbm_rigid_coupling.py](/Users/Alexandrite/Workspace/LBM-Fluid-Structure-Interaction/newton/tests/test_lbm_rigid_coupling.py)

Shan–Chen 测试检查：

- 固体伪势不同取值时，力的方向正确；
- 中性固壁接近零力；
- force stride 不产生 NaN/Inf；
- 参数校验正常；
- Solver 确实把边界参数传给 kernel。

刚体测试检查旧 FullF 兼容路径：

- SDF 光栅化；
- moving-wall 速度；
- 流体扰动；
- MEM 力和力矩方向；
- 所有输出保持有限。

这些刚体测试验证的是保留的旧融合 FullF 路径，不代表新的 post-collision coupling 已经实现。

## 这些测试目前能证明什么？

可以证明：

- 2×2 架构路由已接通；
- HOME 内存布局符合约定；
- 基础守恒关系成立；
- HOME reconstruction 和 NOCM kernel 与 NumPy 公式一致；
- 六种组合可以构造并完成一步；
- 旧 SC 和刚体兼容路径没有被直接破坏。

不能证明：

- HOME-NOCM 已经完成论文级流场验证；
- HOME+SRT/TRT 交叉组合具有足够研究价值或长期稳定性；
- 黏性、收敛阶、各向同性完全正确；
- 高 Reynolds 数、湍流或复杂边界稳定；
- CUDA 性能达到 HOME 论文结果；
- FullF+HOME-NOCM 与 HOME+HOME-NOCM 在长期非均匀流动中完全等价。

因此当前测试属于：

```text
架构正确性
+ 公式实现正确性
+ 基础数值不变量
+ 旧功能回归
```

下一层应当是物理验证，例如 Taylor–Green vortex、Poiseuille flow、shear-wave decay，再往后才是湍流和性能 benchmark。